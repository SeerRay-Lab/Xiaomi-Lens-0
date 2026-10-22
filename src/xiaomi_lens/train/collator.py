# Copyright 2026 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime collator for the lazy JSONL pipeline.

This mirrors ms-swift's large-dataset behavior: keep startup cheap and do
tokenization/image preprocessing inside DataLoader workers while training
is running. It reuses `encode_sample` so prompt rendering, truncation, and
mask target construction have one implementation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from .data.preprocess import encode_sample

logger = logging.getLogger(__name__)


class LazyCollator:
    def __init__(
        self,
        processor,
        *,
        mask_token_id: int,
        model_max_length: int,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
    ) -> None:
        self.processor = processor
        self.mask_token_id = mask_token_id
        self.model_max_length = model_max_length
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self._drop_warnings = 0
        self._dummy_warnings = 0

    def _fallback_token_id(self) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
            value = getattr(tokenizer, attr, None)
            if value is not None:
                return int(value)
        return 0

    @staticmethod
    def _sample_tag(row: dict) -> str:
        return (
            f"dataset={row.get('_dataset_name', '?')!r} "
            f"tag={row.get('_dataset_tag', '')!r} "
            f"images={len(row.get('images') or [])} "
            f"messages={len(row.get('messages') or [])}"
        )

    def _dummy_batch(self, batch_size: int) -> dict:
        """Return a graph-safe zero-loss batch when every sampled row fails.

        The trainer/model handle K=0 and no images by running the LM path
        with all labels ignored plus the rank-symmetric dummy mask path.
        """
        n = max(1, int(batch_size))
        input_ids = torch.full((n, 1), self._fallback_token_id(), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones((n, 1), dtype=torch.long),
            "labels": torch.full((n, 1), -100, dtype=torch.long),
            "mm_token_type_ids": torch.zeros((n, 1), dtype=torch.long),
            "_per_sample_mask_positions": [[] for _ in range(n)],
            "_per_sample_mask_valid_mask": [[] for _ in range(n)],
            "_per_sample_mask_pixel_masks": [[] for _ in range(n)],
            "_per_sample_mask_image_index": [[] for _ in range(n)],
            "_per_sample_mask_grid_hw": [[] for _ in range(n)],
        }

    def __call__(self, features: list[dict]) -> dict:
        encoded: list[dict] = []
        for row in features:
            tag = self._sample_tag(row)
            reported_drop = False
            try:
                out = encode_sample(
                    row,
                    processor=self.processor,
                    mask_token_id=self.mask_token_id,
                    model_max_length=self.model_max_length,
                    max_pixels=self.max_pixels,
                    min_pixels=self.min_pixels,
                )
            except Exception as exc:
                out = None
                if self._drop_warnings < 10:
                    logger.warning(
                        "[lazy-collator] dropped sample during encode (%s): %s",
                        tag, exc,
                    )
                    self._drop_warnings += 1
                    reported_drop = True
            if out is None and not reported_drop and self._drop_warnings < 10:
                logger.warning("[lazy-collator] dropped sample during encode (%s)", tag)
                self._drop_warnings += 1
            if out is not None:
                encoded.append(out)

        if not encoded:
            if self._dummy_warnings < 10:
                logger.warning(
                    "[lazy-collator] every sample in a batch failed; "
                    "returning a zero-loss dummy batch of size %d.",
                    len(features),
                )
                self._dummy_warnings += 1
            return self._dummy_batch(len(features))

        max_len = max(len(r["input_ids"]) for r in encoded)

        def _pad(seq, value, length):
            return list(seq) + [value] * (length - len(seq))

        input_ids = torch.tensor(
            [_pad(r["input_ids"], 0, max_len) for r in encoded],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [_pad(r["attention_mask"], 0, max_len) for r in encoded],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [_pad(r["labels"], -100, max_len) for r in encoded],
            dtype=torch.long,
        )
        mm_token_type_ids = torch.tensor(
            [_pad(r["mm_token_type_ids"], 0, max_len) for r in encoded],
            dtype=torch.long,
        )

        pixel_chunks = [
            torch.as_tensor(r["pixel_values"])
            for r in encoded
            if "pixel_values" in r and np.asarray(r["pixel_values"]).size > 0
        ]
        if pixel_chunks:
            pixel_values = torch.cat(pixel_chunks, dim=0)
            image_grid_thw = torch.cat(
                [
                    torch.as_tensor(r["image_grid_thw"], dtype=torch.long)
                    for r in encoded
                    if len(r["image_grid_thw"]) > 0
                ],
                dim=0,
            )
        else:
            pixel_values = None
            image_grid_thw = None

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "mm_token_type_ids": mm_token_type_ids,
            "_per_sample_mask_positions": [
                list(r["mask_positions"]) for r in encoded
            ],
            "_per_sample_mask_valid_mask": [
                list(r["mask_valid_mask"]) for r in encoded
            ],
            "_per_sample_mask_pixel_masks": [
                r["mask_pixel_masks"] for r in encoded
            ],
            # Per-MASK, not per-sample: each `<|mask|>` binds to its nearest
            # preceding image, so its target canvas is that image's grid.
            "_per_sample_mask_image_index": [
                list(r["mask_image_index"]) for r in encoded
            ],
            "_per_sample_mask_grid_hw": [
                list(r["mask_grid_hw"]) for r in encoded
            ],
        }
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw
        return batch
