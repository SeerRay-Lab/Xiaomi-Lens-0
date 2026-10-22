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

"""Per-sample preprocess for the lazy data pipeline.

`encode_sample` produces the per-sample tensor dict consumed by
`LazyCollator` inside DataLoader workers:

1. Normalize messages via the render helpers (`ensure_user_message`,
   `align_image_placeholders`, `render_prompt_and_completion`).
2. Decode the real images and run them through the Qwen processor so
   `pixel_values`, `image_grid_thw`, and `mm_token_type_ids` come from
   the processor's own code path.
3. Right-truncate `[prompt, completion]` to `model_max_length`.
4. Scan the un-truncated token stream for `<|mask|>` positions and decode
   each mask annotation onto the full-resolution canvas, stored bit-packed via
   `np.packbits`.

LM loss covers EVERY assistant turn (image and pure-text alike), and the
`<|mask|>` scan spans every assistant turn too — both derive from the
ChatML spans returned by `_assistant_spans`.

Each `<|mask|>` binds to the NEAREST PRECEDING image (`mask_image_index`), so
its target is rasterized at THAT image's canvas (`mask_grid_hw`), not the
sample's first image. Single-image samples bind to image 0 either way.

At batch time `XiaomiLensTrainer.compute_loss` inflates the bit-packed masks
on GPU (single packed H2D + shift unpack, equivalent to
`np.unpackbits(..., count=h*w)`) into `(canvas_h, canvas_w)` targets, where
canvas == (grid_h*patch_size, grid_w*patch_size). `_unpack_masks` below is
the CPU reference implementation of that inverse, kept for debugging.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import PIL.Image

logger = logging.getLogger(__name__)

# Per-worker bounded counter so a systematically unrenderable source surfaces
# its real exception instead of vanishing behind `return None`. Capped to avoid
# log spam.
_render_warn_budget = 20

# Same idea for `<|mask|>` tokens found outside any assistant turn — a data bug
# worth surfacing once, not once per sample.
_stray_mask_warn_budget = 20

# And for samples so long that right-truncation would split an image apart. The
# image is dropped rather than half-fed; frequent hits mean --max_pixels /
# --model_max_length are mismatched to the corpus, which is worth seeing.
_truncated_image_warn_budget = 20


def encode_sample(
    sample: dict,
    *,
    processor,
    mask_token_id: int,
    model_max_length: int,
    max_pixels: Optional[int] = None,
    min_pixels: Optional[int] = None,
    image_token_reserve: int = 1024,
) -> Optional[dict]:
    from .render import (
        ensure_user_message,
        align_image_placeholders,
        render_prompt_and_completion,
    )

    image_paths = list(sample.get("images") or [])
    sample_id = sample.get("_dataset_name")

    messages = ensure_user_message(
        sample["messages"], len(image_paths), sample_id=sample_id,
    )
    messages = align_image_placeholders(
        messages, len(image_paths), sample_id=sample_id,
    )

    pil_images: list[PIL.Image.Image] = []
    for p in image_paths:
        try:
            with PIL.Image.open(p) as im:
                img = im.convert("RGB")
                img.load()
                pil_images.append(img)
        except (FileNotFoundError, PIL.UnidentifiedImageError):
            return None

    try:
        prompt_text, completion_text = render_prompt_and_completion(
            processor, messages,
            vision_start=processor.vision_start_token,
            image_token=processor.image_token,
            vision_end=processor.vision_end_token,
        )
    except Exception as exc:
        global _render_warn_budget
        if _render_warn_budget > 0:
            _render_warn_budget -= 1
            logger.warning(
                "[encode_sample] render failed, dropping sample "
                "(dataset=%r images=%d messages=%d): %s: %s",
                sample_id, len(image_paths), len(sample.get("messages") or []),
                type(exc).__name__, exc,
            )
        return None

    # `max_pixels` is a PER-IMAGE ceiling, so a multi-image sample multiplies it:
    # at 4194304 px each image costs up to 4194304/(16²·2²) = 4096 tokens, and two
    # of them alone fill an 8192-token context.
    #
    # So divide an image-token BUDGET among the sample's images instead. A
    # single-image sample keeps the full `max_pixels` (budget ≥ one image's cap);
    # only multi-image samples shrink, and just enough to fit.
    image_processor_kwargs = {}
    if min_pixels is not None:
        image_processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        effective_max_pixels = max_pixels
        if pil_images:
            px_per_token = _pixels_per_token(processor)
            budget = max(int(model_max_length) - int(image_token_reserve), 1)
            per_image_cap = (budget // len(pil_images)) * px_per_token
            effective_max_pixels = min(max_pixels, per_image_cap)
            if min_pixels is not None:
                effective_max_pixels = max(effective_max_pixels, min_pixels)
        image_processor_kwargs["max_pixels"] = effective_max_pixels

    if pil_images:
        prompt_inputs = processor(
            text=prompt_text, images=pil_images,
            return_tensors="np", **image_processor_kwargs,
        )
        image_grid_thw = np.asarray(prompt_inputs["image_grid_thw"], dtype=np.int64)
        prompt_mm = np.asarray(prompt_inputs["mm_token_type_ids"][0], dtype=np.int64)
        pixel_values = np.asarray(prompt_inputs["pixel_values"])
    else:
        prompt_inputs = processor.tokenizer(prompt_text, return_tensors="np")
        image_grid_thw = np.zeros((0, 3), dtype=np.int64)
        prompt_mm = np.zeros(len(prompt_inputs["input_ids"][0]), dtype=np.int64)
        pixel_values = None

    prompt_ids = np.asarray(prompt_inputs["input_ids"][0], dtype=np.int64)
    prompt_mask = np.asarray(prompt_inputs["attention_mask"][0], dtype=np.int64)
    prompt_length = int(prompt_ids.shape[0])

    completion_inputs = processor.tokenizer(completion_text, return_tensors="np")
    completion_ids = np.asarray(completion_inputs["input_ids"][0], dtype=np.int64)
    completion_mask = np.asarray(completion_inputs["attention_mask"][0], dtype=np.int64)

    input_ids = np.concatenate([prompt_ids, completion_ids])
    attention_mask = np.concatenate([prompt_mask, completion_mask])
    mm_token_type_ids = np.concatenate([
        prompt_mm,
        np.zeros(completion_ids.shape[0], dtype=np.int64),
    ])
    # Every assistant turn carries LM loss — image and pure-text alike.
    #
    # The prompt/completion split gives the baseline (last assistant turn
    # supervised, everything before it masked); earlier assistant turns are
    # then OR-ed back in from the ChatML spans. No span detected at all
    # (non-ChatML template) → the baseline stands untouched.
    #
    # `gen_prefix` is whatever the template auto-emits after the `assistant\n`
    # header — on Qwen3.5 the `<think>\n` opener. `add_generation_prompt=True`
    # hands it to the model for free at inference, so the model never generates
    # it and it must NOT be supervised. Read it straight off THIS sample's own
    # render (the last span's body start → where the prompt ends), so the rule
    # is exact under any template and empty under templates that emit nothing.
    #
    # Skip it only where a span ACTUALLY starts with it: Qwen3.5's template
    # strips `<think>…</think>` from history turns and keeps it only on the
    # last one, so earlier spans usually have no such prefix — blindly dropping
    # N tokens there would eat real content.
    assistant_spans = _assistant_spans(input_ids, processor.tokenizer)
    gen_prefix = (
        input_ids[assistant_spans[-1][0]:prompt_length].tolist()
        if assistant_spans and prompt_length > assistant_spans[-1][0]
        else []
    )
    g = len(gen_prefix)
    labels = input_ids.copy()
    labels[:prompt_length] = -100
    for span_start, span_end in assistant_spans:
        body_start = span_start
        if g and span_start + g <= span_end \
                and input_ids[span_start:span_start + g].tolist() == gen_prefix:
            body_start = span_start + g
        labels[body_start:span_end] = input_ids[body_start:span_end]
    labels[attention_mask == 0] = -100

    # Keep the pre-truncation stream: `mask_positions` must be scanned on it,
    # so a `<|mask|>` that right-truncation drops still occupies its slot k and
    # `segmentation[k]` stays aligned (the slot is then marked invalid below).
    full_input_ids = input_ids

    if input_ids.shape[0] > model_max_length:
        # An image is INDIVISIBLE. The model asserts `#<|image_pad|> == #features`
        # (`get_placeholder_mask`), so cutting a placeholder run in half while the
        # ViT still encodes the whole image raises "Image features and image tokens
        # do not match". Pull the cut back to the `<|vision_start|>` of the first
        # image that would not survive whole, then drop that image and every one
        # after it from `pixel_values` / `image_grid_thw` as well.
        end, n_keep = _truncate_keeping_whole_images(
            full_input_ids,
            model_max_length=model_max_length,
            image_grid_thw=image_grid_thw,
            processor=processor,
            sample_id=sample_id,
        )
        if n_keep < int(image_grid_thw.shape[0]):
            # pixel_values is (total_patches, dim) with t*h*w PRE-merge patches
            # per image, laid out image-major — so the kept prefix is contiguous.
            patch_rows = int(np.prod(image_grid_thw[:n_keep], axis=1).sum()) if n_keep else 0
            pixel_values = pixel_values[:patch_rows] if patch_rows else None
            image_grid_thw = image_grid_thw[:n_keep]

        input_ids = input_ids[:end]
        attention_mask = attention_mask[:end]
        labels = labels[:end]
        mm_token_type_ids = mm_token_type_ids[:end]

    # `mask_positions` index into the un-truncated token stream; right-
    # truncation may drop a position past `input_ids.shape[0]`, in which
    # case `mask_valid_mask[k]=0` and compute_loss skips that slot.
    #
    # Scan every assistant turn, in reading order, matching the documented
    # contract `segmentation[k] = annotation for the k-th <|mask|> in messages`.
    # The scan is CONFINED to assistant spans: a `<|mask|>` echoed into a
    # user/system turn is not something the model emits, and counting it
    # would shift every later `segmentation[k]` onto the wrong token.
    if assistant_spans:
        in_assistant = np.zeros(full_input_ids.shape[0], dtype=bool)
        for span_start, span_end in assistant_spans:
            in_assistant[span_start:span_end] = True
        is_mask_tok = full_input_ids == mask_token_id
        mask_positions = np.where(is_mask_tok & in_assistant)[0].astype(np.int64)
        stray = int(np.count_nonzero(is_mask_tok & ~in_assistant))
        if stray:
            global _stray_mask_warn_budget
            if _stray_mask_warn_budget > 0:
                _stray_mask_warn_budget -= 1
                logger.warning(
                    "[encode_sample] ignoring %d <|mask|> token(s) outside any "
                    "assistant turn (dataset=%r); segmentation[k] pairs only with "
                    "assistant-emitted masks.",
                    stray, sample_id,
                )
    else:
        # No ChatML span detected — fall back to the completion-only scan,
        # offset into `[prompt_ids ++ completion_ids]` coordinates.
        completion_mask_offsets = np.where(completion_ids == mask_token_id)[0]
        mask_positions = (completion_mask_offsets + prompt_length).astype(np.int64)
    mask_valid_mask = (mask_positions < input_ids.shape[0]).astype(np.float32)

    # Each `<|mask|>` binds to the NEAREST PRECEDING image: count the
    # `<|vision_start|>` markers before it. `align_image_placeholders` keeps
    # `#<|vision_start|> == len(images)`, so the running count indexes `images`
    # directly. A `<|mask|>` with no image before it (count 0 → -1) falls back
    # to image 0. Counted on `full_input_ids` because `mask_positions` index the
    # un-truncated stream; for any slot that survives truncation the prefix is
    # identical either way, so the model can re-derive the same value at
    # inference from its own `input_ids` (see `_decode_pixel_masks`).
    num_images = int(image_grid_thw.shape[0])
    vision_start_id = processor.tokenizer.convert_tokens_to_ids(
        processor.vision_start_token
    )
    if num_images > 0 and vision_start_id is not None and mask_positions.shape[0] > 0:
        seen_images = np.cumsum(full_input_ids == vision_start_id)
        mask_image_index = np.clip(
            seen_images[mask_positions] - 1, 0, num_images - 1,
        ).astype(np.int64)
    else:
        mask_image_index = np.zeros(mask_positions.shape[0], dtype=np.int64)
    # Grid of each mask's OWN image — the canvas its target is rasterized at and
    # the resolution the decoder will emit it at.
    mask_grid_hw = [
        (int(image_grid_thw[i, 1]), int(image_grid_thw[i, 2])) if num_images > 0
        else (0, 0)
        for i in mask_image_index
    ]

    mask_pixel_masks: list[bytes] = []
    if pil_images and sample.get("segmentation") is not None and mask_positions.shape[0] > 0:
        from .segmentation import segmentation_to_pixel_mask, split_segmentation_masks
        per_mask = split_segmentation_masks(sample["segmentation"])
        ps = (
            processor.image_processor.patch_size[0]
            if isinstance(processor.image_processor.patch_size, (list, tuple))
            else processor.image_processor.patch_size
        )
        # Targets are rasterized at full canvas (grid*patch_size), the
        # resolution PixelMaskDecoder emits inside `_decode_pixel_masks`.
        if int(ps) != 16:
            raise ValueError(
                f"Xiaomi-Lens-0 expects image_processor.patch_size=16 but got {ps}; "
                "the full-canvas rasterizer and `MASK_CANVAS_SCALE=16` in "
                "`model/modeling_xiaomi_lens.py` both assume 16."
            )
        # Canvas is PER MASK — the processed size of the image that mask binds
        # to, not the sample's first image. RLE is decoded at its declared
        # source size and nearest-neighbor resized to this canvas.
        for k in range(mask_positions.shape[0]):
            mask_annotation = per_mask[k] if k < len(per_mask) else None
            if not mask_valid_mask[k] or not mask_annotation:
                mask_pixel_masks.append(b"")
                continue
            image_index = int(mask_image_index[k])
            source_width, source_height = pil_images[image_index].size
            rle_size = tuple(int(value) for value in mask_annotation["size"])
            if rle_size != (source_height, source_width):
                raise ValueError(
                    f"segmentation[{k}] RLE size={list(rle_size)} does not "
                    f"match bound image size={[source_height, source_width]}"
                )
            grid_h, grid_w = mask_grid_hw[k]
            canvas = (grid_w * ps, grid_h * ps)
            pixel_mask = segmentation_to_pixel_mask(mask_annotation, canvas)
            mask_pixel_masks.append(np.packbits(pixel_mask.astype(bool).flatten()).tobytes())

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "mm_token_type_ids": mm_token_type_ids,
        "image_grid_thw": image_grid_thw,
        "mask_positions": mask_positions,
        "mask_valid_mask": mask_valid_mask,
        "mask_image_index": mask_image_index,
        "mask_grid_hw": mask_grid_hw,
        "mask_pixel_masks": mask_pixel_masks,
    }
    if pixel_values is not None:
        out["pixel_values"] = pixel_values
    return out


def _pixels_per_token(processor) -> int:
    """How many source pixels one `<|image_pad|>` token stands for.

    A patch is `patch_size²` pixels and `merge_size²` patches collapse into one
    token, so the token covers `(patch_size · merge_size)²` pixels — 1024 on the
    Qwen family (16 · 2)². Dividing a pixel budget by this yields a token count,
    and multiplying a token budget by it yields the `max_pixels` to ask for.
    """
    ps = processor.image_processor.patch_size
    if isinstance(ps, (list, tuple)):
        ps = ps[0]
    merge = getattr(processor.image_processor, "merge_size", 2) or 2
    return int(ps) * int(ps) * int(merge) * int(merge)


def _truncate_keeping_whole_images(
    full_input_ids: np.ndarray,
    *,
    model_max_length: int,
    image_grid_thw: np.ndarray,
    processor,
    sample_id,
) -> tuple[int, int]:
    """Pick a right-truncation point that never splits an image.

    Returns `(end, n_keep)`: cut `input_ids[:end]`, keep `image_grid_thw[:n_keep]`.

    Each image occupies a contiguous run of `<|image_pad|>` tokens whose length is
    `t*h*w / merge_size**2` (the grid AFTER spatial merging). A naive
    `input_ids[:model_max_length]` can land inside such a run, leaving fewer
    placeholders than the ViT emits features — which the model rejects outright.
    So: keep every image whose run ends before the limit, and pull `end` back to
    the `<|vision_start|>` of the first image that would be split.
    """
    limit = int(model_max_length)
    num_images = int(image_grid_thw.shape[0])
    if num_images == 0:
        return limit, 0

    image_pad_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    vision_start_id = processor.tokenizer.convert_tokens_to_ids(
        processor.vision_start_token
    )
    if image_pad_id is None or vision_start_id is None:
        return limit, num_images

    merge = getattr(processor.image_processor, "merge_size", 2) or 2
    pads_per_image = np.prod(image_grid_thw, axis=1) // (merge * merge)
    cum_pads = np.cumsum(pads_per_image)

    pad_positions = np.where(full_input_ids == image_pad_id)[0]
    if pad_positions.shape[0] != int(cum_pads[-1]):
        # Placeholder count already disagrees with the grid — nothing sane to do
        # here; leave it to the model's own assertion rather than guess.
        return limit, num_images

    kept_pads = int(np.count_nonzero(pad_positions < limit))
    n_keep = int(np.count_nonzero(cum_pads <= kept_pads))
    if n_keep == num_images:
        return limit, num_images  # cut lands after the last image: nothing to do

    # Image `n_keep` is the first one that would be split (or fall outside).
    # Drop it and everything after: cut at its `<|vision_start|>`.
    first_pad_of_split = int(pad_positions[int(cum_pads[n_keep - 1]) if n_keep else 0])
    vision_starts = np.where(full_input_ids[:first_pad_of_split] == vision_start_id)[0]
    end = int(vision_starts[-1]) if vision_starts.shape[0] else first_pad_of_split
    end = min(end, limit)

    global _truncated_image_warn_budget
    if _truncated_image_warn_budget > 0:
        _truncated_image_warn_budget -= 1
        logger.warning(
            "[encode_sample] sample is %d tokens > model_max_length=%d; dropping "
            "%d of %d image(s) so no placeholder run is split (dataset=%r). "
            "Lower --max_pixels or raise --model_max_length if this is frequent.",
            int(full_input_ids.shape[0]), limit, num_images - n_keep, num_images,
            sample_id,
        )
    return end, n_keep


def _assistant_spans(input_ids: np.ndarray, tokenizer) -> list[tuple[int, int]]:
    """Locate every assistant turn's body span in a ChatML token stream.

    Scans for `<|im_start|>assistant\\n … <|im_end|>` and returns
    `[(start, end), ...]` in ascending order, where `start` is the first
    body token (right after the `assistant\\n` header) and `end` is
    exclusive and INCLUDES the closing `<|im_end|>` — supervising it is what
    teaches the model to stop. A turn whose `<|im_end|>` was cut off the tail
    ends at `len(input_ids)`.

    Two callers share this one scan:
    - LM labels: every returned span is supervised, so multi-turn samples
      train all assistant replies (ms-swift / LLaMA-Factory convention),
      not just the last one.
    - `<|mask|>` positions: only tokens inside these spans count, so a
      `<|mask|>` echoed into a user/system turn can't shift the positional
      `segmentation[k]` pairing.

    Returns `[]` if no assistant span is found (non-ChatML template, missing
    special tokens); callers must keep their prompt/completion fallback.
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    hdr = tokenizer.encode("assistant\n", add_special_tokens=False)
    spans: list[tuple[int, int]] = []
    if im_start is None or im_end is None or not hdr:
        return spans
    n = input_ids.shape[0]
    h = len(hdr)
    ids = input_ids.tolist()
    i = 0
    while i < n:
        if ids[i] == im_start and i + 1 + h <= n and ids[i + 1:i + 1 + h] == hdr:
            start = i + 1 + h
            j = start
            while j < n and ids[j] != im_end:
                j += 1
            end = j + 1 if j < n else n
            spans.append((start, end))
            i = end
        else:
            i += 1
    return spans


def _unpack_masks(packed: list[bytes], shapes: list[tuple[int, int]]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for blob, (h, w) in zip(packed, shapes):
        if not blob:
            out.append(np.zeros((h, w), dtype=bool))
        else:
            arr = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), count=h * w)
            out.append(arr.reshape(h, w).astype(bool))
    return out
