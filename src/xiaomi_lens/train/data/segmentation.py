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

"""COCO RLE helpers for segmentation targets.

Every ``<|mask|>`` in an assistant turn is supervised by one COCO RLE object
``{"size": [height, width], "counts": ...}``. Compressed ``counts`` strings are
decoded without requiring pycocotools.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PIL import Image


def is_coco_rle(segmentation) -> bool:
    """Return whether *segmentation* is a JSON-serializable COCO RLE object."""
    return (
        isinstance(segmentation, Mapping)
        and "size" in segmentation
        and "counts" in segmentation
    )


def split_segmentation_masks(segmentation) -> list:
    """Return one RLE per assistant ``<|mask|>``, in reading order.

    Accepts ``None`` (no masks), a single RLE, or a list of RLEs.
    """
    if segmentation is None:
        return []
    if is_coco_rle(segmentation):
        return [segmentation]
    if isinstance(segmentation, (list, tuple)) and all(
        is_coco_rle(value) for value in segmentation
    ):
        return list(segmentation)
    raise ValueError(
        "`segmentation` must be a COCO RLE object or a list of COCO RLE objects "
        "with `size` and `counts`."
    )


def _decode_compressed_rle_counts(encoded: str | bytes) -> list[int]:
    """Decode the compact COCO ``counts`` string without requiring pycocotools."""
    if isinstance(encoded, bytes):
        try:
            encoded = encoded.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("COCO RLE counts bytes must be ASCII") from exc
    if not isinstance(encoded, str):
        raise TypeError("compressed COCO RLE counts must be str or bytes")

    counts: list[int] = []
    position = 0
    while position < len(encoded):
        value = 0
        shift = 0
        while True:
            code = ord(encoded[position]) - 48
            position += 1
            if code < 0 or code > 0x3F:
                raise ValueError("invalid character in compressed COCO RLE counts")
            value |= (code & 0x1F) << shift
            shift += 5
            more = bool(code & 0x20)
            if not more:
                if code & 0x10:
                    value |= -1 << shift
                break
            if position >= len(encoded):
                raise ValueError("truncated compressed COCO RLE counts")

        # COCO stores a second-order delta from the fourth run onward.
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError("compressed COCO RLE contains a negative run")
        counts.append(value)
    return counts


def decode_coco_rle(rle: Mapping) -> np.ndarray:
    """Decode compressed or uncompressed COCO RLE into an ``(H, W)`` uint8 mask."""
    if not is_coco_rle(rle):
        raise TypeError("COCO RLE must be an object containing 'size' and 'counts'")

    size = rle["size"]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("COCO RLE size must be [height, width]")
    try:
        height, width = (int(size[0]), int(size[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError("COCO RLE size values must be integers") from exc
    if height <= 0 or width <= 0:
        raise ValueError("COCO RLE size values must be positive")

    raw_counts = rle["counts"]
    if isinstance(raw_counts, (str, bytes)):
        counts = _decode_compressed_rle_counts(raw_counts)
    elif isinstance(raw_counts, (list, tuple)):
        counts = []
        for run in raw_counts:
            if isinstance(run, (bool, np.bool_)) or not isinstance(run, (int, np.integer)):
                raise ValueError("uncompressed COCO RLE counts must be non-negative integers")
            run = int(run)
            if run < 0:
                raise ValueError("uncompressed COCO RLE counts must be non-negative integers")
            counts.append(run)
    else:
        raise TypeError("COCO RLE counts must be a compressed string or integer list")

    total = height * width
    if sum(counts) != total:
        raise ValueError(
            f"COCO RLE run total {sum(counts)} does not match size product {total}"
        )

    flat = np.zeros(total, dtype=np.uint8)
    offset = 0
    for index, run in enumerate(counts):
        end = offset + run
        if index % 2:
            flat[offset:end] = 1
        offset = end
    return flat.reshape((height, width), order="F")


def segmentation_to_pixel_mask(segmentation, image_size: tuple[int, int]) -> np.ndarray:
    """Convert COCO RLE to a ``(height, width)`` uint8 mask at *image_size*.

    *image_size* is ``(width, height)`` of the processed-image canvas. RLE is
    decoded at its declared source ``size`` and nearest-neighbor resized. A
    list of RLE objects is decoded as their union.
    """
    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    pixel_mask = np.zeros((height, width), dtype=np.uint8)
    for rle in split_segmentation_masks(segmentation):
        decoded = decode_coco_rle(rle)
        if decoded.shape != (height, width):
            decoded = np.asarray(
                Image.fromarray(decoded, mode="L").resize(
                    (width, height), resample=Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
        pixel_mask |= decoded
    return pixel_mask
