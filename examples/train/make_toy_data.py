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

"""Generate a small synthetic dataset for smoke-testing the training pipeline.

Each image contains a few colored shapes on a plain background. The script
writes PNG images plus two JSONL files that cover the Stage 1 and Stage 2
sample shapes described in examples/train/README.md:

- stage1.jsonl: assistant-only rows with one `<|mask|>` each.
- stage2.jsonl: single- and multi-instance referring segmentation, a
  multi-turn conversation, box-only VQA, and a text-only row.

Masks are stored as uncompressed COCO RLE. The toy data checks that training
runs end to end; it is not meant to produce a useful model.

Usage:
    python examples/train/make_toy_data.py --output examples/train/toy --num-images 48
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLORS = {
    "red": (220, 40, 40),
    "green": (40, 170, 60),
    "blue": (40, 80, 220),
    "yellow": (230, 200, 30),
    "purple": (150, 60, 190),
}
SHAPES = ("circle", "square")
IMAGE_SIZES = ((320, 240), (256, 256), (400, 300), (240, 320))
EMPTY_THINK = "<think>\n\n</think>\n\n"


def encode_rle(mask: np.ndarray) -> dict:
    """Encode an (H, W) binary mask as uncompressed COCO RLE (column-major)."""
    flat = mask.astype(np.uint8).flatten(order="F")
    counts: list[int] = []
    current, run = 0, 0
    for value in flat:
        if value == current:
            run += 1
        else:
            counts.append(run)
            current, run = int(value), 1
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def per_mille_box(mask: np.ndarray) -> list[int]:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    return [
        round(xs.min() / width * 1000),
        round(ys.min() / height * 1000),
        round((xs.max() + 1) / width * 1000),
        round((ys.max() + 1) / height * 1000),
    ]


def grounding_block(mask: np.ndarray) -> str:
    return f"<bbox>{json.dumps(per_mille_box(mask))}</bbox><mask><|mask|></mask>"


def draw_image(rng: random.Random, size: tuple[int, int]) -> tuple[Image.Image, list[dict]]:
    """Draw 2–3 non-overlapping shapes and return the image with per-shape masks."""
    width, height = size
    image = Image.new("RGB", size, (235, 235, 230))
    draw = ImageDraw.Draw(image)
    objects: list[dict] = []
    cell_width = width // 3
    for cell in rng.sample(range(3), rng.randint(2, 3)):
        color = rng.choice(list(COLORS))
        shape = rng.choice(SHAPES)
        extent = rng.randint(min(cell_width, height) // 3, min(cell_width, height) // 2)
        x0 = cell * cell_width + rng.randint(0, cell_width - extent)
        y0 = rng.randint(0, height - extent)
        box = (x0, y0, x0 + extent, y0 + extent)

        mask_image = Image.new("L", size, 0)
        mask_draw = ImageDraw.Draw(mask_image)
        if shape == "circle":
            draw.ellipse(box, fill=COLORS[color])
            mask_draw.ellipse(box, fill=1)
        else:
            draw.rectangle(box, fill=COLORS[color])
            mask_draw.rectangle(box, fill=1)
        objects.append({"name": f"{color} {shape}", "mask": np.asarray(mask_image, dtype=np.uint8)})
    return image, objects


def stage1_row(image_name: str, obj: dict) -> dict:
    content = (
        f'{EMPTY_THINK}The mask of all regions matching the description "{obj["name"]}" '
        "in <image> is <|mask|>."
    )
    return {
        "images": [image_name],
        "messages": [{"role": "assistant", "content": content}],
        "segmentation": [encode_rle(obj["mask"])],
    }


def stage2_rows(rng: random.Random, image_name: str, objects: list[dict]) -> list[dict]:
    first, second = objects[0], objects[1]
    rows = [
        {
            "images": [image_name],
            "messages": [
                {
                    "role": "user",
                    "content": f'<image>\nLocate the instance that matches the description "{first["name"]}", report bbox coordinates and masks.',
                },
                {"role": "assistant", "content": EMPTY_THINK + grounding_block(first["mask"])},
            ],
            "segmentation": [encode_rle(first["mask"])],
        },
        {
            "images": [image_name],
            "messages": [
                {
                    "role": "user",
                    "content": "<image>\nLocate every shape in the image, report bbox coordinates and masks.",
                },
                {
                    "role": "assistant",
                    "content": f"<think>\nI see {len(objects)} shapes.\n</think>\n\n"
                    + "".join(grounding_block(obj["mask"]) for obj in objects),
                },
            ],
            "segmentation": [encode_rle(obj["mask"]) for obj in objects],
        },
        {
            "images": [image_name],
            "messages": [
                {
                    "role": "user",
                    "content": f'<image>\nLocate the instance that matches the description "{first["name"]}", report bbox coordinates and masks.',
                },
                {"role": "assistant", "content": EMPTY_THINK + grounding_block(first["mask"])},
                {"role": "user", "content": f'Now the {second["name"]}.'},
                {"role": "assistant", "content": EMPTY_THINK + grounding_block(second["mask"])},
            ],
            "segmentation": [encode_rle(first["mask"]), encode_rle(second["mask"])],
        },
        {
            "images": [image_name],
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>\nWhat is at <bbox>{json.dumps(per_mille_box(first['mask']))}</bbox>?",
                },
                {"role": "assistant", "content": f"{EMPTY_THINK}A {first['name']}."},
            ],
        },
    ]
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    rows.append({
        "messages": [
            {"role": "user", "content": f"What is {a} + {b}?"},
            {"role": "assistant", "content": f"{EMPTY_THINK}{a + b}"},
        ],
    })
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("examples/train/toy"))
    parser.add_argument("--num-images", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    stage1: list[dict] = []
    stage2: list[dict] = []
    for index in range(args.num_images):
        image, objects = draw_image(rng, IMAGE_SIZES[index % len(IMAGE_SIZES)])
        image_name = f"{index:04d}.png"
        image.save(image_dir / image_name)
        stage1.extend(stage1_row(image_name, obj) for obj in objects)
        stage2.extend(stage2_rows(rng, image_name, objects))

    write_jsonl(args.output / "stage1.jsonl", stage1)
    write_jsonl(args.output / "stage2.jsonl", stage2)
    print(f"Wrote {len(stage1)} Stage 1 rows and {len(stage2)} Stage 2 rows to {args.output}")


if __name__ == "__main__":
    main()
