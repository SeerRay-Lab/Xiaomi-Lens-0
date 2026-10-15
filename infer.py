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

"""Run Xiaomi-Lens-0 inference and save pixel-mask overlays.

Pass a complete instruction with --text. The checkpoint's chat template places
the image before the text; no <image> placeholder is needed in the instruction.
Grounding prompts should request bbox coordinates and masks, as shown below.

Inference first generates text, then replays the full sequence to decode each
<|mask|> token into a pixel mask. Only final-answer masks are visualized; masks
inside <think> are excluded. Overlays use the vision encoder's input resolution.
The parsing, decoding, and visualization functions are also used by vLLM serving.

Examples:
    # Locate instances matching a referring expression.
    python infer.py \\
        --checkpoint SeerRay-Lab/Xiaomi-Lens-0 \\
        --image /path/to/image.jpg \\
        --text 'Locate all instances in the image that match the description "the cat on the left". Report bbox coordinates and masks.' \\
        --output result.jpg --draw-bbox

    # Locate all people and save one overlay per predicted instance.
    python infer.py \\
        --checkpoint SeerRay-Lab/Xiaomi-Lens-0 \\
        --image /path/to/image.jpg \\
        --text 'Locate all instances in the image that match the description "person". Report bbox coordinates and masks.' \\
        --output result.jpg --draw-bbox --per-mask-output

By default, all final-answer masks share one overlay with distinct colors.
--per-mask-output saves result_mask_0.jpg, result_mask_1.jpg, etc.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Make the src-layout package importable without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw
from transformers import AutoProcessor

from xiaomi_lens.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    DEFAULT_MAX_NEW_TOKENS,
    MASK_TOKEN,
)

# Import the HF model only when loading a checkpoint. Serving also uses the
# parsing and visualization helpers without loading the HF model class.
if TYPE_CHECKING:
    from xiaomi_lens.model.modeling_xiaomi_lens import XiaomiLensQwen3_5ForConditionalGeneration

# Overlay colors repeat after eight masks.
DEFAULT_MASK_PALETTE: tuple[str, ...] = (
    "#ff4444",  # red
    "#44ff44",  # green
    "#4488ff",  # blue
    "#ffaa00",  # orange
    "#aa44ff",  # purple
    "#00cccc",  # cyan
    "#ff44aa",  # pink
    "#aaff44",  # yellow-green
)

_BOX_RE = re.compile(
    rf"{re.escape(BOX_START_TOKEN)}\s*\[\s*(\d+)\s*,\s*(\d+)\s*,"
    rf"\s*(\d+)\s*,\s*(\d+)\s*\]\s*{re.escape(BOX_END_TOKEN)}"
)


def parse_bboxes(generated_text: str) -> list[list[int]]:
    """Extract all <bbox>[x1, y1, x2, y2]</bbox> boxes in text order.

    Coordinates are normalized to 0–1000. This scans both reasoning and answer
    text; use :func:`parse_answer_bboxes` for final-answer boxes only.
    """
    return [[int(v) for v in m.groups()] for m in _BOX_RE.finditer(generated_text)]


def split_answer_content(generated_text: str) -> tuple[str, int]:
    """Return ``(answer_text, thought_mask_count)`` for a generated completion.

    Split at the first ``</think>`` and count the auxiliary ``<|mask|>`` tokens
    before it so callers can exclude reasoning masks from the visualization.
    Completions without ``</think>`` are treated as all answer.
    """
    thought, separator, answer = generated_text.partition("</think>")
    if not separator:
        return generated_text, 0
    return answer, thought.count(MASK_TOKEN)


def parse_answer_bboxes(generated_text: str) -> list[list[int]]:
    """Extract final-answer boxes, excluding a preceding thought block."""
    answer, _ = split_answer_content(generated_text)
    return parse_bboxes(answer)


def load_model_and_processor(
    checkpoint: Path,
    device: torch.device,
) -> tuple[XiaomiLensQwen3_5ForConditionalGeneration, AutoProcessor, int]:
    """Load the checkpoint and processor, and resolve the ``<|mask|>`` token ID."""
    from xiaomi_lens.model.modeling_xiaomi_lens import XiaomiLensQwen3_5ForConditionalGeneration

    print(f"Loading checkpoint: {checkpoint}")
    model = XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=str(device),
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(str(checkpoint))
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [MASK_TOKEN]}
    )
    model.resize_token_embeddings(len(processor.tokenizer))

    new_vocab_size = len(processor.tokenizer)
    if hasattr(model.config, "text_config") and model.config.text_config.vocab_size != new_vocab_size:
        model.config.text_config.vocab_size = new_vocab_size

    mask_token_id = processor.tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    print(f"<|mask|> token id: {mask_token_id}")
    return model, processor, mask_token_id


def build_inputs(
    processor: AutoProcessor,
    image: Image.Image,
    text: str,
    device: torch.device,
) -> tuple[dict, str]:
    """Apply the checkpoint's chat template to an image and full instruction.

    ``text`` is passed through unchanged. The template supplies image tokens and
    the assistant generation prefix. Returns ``(inputs, prompt_text)``.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    processor.tokenizer.padding_side = "left"
    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs, prompt_text


def generate_completion(
    model: XiaomiLensQwen3_5ForConditionalGeneration,
    processor: AutoProcessor,
    prompt_inputs: dict,
    max_new_tokens: int,
    mask_token_id: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> tuple[torch.Tensor, str, int]:
    """Decode the assistant completion (greedy by default, or sampled).

    Returns `(completion_ids, decoded_text, mask_count)`:
    - `completion_ids`: (1, completion_len) — only the new tokens.
    - `mask_count`: total `<|mask|>` count, including reasoning and answer masks.
      A count of zero means there are no masks to decode.
    """
    prompt_length = prompt_inputs["input_ids"].shape[1]

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k and top_k > 0:
            gen_kwargs["top_k"] = top_k

    with torch.inference_mode():
        output_ids = model.generate(**prompt_inputs, **gen_kwargs)

    completion_ids = output_ids[:, prompt_length:]
    mask_count = int((completion_ids == mask_token_id).sum(dim=1)[0].item())

    generated_text = processor.tokenizer.decode(completion_ids[0], skip_special_tokens=False)
    return completion_ids, generated_text, mask_count


def run_forward(
    model: XiaomiLensQwen3_5ForConditionalGeneration,
    prompt_inputs: dict,
    completion_ids: torch.Tensor,
    mask_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Run a second forward pass with the full sequence to extract pred_masks.

    Decode every `<|mask|>` token in completion order, including reasoning masks.
    Callers select final-answer masks using :func:`split_answer_content`.

    Returns pred_masks: (1, K, grid_h*16, grid_w*16) — raw logits at canvas
    (full ViT input pixel) resolution, before sigmoid.
    """
    prompt_ids = prompt_inputs["input_ids"]
    prompt_mask = prompt_inputs["attention_mask"]
    prompt_length = prompt_ids.shape[1]

    completion_mask = torch.ones_like(completion_ids)
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    mm_token_type_ids = prompt_inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        completion_mm = torch.zeros(
            completion_ids.shape,
            dtype=mm_token_type_ids.dtype,
            device=device,
        )
        mm_token_type_ids = torch.cat([mm_token_type_ids, completion_mm], dim=1)

    completion_mask_indices = (
        (completion_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
    )
    if completion_mask_indices.numel() == 0:
        raise RuntimeError(
            "run_forward called but completion contains no <|mask|> tokens "
            "(K=0 should be filtered by the caller before reaching here)."
        )
    mask_positions = (completion_mask_indices + prompt_length).unsqueeze(0)  # (1, K)

    with torch.inference_mode():
        model_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=prompt_inputs["pixel_values"],
            image_grid_thw=prompt_inputs["image_grid_thw"],
            mm_token_type_ids=mm_token_type_ids,
            mask_positions=mask_positions,
        )

    pred_masks = model_output.get("pred_masks")
    if pred_masks is None:
        raise RuntimeError("model.forward() returned no pred_masks. Check that mask_positions is correct.")
    return pred_masks  # (1, K, grid_h*16, grid_w*16)


def visualize_pixel_mask(
    image: Image.Image,
    pred_masks: torch.Tensor,
    image_grid_thw: torch.Tensor,
    patch_size: int = 16,
    merge_size: int = 2,
    threshold: float = 0.5,
    palette: tuple[str, ...] = DEFAULT_MASK_PALETTE,
    alpha: float = 0.5,
    bboxes: list[list[int]] | None = None,
    draw_bbox: bool = False,
) -> Image.Image:
    """Resize image to model canvas and composite K mask overlays.

    ``pred_masks`` contains pixel logits at canvas resolution
    (``raw_h*patch_size x raw_w*patch_size``). Resize the image to that resolution
    before compositing. ``merge_size`` is retained for compatibility and unused.

    Supports a single image with masks shaped ``(1, K, H, W)`` and the canvas
    defined by ``image_grid_thw[0]``.
    """
    del merge_size
    thw = image_grid_thw[0]
    raw_h = int(thw[1].item())
    raw_w = int(thw[2].item())
    pixel_h = raw_h * patch_size
    pixel_w = raw_w * patch_size

    if pred_masks.dim() != 4 or pred_masks.shape[0] != 1:
        raise ValueError(
            f"pred_masks expected shape (1, K, H, W), got {tuple(pred_masks.shape)}"
        )
    K = pred_masks.shape[1]
    mh, mw = pred_masks.shape[2], pred_masks.shape[3]
    if (mh, mw) != (pixel_h, pixel_w):
        raise ValueError(
            f"pred_masks spatial dims {(mh, mw)} != canvas {(pixel_h, pixel_w)}; "
            "expected canvas-resolution masks from `_decode_pixel_masks`."
        )

    print(f"Image grid: {raw_h}x{raw_w} raw patches; canvas {pixel_w}x{pixel_h}px")
    print(f"K = {K} <|mask|> mask(s)")

    canvas = image.resize((pixel_w, pixel_h), Image.LANCZOS).convert("RGBA")
    fill_alpha = int(alpha * 255)
    total = pixel_h * pixel_w

    for k in range(K):
        probs = torch.sigmoid(pred_masks[0, k]).float().cpu().numpy()  # (H, W)
        fg_color = ImageColor.getrgb(palette[k % len(palette)])
        if threshold > 0:
            binary = probs >= threshold
            a = binary.astype("uint8") * fill_alpha
            fg = int(binary.sum())
        else:
            a = (probs * fill_alpha).astype("uint8")
            fg = int((a > 0).sum())
        rgba = np.zeros((pixel_h, pixel_w, 4), dtype="uint8")
        rgba[..., 0] = fg_color[0]
        rgba[..., 1] = fg_color[1]
        rgba[..., 2] = fg_color[2]
        rgba[..., 3] = a
        overlay = Image.fromarray(rgba, mode="RGBA")
        canvas = Image.alpha_composite(canvas, overlay)
        print(f"  mask[{k}] color={palette[k % len(palette)]} foreground={fg}/{total} ({fg / total * 100:.1f}%)")

    if draw_bbox and bboxes:
        # Scale normalized [0, 1000] box coordinates to the mask canvas.
        draw = ImageDraw.Draw(canvas)
        line_w = max(2, pixel_w // 320)
        for k, bb in enumerate(bboxes):
            if not bb or len(bb) < 4:
                continue
            x0 = bb[0] / 1000.0 * pixel_w
            y0 = bb[1] / 1000.0 * pixel_h
            x1 = bb[2] / 1000.0 * pixel_w
            y1 = bb[3] / 1000.0 * pixel_h
            color = ImageColor.getrgb(palette[k % len(palette)]) + (255,)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=line_w)
            print(f"  bbox[{k}] normalized={bb} -> px=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")

    return canvas.convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local checkpoint directory or Hugging Face repo ID (e.g. SeerRay-Lab/Xiaomi-Lens-0).")
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Complete instruction, passed unchanged to the chat template. "
             "For grounding, request bbox coordinates and masks (see examples). "
             "The template inserts the image before the text.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: infer_result.jpg next to the input image.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for pixel-mask binarization. Set to 0 for a soft overlay. Default: 0.5.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mask overlay opacity: 0 = invisible, 1 = opaque. Default: 0.5.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum new tokens for generation. Default: {DEFAULT_MAX_NEW_TOKENS}.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Vision encoder patch size in pixels (16 for Qwen3.5). Default: 16.",
    )
    parser.add_argument(
        "--merge-size",
        type=int,
        default=2,
        help="Retained for compatibility; unused for canvas-resolution pixel masks.",
    )
    parser.add_argument(
        "--draw-bbox",
        action="store_true",
        help="Draw final-answer <bbox> coordinates (normalized to 0–1000) "
             "in the corresponding mask's color.",
    )
    parser.add_argument(
        "--per-mask-output",
        action="store_true",
        help="Save one overlay per final-answer mask (suffix '_mask_<i>'). "
             "Default: one composite image with distinct mask colors.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 = greedy (deterministic). >0 enables "
             "stochastic sampling (do_sample=True). Default: 0.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p (only used when temperature>0). Default: 1.0.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling cutoff (only used when temperature>0; 0=off). Default: 0.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of generations. The script reports final-answer mask counts "
             "and visualizes the generation with the most answer masks "
             "(ties use the earliest). Use temperature>0 for sampling. Default: 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Manual RNG seed for reproducible sampling rollouts. Default: none.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.image.parent / "infer_result.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, processor, mask_token_id = load_model_and_processor(args.checkpoint, device)

    image = Image.open(args.image).convert("RGB")

    prompt_inputs, _ = build_inputs(processor, image, args.text, device)

    do_sample = args.temperature is not None and args.temperature > 0
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Rank generations by final-answer mask count, excluding reasoning masks.
    from collections import Counter
    rollouts = []  # (idx, answer_K, thought_K, generated_text, completion_ids)
    for i in range(max(1, args.num_rollouts)):
        completion_ids, generated_text, total_mask_count = generate_completion(
            model, processor, prompt_inputs, args.max_new_tokens, mask_token_id,
            do_sample=do_sample, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k,
        )
        _, thought_mask_count = split_answer_content(generated_text)
        answer_mask_count = max(0, total_mask_count - thought_mask_count)
        rollouts.append(
            (i, answer_mask_count, thought_mask_count, generated_text, completion_ids)
        )
        mode = f"sample(T={args.temperature},top_p={args.top_p},top_k={args.top_k})" if do_sample else "greedy"
        print(
            f"[rollout {i} | {mode}] K_answer={answer_mask_count} "
            f"K_thought={thought_mask_count}  {generated_text!r}"
        )

    k_dist = Counter(r[1] for r in rollouts)
    print(f"\n=== answer K (<|mask|> count) distribution over {len(rollouts)} rollout(s): "
          f"{dict(sorted(k_dist.items()))} ===")
    n_multi = sum(1 for r in rollouts if r[1] >= 2)
    print(f"rollouts with answer K>=2: {n_multi}/{len(rollouts)}; max K = {max(r[1] for r in rollouts)}")

    # Select the most final-answer masks, breaking ties by generation order.
    best = max(rollouts, key=lambda r: r[1])
    _, mask_count, thought_mask_count, generated_text, completion_ids = best
    print(
        f"\nVisualizing rollout {best[0]} "
        f"(K_answer={mask_count}, K_thought={thought_mask_count}): {generated_text!r}"
    )

    if mask_count == 0:
        print("No answer <|mask|> token in any rollout; skipping mask visualization.")
        return

    pred_masks = run_forward(model, prompt_inputs, completion_ids, mask_token_id, device)
    pred_masks = pred_masks[:, thought_mask_count:]
    K = pred_masks.shape[1]

    bboxes = parse_answer_bboxes(generated_text) if args.draw_bbox else None
    if args.draw_bbox and bboxes is not None and len(bboxes) != K:
        print(f"Warning: parsed {len(bboxes)} bbox(es) but K={K} mask(s); "
              "drawing the overlapping prefix.")

    if args.per_mask_output:
        # Save each answer mask separately, using the same color as the composite.
        for k in range(K):
            per_mask_path = output_path.with_name(
                f"{output_path.stem}_mask_{k}{output_path.suffix}"
            )
            single = visualize_pixel_mask(
                image=image,
                pred_masks=pred_masks[:, k : k + 1, :, :],
                image_grid_thw=prompt_inputs["image_grid_thw"],
                patch_size=args.patch_size,
                merge_size=args.merge_size,
                threshold=args.mask_threshold,
                palette=(DEFAULT_MASK_PALETTE[k % len(DEFAULT_MASK_PALETTE)],),
                alpha=args.alpha,
                bboxes=([bboxes[k]] if bboxes and k < len(bboxes) else None),
                draw_bbox=args.draw_bbox,
            )
            single.save(per_mask_path)
            print(f"Saved: {per_mask_path}")
    else:
        result = visualize_pixel_mask(
            image=image,
            pred_masks=pred_masks,
            image_grid_thw=prompt_inputs["image_grid_thw"],
            patch_size=args.patch_size,
            merge_size=args.merge_size,
            threshold=args.mask_threshold,
            alpha=args.alpha,
            bboxes=bboxes,
            draw_bbox=args.draw_bbox,
        )
        result.save(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
