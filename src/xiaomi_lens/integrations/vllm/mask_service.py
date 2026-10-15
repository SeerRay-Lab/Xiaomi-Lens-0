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

"""Hugging Face mask-decoding HTTP service for Xiaomi-Lens-0.

This optional fallback accepts an image, the original instruction, and a text
completion from a separate generation server. `infer.run_forward` recomputes
hidden states and visual features to decode the completion's <|mask|> tokens.
The response includes only final-answer masks and boxes; reasoning masks are
excluded. This service uses `/decode_mask`, while the vLLM plugin uses `/pooling`.

Run:
    PYTHONPATH=src python -m xiaomi_lens.integrations.vllm.mask_service \\
        --checkpoint SeerRay-Lab/Xiaomi-Lens-0 --port 8100

Endpoint:
    POST /decode_mask
    {
      "image_path": "/path/to/image.jpg",
      "text": "Locate all instances in the image that match the description \\"the cat on the left\\". Report bbox coordinates and masks.",
      "completion": "</think><bbox>[100, 200, 400, 800]</bbox><mask><|mask|></mask>",
      "threshold": 0.5,
      "return_overlay": true
    }
    -> {
      "num_masks": K, "canvas": [w, h],
      "bboxes": [[x1,y1,x2,y2], ...],          # normalized to 0–1000
      "masks_b64": ["<base64 PNG L-mode>", ...],  # K binary masks at canvas res
      "overlay_b64": "<base64 PNG>"            # composited box+mask, if requested
    }

The completion above illustrates the format; send the generation server's
actual response. `text` must match the instruction used for generation.
Use `image_b64` instead of `image_path` to send base64-encoded PNG/JPEG bytes.
`threshold` and `return_overlay` default to 0.5 and true. With no answer masks,
the response contains `num_masks: 0`, `canvas: null`, and an empty mask list.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Make the package and root inference module importable when run as a file.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from infer import (  # noqa: E402
    build_inputs,
    load_model_and_processor,
    parse_answer_bboxes,
    run_forward,
    split_answer_content,
    visualize_pixel_mask,
)

# Load the model once in main() and share it across request handlers.
_MODEL = None
_PROCESSOR = None
_MASK_TOKEN_ID = None
_DEVICE = None


def _png_b64(arr: np.ndarray, mode: str) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_image(payload: dict) -> Image.Image:
    if payload.get("image_path"):
        return Image.open(payload["image_path"]).convert("RGB")
    if payload.get("image_b64"):
        raw = base64.b64decode(payload["image_b64"])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise ValueError("request must provide 'image_path' or 'image_b64'")


def decode_mask(payload: dict) -> dict:
    """Phase-2 decode for one (image, prompt, completion). Reads model globals."""
    text = payload["text"]
    completion = payload["completion"]
    threshold = float(payload.get("threshold", 0.5))
    return_overlay = bool(payload.get("return_overlay", True))

    image = _load_image(payload)
    prompt_inputs, _ = build_inputs(_PROCESSOR, image, text, _DEVICE)

    completion_ids = _PROCESSOR.tokenizer(
        completion, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(_DEVICE)

    bboxes = parse_answer_bboxes(completion)
    _, thought_mask_count = split_answer_content(completion)
    total_mask_count = int((completion_ids[0] == _MASK_TOKEN_ID).sum().item())
    answer_mask_count = max(0, total_mask_count - thought_mask_count)
    if answer_mask_count == 0:
        # No final-answer mask — auxiliary masks inside <think> are not returned.
        return {"num_masks": 0, "canvas": None, "bboxes": bboxes,
                "masks_b64": [], "overlay_b64": None}

    pred_masks = run_forward(_MODEL, prompt_inputs, completion_ids, _MASK_TOKEN_ID, _DEVICE)
    pred_masks = pred_masks[:, thought_mask_count:]
    K = int(pred_masks.shape[1])
    grid = prompt_inputs["image_grid_thw"]
    canvas_w = int(grid[0][2].item()) * 16
    canvas_h = int(grid[0][1].item()) * 16

    masks_b64 = []
    for k in range(K):
        prob = torch.sigmoid(pred_masks[0, k]).float().cpu().numpy()
        binary = (prob >= threshold).astype("uint8") * 255
        masks_b64.append(_png_b64(binary, mode="L"))

    overlay_b64 = None
    if return_overlay:
        overlay = visualize_pixel_mask(
            image=image,
            pred_masks=pred_masks,
            image_grid_thw=grid,
            threshold=threshold,
            bboxes=bboxes if bboxes else None,
            draw_bbox=bool(bboxes),
        )
        overlay_b64 = _png_b64(np.asarray(overlay), mode="RGB")

    return {
        "num_masks": K,
        "canvas": [canvas_w, canvas_h],
        "bboxes": bboxes,
        "masks_b64": masks_b64,
        "overlay_b64": overlay_b64,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok", "device": str(_DEVICE)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/decode_mask":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, decode_mask(payload))
        except Exception as exc:  # noqa: BLE001 — surface the error to the caller
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *args) -> None:  # quiet default access logging
        return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()

    global _MODEL, _PROCESSOR, _MASK_TOKEN_ID, _DEVICE
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mask_service] device={_DEVICE}", flush=True)
    _MODEL, _PROCESSOR, _MASK_TOKEN_ID = load_model_and_processor(args.checkpoint, _DEVICE)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[mask_service] listening on {args.host}:{args.port} "
          f"(POST /decode_mask, GET /health)", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
