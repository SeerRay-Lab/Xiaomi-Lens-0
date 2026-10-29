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

"""Generate text and decode masks through two Xiaomi-Lens-0 vLLM servers.

  prompt + image ── vLLM generate (:8000, /v1/chat/completions) ──> completion text
                                                              │
  if completion has <|mask|>:  image+text+completion ── vLLM pool (:8001, /pooling) ──> masks
                                                              │
                                          save composited box+mask overlay

Launch `run_vllm.sh` for text on port 8000 and `run_vllm_mask.sh` for masks on
port 8001. Both processes use the vLLM environment and checkpoint views derived
from the same model. The second request uses `/pooling`; the `xiaomi_lens_mask`
IO processor serializes masks decoded by PixelMaskDecoder inside vLLM.

Uses only the stdlib (urllib) so it runs anywhere without `requests`.

Run:
    PYTHONPATH=src python -m xiaomi_lens.integrations.vllm.client \\
        --image /path/to/image.jpg \\
        --query "the cat on the left" \\
        --model xiaomi-lens \\
        --output out.png

`--query` is the target description. The client wraps it in the grounding
instruction below and sends the same instruction to both servers.
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from pathlib import Path

from xiaomi_lens.constants import DEFAULT_MAX_NEW_TOKENS

# Shared instruction for referring-expression and category grounding.
GROUNDING_TEMPLATE = (
    'Locate all instances in the image that match the description "{query}". '
    'Report bbox coordinates and masks.'
)
# Both template names remain available for CLI compatibility.
TEMPLATES = {
    "description": GROUNDING_TEMPLATE,
    "category": GROUNDING_TEMPLATE,
}


def _post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise RuntimeError(f"HTTP request failed with status {e.code}: {error_body}") from e


def _image_data_uri(image_path: Path) -> str:
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    suffix = image_path.suffix.lstrip(".").lower() or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{b64}"


def call_vllm(vllm_url: str, model: str, image_path: Path, query_text: str,
              max_tokens: int) -> tuple[str, list[int]]:
    """One greedy chat completion with its exact generated token IDs."""
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_data_uri(image_path)}},
                {"type": "text", "text": query_text},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        # MUST keep special tokens: the remapped bbox is wrapped in the native
        # IDs decoded as <bbox>...</bbox>, and the mask slot is <|mask|>. vLLM
        # strips both by default, which would break parsing and mask decoding.
        "skip_special_tokens": False,
        # Replaying exact IDs avoids a decode/encode round trip and prevents BPE
        # merges across the template-prefix/completion boundary in phase 2.
        "return_token_ids": True,
    }
    out = _post_json(vllm_url.rstrip("/") + "/chat/completions", body)
    choice = out["choices"][0]
    return choice["message"]["content"], choice["token_ids"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--query", required=True, help="e.g. 'close this window'")
    ap.add_argument("--template", choices=sorted(TEMPLATES), default="description")
    ap.add_argument("--model", default="xiaomi-lens", help="served model name in vLLM")
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1",
                    help="OpenAI-compatible base (phase 1, chat completions)")
    ap.add_argument("--pool-url", default="http://localhost:8001",
                    help="mask vLLM base for the /pooling endpoint (phase 2; "
                         "run_vllm_mask.sh defaults to :8001)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--output", type=Path, default=Path("hybrid_result.png"))
    args = ap.parse_args()

    query_text = TEMPLATES[args.template].format(query=args.query)

    # Phase 1: text (vLLM). vLLM injects the image placeholder from the chat
    # template, so we send the bare instruction here.
    completion, completion_token_ids = call_vllm(
        args.vllm_url, args.model, args.image, query_text, args.max_tokens
    )
    print(f"[vLLM completion] {completion!r}")

    if "<|mask|>" not in completion:
        print("No <|mask|> in completion — text-only response, skipping mask service.")
        return 0

    # Phase 2: mask (same vLLM, /pooling + xiaomi_lens_mask io_processor). The
    # io_processor rebuilds the same image + text messages from `data`, then
    # appends the generated completion (with the <|mask|> the pooler decodes).
    resp = _post_json(args.pool_url.rstrip("/") + "/pooling", {
        "model": args.model,
        "data": {
            "image_path": str(args.image),
            "text": query_text,
            "completion": completion,
            "completion_token_ids": completion_token_ids,
            "threshold": args.threshold,
            "return_overlay": True,
        },
    })
    if "error" in resp:
        print(f"[vLLM /pooling error] {resp['error']}")
        return 1

    # The io_processor result is nested under `data` in the /pooling response.
    mask = resp.get("data", resp)
    print(f"[vLLM mask] K={mask['num_masks']} canvas={mask['canvas']}")
    if mask.get("overlay_b64"):
        args.output.write_bytes(base64.b64decode(mask["overlay_b64"]))
        print(f"Saved overlay: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
