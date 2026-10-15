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

"""Prepare checkpoint views for Xiaomi-Lens-0's vLLM launchers.

The default mode prepares indexed safetensors shards for the text server:
remap nested backbone keys to vLLM's Qwen3.5 layout, remove `mask_decoder.*`,
and omit `lm_head.weight` when embeddings are tied. The mask decoder runs in
the separate pooling server or the optional HF mask service.

For pooling, `--tokenizer-only` symlinks all weights and preserves the model's
architecture. `XiaomiLensForVLLM.load_weights` remaps backbone names and loads
the mask decoder inside vLLM.

Both modes write a tokenizer configuration using `Qwen2Tokenizer`, removing
`extra_special_tokens` and `backend` entries for compatibility with the serving
tokenizer loader. Special tokens remain defined in the linked `tokenizer.json`.

Source checkpoint files are read only. Unchanged files are symlinked into `out`;
default mode writes copies of the indexed weight shards, while tokenizer-only
mode copies no weights. Use separate output directories for the two modes.

Run:
    python -m xiaomi_lens.integrations.vllm.prepare_served \\
        --checkpoint /path/to/checkpoint --out /tmp/xiaomi_lens_served

    python -m xiaomi_lens.integrations.vllm.prepare_served \\
        --checkpoint /path/to/checkpoint --out /tmp/xiaomi_lens_pool --tokenizer-only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Longest prefix first so the text rule (3x language_model) wins over any shorter
# accidental match. Applied to the FIRST matching prefix only.
RENAMES: tuple[tuple[str, str], ...] = (
    ("model.language_model.language_model.language_model.", "model.language_model."),
    ("model.language_model.visual.", "model.visual."),
)
DROP_PREFIXES = ("mask_decoder.",)      # Omit mask weights from the text-server view.


def remap_key(key: str, *, drop_lm_head: bool) -> str | None:
    """Xiaomi-Lens-0 tensor name -> vLLM Qwen3.5-VL name, or None to drop it.

    `drop_lm_head` MUST reflect the checkpoint's `tie_word_embeddings`: on a tied
    model (4B) `lm_head.weight` is a redundant copy of the embedding and vLLM
    re-ties it, so we drop it; on an untied model (9B) it is an independently
    trained tensor that vLLM's output head needs, so we keep and rename it.
    """
    if any(key.startswith(p) for p in DROP_PREFIXES):
        return None
    if key == "lm_head.weight" and drop_lm_head:
        return None
    for old, new in RENAMES:
        if key.startswith(old):
            return new + key[len(old):]
    return key


def _corrected_tokenizer_config(src: Path) -> dict:
    d = json.loads(src.read_text())
    d["tokenizer_class"] = "Qwen2Tokenizer"
    for bad in ("extra_special_tokens", "backend"):
        d.pop(bad, None)
    return d


def _rewrite_shard(src: Path, dst: Path, *, drop_lm_head: bool) -> tuple[int, int]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    kept, dropped = {}, 0
    with safe_open(str(src), framework="pt") as f:
        meta = f.metadata()
        for key in f.keys():
            new = remap_key(key, drop_lm_head=drop_lm_head)
            if new is None:
                dropped += 1
                continue
            kept[new] = f.get_tensor(key)
    save_file(kept, str(dst), metadata=meta)
    return len(kept), dropped


def _tie_word_embeddings(checkpoint: Path) -> bool:
    """Read the checkpoint's tie_word_embeddings (default True, as Qwen HF does)."""
    cfg = json.loads((checkpoint / "config.json").read_text())
    text_cfg = cfg.get("text_config") or {}
    # HF resolves the tie flag from the text config first, then the top level.
    return bool(text_cfg.get("tie_word_embeddings", cfg.get("tie_word_embeddings", True)))


def prepare(checkpoint: Path, out: Path, *, tokenizer_only: bool = False) -> Path:
    """Materialize a vLLM-servable view of `checkpoint` under `out`.

    Two modes:
    - Default (text server): rewrite indexed weight shards with backbone keys
      remapped to the base Qwen3.5 layout, dropping mask weights and a tied LM head.
      The launcher supplies the base architecture override.
    - `tokenizer_only` (pooling server): symlink weights, index, and model config;
      rewrite only `tokenizer_config.json`. The custom vLLM model handles weight
      remapping and mask decoding.
    """
    checkpoint = checkpoint.resolve()  # so symlink targets are absolute, not relative to `out`
    out.mkdir(parents=True, exist_ok=True)
    index_name = "model.safetensors.index.json"
    drop_lm_head = _tie_word_embeddings(checkpoint)

    index_path = checkpoint / index_name
    shard_files: set[str] = set()
    weight_map: dict = {}
    if index_path.exists() and not tokenizer_only:
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shard_files = set(weight_map.values())

    for entry in sorted(checkpoint.iterdir()):
        name, dst = entry.name, out / entry.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        if name == "tokenizer_config.json":
            dst.write_text(json.dumps(_corrected_tokenizer_config(entry)))
        elif name == index_name and shard_files:
            new_map = {}
            for key, shard in weight_map.items():
                nk = remap_key(key, drop_lm_head=drop_lm_head)
                if nk is not None:
                    new_map[nk] = shard
            dst.write_text(json.dumps({"weight_map": new_map}))
        elif name in shard_files:
            kept, dropped = _rewrite_shard(entry, dst, drop_lm_head=drop_lm_head)
            print(f"[prepare_served] {name}: kept {kept}, dropped {dropped}")
        else:
            os.symlink(entry, dst)  # entry is absolute (checkpoint was resolved)

    if tokenizer_only:
        print(f"[prepare_served] {out}  (tokenizer-only: all weights symlinked, "
              f"mask_decoder kept; custom vLLM model remaps + loads them)")
    else:
        print(f"[prepare_served] {out}  (tie_word_embeddings={drop_lm_head}, "
              f"lm_head {'dropped' if drop_lm_head else 'kept'})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer-only", action="store_true",
                    help="Unified pooling path: symlink all weights (keep arch + "
                         "mask_decoder), fix only tokenizer_config.json. The custom "
                         "vLLM model does the weight remap + mask_decoder load.")
    args = ap.parse_args()
    prepare(args.checkpoint, args.out, tokenizer_only=args.tokenizer_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
