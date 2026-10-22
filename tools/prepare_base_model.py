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

"""Prepare a base Qwen3.5 checkpoint for Xiaomi-Lens-0 training.

Xiaomi-Lens-0 writes boxes as `<bbox>[x1, y1, x2, y2]</bbox>`, reusing Qwen's
native box special tokens. This tool renames them in the tokenizer metadata
while preserving their token IDs and weights:

    <|box_start|> -> <bbox>
    <|box_end|>   -> </bbox>

Model tensors are not touched. `<|mask|>` is added later by the trainer.
Without `--in-place` the tool only validates and reports. Writes are atomic
and create `.legacy-box-tokens.bak` copies by default. A directory containing
several models is processed recursively.

Usage:
    hf download Qwen/Qwen3.5-4B --local-dir checkpoints/Qwen3.5-4B
    python tools/prepare_base_model.py checkpoints/Qwen3.5-4B --in-place
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

TOKEN_MAP = {
    "<|box_start|>": "<bbox>",
    "<|box_end|>": "</bbox>",
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
DEFAULT_BACKUP_SUFFIX = ".legacy-box-tokens.bak"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Model directory or its parent")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite tokenizer files; default is a read-only validation",
    )
    parser.add_argument(
        "--backup-suffix",
        default=DEFAULT_BACKUP_SUFFIX,
        help=f"Backup suffix (default: {DEFAULT_BACKUP_SUFFIX!r}); empty disables backups",
    )
    return parser.parse_args()


def find_model_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        raise NotADirectoryError(path)
    if all((path / name).is_file() for name in TOKENIZER_FILES):
        return [path]
    return sorted(
        tokenizer.parent
        for tokenizer in path.rglob("tokenizer.json")
        if (tokenizer.parent / "tokenizer_config.json").is_file()
    )


def token_entries(model_dir: Path) -> tuple[dict, dict]:
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    config = json.loads(
        (model_dir / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    return tokenizer, config


def inspect_model(model_dir: Path) -> dict:
    tokenizer, config = token_entries(model_dir)
    added = tokenizer.get("added_tokens")
    decoder = config.get("added_tokens_decoder") or {}
    specials = config.get("additional_special_tokens") or []
    if not isinstance(added, list):
        raise ValueError(f"{model_dir}: tokenizer.json has no added_tokens list")
    if not isinstance(decoder, dict):
        raise ValueError(f"{model_dir}: tokenizer_config.json has invalid added_tokens_decoder")
    if not isinstance(specials, list):
        raise ValueError(f"{model_dir}: tokenizer_config.json has invalid additional_special_tokens")

    ids: dict[str, int] = {}
    states = set()
    for legacy, canonical in TOKEN_MAP.items():
        legacy_added = [item for item in added if item.get("content") == legacy]
        canonical_added = [item for item in added if item.get("content") == canonical]
        legacy_decoder = [
            (token_id, item)
            for token_id, item in decoder.items()
            if item.get("content") == legacy
        ]
        canonical_decoder = [
            (token_id, item)
            for token_id, item in decoder.items()
            if item.get("content") == canonical
        ]
        legacy_specials = specials.count(legacy)
        canonical_specials = specials.count(canonical)

        shape = (
            len(legacy_added),
            len(legacy_decoder),
            legacy_specials,
            len(canonical_added),
            len(canonical_decoder),
            canonical_specials,
        )
        if (
            len(legacy_added) == 1
            and not canonical_added
            and len(legacy_decoder) <= 1
            and not canonical_decoder
            and legacy_specials <= 1
            and canonical_specials == 0
        ):
            state = "legacy"
            entry = legacy_added[0]
            config_entries = legacy_decoder
        elif (
            len(canonical_added) == 1
            and not legacy_added
            and len(canonical_decoder) <= 1
            and not legacy_decoder
            and canonical_specials <= 1
            and legacy_specials == 0
        ):
            state = "canonical"
            entry = canonical_added[0]
            config_entries = canonical_decoder
        else:
            raise ValueError(
                f"{model_dir}: inconsistent metadata for {legacy!r}/{canonical!r}: "
                f"{shape}"
            )
        added_id = entry.get("id")
        if not isinstance(added_id, int):
            raise ValueError(f"{model_dir}: invalid token ID {added_id!r}")
        if config_entries:
            decoder_id = int(config_entries[0][0])
            if added_id != decoder_id:
                raise ValueError(
                    f"{model_dir}: token ID mismatch for {legacy!r}/{canonical!r}: "
                    f"tokenizer={added_id!r}, config={decoder_id!r}"
                )
            if config_entries[0][1].get("special") is not True:
                raise ValueError(f"{model_dir}: config box delimiter is not special")
        if entry.get("special") is not True:
            raise ValueError(f"{model_dir}: tokenizer box delimiter is not special")
        states.add(state)
        ids[canonical] = added_id

    if len(states) != 1:
        raise ValueError(f"{model_dir}: box tokens have mixed migration states: {states}")
    return {"model": str(model_dir), "state": states.pop(), "token_ids": ids}


def atomic_write(path: Path, content: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".box-token-remap.tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), path.stat().st_mode & 0o7777)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def rewrite_model(model_dir: Path, backup_suffix: str) -> None:
    pending: dict[Path, bytes] = {}
    for name in TOKENIZER_FILES:
        path = model_dir / name
        original = path.read_bytes()
        converted = original
        for legacy, canonical in TOKEN_MAP.items():
            converted = converted.replace(legacy.encode(), canonical.encode())
        if converted == original:
            continue
        json.loads(converted)
        pending[path] = converted

    if not pending:
        raise AssertionError(f"{model_dir}: validated as legacy but no bytes changed")

    if backup_suffix:
        for path in pending:
            backup = path.with_name(path.name + backup_suffix)
            if backup.exists():
                raise FileExistsError(f"refusing to overwrite backup: {backup}")
        for path in pending:
            shutil.copy2(path, path.with_name(path.name + backup_suffix))

    for path, converted in pending.items():
        atomic_write(path, converted)


def main() -> int:
    args = parse_args()
    model_dirs = find_model_dirs(args.path)
    if not model_dirs:
        raise ValueError(f"{args.path}: found no tokenizer model directories")

    before_by_dir = {model_dir: inspect_model(model_dir) for model_dir in model_dirs}
    if args.in_place and args.backup_suffix:
        for model_dir, before in before_by_dir.items():
            if before["state"] != "legacy":
                continue
            for name in TOKENIZER_FILES:
                path = model_dir / name
                original = path.read_bytes()
                if not any(legacy.encode() in original for legacy in TOKEN_MAP):
                    continue
                backup = path.with_name(path.name + args.backup_suffix)
                if backup.exists():
                    raise FileExistsError(f"refusing to overwrite backup: {backup}")

    reports = []
    for model_dir, before in before_by_dir.items():
        if args.in_place and before["state"] == "legacy":
            rewrite_model(model_dir, args.backup_suffix)
        after = inspect_model(model_dir)
        if args.in_place and after["state"] != "canonical":
            raise AssertionError(f"{model_dir}: post-write validation failed")
        reports.append({"before": before, "after": after})

    print(json.dumps({"written": args.in_place, "models": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
