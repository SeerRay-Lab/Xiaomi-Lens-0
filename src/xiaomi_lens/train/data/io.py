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

"""Dataset config + image-path helpers for the lazy JSONL pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetEntry:
    """One dataset source resolved from CLI flags or a YAML config."""
    path: str
    image_folder: str = ""
    name: str = ""        # optional display name
    tag: str = ""         # optional free-form label (e.g. "seg", "vqa", "text")
    sampling_strategy: str = "all"  # "all" | "first:N" | "end:N" | "random:N[%]"

    @property
    def display_name(self) -> str:
        return self.name or Path(self.path).stem


def parse_path_spec(path_spec: str | None) -> list[str]:
    if not path_spec:
        return []
    return [part for part in path_spec.split(":") if part]


def resolve_dataset_entries(script_args) -> list[DatasetEntry]:
    """Resolve dataset entries from `--data_config` or CLI path lists.

    YAML configs carry per-entry `image_folder`, `name`, `tag`, and
    `sampling_strategy`. The CLI form accepts colon-separated
    `data_file_paths` plus optional colon-separated `image_folders`.
    Empty folder entries are meaningful, so `image_folders` is split
    directly instead of going through `parse_path_spec`.
    """
    if getattr(script_args, "data_config", None):
        return list(load_dataset_config(script_args.data_config, split="train"))
    if not getattr(script_args, "data_file_paths", None):
        raise ValueError(
            "Either --data_config or --data_file_paths must be set."
        )

    paths = parse_path_spec(script_args.data_file_paths)
    raw_folders = getattr(script_args, "image_folders", None) or ""
    folders = raw_folders.split(":") if raw_folders else []
    if folders and len(folders) != len(paths):
        raise ValueError(
            f"--data_file_paths ({len(paths)}) and --image_folders "
            f"({len(folders)}) must contain the same number of entries."
        )
    if not folders:
        folders = [""] * len(paths)

    return [
        DatasetEntry(
            path=os.path.expandvars(path),
            image_folder=os.path.expandvars(folder),
        )
        for path, folder in zip(paths, folders)
    ]


def resolve_sample_images(item: dict, image_folder: str) -> dict:
    """Normalize the `images` field.

    Accepts `images` (list or single str), `image` (single str), or no
    image field at all (pure-text sample). Relative paths resolve against
    `image_folder`. Always returns a `sample` with
    a `images: list[str]` field, possibly empty.
    """
    sample = dict(item)

    if "images" in sample:
        images = sample["images"]
    elif "image" in sample:
        images = sample.pop("image")
    else:
        images = []

    if images is None:
        images = []
    if isinstance(images, str):
        images = [images]

    sample["images"] = [
        image_path if os.path.isabs(image_path) else os.path.join(image_folder, image_path)
        for image_path in images
    ]
    return sample


PRESET_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def resolve_dataset_config_path(config_path: str) -> Path:
    """Resolve a data config path; fall back to the shipped `train/configs/<name>`.

    Lookup order:
        1. `config_path` as-is (absolute or cwd-relative) if it exists.
        2. `<PRESET_CONFIG_DIR>/<config_path>`.
    """
    direct = Path(config_path)
    if direct.is_file():
        return direct
    preset = PRESET_CONFIG_DIR / config_path
    if preset.is_file():
        return preset
    raise FileNotFoundError(
        f"Data config `{config_path}` not found. Tried `{direct}` and `{preset}`."
    )


def _relative_to(base: Path, path: str) -> str:
    """Resolve a relative manifest path against the manifest's directory."""
    return path if os.path.isabs(path) else str(base / path)


def load_dataset_config(
    config_path: str,
    *,
    split: str | None = None,
) -> list[DatasetEntry]:
    """Parse dataset entries from a YAML manifest.

    A train-only manifest uses a top-level ``datasets:`` list. A combined
    training and evaluation manifest uses ``train: {datasets: [...]}`` and
    ``eval: {datasets: [...]}``; pass ``split="train"`` or ``split="eval"``
    to select one. Without ``split``, the train section is used.

    Relative entry paths resolve against the manifest's directory. Bare
    manifest filenames fall back to the shipped `train/configs/` presets.

    Schema (each entry):
        path: str                — required, absolute or resolvable path to .json/.jsonl
        image_folder: str        — optional, default ""
        name: str                — optional, for logs/preflight; defaults to filename stem
        tag: str                 — optional, free-form label (e.g. "seg", "vqa", "text")
        sampling_strategy: str   — optional, one of:
            "all"       (default — keep every sample)
            "first:N"   (keep the first N samples)
            "end:N"     (keep the last N samples)
            "random:N"  (seeded shuffle, keep the first N afterwards)
            "random:P%" (random strategy with percentage of dataset size)
    """
    resolved = resolve_dataset_config_path(config_path)
    raw = yaml.safe_load(resolved.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Data config `{resolved}` must contain a YAML mapping.")

    selected_split = split
    if selected_split is None and "datasets" not in raw and "train" in raw:
        selected_split = "train"
    if selected_split is not None:
        if selected_split in raw:
            raw = raw[selected_split]
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Split `{selected_split}` in data config `{resolved}` must "
                    "be a mapping containing `datasets:`."
                )
        elif selected_split != "train" or "datasets" not in raw:
            raise ValueError(
                f"Data config `{resolved}` has no top-level "
                f"`{selected_split}:` split containing `datasets:`."
            )

    if "datasets" not in raw:
        location = f"split `{selected_split}`" if selected_split else "top level"
        raise ValueError(
            f"Data config `{resolved}` must contain a `datasets:` list in {location}."
        )
    entries_raw = raw["datasets"]
    if not isinstance(entries_raw, list) or not entries_raw:
        raise ValueError(f"`datasets` in `{resolved}` must be a non-empty list.")

    entries: list[DatasetEntry] = []
    allowed = {"path", "image_folder", "name", "tag", "sampling_strategy"}
    for idx, raw_entry in enumerate(entries_raw):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Entry #{idx} in `{resolved}` must be a mapping.")
        unknown = set(raw_entry) - allowed
        if unknown:
            raise ValueError(
                f"Entry #{idx} in `{resolved}` has unknown keys: {sorted(unknown)}. "
                f"Allowed: {sorted(allowed)}."
            )
        if "path" not in raw_entry:
            raise ValueError(f"Entry #{idx} in `{resolved}` is missing required `path`.")
        image_folder = os.path.expandvars(str(raw_entry.get("image_folder", "") or ""))
        base = resolved.resolve().parent
        entries.append(DatasetEntry(
            path=_relative_to(base, os.path.expandvars(str(raw_entry["path"]))),
            image_folder=_relative_to(base, image_folder) if image_folder else "",
            name=str(raw_entry.get("name", "") or ""),
            tag=str(raw_entry.get("tag", "") or ""),
            sampling_strategy=str(raw_entry.get("sampling_strategy", "all") or "all"),
        ))
    return entries
