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

"""Lightweight JSONL offset-index dataset.

This is the ms-swift-like path for large training mixes: scan cheap line
offsets and lengths into memory, then parse/process one row at a time in
DataLoader workers. No dataset artifact is written to disk.
"""

from __future__ import annotations

import json
import logging
import math
import mmap
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .io import DatasetEntry, resolve_sample_images

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JsonlIndex:
    """Byte offsets for every non-empty line in one JSONL file."""

    path: Path
    offsets: np.ndarray
    lengths: np.ndarray
    size: int
    mtime_ns: int

    def __len__(self) -> int:
        return int(self.offsets.shape[0])


def scan_jsonl_offsets(path: str | Path) -> JsonlIndex:
    """Single-pass scan of non-empty JSONL line byte ranges."""
    p = Path(path).resolve()
    st = p.stat()
    offsets: list[int] = []
    lengths: list[int] = []
    cursor = 0
    with p.open("rb") as f:
        for line in f:
            line_len = len(line)
            stripped = line.rstrip(b"\r\n")
            if stripped.strip():
                offsets.append(cursor)
                lengths.append(len(stripped))
            cursor += line_len
    return JsonlIndex(
        path=p,
        offsets=np.asarray(offsets, dtype=np.uint64),
        lengths=np.asarray(lengths, dtype=np.uint32),
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
    )


def apply_sampling_strategy(
    indices: np.ndarray,
    strategy: str,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Apply the data_config sampling DSL to an index array."""
    if not strategy or strategy == "all":
        return indices
    if ":" not in strategy:
        raise ValueError(
            f"Unsupported sampling_strategy {strategy!r}; expected mode:count."
        )
    mode, count_str = strategy.split(":", 1)
    n = int(indices.shape[0])
    if count_str.endswith("%"):
        pct = int(count_str[:-1])
        count = math.ceil(n * pct / 100)
    else:
        count = int(count_str)
    count = max(0, min(count, n))

    if mode == "first":
        return indices[:count]
    if mode == "end":
        return indices[-count:] if count else indices[:0]
    if mode == "random":
        rng = np.random.default_rng(seed)
        return indices[rng.permutation(n)[:count]]
    raise ValueError(
        f"Unsupported sampling mode {mode!r}; valid: first | end | random."
    )


def stride_for_rank(
    indices: np.ndarray,
    *,
    rank: int,
    world_size: int,
) -> np.ndarray:
    """Deterministically keep rank::world_size after trimming tails."""
    if world_size <= 1:
        return indices
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} not in [0, {world_size})")
    n = int(indices.shape[0])
    if n < world_size:
        return indices
    trimmed = (n // world_size) * world_size
    return indices[rank:trimmed:world_size]


class JsonlIndexShard:
    """One JSONL file plus offset index and dataset metadata."""

    def __init__(
        self,
        index: JsonlIndex,
        *,
        image_folder: str = "",
        name: str = "",
        tag: str = "",
    ) -> None:
        self.index = index
        self.image_folder = image_folder
        self.name = name or index.path.stem
        self.tag = tag
        self._fd: int | None = None
        self._mmap: mmap.mmap | None = None
        self._default_modality: bool | None = None
        self._open()

    def _open(self) -> None:
        if self._mmap is not None:
            return
        fd = os.open(str(self.index.path), os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._mmap = mm

    def __getstate__(self) -> dict:
        return {
            "index": self.index,
            "image_folder": self.image_folder,
            "name": self.name,
            "tag": self.tag,
            "default_modality": self._default_modality,
        }

    def __setstate__(self, state: dict) -> None:
        self.index = state["index"]
        self.image_folder = state["image_folder"]
        self.name = state["name"]
        self.tag = state["tag"]
        self._default_modality = state.get("default_modality")
        self._fd = None
        self._mmap = None
        self._open()

    def __len__(self) -> int:
        return len(self.index)

    def get(self, i: int) -> dict:
        self._open()
        assert self._mmap is not None
        off = int(self.index.offsets[i])
        ln = int(self.index.lengths[i])
        sample = json.loads(self._mmap[off:off + ln])
        sample = resolve_sample_images(sample, self.image_folder)
        sample.setdefault("_dataset_name", self.name)
        sample.setdefault("_dataset_tag", self.tag)
        return sample

    @property
    def default_modality(self) -> bool:
        if self._default_modality is None:
            self._default_modality = (
                bool((self.get(0).get("images") or [])) if len(self) else False
            )
        return self._default_modality

    def close(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None

    def __del__(self) -> None:
        self.close()


def open_jsonl_shard(
    path: str | Path,
    *,
    image_folder: str = "",
    name: str = "",
    tag: str = "",
) -> JsonlIndexShard:
    idx = scan_jsonl_offsets(path)
    return JsonlIndexShard(idx, image_folder=image_folder, name=name, tag=tag)


class JsonlIndexDataset:
    """Concatenated JSONL shards with bounded retry on bad rows."""

    def __init__(
        self,
        shards: list[JsonlIndexShard],
        *,
        n_try_fetch: int = 10,
        seed: int = 0,
        is_rank_sharded: bool = False,
        shard_world: int = 1,
        shard_rank: int = 0,
        global_indices: np.ndarray | None = None,
    ) -> None:
        if not shards:
            raise ValueError("JsonlIndexDataset requires at least one shard.")
        self._shards = shards
        shard_ids: list[np.ndarray] = []
        local_idx: list[np.ndarray] = []
        for sid, shard in enumerate(shards):
            n = len(shard)
            shard_ids.append(np.full(n, sid, dtype=np.uint16))
            local_idx.append(np.arange(n, dtype=np.uint32))
        sid_flat = (
            np.concatenate(shard_ids) if shard_ids else np.zeros(0, dtype=np.uint16)
        )
        loc_flat = (
            np.concatenate(local_idx) if local_idx else np.zeros(0, dtype=np.uint32)
        )
        if global_indices is None:
            self._shard_ids = sid_flat
            self._local_idx = loc_flat
        else:
            self._shard_ids = sid_flat[global_indices]
            self._local_idx = loc_flat[global_indices]

        self.n_try_fetch = max(1, int(n_try_fetch))
        rng = np.random.default_rng(seed)
        n = len(self)
        self._fallback_perm = (
            rng.permutation(n).astype(np.int64) if n else np.zeros(0, dtype=np.int64)
        )
        self._fallback_cursor = 0
        self._warning_count = 0
        self.is_rank_sharded = is_rank_sharded
        self.shard_world = shard_world
        self.shard_rank = shard_rank

    def __len__(self) -> int:
        return int(self._shard_ids.shape[0])

    def __getitem__(self, idx: int) -> dict:
        n = len(self)
        if n == 0:
            raise IndexError("JsonlIndexDataset is empty")
        last_exc: Exception | None = None
        for attempt in range(self.n_try_fetch):
            if attempt == 0:
                try_idx = int(idx) % n
            else:
                try_idx = int(self._fallback_perm[self._fallback_cursor % n])
                self._fallback_cursor += 1
            try:
                sid = int(self._shard_ids[try_idx])
                loc = int(self._local_idx[try_idx])
                return self._shards[sid].get(loc)
            except Exception as exc:
                last_exc = exc
                if self._warning_count < 10:
                    logger.warning(
                        "[jsonl-index] failed idx=%d attempt=%d/%d: %s",
                        try_idx,
                        attempt + 1,
                        self.n_try_fetch,
                        exc,
                    )
                    self._warning_count += 1
        raise RuntimeError(
            f"JsonlIndexDataset exhausted {self.n_try_fetch} retries from "
            f"idx={idx}; last error: {last_exc!r}"
        )

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]

    @property
    def column_names(self) -> list[str]:
        return ["messages", "images", "segmentation", "_dataset_name", "_dataset_tag"]

    @property
    def lengths(self) -> list[int]:
        return [
            int(self._shards[int(sid)].index.lengths[int(loc)])
            for sid, loc in zip(self._shard_ids, self._local_idx)
        ]

    @property
    def modalities(self) -> list[bool]:
        return [
            self._shards[int(sid)].default_modality for sid in self._shard_ids
        ]


def build_jsonl_index_dataset(
    entries: list[DatasetEntry],
    *,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
    n_try_fetch: int = 10,
) -> JsonlIndexDataset:
    """Build an indexed lazy dataset from DatasetEntry objects."""
    if not entries:
        raise ValueError("build_jsonl_index_dataset: `entries` is empty.")

    shards: list[JsonlIndexShard] = []
    per_entry_indices: list[np.ndarray] = []
    offset = 0
    for entry in entries:
        shard = open_jsonl_shard(
            entry.path,
            image_folder=entry.image_folder,
            name=entry.display_name,
            tag=entry.tag,
        )
        shards.append(shard)
        local = np.arange(len(shard), dtype=np.int64)
        local = apply_sampling_strategy(
            local,
            entry.sampling_strategy or "all",
            seed=seed,
        )
        per_entry_indices.append(local + offset)
        offset += len(shard)

    global_indices = (
        np.concatenate(per_entry_indices)
        if per_entry_indices else np.zeros(0, dtype=np.int64)
    )
    global_indices = stride_for_rank(
        global_indices,
        rank=rank,
        world_size=world_size,
    )
    return JsonlIndexDataset(
        shards,
        n_try_fetch=n_try_fetch,
        seed=seed,
        is_rank_sharded=(world_size > 1),
        shard_world=world_size,
        shard_rank=rank,
        global_indices=global_indices,
    )
