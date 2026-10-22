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

"""Batch sampler with length-grouped + rank-strided sharding.

Adapted from ms-swift swift/dataloader/shard.py. Difference: explicit
(rank, world_size) accepted at construction so unit tests don't need
torch.distributed initialized.

When `modalities` is supplied alongside `group_by_length=True`, the
sampler groups image-bearing and text-only samples separately (LLaVA's
recipe) before merging — keeps each global mega-batch homogeneous in
both modality and length, which matters because image samples expand
into hundreds of vision tokens that text-only samples don't have.

torch is imported lazily inside __iter__ so unit tests that only
exercise the no-shuffle path can run without torch.
"""

from __future__ import annotations

from typing import Optional


def _length_grouped_indices(
    lengths: list[int], batch_size: int, world_size: int, generator,
) -> list[int]:
    """LLaVA-style length grouping: permute → mega-batch → sort desc."""
    import torch
    indices = torch.randperm(len(lengths), generator=generator).tolist()
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i:i + megabatch_size]
        for i in range(0, len(indices), megabatch_size)
    ]
    megabatches = [
        sorted(mb, key=lambda idx: lengths[idx], reverse=True)
        for mb in megabatches
    ]
    return [idx for mb in megabatches for idx in mb]


def _modality_length_grouped_indices(
    lengths: list[int], modalities: list[bool],
    batch_size: int, world_size: int, generator,
) -> list[int]:
    """Length grouping run per modality, mega-batches merged & shuffled.

    Mirrors LLaVA's modality-aware length grouping recipe.
    """
    import torch

    if all(modalities) or not any(modalities):
        return _length_grouped_indices(lengths, batch_size, world_size, generator)

    mm_idx = [i for i, m in enumerate(modalities) if m]
    text_idx = [i for i, m in enumerate(modalities) if not m]

    mm_order = _length_grouped_indices(
        [lengths[i] for i in mm_idx], batch_size, world_size, generator,
    )
    text_order = _length_grouped_indices(
        [lengths[i] for i in text_idx], batch_size, world_size, generator,
    )
    mm_global = [mm_idx[i] for i in mm_order]
    text_global = [text_idx[i] for i in text_order]

    megabatch_size = world_size * batch_size
    mm_batches = [
        mm_global[i:i + megabatch_size]
        for i in range(0, len(mm_global), megabatch_size)
    ]
    text_batches = [
        text_global[i:i + megabatch_size]
        for i in range(0, len(text_global), megabatch_size)
    ]

    leftovers: list[int] = []
    if mm_batches and len(mm_batches[-1]) < megabatch_size:
        leftovers.extend(mm_batches.pop())
    if text_batches and len(text_batches[-1]) < megabatch_size:
        leftovers.extend(text_batches.pop())

    all_batches = mm_batches + text_batches
    if all_batches:
        perm = torch.randperm(len(all_batches), generator=generator).tolist()
        all_batches = [all_batches[i] for i in perm]
    if leftovers:
        all_batches.append(sorted(leftovers, key=lambda i: lengths[i], reverse=True))

    return [idx for mb in all_batches for idx in mb]


class BatchSamplerShard:
    def __init__(
        self, *,
        total_samples: int, batch_size: int,
        shuffle: bool, drop_last: bool,
        data_seed: Optional[int],
        rank: Optional[int] = None, world_size: Optional[int] = None,
        group_by_length: bool = False, lengths: Optional[list[int]] = None,
        modalities: Optional[list[bool]] = None,
    ):
        if group_by_length and not shuffle:
            raise ValueError("shuffle=True is required when group_by_length=True")
        if (
            modalities is not None and lengths is not None
            and len(modalities) != len(lengths)
        ):
            raise ValueError(
                f"lengths ({len(lengths)}) and modalities "
                f"({len(modalities)}) must align"
            )
        self._explicit_rank = rank
        self._explicit_world = world_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_seed = data_seed if data_seed is not None else 0
        self.curr_seed = self.base_seed
        self.group_by_length = group_by_length
        self.lengths = lengths
        self.modalities = modalities
        self.total_samples = total_samples // self.world_size

    @property
    def rank(self) -> int:
        if self._explicit_rank is not None:
            return self._explicit_rank
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def world_size(self) -> int:
        if self._explicit_world is not None:
            return self._explicit_world
        import torch.distributed as dist
        return dist.get_world_size() if dist.is_initialized() else 1

    def __iter__(self):
        import torch
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.curr_seed)
            if self.group_by_length and self.modalities is not None:
                total_idx = _modality_length_grouped_indices(
                    self.lengths, self.modalities,
                    self.batch_size, self.world_size, generator,
                )
            elif self.group_by_length:
                total_idx = _length_grouped_indices(
                    self.lengths, self.batch_size, self.world_size, generator,
                )
            else:
                total_idx = torch.randperm(
                    self.total_samples * self.world_size, generator=generator,
                ).tolist()
            total_idx = total_idx[self.rank::self.world_size]
        else:
            total_idx = list(range(
                self.rank, self.total_samples * self.world_size, self.world_size,
            ))

        batch = []
        for idx in total_idx:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last and batch:
            yield batch

    def set_epoch(self, epoch: int) -> None:
        self.curr_seed = self.base_seed + epoch

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_samples // self.batch_size
        return (self.total_samples + self.batch_size - 1) // self.batch_size
