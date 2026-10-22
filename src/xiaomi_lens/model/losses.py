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

"""Mask loss modules — InstructSAM-style dice + sigmoid BCE.

Mirrors `DCDmllm/InstructSAM` (`instructsam/models/utils.py`): the
foreground/background term is `CrossEntropyLoss(use_sigmoid=True)` —
i.e., per-pixel `BCEWithLogits` with reduction='mean' — and the region
term is naive dice. ``MaskLoss`` composes them with InstructSAM
defaults ``dice_weight=1.0``, ``bce_weight=2.0`` (callers may override
both via config). Each per-mask loss is summed and divided by
``num_masks`` (DETR / Mask2Former / SAM3 convention; see
``_global_num_masks`` for the cross-rank reduction).

The class is resolution-agnostic: it operates on whatever ``(B, 1, H, W)``
logits and targets the caller provides. A ``valid_mask`` argument is supported to exclude padded regions introduced
when batching images with different patch-grid sizes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


def _global_num_masks(local_num_masks: torch.Tensor) -> torch.Tensor:
    """All-reduce ``local_num_masks`` and return the per-rank divisor.

    Inputs and outputs are zero-dimensional device tensors and ``.item()`` is
    never called, so the reduction adds no host synchronization per microbatch.

    DDP and DeepSpeed ZeRO-1/2 **average** gradients
    across data-parallel ranks. Let world size be ``W``, the global number of
    masks be ``N``, and rank r's sum of per-mask losses be ``S_r``. This function
    returns ``N/W``, so the local loss is ``L_r = W*S_r/N`` and:

    ``mean_r(∇L_r) = (1/W) * Σ_r(W/N * ∇S_r) = ∇(Σ_r S_r/N)``.

    Uneven mask counts across ranks therefore do not change the direction or
    scale of a microbatch's global per-mask gradient; only the local scalar loss
    differs by rank. Logging must take the cross-rank mean of those local
    scalars; see ``XiaomiLensTrainer._distributed_metric_means``. The all_reduce
    collective itself is part of DDP gradient semantics and cannot be removed.

    ``clamp_min(1)`` scales the entire mask gradient by ``N/W`` only when
    ``0 < N < W``. The mask component's direction is unchanged, but its weight
    relative to LM loss decreases. This is a common sparse-target safeguard in
    DETR-family models. Note also that gradient accumulation still averages the
    per-mask mean of each microbatch. When N differs across microbatches, this is
    not exactly the same as averaging all masks in the complete accumulation
    window; that issue is independent of rank-0 logging bias.
    """
    local = local_num_masks.to(torch.float32)
    if not (dist.is_available() and dist.is_initialized()):
        return local.double().clamp_min(1.0).float()
    world_size = dist.get_world_size()
    if world_size <= 1:
        return local.double().clamp_min(1.0).float()
    t = local.reshape(1)
    dist.all_reduce(t)
    return (t.double() / world_size).clamp_min(1.0).float().reshape(())


class DiceLoss(nn.Module):
    """DICE loss on sigmoid probabilities."""

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = inputs.sigmoid().flatten(1)
        tgt = targets.flatten(1)
        if valid_mask is not None:
            valid = valid_mask.flatten(1)
            probs = probs * valid
            tgt = tgt * valid
        numerator = 2 * (probs * tgt).sum(dim=-1)
        denominator = probs.sum(dim=-1) + tgt.sum(dim=-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks.clamp_min(1.0)


class BCEWithLogitsLoss(nn.Module):
    """Sigmoid binary cross-entropy on raw logits (InstructSAM-style).

    Named to mirror `torch.nn.BCEWithLogitsLoss`: it consumes RAW logits and
    applies the sigmoid internally (via `F.binary_cross_entropy_with_logits`).
    Do NOT pre-sigmoid the inputs.

    Matches the foreground/background term in `instructsam/models/utils.py`
    `CrossEntropyLoss(use_sigmoid=True, reduction='mean')` — i.e., plain
    `F.binary_cross_entropy_with_logits` reduced as a mean across spatial
    positions, then summed across masks and divided by ``num_masks``.

    Without ``valid_mask`` the spatial reduction is a plain mean. With
    ``valid_mask`` the reduction is ``sum / num_valid_per_mask`` so padded
    positions do not dilute the per-mask loss.
    """

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Fold masking and reduction into the kernel instead of materializing a
        # full-canvas `reduction="none"` FP32 tensor per mask. `weight` is the
        # row-normalized valid mask, so a plain `sum` equals the sum of
        # per-mask means:
        #   Σ_ij bce_ij · valid_ij / count_i == Σ_i per_mask_i.
        flat_inputs = inputs.flatten(1)
        flat_targets = targets.flatten(1)
        if valid_mask is None:
            # Mean over B·N elements times B is the same sum of per-mask means.
            total = F.binary_cross_entropy_with_logits(
                flat_inputs, flat_targets, reduction="mean"
            ) * flat_inputs.shape[0]
        else:
            valid = valid_mask.flatten(1)
            weight = valid / valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
            total = F.binary_cross_entropy_with_logits(
                flat_inputs, flat_targets, weight=weight, reduction="sum"
            )
        return total / num_masks.clamp_min(1.0)


class MaskLoss(nn.Module):
    """InstructSAM-style combined mask loss: ``dice_weight * dice + bce_weight * bce``.

    Resolution-agnostic — operates on whatever ``(B, 1, H, W)`` logits +
    targets the caller passes. FP32 arithmetic is retained for numerical
    stability, but masks are sliced *before* conversion so no full ``B*H*W``
    FP32 copy is materialized. With gradient checkpointing, every slice is an
    independent checkpoint: backward recomputes and releases one slice at a
    time instead of rebuilding the complete mask set at once. Without gradient
    checkpointing, eager autograd still retains the saved tensors from every
    slice until backward; chunking then reduces transient workspace, but not
    the aggregate saved-tensor memory.

    Defaults follow `DCDmllm/InstructSAM`'s `instructsam.py`
    (``dice_weight=1.0``, ``bce_weight=2.0``). ``chunk_size <= 0`` restores the
    unchunked reference path for parity checks and debugging.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 2.0,
        chunk_size: int = 8,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.chunk_size = int(chunk_size)
        self.dice = DiceLoss()
        self.bce = BCEWithLogitsLoss()
        # Declared so Transformers' `_set_gradient_checkpointing` opts this
        # module in along with the rest of the model. Each chunk gets its own
        # checkpoint below; a single checkpoint around the whole loss would
        # still recreate every full-canvas FP32 temporary at the same time.
        self.gradient_checkpointing = False

    def forward(
        self,
        pred_masks: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_masks: ``(B, 1, H, W)`` or ``(B, H, W)`` raw logits.
            targets: matching shape, 0/1.
            valid_mask: optional, matching shape, 0/1.
        """
        pred = pred_masks.unsqueeze(1) if pred_masks.dim() == 3 else pred_masks
        tgt = targets.unsqueeze(1) if targets.dim() == 3 else targets
        if pred.shape != tgt.shape:
            raise ValueError(
                f"Shape mismatch between pred_masks {tuple(pred.shape)} and targets {tuple(tgt.shape)}"
            )
        valid = valid_mask
        if valid is not None:
            valid = valid.unsqueeze(1) if valid.dim() == 3 else valid
            if valid.shape != tgt.shape:
                raise ValueError(
                    f"Shape mismatch between valid_mask {tuple(valid.shape)} and targets {tuple(tgt.shape)}"
                )

        if valid is not None:
            # Count only samples whose valid_mask has any positive pixel;
            # others are padding-only rows (e.g., mixed-modality batches
            # where text-only samples sit next to segmentation samples).
            # Keep this as a GPU tensor without calling .item(); let
            # `_global_num_masks` handle it to avoid a host sync per microbatch.
            local_num_masks = valid.flatten(1).any(dim=1).sum()
        else:
            local_num_masks = torch.full(
                (), pred.shape[0], dtype=torch.int64, device=pred.device
            )
        # Cross-rank normalization: see `_global_num_masks` for why local counts
        # produce skewed gradients under DDP/ZeRO. This stays OUTSIDE the chunk
        # loop — its `all_reduce` must run exactly once per microbatch, not once
        # per chunk or again during checkpoint recomputation.
        num_masks = _global_num_masks(local_num_masks)

        num_rows = pred.shape[0]
        if num_rows == 0:
            return pred.sum() * 0.0
        chunk_size = num_rows if self.chunk_size <= 0 else min(self.chunk_size, num_rows)
        total = None
        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            pred_chunk = pred[start:end]
            target_chunk = tgt[start:end]
            valid_chunk = None if valid is None else valid[start:end]
            if self.gradient_checkpointing and self.training:
                chunk_loss = torch.utils.checkpoint.checkpoint(
                    self._compute_chunk,
                    pred_chunk,
                    target_chunk,
                    valid_chunk,
                    num_masks,
                    use_reentrant=False,
                )
            else:
                chunk_loss = self._compute_chunk(
                    pred_chunk, target_chunk, valid_chunk, num_masks
                )
            total = chunk_loss if total is None else total + chunk_loss
        return total

    def _compute_chunk(
        self,
        pred_masks: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        num_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Run FP32 dice + BCE for one mask slice and return its global share."""
        with torch.autocast(device_type=pred_masks.device.type, enabled=False):
            # Slice first, convert second. Targets stay uint8 and validity stays
            # bool outside this function, so their full-canvas storage remains
            # compact and the only FP32 temporaries are chunk-sized.
            logits = pred_masks.float()
            tgt = targets.float()
            valid = None if valid_mask is None else valid_mask.float()
            dice = self.dice(logits, tgt, num_masks=num_masks, valid_mask=valid)
            bce = self.bce(logits, tgt, num_masks=num_masks, valid_mask=valid)
            return self.dice_weight * dice + self.bce_weight * bce
