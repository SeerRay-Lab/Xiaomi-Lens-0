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

"""MaskFormer-style dynamic classifier that decodes mask tokens at canvas resolution.

Each mask-token hidden state passes through `mask_embed` to produce a dynamic
classifier, which is dotted with the pixel embeddings. Patch self-attention
provides spatial context; the mask queries enter at the classification step.

Data flow:
    mask_hidden    → mask_embed MLP                          → (B, K, C_emb)
    image_features → patch_proj → patch_self_attn            → (B, P, llm_hidden)
                     reshape + channel_reduce                → (B, C, h, w)
                     2 × ConvTranspose2d(stride=2)           → (B, C_emb, 4h, 4w)
    einsum(mask_embedding, pixel_embedding) / √C_emb         → (B, K, 4h, 4w)
    bilinear interpolate(scale_factor=4)                     → (B, K, 16h, 16w)

Learned 4× upsampling followed by fixed 4× bilinear interpolation gives exactly
`MASK_CANVAS_SCALE=16`. The `(16h, 16w)` output therefore matches the resolution
of the ViT-processed input canvas.

Batching constraints:
- `_decode_pixel_masks` packs `(sample, image)` groups into a batch. Differently
  sized images within one sample also require padding, so batch size 1 does not
  guarantee a padding-free input.
- Smaller grids are zero-padded to the batch maximum and accompanied by
  `patch_key_padding_mask`, where True means padding. Fully padded placeholder
  rows must pass all False to prevent NaNs from attention softmax.
- The transposed convolutions use non-overlapping 2×2 kernels. The final bilinear
  interpolation mixes neighboring outputs, so boundary pixels of smaller grids
  can depend on padding. Callers must crop to each image's valid region.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

AttentionFunction: TypeAlias = Callable[..., torch.Tensor]
FlashFunctions: TypeAlias = tuple[AttentionFunction, AttentionFunction]

_SUPPORTED_ATTENTION_BACKENDS = ("mha", "sdpa", "flash")
_FLASH_DTYPES = (torch.float16, torch.bfloat16)


def _resolve_attention_backend() -> str:
    """Resolve the decoder attention backend from the environment (default: flash)."""
    backend = os.environ.get("XIAOMI_LENS_DECODER_ATTN", "").lower() or "flash"
    if backend not in _SUPPORTED_ATTENTION_BACKENDS:
        choices = "|".join(_SUPPORTED_ATTENTION_BACKENDS)
        raise ValueError(
            f"XIAOMI_LENS_DECODER_ATTN must be one of {choices}, got {backend!r}"
        )
    return backend


# All three implementations share MultiheadAttention projection weights, so
# switching kernels does not change checkpoints. With padding, the flash path
# packs only k/v and leaves q at full length to preserve MHA semantics for padded
# queries.
_DECODER_ATTN = _resolve_attention_backend()

# flash-attn is optional; import it lazily so CPU environments are unaffected.
_FLASH_FUNCS: FlashFunctions | None = None
_FLASH_IMPORT_ATTEMPTED = False
_FLASH_WARNED: set[str] = set()


def _flash_warn_once(key: str, message: str) -> None:
    """Log each fallback reason only once."""
    if key in _FLASH_WARNED:
        return
    _FLASH_WARNED.add(key)
    logger.warning(
        "PatchSelfAttention flash path disabled (%s); falling back to SDPA",
        message,
    )


def _get_flash_funcs() -> FlashFunctions | None:
    """Import and cache flash-attn functions on demand."""
    global _FLASH_FUNCS, _FLASH_IMPORT_ATTEMPTED
    if _FLASH_FUNCS is None and not _FLASH_IMPORT_ATTEMPTED:
        _FLASH_IMPORT_ATTEMPTED = True
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func

            _FLASH_FUNCS = (flash_attn_func, flash_attn_varlen_func)
        except Exception:
            # An incompatible binary extension may raise exceptions other than ImportError.
            _FLASH_FUNCS = None
    return _FLASH_FUNCS


class MLP(nn.Module):
    """Two-layer MLP: Linear → GELU → Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LayerNorm2d(nn.Module):
    """Apply LayerNorm over the channel dimension of `(B, C, H, W)`."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize channels at each spatial position with PyTorch's LayerNorm.
        y = x.permute(0, 2, 3, 1)
        y = F.layer_norm(y, (x.shape[1],), self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2)


class PatchSelfAttention(nn.Module):
    """Single-layer pre-norm Transformer block over patch tokens.

    This module performs only patch self-attention. It has no `<|mask|>`-to-patch
    cross-attention and does not upsample. Checkpoint keys from the former
    `PatchDecoder` name are converted at load time by
    `PixelMaskDecoder._remap_legacy_keys`.
    """

    def __init__(self, hidden_dim: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, intermediate_size, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform patch self-attention.

        Args:
            x: A `(B, P, hidden_dim)` patch sequence.
            key_padding_mask: Optional `(B, P)` bool mask where True denotes
                padding. By convention, fully padded placeholder rows must pass
                all False to avoid NaNs from an all-`-inf` softmax.
        """
        attention_output = self._self_attention(self.norm1(x), key_padding_mask)
        x = x + attention_output
        return x + self.mlp(self.norm2(x))

    def _self_attention(
        self,
        normed: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Select the configured attention kernel, falling back to SDPA when flash is unavailable.

        MultiheadAttention with a key padding mask materializes the (P, P) attention
        weights of every head, which does not fit in memory for large images; SDPA
        does not.
        """
        if _DECODER_ATTN == "mha":
            return self._mha_self_attention(normed, key_padding_mask)
        if _DECODER_ATTN == "flash":
            output = self._flash_self_attention(normed, key_padding_mask)
            if output is not None:
                return output
        return self._sdpa_self_attention(normed, key_padding_mask)

    def _mha_self_attention(
        self,
        normed: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.attn(
            normed,
            normed,
            normed,
            need_weights=False,
            key_padding_mask=key_padding_mask,
        )[0]

    def _project_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse MultiheadAttention parameters and return `(B, P, H, D)` q/k/v."""
        batch_size, num_patches, embed_dim = x.shape
        num_heads = self.attn.num_heads
        head_dim = embed_dim // num_heads
        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (batch_size, num_patches, num_heads, head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    def _project_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.attn.out_proj.weight, self.attn.out_proj.bias)

    def _sdpa_self_attention(
        self,
        normed: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run PyTorch SDPA using the MHA projection weights."""
        batch_size, num_patches, embed_dim = normed.shape
        q, k, v = self._project_qkv(normed)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attention_mask = None
        if key_padding_mask is not None:
            # Convert True padding to additive -inf; all-False rows get a zero mask.
            attention_mask = torch.zeros(
                (batch_size, 1, 1, num_patches),
                dtype=normed.dtype,
                device=normed.device,
            ).masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
        )
        output = output.transpose(1, 2).reshape(
            batch_size,
            num_patches,
            embed_dim,
        )
        return self._project_output(output)

    def _flash_self_attention(
        self,
        normed: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Run flash-attn FA2, returning None when the environment is unsupported.

        With a mask, use asymmetric varlen: retain all P q positions and pack only
        valid k/v positions. Padded queries can still attend to valid keys,
        matching MHA key-padding-mask semantics. Packing q and padding it back
        with zeros would change boundary values after final bilinear interpolation.
        """
        if not normed.is_cuda:
            _flash_warn_once("cuda", "input not on CUDA")
            return None

        # FA2 requires sm80+. Older GPUs such as V100/T4 can import it but fail at launch.
        major, minor = torch.cuda.get_device_capability(normed.device)
        if (major, minor) < (8, 0):
            _flash_warn_once(
                "arch",
                f"compute capability {major}.{minor} < 8.0 unsupported by FA2",
            )
            return None

        head_dim = self.attn.head_dim
        if head_dim % 8 != 0 or not 8 <= head_dim <= 256:
            _flash_warn_once("headdim", f"head_dim={head_dim} unsupported by FA2")
            return None

        flash_functions = _get_flash_funcs()
        if flash_functions is None:
            _flash_warn_once("import", "flash_attn not importable")
            return None

        q, k, v = self._project_qkv(normed)
        # LayerNorm may emit fp32 under autocast; the projected dtype controls the kernel.
        if q.dtype not in _FLASH_DTYPES:
            _flash_warn_once("dtype", f"projected dtype {q.dtype} not fp16/bf16")
            return None

        flash_attn_func, flash_attn_varlen_func = flash_functions
        batch_size, num_patches, _, _ = q.shape
        # Model forward always passes a mask tensor. For all-False masks without
        # padding, use the fixed-length kernel to avoid nonzero plus two full K/V copies.
        if key_padding_mask is None or not key_padding_mask.any():
            output = flash_attn_func(q, k, v)
            output = output.reshape(batch_size, num_patches, -1)
        else:
            output = self._flash_varlen_attention(
                q,
                k,
                v,
                key_padding_mask,
                flash_attn_varlen_func,
            )
        return self._project_output(output)

    @staticmethod
    def _flash_varlen_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor,
        flash_attn_varlen_func: AttentionFunction,
    ) -> torch.Tensor:
        """Pack valid k/v and run FA2 with fixed-length q and variable-length k/v."""
        batch_size, num_patches, num_heads, head_dim = q.shape
        valid_mask = ~key_padding_mask
        valid_indices = valid_mask.reshape(-1).nonzero(as_tuple=True)[0]

        flat_shape = (batch_size * num_patches, num_heads, head_dim)
        unpadded_k = k.reshape(flat_shape).index_select(0, valid_indices)
        unpadded_v = v.reshape(flat_shape).index_select(0, valid_indices)

        # Fully padded placeholders pass all False, so every row has valid keys.
        key_lengths = valid_mask.sum(dim=1, dtype=torch.int32)
        cumulative_k = F.pad(key_lengths.cumsum(0), (1, 0)).to(torch.int32)
        cumulative_q = torch.arange(
            0,
            (batch_size + 1) * num_patches,
            num_patches,
            device=q.device,
            dtype=torch.int32,
        )
        output = flash_attn_varlen_func(
            q.reshape(flat_shape),
            unpadded_k,
            unpadded_v,
            cu_seqlens_q=cumulative_q,
            cu_seqlens_k=cumulative_k,
            # Using P as the upper bound avoids a host sync from lengths.max().item().
            max_seqlen_q=num_patches,
            max_seqlen_k=num_patches,
        )
        return output.view(batch_size, num_patches, num_heads * head_dim)


class PixelMaskDecoder(nn.Module):
    """Decode `<|mask|>` hidden states into full-resolution mask logits.

    Configuration fields:
        llm_hidden_size, llm_intermediate_size, llm_num_heads: Text-model parameters.
        vision_hidden_size: Pre-merge ViT hidden size.
        mask_decoder_channels: Channel count C after 1×1 reduction.
        mask_decoder_embed_dim: Pixel-embedding channel count C_emb, which must
            match the mask_embed output dimension.
    """

    def __init__(self, config: Mapping[str, int]):
        super().__init__()
        llm_hidden = int(config["llm_hidden_size"])
        llm_intermediate = int(config["llm_intermediate_size"])
        llm_num_heads = int(config["llm_num_heads"])
        vision_hidden = int(config["vision_hidden_size"])
        channels = int(config["mask_decoder_channels"])
        embed_dim = int(config["mask_decoder_embed_dim"])

        # Upsampling channels are C → C/2 → C_emb. If C/2 has only one channel,
        # LayerNorm2d always emits zero and makes the decoder image-independent,
        # so require at least two intermediate channels.
        if channels < 4 or channels % 2 != 0:
            raise ValueError(
                "mask_decoder_channels must be even and >= 4 "
                f"(so the mid layer C/2 has >= 2 channels); got {channels}"
            )

        self.embed_dim = embed_dim
        self.mask_embed = MLP(llm_hidden, llm_hidden, embed_dim)
        self.patch_proj = nn.Linear(vision_hidden, llm_hidden)
        self.patch_self_attn = PatchSelfAttention(
            llm_hidden,
            llm_num_heads,
            llm_intermediate,
        )
        self.channel_reduce = nn.Conv2d(llm_hidden, channels, kernel_size=1)

        # Two stride-2 deconvolutions learn 4× upsampling, followed by fixed 4×
        # interpolation in forward to produce a 16× patch-grid canvas.
        intermediate_channels = channels // 2
        self.upscaler = nn.Sequential(
            nn.ConvTranspose2d(
                channels,
                intermediate_channels,
                kernel_size=2,
                stride=2,
            ),
            LayerNorm2d(intermediate_channels),
            nn.GELU(),
            nn.ConvTranspose2d(
                intermediate_channels,
                embed_dim,
                kernel_size=2,
                stride=2,
            ),
        )

        # Support the legacy `patch_decoder.*` checkpoint parameter names.
        self._register_load_state_dict_pre_hook(self._remap_legacy_keys)

    @staticmethod
    def _remap_legacy_keys(
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        *_args,
        **_kwargs,
    ) -> None:
        old_prefix = prefix + "patch_decoder."
        new_prefix = prefix + "patch_self_attn."
        old_keys = [key for key in state_dict if key.startswith(old_prefix)]
        for old_key in old_keys:
            suffix = old_key[len(old_prefix) :]
            state_dict[new_prefix + suffix] = state_dict.pop(old_key)

    def forward(
        self,
        mask_hidden: torch.Tensor,
        image_features: torch.Tensor,
        grid_h: int,
        grid_w: int,
        patch_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate mask logits in a batch.

        Args:
            mask_hidden: `(B, K, llm_hidden)` with K mask queries per group.
            image_features: `(B, grid_h * grid_w, vision_hidden)`, with smaller
                grids zero-padded to the batch maximum.
            grid_h: Shared patch-grid height for the batch.
            grid_w: Shared patch-grid width for the batch.
            patch_key_padding_mask: Optional `(B, grid_h * grid_w)` bool mask
                where True denotes padding. Fully padded placeholder rows must
                pass all False.

        Returns:
            `(B, K, grid_h * 16, grid_w * 16)` mask logits.
        """
        batch_size = self._validate_inputs(
            mask_hidden,
            image_features,
            grid_h,
            grid_w,
            patch_key_padding_mask,
        )

        mask_embedding = self.mask_embed(mask_hidden)
        patch_features = self.patch_proj(image_features)
        patch_embedding = self.patch_self_attn(
            patch_features,
            patch_key_padding_mask,
        )
        patch_embedding = patch_embedding.reshape(
            batch_size,
            grid_h,
            grid_w,
            -1,
        ).permute(0, 3, 1, 2)

        reduced_features = self.channel_reduce(patch_embedding)
        pixel_embedding = self.upscaler(reduced_features)

        # Divide by √C_emb to prevent high initial dot-product variance from saturating sigmoid.
        mask_logits = torch.einsum(
            "bkc,bchw->bkhw",
            mask_embedding,
            pixel_embedding,
        ) / (self.embed_dim**0.5)

        # Interpolate K independently as channels; a fixed integer scale stabilizes
        # sampling offsets.
        return F.interpolate(
            mask_logits,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _validate_inputs(
        mask_hidden: torch.Tensor,
        image_features: torch.Tensor,
        grid_h: int,
        grid_w: int,
        patch_key_padding_mask: torch.Tensor | None,
    ) -> int:
        """Validate decoder inputs and return the batch size."""
        if mask_hidden.dim() != 3:
            raise ValueError(
                "mask_hidden must be (B, K, llm_hidden); "
                f"got {tuple(mask_hidden.shape)}"
            )
        if image_features.dim() != 3:
            raise ValueError(
                "image_features must be (B, P, vision_hidden); "
                f"got {tuple(image_features.shape)}"
            )

        batch_size = mask_hidden.shape[0]
        image_batch_size, num_patches = image_features.shape[:2]
        if image_batch_size != batch_size:
            raise ValueError(
                "mask_hidden and image_features must share batch dim; "
                f"got {batch_size} vs {image_batch_size}"
            )

        expected_patches = grid_h * grid_w
        if num_patches != expected_patches:
            raise ValueError(
                f"Patch feature count {num_patches} does not match grid "
                f"({grid_h}, {grid_w})"
            )

        if patch_key_padding_mask is not None:
            expected_shape = (batch_size, num_patches)
            if tuple(patch_key_padding_mask.shape) != expected_shape:
                raise ValueError(
                    "patch_key_padding_mask must have shape "
                    f"{expected_shape}; got {tuple(patch_key_padding_mask.shape)}"
                )
            if patch_key_padding_mask.dtype != torch.bool:
                raise TypeError(
                    "patch_key_padding_mask must use torch.bool; "
                    f"got {patch_key_padding_mask.dtype}"
                )

        return batch_size
