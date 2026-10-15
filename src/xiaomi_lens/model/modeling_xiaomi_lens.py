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

"""Extend Transformers' Qwen3.5 model with a <|mask|> token and pixel-mask decoder.

Core design:
1. Add the <|mask|> special token to the vocabulary.
2. PixelMaskDecoder applies self-attention to pre-merge ViT patch features and
   upsamples them 4× with two stride-2 transposed convolutions.
3. Each <|mask|> hidden state produces a dynamic classifier over the resulting
   pixel embeddings. A further 4× bilinear interpolation brings the mask logits
   to `(grid_h*16, grid_w*16)`, matching the vision encoder's input resolution.

This module contains the inference path only.
"""

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)

from .config import build_mask_decoder_config, get_qwen_vision_config
from .pixel_mask_decoder import PixelMaskDecoder

# Each ViT patch spans 16 input pixels. Learned 4× upsampling followed by 4×
# bilinear interpolation restores masks to the vision encoder's input resolution.
MASK_CANVAS_SCALE = 16


@dataclass
class XiaomiLensOutput(ModelOutput):
    """Output type for XiaomiLensQwen3_5ForConditionalGeneration."""
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[List] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    rope_deltas: Optional[torch.Tensor] = None
    pred_masks: Optional[torch.Tensor] = None
    pred_mask_valid: Optional[torch.Tensor] = None


class XiaomiLensQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        decoder_config = getattr(config, 'mask_decoder', None)
        if decoder_config is None:
            decoder_config = build_mask_decoder_config(config)
        self.mask_decoder = PixelMaskDecoder(decoder_config)

    @staticmethod
    def _get_vision_config(config):
        return get_qwen_vision_config(config)

    def _get_spatial_merge_size(self) -> int:
        vision_config = self._get_vision_config(self.config)
        return int(getattr(vision_config, "spatial_merge_size", 1) or 1)

    @staticmethod
    def _flatten_image_features(visual_outputs) -> list[torch.Tensor]:
        image_features = visual_outputs.pooler_output
        if image_features is None:
            raise ValueError(
                "visual_outputs.pooler_output is None; ensure get_image_features returns patch features."
            )
        if isinstance(image_features, torch.Tensor):
            return [image_features]
        return list(image_features)

    def _extract_pre_merge_features(
        self,
        visual_outputs,
        image_grid_thw: torch.LongTensor,
    ) -> list[torch.Tensor]:
        """Split ViT `last_hidden_state` per-image and reorder to row-major spatial layout.

        Qwen3.5 ViT stores tokens grouped by merge blocks:
            (merged_h, merged_w, merge_size, merge_size, hidden)
        We permute back to row-major (grid_h, grid_w, hidden) at patch_size=16.
        """
        raw = visual_outputs.last_hidden_state
        if raw is None:
            raise ValueError(
                "visual_outputs.last_hidden_state is None; required for pre-merge patch features."
            )
        merge_size = self._get_spatial_merge_size()
        # One host copy covers all images, replacing three `.item()` syncs per image.
        thw_l = image_grid_thw.tolist()
        token_counts = [int(t) * int(h) * int(w) for t, h, w in thw_l]
        per_image = list(torch.split(raw, token_counts, dim=0))

        reordered: list[torch.Tensor] = []
        for batch_idx, tokens in enumerate(per_image):
            grid_t = int(thw_l[batch_idx][0])
            grid_h = int(thw_l[batch_idx][1])
            grid_w = int(thw_l[batch_idx][2])
            if grid_t != 1:
                raise ValueError(f"Xiaomi-Lens-0 expects image_grid_thw[:, 0] == 1 for images, got {grid_t}.")
            if grid_h % merge_size != 0 or grid_w % merge_size != 0:
                raise ValueError(
                    f"Image grid ({grid_h}, {grid_w}) must be divisible by merge_size={merge_size}."
                )
            merged_h = grid_h // merge_size
            merged_w = grid_w // merge_size
            hidden = tokens.shape[-1]
            # (merged_h, merged_w, m, m, hidden) -> (merged_h, m, merged_w, m, hidden) -> (grid_h, grid_w, hidden)
            spatial = (
                tokens.view(merged_h, merged_w, merge_size, merge_size, hidden)
                .permute(0, 2, 1, 3, 4)
                .reshape(grid_h * grid_w, hidden)
                .contiguous()
            )
            reordered.append(spatial)
        return reordered

    def _prepare_multimodal_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.LongTensor],
        mm_token_type_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.FloatTensor],
    ):
        visual_outputs = None
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        has_images = (
            pixel_values is not None
            and image_grid_thw is not None
            and pixel_values.numel() > 0
            and image_grid_thw.numel() > 0
        )

        if has_images:
            visual_outputs = self.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_features = self._flatten_image_features(visual_outputs)
            image_embeds = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            # Insert spatially merged visual features into the image-token slots.

        if position_ids is None:
            position_ids = self.model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw if has_images else None,
                video_grid_thw=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        return inputs_embeds, visual_outputs, position_ids

    def _decode_pixel_masks(
        self,
        hidden_states: torch.Tensor,
        mask_positions: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        visual_outputs,
        images_per_sample: Optional[torch.Tensor] = None,
        mask_image_index: Optional[torch.Tensor] = None,
    ):
        """Decode pixel masks via a single batched mask_decoder call.

        **The decoder's batch dim is a `(sample, image)` GROUP, not a sample.**
        `mask_image_index[b, k]` (sample-local) says which of row `b`'s images
        the k-th `<|mask|>` binds to — the nearest image preceding it in the
        token stream. Every `<|mask|>` bound to the same image of the same
        sample shares one group, so that image's patch features are lifted and
        attended over exactly once no matter how many queries hit it.

        `images_per_sample[i]` is the number of images owned by batch row `i`
        (0 / 1 / ≥2); a row with 0 images produces no group and its outputs
        stay zero. The canvas is sized to the largest image any `<|mask|>`
        binds to.

        Returns `(pred_masks, pred_mask_valid)`, both
        `(B, K, max_grid_h*16, max_grid_w*16)`: raw mask logits, and a 0/1 map
        of each mask's real (unpadded) area.
        """
        if mask_positions.dim() == 1:
            mask_positions = mask_positions.unsqueeze(1)
        mask_positions = mask_positions.to(hidden_states.device)

        batch_size = hidden_states.shape[0]
        K = mask_positions.shape[1]
        num_images = int(image_grid_thw.shape[0])

        if images_per_sample is None:
            # Back-compat: assume every sample contributes exactly one image.
            images_per_sample = torch.ones(batch_size, dtype=torch.long, device=hidden_states.device)
        else:
            images_per_sample = images_per_sample.to(hidden_states.device).long()

        # One host sync up front rather than an `.item()` per element in the loops.
        counts = images_per_sample.tolist()
        if sum(counts) != num_images:
            raise ValueError(
                f"images_per_sample sums to {sum(counts)} but "
                f"image_grid_thw has {num_images} rows."
            )

        positions_l = mask_positions.tolist()
        if mask_image_index is None:
            img_idx_l = [[0] * K for _ in range(batch_size)]
        else:
            img_idx_l = mask_image_index.to(hidden_states.device).long().tolist()

        # Global image row of each sample's image 0.
        offsets, _cursor = [], 0
        for count in counts:
            offsets.append(_cursor)
            _cursor += count

        # Group the valid (b, k) slots by the IMAGE they bind to. Insertion order
        # makes the group index deterministic.
        groups: dict = {}
        for b in range(batch_size):
            n_img = counts[b]
            if n_img == 0:
                continue  # zero-image row: no group, its outputs stay zero
            for k in range(K):
                local = min(max(int(img_idx_l[b][k]), 0), n_img - 1)
                groups.setdefault((b, offsets[b] + local), []).append(k)

        group_keys = list(groups.keys())
        # Clamp to ≥1: nn.MultiheadAttention on a zero-size batch is a landmine.
        num_groups = max(1, len(group_keys))
        max_queries = max(1, max((len(v) for v in groups.values()), default=1))

        if group_keys:
            # The canvas covers every image a <|mask|> binds to.
            _rows = torch.tensor(
                [row for _, row in group_keys], device=image_grid_thw.device, dtype=torch.long,
            )
            max_grid_h = int(image_grid_thw[_rows, 1].max().item())
            max_grid_w = int(image_grid_thw[_rows, 2].max().item())
        else:
            max_grid_h = int(image_grid_thw[:, 1].max().item())
            max_grid_w = int(image_grid_thw[:, 2].max().item())
        max_patches = max_grid_h * max_grid_w
        vision_hidden = int(self._get_vision_config(self.config).hidden_size)
        llm_hidden = hidden_states.shape[-1]

        image_patch_features = self._extract_pre_merge_features(visual_outputs, image_grid_thw)
        if len(image_patch_features) != num_images:
            raise ValueError(
                f"Expected {num_images} image feature groups, but got {len(image_patch_features)}."
            )

        # Decoder inputs are per-GROUP; outputs are scattered back per-(sample, k).
        grouped_mask_hidden = hidden_states.new_zeros((num_groups, max_queries, llm_hidden))
        grouped_image_features = hidden_states.new_zeros((num_groups, max_patches, vision_hidden))
        # Mark each mask's unpadded area at the decoder's output resolution,
        # which is 16× the patch grid along each spatial axis.
        mask_h = max_grid_h * MASK_CANVAS_SCALE
        mask_w = max_grid_w * MASK_CANVAS_SCALE
        pred_mask_valid = hidden_states.new_zeros((batch_size, K, mask_h, mask_w))
        # True marks padded keys ignored by self-attention. Only groups with
        # real image features receive padding masks; an empty placeholder stays
        # all-False and avoids the all-`-inf` softmax NaN.
        patch_key_padding_mask = torch.zeros(
            (num_groups, max_patches), dtype=torch.bool, device=hidden_states.device
        )

        # Scatter coordinates: grouped[g_idx, j_idx] → pred_masks[b_idx, k_idx].
        b_idx: list = []
        k_idx: list = []
        g_idx: list = []
        j_idx: list = []

        # Share one host copy across the scatter loop to avoid per-group `.item()` syncs.
        thw_l = image_grid_thw.tolist()
        for g, (b, image_row) in enumerate(group_keys):
            grid_h = int(thw_l[image_row][1])
            grid_w = int(thw_l[image_row][2])
            feats = image_patch_features[image_row]
            expected = grid_h * grid_w
            if feats.shape[0] != expected:
                raise ValueError(
                    f"Expected {expected} pre-merge ViT patch features, got {feats.shape[0]}."
                )
            if grid_h > max_grid_h or grid_w > max_grid_w:
                raise ValueError(
                    f"Image grid ({grid_h}, {grid_w}) exceeds the mask canvas grid "
                    f"({max_grid_h}, {max_grid_w})."
                )

            # This image's features, shared by every query in the group via einsum.
            feat_2d = feats.reshape(grid_h, grid_w, vision_hidden)
            padded = F.pad(
                feat_2d,
                (0, 0, 0, max_grid_w - grid_w, 0, max_grid_h - grid_h),
            )
            grouped_image_features[g] = padded.reshape(max_patches, vision_hidden)

            # Mark the padded patches (outside this image's real grid) so the
            # decoder's self-attention ignores them. Layout matches the F.pad
            # above: position (r, c) in the (max_grid_h, max_grid_w) row-major
            # flatten is real iff r < grid_h and c < grid_w.
            pad_mask_2d = torch.ones(
                (max_grid_h, max_grid_w), dtype=torch.bool, device=hidden_states.device
            )
            pad_mask_2d[:grid_h, :grid_w] = False
            patch_key_padding_mask[g] = pad_mask_2d.reshape(max_patches)

            for j, k in enumerate(groups[(b, image_row)]):
                pos = int(positions_l[b][k])
                grouped_mask_hidden[g, j] = hidden_states[b, pos]
                pred_mask_valid[
                    b, k,
                    : grid_h * MASK_CANVAS_SCALE,
                    : grid_w * MASK_CANVAS_SCALE,
                ] = 1
                b_idx.append(b)
                k_idx.append(k)
                g_idx.append(g)
                j_idx.append(j)

        grouped = self.mask_decoder(
            grouped_mask_hidden,         # (N, Kg, llm_hidden)
            grouped_image_features,      # (N, max_patches, vision_hidden)
            grid_h=max_grid_h,
            grid_w=max_grid_w,
            patch_key_padding_mask=patch_key_padding_mask,  # (N, max_patches) True=pad
        )  # (N, Kg, max_grid_h*16, max_grid_w*16) — already canvas resolution

        # Scatter the groups back into the (sample, k) layout.
        pred_masks = grouped.new_zeros((batch_size, K, mask_h, mask_w))
        if b_idx:
            pred_masks[b_idx, k_idx] = grouped[g_idx, j_idx]
        del grouped
        return pred_masks, pred_mask_valid

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        mask_positions: Optional[torch.Tensor] = None,
        mask_image_index: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Run the LM and, when `mask_positions` is given, decode one mask per `<|mask|>`.

        `mask_positions` is `(B, K)` token indices of `<|mask|>` slots in
        `input_ids`. `mask_image_index` optionally binds each slot to one of the
        sample's images; by default each slot binds to the nearest preceding
        image.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise ValueError("Xiaomi-Lens-0 currently supports image inputs only.")

        inputs_embeds, visual_outputs, position_ids = self._prepare_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        outputs = self.model.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden)

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        pred_masks = None
        pred_mask_valid = None
        has_mask_work = bool(
            mask_positions is not None
            and mask_positions.numel() > 0
            and image_grid_thw is not None
            and image_grid_thw.numel() > 0
            and visual_outputs is not None
        )
        if has_mask_work:
            images_per_sample = None
            vision_start_id = getattr(self.config, "vision_start_token_id", None)
            if input_ids is not None and vision_start_id is not None:
                is_vision_start = (input_ids == vision_start_id)
                images_per_sample = is_vision_start.sum(dim=1)
                if mask_image_index is None:
                    # Each `<|mask|>` binds to the nearest image preceding it.
                    seq_len = input_ids.shape[1]
                    pos = mask_positions
                    if pos.dim() == 1:
                        pos = pos.unsqueeze(1)
                    pos = pos.to(input_ids.device).clamp(0, seq_len - 1)
                    seen = is_vision_start.long().cumsum(dim=1)
                    mask_image_index = (seen.gather(1, pos) - 1).clamp_min(0)
            pred_masks, pred_mask_valid = self._decode_pixel_masks(
                hidden_states=hidden_states,
                mask_positions=mask_positions,
                image_grid_thw=image_grid_thw,
                visual_outputs=visual_outputs,
                images_per_sample=images_per_sample,
                mask_image_index=mask_image_index,
            )

        if not return_dict:
            return (logits,) + outputs[1:]

        return XiaomiLensOutput(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=getattr(self.model, "rope_deltas", None),
            pred_masks=pred_masks,
            pred_mask_valid=pred_mask_valid,
        )
