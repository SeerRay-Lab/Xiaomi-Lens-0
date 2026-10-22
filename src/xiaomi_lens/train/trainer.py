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

"""Xiaomi-Lens-0 trainer.

Responsibilities:
1. Build the Qwen3.5 backbone with a `<|mask|>` token and a PixelMaskDecoder.
2. Apply the `--unfreeze_*` trainability recipe.
3. Stream JSONL samples through `LazyCollator` and run LM + mask training.
"""

from __future__ import annotations

from collections import defaultdict
import os
import time
from typing import Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, PreTrainedModel, Trainer, TrainerCallback
from transformers.trainer import TRAINING_ARGS_NAME, logger

from ..constants import MASK_TOKEN
from ..model.config import build_mask_decoder_config
from ..model.lm_loss import LM_LOSS_IMPLS
from ..model.modeling_xiaomi_lens import (
    MASK_CANVAS_SCALE,
    XiaomiLensQwen3_5ForConditionalGeneration,
)
from .config import XiaomiLensModelConfig, XiaomiLensScriptArguments, XiaomiLensTrainingConfig


class _FreezeNonTrainableEmbedRows:
    """Zero `embed_tokens.weight.grad` on every row except `trainable_token_ids`.

    A `Tensor.register_hook` fires during backward and returns the masked
    gradient before autograd accumulates it. This runs before DeepSpeed's
    post-accumulate hook copies `.grad` into its bucket, and before
    `clip_grad_norm_`, so frozen rows neither update nor inflate the gradient
    norm. Supported with single GPU, DDP and DeepSpeed ZeRO-1/2, where the
    gradient has the full `(V, H)` shape on every rank.
    """

    def __init__(self, embed_tokens, trainable_token_ids: tuple[int, ...]) -> None:
        self._trainable_token_ids = list(trainable_token_ids)
        self._vocab_size: int = embed_tokens.num_embeddings
        self._hidden_size: int = embed_tokens.embedding_dim
        for tid in self._trainable_token_ids:
            if not (0 <= tid < self._vocab_size):
                raise ValueError(
                    f"trainable_token_id {tid} out of range [0, {self._vocab_size})."
                )
        self._mask_cache: dict[tuple, torch.Tensor] = {}
        self._warned_unknown_shape = False

        param = embed_tokens.weight
        if not param.requires_grad:
            logger.warning(
                "[freeze-embed-rows] embed_tokens.weight requires_grad=False; "
                "grad hook not installed."
            )
            return
        param.register_hook(self._hook)
        logger.info(
            "[freeze-embed-rows] grad hook installed on embed_tokens.weight "
            "(trainable_token_ids=%s, V=%d, H=%d).",
            tuple(self._trainable_token_ids), self._vocab_size, self._hidden_size,
        )

    def _hook(self, grad: torch.Tensor) -> Optional[torch.Tensor]:
        V, H = self._vocab_size, self._hidden_size
        cache_key = (grad.device, grad.dtype, tuple(grad.shape))
        keep = self._mask_cache.get(cache_key)
        if keep is None:
            if grad.dim() == 2 and grad.shape == (V, H):
                keep = torch.zeros(V, 1, dtype=grad.dtype, device=grad.device)
                for tid in self._trainable_token_ids:
                    keep[tid, 0] = 1
            elif grad.dim() == 1 and grad.numel() == V * H:
                keep = torch.zeros(V * H, dtype=grad.dtype, device=grad.device)
                for tid in self._trainable_token_ids:
                    keep[tid * H : (tid + 1) * H] = 1
            else:
                # A partitioned gradient (e.g. ZeRO-3) cannot be aligned to rows.
                if not self._warned_unknown_shape:
                    self._warned_unknown_shape = True
                    logger.warning(
                        "[freeze-embed-rows] unsupported grad shape %s (V=%d, H=%d); "
                        "masking is SKIPPED and frozen rows may drift. Use "
                        "DeepSpeed ZeRO-1 or ZeRO-2.",
                        tuple(grad.shape), V, H,
                    )
                return None
            self._mask_cache[cache_key] = keep
        return grad * keep


class XiaomiLensTrainer(Trainer):

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        script_args: XiaomiLensScriptArguments = None,
        training_args: XiaomiLensTrainingConfig = None,
        model_args: XiaomiLensModelConfig = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        **kwargs,
    ):
        self.script_args = script_args
        self.model_args = model_args
        self._metrics = defaultdict(list)
        uses_row_mask = (
            not model_args.unfreeze_all_parameters
            and not model_args.unfreeze_embed_tokens
            and model_args.unfreeze_new_tokens
        )
        # The row-mask hook zeros gradients, but decoupled weight decay would
        # still shrink every frozen row of the shared embedding parameter.
        if uses_row_mask and training_args.weight_decay > 0:
            raise ValueError(
                "`--unfreeze_new_tokens` trains one row of embed_tokens.weight; "
                "set `--weight_decay 0` so AdamW does not decay the frozen rows."
            )

        processor_source = model if isinstance(model, str) else model.config._name_or_path
        processing_class = AutoProcessor.from_pretrained(processor_source)
        self._configure_image_processor(processing_class.image_processor, script_args)
        if script_args.model_max_length is not None:
            processing_class.tokenizer.model_max_length = script_args.model_max_length

        # Only `<|mask|>` adds a tokenizer row; `<bbox>` / `</bbox>` reuse the
        # native box-token IDs (see tools/prepare_base_model.py). Derive the
        # old/new row partition from the tokenizer, not the embedding shape:
        # Qwen embeddings are padded past the real vocabulary size.
        num_new_tokens = processing_class.tokenizer.add_special_tokens(
            {"additional_special_tokens": [MASK_TOKEN]}
        )
        self.mask_token_id = processing_class.tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        target_vocab_size = len(processing_class.tokenizer)
        original_vocab_size = target_vocab_size - num_new_tokens
        logger.info(
            "[vocab-resize] tokenizer added %d new token(s); "
            "original_vocab_size=%d, target_vocab_size=%d",
            num_new_tokens, original_vocab_size, target_vocab_size,
        )

        model_instance = self._build_model(model, training_args, model_args)
        # New rows are overwritten with the vocabulary mean right after, so
        # skip Transformers' covariance-based initialization.
        model_instance.resize_token_embeddings(target_vocab_size, mean_resizing=False)
        self._mean_init_new_token_embeddings(model_instance, original_vocab_size)
        if (
            hasattr(model_instance.config, "text_config")
            and model_instance.config.text_config.vocab_size != target_vocab_size
        ):
            model_instance.config.text_config.vocab_size = target_vocab_size
        self._apply_param_trainability(model_instance, model_args)
        self._disable_unneeded_gradient_checkpointing(model_instance, training_args)

        from .collator import LazyCollator
        from .data.io import load_dataset_config, resolve_dataset_entries
        from .data.jsonl_index import build_jsonl_index_dataset

        if train_dataset is None:
            train_dataset = build_jsonl_index_dataset(
                resolve_dataset_entries(script_args),
                seed=training_args.seed or 0,
            )
        eval_strategy = getattr(
            training_args.eval_strategy, "value", training_args.eval_strategy
        )
        if eval_dataset is None and eval_strategy != "no":
            if not script_args.data_config:
                raise ValueError(
                    "Evaluation requires --data_config with explicit `train:` "
                    "and `eval:` sections."
                )
            eval_dataset = build_jsonl_index_dataset(
                load_dataset_config(script_args.data_config, split="eval"),
                seed=training_args.seed or 0,
            )
        data_collator = LazyCollator(
            processing_class,
            mask_token_id=self.mask_token_id,
            model_max_length=int(processing_class.tokenizer.model_max_length),
            max_pixels=getattr(script_args, "max_pixels", None),
            min_pixels=getattr(script_args, "min_pixels", None),
        )

        super().__init__(
            model=model_instance,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # The row mask is keyed to `mask_token_id`, so it also works when the
        # tokenizer already contains `<|mask|>` (resuming a Xiaomi-Lens-0
        # checkpoint, where `num_new_tokens == 0`).
        self._row_mask_helper: Optional[_FreezeNonTrainableEmbedRows] = None
        if uses_row_mask:
            embed_tokens = self.model.get_input_embeddings()
            self._row_mask_helper = _FreezeNonTrainableEmbedRows(
                embed_tokens, trainable_token_ids=(self.mask_token_id,)
            )

    @staticmethod
    def _disable_unneeded_gradient_checkpointing(
        model: PreTrainedModel,
        training_args: XiaomiLensTrainingConfig,
    ) -> None:
        """Keep the frozen backbone out of the backward graph in decoder-only training.

        With `gradient_checkpointing=True`, Transformers makes embedding
        outputs require gradients, which reconnects a fully frozen ViT/LLM to
        the backward graph. When every trainable tensor lives under
        `mask_decoder`, turn the Trainer-level flag off and opt the decoder and
        `MaskLoss` in to their own recomputation instead. Their inputs already
        carry `grad_fn` from trainable decoder parameters.
        """
        if not training_args.gradient_checkpointing:
            return
        trainable_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if trainable_names and all(
            name.startswith("mask_decoder.") for name in trainable_names
        ):
            training_args.gradient_checkpointing = False
            for attribute in ("mask_decoder", "mask_loss_fn"):
                getattr(model, attribute).gradient_checkpointing = True
            logger.info(
                "[decoder-only] disabled HF gradient checkpointing: all %d trainable "
                "tensors are under mask_decoder. Kept recomputation for "
                "mask_decoder and mask_loss_fn.",
                len(trainable_names),
            )

    @staticmethod
    def _set_requires_grad(module, value: bool, predicate=None) -> None:
        if module is None:
            return
        for name, parameter in module.named_parameters():
            if predicate is None or predicate(name):
                parameter.requires_grad = value

    def _apply_param_trainability(
        self,
        model: PreTrainedModel,
        model_args: XiaomiLensModelConfig,
    ) -> None:
        """Freeze every parameter, then re-enable the modules named by `--unfreeze_*`.

        Flags are additive. `unfreeze_embed_tokens=True` wins over
        `unfreeze_new_tokens=True`, so no row-mask hook is installed and every
        embedding row trains.
        """
        if model_args.unfreeze_all_parameters:
            for p in model.parameters():
                p.requires_grad = True
            logger.info("[param-trainability] unfreeze_all_parameters=True — entire model trainable.")
            return

        for p in model.parameters():
            p.requires_grad = False

        base_model = getattr(model, "model", None)
        visual = getattr(base_model, "visual", None) if base_model else None
        language_model = getattr(base_model, "language_model", None) if base_model else None

        if model_args.unfreeze_vision_tower:
            if visual is None:
                raise ValueError("Qwen3.5 visual tower is required for `unfreeze_vision_tower`.")
            self._set_requires_grad(
                visual, True, predicate=lambda name: not name.startswith("merger.")
            )

        if model_args.unfreeze_vision_projection:
            projection = getattr(visual, "merger", None) if visual else None
            if projection is None:
                raise ValueError("Qwen3.5 visual projection module `model.visual.merger` was not found.")
            self._set_requires_grad(projection, True)

        if model_args.unfreeze_language_model_backbone:
            if language_model is None:
                raise ValueError(
                    "Qwen3.5 language model is required for `unfreeze_language_model_backbone`."
                )
            self._set_requires_grad(
                language_model, True,
                predicate=lambda name: not name.startswith("embed_tokens"),
            )

        if model_args.unfreeze_language_model_head:
            lm_head = getattr(model, "lm_head", None)
            if lm_head is None:
                raise ValueError(
                    "`lm_head` was not found on the model; cannot apply `unfreeze_language_model_head`."
                )
            self._set_requires_grad(lm_head, True)

        embed_tokens = getattr(language_model, "embed_tokens", None) if language_model else None
        if model_args.unfreeze_embed_tokens:
            if embed_tokens is None:
                raise ValueError(
                    "`embed_tokens` not found on `language_model`; cannot apply `unfreeze_embed_tokens`."
                )
            self._set_requires_grad(embed_tokens, True)
        elif model_args.unfreeze_new_tokens:
            if embed_tokens is None:
                raise ValueError(
                    "`embed_tokens` not found on `language_model`; cannot apply `unfreeze_new_tokens`."
                )
            vocab_size = embed_tokens.weight.shape[0]
            if not (0 <= self.mask_token_id < vocab_size):
                raise ValueError(
                    f"`<|mask|>` token id ({self.mask_token_id}) is not within "
                    f"embed_tokens vocab_size ({vocab_size})."
                )
            # On a tied-embedding model the row-mask hook would also zero every
            # non-<|mask|> row of lm_head, silently undoing the head unfreeze.
            if model_args.unfreeze_language_model_head:
                output_embeddings = model.get_output_embeddings()
                if (
                    output_embeddings is not None
                    and output_embeddings.weight is embed_tokens.weight
                ):
                    raise ValueError(
                        "`unfreeze_language_model_head=True` and "
                        "`unfreeze_new_tokens=True` cannot be combined on a "
                        "tied-embedding model (`lm_head.weight is "
                        "embed_tokens.weight`). Use `unfreeze_embed_tokens=True` "
                        "to train the shared parameter, or drop one flag."
                    )
            embed_tokens.weight.requires_grad = True

        if model_args.unfreeze_mask_decoder:
            self._set_requires_grad(model.mask_decoder, True)

    @staticmethod
    def _mean_init_new_token_embeddings(
        model: PreTrainedModel,
        original_vocab_size: int,
    ) -> None:
        """Initialize new-token rows to the mean of the original vocabulary rows.

        `resize_token_embeddings` appends random rows. Starting `<|mask|>` at
        the vocabulary mean keeps it in-distribution at the first forward.
        Tied and untied output embeddings are both handled.
        """
        input_param = model.get_input_embeddings().weight
        if input_param is None or original_vocab_size >= input_param.shape[0]:
            return

        with torch.no_grad():
            input_mean = input_param.data[:original_vocab_size].mean(dim=0, keepdim=True)
            input_param.data[original_vocab_size:] = input_mean

            output = model.get_output_embeddings()
            output_param = output.weight if output is not None else None
            if output_param is not None and output_param is not input_param:
                output_mean = output_param.data[:original_vocab_size].mean(dim=0, keepdim=True)
                output_param.data[original_vocab_size:] = output_mean

    def _build_model(
        self,
        model_id: Union[str, PreTrainedModel],
        training_args: XiaomiLensTrainingConfig,
        model_args: XiaomiLensModelConfig,
    ) -> PreTrainedModel:
        if isinstance(model_id, PreTrainedModel):
            # Trainer counts `num_items_in_batch` using this flag, and the LM
            # loss reads it from the model config; they must agree.
            model_id.config.average_tokens_across_devices = getattr(
                training_args, "average_tokens_across_devices", True
            )
            return model_id

        if training_args.lm_loss_impl not in LM_LOSS_IMPLS:
            raise ValueError(
                f"--lm_loss_impl must be one of {LM_LOSS_IMPLS}, "
                f"got {training_args.lm_loss_impl!r}."
            )

        model_init_kwargs = dict(training_args.model_init_kwargs or {})
        model_init_kwargs["attn_implementation"] = model_args.attn_implementation
        torch_dtype = model_init_kwargs.get("torch_dtype") or model_args.torch_dtype or "bfloat16"
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            torch_dtype = getattr(torch, torch_dtype)
        model_init_kwargs["torch_dtype"] = torch_dtype

        model_cls = XiaomiLensQwen3_5ForConditionalGeneration
        model_config, _ = model_cls.config_class.from_pretrained(
            model_id,
            return_unused_kwargs=True,
            **model_init_kwargs,
        )
        if training_args.gradient_checkpointing:
            model_config.use_cache = False
        model_config.update({
            "mask_decoder": getattr(model_config, "mask_decoder", None)
            or build_mask_decoder_config(model_config),
            "lm_loss_weight": training_args.lm_loss_weight,
            "mask_loss_weight": training_args.mask_loss_weight,
            "mask_dice_weight": training_args.mask_dice_weight,
            "mask_bce_weight": training_args.mask_bce_weight,
            "mask_loss_chunk_size": training_args.mask_loss_chunk_size,
            "lm_loss_impl": training_args.lm_loss_impl,
            "average_tokens_across_devices": getattr(
                training_args, "average_tokens_across_devices", True
            ),
        })
        # HF Trainer enables gradient checkpointing when train() starts, after
        # `_disable_unneeded_gradient_checkpointing` has had a chance to run.
        return model_cls.from_pretrained(model_id, config=model_config, **model_init_kwargs)

    @staticmethod
    def _configure_image_processor(image_processor, script_args: XiaomiLensScriptArguments) -> None:
        if script_args.max_pixels is not None:
            image_processor.max_pixels = script_args.max_pixels
        if script_args.min_pixels is not None:
            image_processor.min_pixels = script_args.min_pixels

        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            updated_size = dict(size)
            if script_args.min_pixels is not None:
                updated_size["shortest_edge"] = script_args.min_pixels
            if script_args.max_pixels is not None:
                updated_size["longest_edge"] = script_args.max_pixels
            image_processor.size = updated_size
            return

        if size is not None:
            if script_args.min_pixels is not None and hasattr(size, "shortest_edge"):
                size.shortest_edge = script_args.min_pixels
            if script_args.max_pixels is not None and hasattr(size, "longest_edge"):
                size.longest_edge = script_args.max_pixels

    def _prepare_inputs(self, inputs):
        # The prepared dataloader already places tensors on device. The ragged
        # `_per_sample_*` metadata is consumed by `compute_loss` as host lists.
        return inputs

    def get_train_dataloader(self):
        from accelerate.data_loader import prepare_data_loader
        from torch.utils.data import DataLoader

        from .sampler import BatchSamplerShard

        group = self.args.group_by_modality_length
        sampler = BatchSamplerShard(
            total_samples=len(self.train_dataset),
            batch_size=self.args.train_batch_size,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
            data_seed=self.args.seed,
            group_by_length=group,
            lengths=self.train_dataset.lengths if group else None,
            modalities=self.train_dataset.modalities if group else None,
        )
        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            pin_memory=True,
        )
        # `BatchSamplerShard` already shards by rank, so Accelerate must not
        # shard again (num_processes=1).
        acc = self.accelerator
        return prepare_data_loader(
            dataloader, device=acc.device,
            num_processes=1, process_index=0,
            split_batches=getattr(acc, "split_batches", False),
            put_on_device=getattr(acc, "device_placement", False),
            rng_types=list(getattr(acc, "rng_types", []) or []),
            dispatch_batches=False,
            even_batches=getattr(acc, "even_batches", True),
            use_seedable_sampler=getattr(acc, "use_seedable_sampler", False),
            data_seed=getattr(acc, "data_seed", None),
            non_blocking=getattr(acc, "non_blocking", False),
            use_stateful_dataloader=getattr(acc, "use_stateful_dataloader", False),
        )

    def _build_mask_inputs(self, inputs) -> dict:
        """Turn the collator's ragged per-sample mask metadata into dense device tensors."""
        K = max((len(p) for p in inputs["_per_sample_mask_positions"]), default=0)
        if K == 0:
            return {}
        B = inputs["input_ids"].shape[0]
        device = self.accelerator.device
        per_grid = inputs["_per_sample_mask_grid_hw"]
        per_image_index = inputs["_per_sample_mask_image_index"]

        # The canvas covers the grids of the images that VALID masks bind to.
        # Passing it to the model keeps prediction and target H/W identical.
        grid_h_max = grid_w_max = 1
        for b in range(B):
            valid = inputs["_per_sample_mask_valid_mask"][b]
            for k, (gh, gw) in enumerate(per_grid[b][:K]):
                if k < len(valid) and valid[k] > 0:
                    grid_h_max = max(grid_h_max, gh)
                    grid_w_max = max(grid_w_max, gw)
        max_h = grid_h_max * MASK_CANVAS_SCALE
        max_w = grid_w_max * MASK_CANVAS_SCALE

        pos_rows: list[list[int]] = []
        valid_rows: list[list[float]] = []
        img_rows: list[list[int]] = []
        slots: list[tuple[int, int, int, int]] = []
        blob_parts: list[bytes] = []
        blob_slots: list[tuple[int, int, int, int, int]] = []  # (b, k, ch, cw, nbytes)
        for b in range(B):
            positions = inputs["_per_sample_mask_positions"][b]
            valid = inputs["_per_sample_mask_valid_mask"][b]
            packed = inputs["_per_sample_mask_pixel_masks"][b]
            n = min(len(positions), K)
            pos_rows.append([int(p) for p in positions[:n]] + [0] * (K - n))
            valid_rows.append([float(v) for v in valid[:n]] + [0.0] * (K - n))
            img_rows.append([int(i) for i in per_image_index[b][:n]] + [0] * (K - n))
            for k in range(min(n, len(packed))):
                # A `<|mask|>` dropped by truncation has no query; scoring its
                # pixels would produce a spurious gradient.
                if k >= len(valid) or valid[k] <= 0:
                    continue
                gh, gw = per_grid[b][k]
                ch, cw = gh * MASK_CANVAS_SCALE, gw * MASK_CANVAS_SCALE
                if ch * cw == 0:
                    continue
                slots.append((b, k, ch, cw))
                if packed[k]:
                    blob_parts.append(packed[k])
                    blob_slots.append((b, k, ch, cw, len(packed[k])))

        # uint8 targets and bool validity are converted to FP32 per chunk
        # inside MaskLoss, keeping the full-canvas tensors compact.
        mask_targets = torch.zeros((B, K, max_h, max_w), dtype=torch.uint8, device=device)
        mask_pixel_valid = torch.zeros((B, K, max_h, max_w), dtype=torch.bool, device=device)
        for b, k, ch, cw in slots:
            mask_pixel_valid[b, k, :ch, :cw] = True

        if blob_parts:
            # One H2D copy for every mask's packed bits, then unpack on device.
            # MSB-first matches np.packbits' default bitorder='big'.
            packed_t = torch.frombuffer(
                bytearray().join(blob_parts), dtype=torch.uint8
            ).to(device)
            shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
            bits = ((packed_t.unsqueeze(1) >> shifts) & 1).flatten()
            bit_off = 0
            for b, k, ch, cw, nbytes in blob_slots:
                mask_targets[b, k, :ch, :cw] = bits[bit_off : bit_off + ch * cw].view(ch, cw)
                bit_off += nbytes * 8

        return {
            "mask_positions": torch.tensor(pos_rows, dtype=torch.long, device=device),
            "mask_valid_mask": torch.tensor(valid_rows, dtype=torch.float32, device=device),
            "mask_image_index": torch.tensor(img_rows, dtype=torch.long, device=device),
            "mask_canvas_grid": (grid_h_max, grid_w_max),
            "mask_targets": mask_targets,
            "mask_pixel_valid": mask_pixel_valid,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = self.accelerator.device
        mask_inputs = self._build_mask_inputs(inputs)

        def _to_device(value):
            if value is None or (hasattr(value, "numel") and value.numel() == 0):
                return None
            return value.to(device, non_blocking=True)

        current_accumulation_steps = max(
            int(
                getattr(self, "current_gradient_accumulation_steps", None)
                or self.args.gradient_accumulation_steps
            ),
            1,
        )
        model_output = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            labels=inputs["labels"].to(device),
            pixel_values=_to_device(inputs.get("pixel_values")),
            image_grid_thw=_to_device(inputs.get("image_grid_thw")),
            mm_token_type_ids=_to_device(inputs.get("mm_token_type_ids")),
            num_items_in_batch=num_items_in_batch,
            # When num_items_in_batch is set, Trainer skips its accumulation
            # division; the model needs the window size to scale the
            # per-microbatch mask term.
            gradient_accumulation_steps=(
                current_accumulation_steps if num_items_in_batch is not None else None
            ),
            **mask_inputs,
        )
        loss = model_output["loss"]
        # Record the weighted components of `loss` for logging. Without
        # num_items_in_batch, Trainer divides the total by the accumulation
        # steps afterwards, so mirror that division here. Each component's
        # log denominator is the sum of 1/G, i.e. the optimizer-step count.
        if getattr(model, "training", False):
            step_fraction = 1.0 / current_accumulation_steps
            outer_loss_scale = 1.0 if num_items_in_batch is not None else step_fraction
            for key in ("lm_loss", "mask_loss"):
                value = model_output.get(key)
                if value is not None:
                    self._metrics[key].append(value.detach() * outer_loss_scale)
                    self._metrics[f"_{key}_step_fraction"].append(step_fraction)
        return (loss, model_output) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        # LazyCollator batches carry ragged `_per_sample_*` keys that the model
        # does not accept; route them through compute_loss, which builds the
        # same dense mask tensors as training.
        if not (isinstance(inputs, dict) and "_per_sample_mask_positions" in inputs):
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        with torch.no_grad():
            with self.compute_loss_context_manager():
                if prediction_loss_only:
                    loss = self.compute_loss(model, inputs)
                    return loss.detach().mean(), None, None
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs
        if isinstance(logits, torch.Tensor):
            logits = logits.detach()
        return loss.detach().mean(), logits, None

    def _distributed_metric_means(self) -> dict[str, float]:
        """Aggregate `lm_loss` / `mask_loss` with the same semantics as HF's logged loss.

        Each key all-reduces one `(sum, denominator)` pair in a fixed order on
        every rank, even when a component is disabled or a rank holds no
        masks, so collective sequences never diverge. The logged invariant is
        `loss == lm_loss + mask_loss`.
        """
        metric_keys = ("lm_loss", "mask_loss")
        flat_stats: list[float] = []
        for key in metric_keys:
            values = self._metrics.get(key, [])
            if values:
                host_vals = torch.stack(values).to(torch.float64).cpu().tolist()
                denominator = float(sum(self._metrics[f"_{key}_step_fraction"]))
                flat_stats.extend((float(sum(host_vals)), denominator))
            else:
                flat_stats.extend((0.0, 0.0))

        dist_ready = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        if dist_ready:
            # NCCL accepts only CUDA tensors; Gloo uses CPU tensors.
            backend = str(torch.distributed.get_backend()).lower()
            stats_device = torch.device("cpu") if "gloo" in backend else self.accelerator.device
        else:
            stats_device = torch.device("cpu")

        stats = torch.tensor(flat_stats, dtype=torch.float64, device=stats_device)
        if dist_ready:
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        stats_l = stats.cpu().tolist()

        metrics: dict[str, float] = {}
        for idx, key in enumerate(metric_keys):
            total, count = stats_l[2 * idx : 2 * idx + 2]
            if count > 0:
                metrics[key] = total / count
        return metrics

    def log(self, logs, start_time=None):
        metrics = self._distributed_metric_means()

        step = self.state.global_step
        max_steps = self.state.max_steps
        # Anchor timing at this process's first log so that speed and ETA
        # reflect post-resume throughput.
        if not hasattr(self, "_training_start_time"):
            self._training_start_time = time.monotonic()
            self._training_start_step = step
        elapsed = time.monotonic() - self._training_start_time
        steps_done = step - self._training_start_step
        if steps_done > 0:
            speed = elapsed / steps_done
            remaining_str = self._format_duration(speed * (max_steps - step))
        else:
            speed = 0.0
            remaining_str = "N/A"

        memory_gib = "N/A"
        if torch.cuda.is_available():
            memory_gib = f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}"

        extra = {
            "global_step/max_steps": f"{step}/{max_steps}",
            "elapsed_time": self._format_duration(elapsed),
            "remaining_time": remaining_str,
            "memory(GiB)": memory_gib,
            "train_speed(s/it)": f"{speed:.3f}",
        }

        # Derive the epoch from global_step: with `ignore_data_skip=True` after
        # auto-resume, HF's dataloader-position epoch restarts near zero.
        num_epochs = float(getattr(self.args, "num_train_epochs", 1) or 1)
        if max_steps > 0:
            logs = {**logs, "epoch": round(step / max_steps * num_epochs, 4)}

        super().log({**logs, **metrics, **extra}, start_time)
        self._metrics.clear()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        return f"{m}m {s:02d}s"

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[dict] = None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        unwrapped_model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        unwrapped_model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
            max_shard_size=self.args.save_max_shard_size,
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
