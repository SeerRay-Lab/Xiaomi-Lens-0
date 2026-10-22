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

"""Preflight summary for Xiaomi-Lens-0 training runs.

Builds a human-readable JSON report covering data splits, model parameters, and
training configuration. Printed once on rank 0 before `Trainer.train()` so any
config issue surfaces before the first step is executed.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Iterable

from ..constants import MASK_TOKEN
from .data.segmentation import split_segmentation_masks


def _format_count(value: int) -> str:
    return f"{value:,}"


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00%"
    return f"{(100.0 * numerator / denominator):.2f}%"


def _summarize_numeric(values: Iterable[int]) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {"avg": 0.0, "min": 0, "max": 0}
    return {
        "avg": round(sum(values) / len(values), 2),
        "min": min(values),
        "max": max(values),
    }


def _normalize_enum(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def is_eval_enabled(training_args: Any) -> bool:
    return _normalize_enum(training_args.eval_strategy) != "no"


def _count_parameters(module: Any) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in module.parameters():
        numel = parameter.numel()
        total += numel
        if parameter.requires_grad:
            trainable += numel
    return total, trainable


def _build_dataset_summary(dataset: Any, split_name: str) -> dict[str, Any]:
    image_counts: list[int] = []
    message_counts: list[int] = []
    mask_annotation_counts: list[int] = []
    assistant_mask_counter: Counter[int] = Counter()
    assistant_missing = 0
    user_missing = 0
    segmentation_present = 0
    unique_images: set[str] = set()
    per_source_counter: Counter[str] = Counter()
    per_tag_counter: Counter[str] = Counter()
    mismatched_mask_counts = 0

    max_samples = int(os.environ.get("XIAOMI_LENS_PREFLIGHT_MAX_SAMPLES", "20000"))
    observed = 0
    total = len(dataset)
    for sample in dataset:
        if max_samples > 0 and observed >= max_samples:
            break
        observed += 1
        source = sample.get("_dataset_name") or ""
        tag = sample.get("_dataset_tag") or ""
        if source:
            per_source_counter[source] += 1
        if tag:
            per_tag_counter[tag] += 1
        images = sample.get("images") or []
        image_counts.append(len(images))
        unique_images.update(images)

        messages = sample.get("messages") or []
        message_counts.append(len(messages))

        user_count = sum(1 for message in messages if message.get("role") == "user")
        assistant_messages = [
            message.get("content", "") for message in messages if message.get("role") == "assistant"
        ]
        if user_count == 0:
            user_missing += 1
        if not assistant_messages:
            assistant_missing += 1

        mask_count = sum(content.count(MASK_TOKEN) for content in assistant_messages)
        assistant_mask_counter[mask_count] += 1

        annotations = split_segmentation_masks(sample.get("segmentation"))
        mask_annotation_counts.append(len(annotations))
        if annotations:
            segmentation_present += 1
        if len(annotations) != mask_count:
            mismatched_mask_counts += 1

    return {
        "split": split_name,
        "num_samples": total,
        "summary_samples": observed,
        "summary_truncated": observed < total,
        "columns": list(dataset.column_names),
        "num_unique_images": len(unique_images),
        "images_per_sample": _summarize_numeric(image_counts),
        "messages_per_sample": _summarize_numeric(message_counts),
        "samples_with_segmentation": segmentation_present,
        "segmentation_coverage": _format_ratio(segmentation_present, observed),
        "mask_annotations_per_sample": _summarize_numeric(mask_annotation_counts),
        "samples_with_mask_count_mismatch": mismatched_mask_counts,
        "samples_missing_user_message": user_missing,
        "samples_missing_assistant_message": assistant_missing,
        "assistant_mask_token_distribution": dict(sorted(assistant_mask_counter.items())),
        "samples_with_exactly_one_mask_token": assistant_mask_counter.get(1, 0),
        "samples_per_source": dict(per_source_counter),
        "samples_per_tag": dict(per_tag_counter),
    }


def _build_model_summary(model: Any) -> dict[str, Any]:
    config = model.config
    vision_config = getattr(config, "vision_config", getattr(config, "visual_config", None))
    text_config = config.text_config
    decoder_config = getattr(config, "mask_decoder", {})

    total_params, trainable_params = _count_parameters(model)
    frozen_params = total_params - trainable_params

    backbone = getattr(model, "model", None)
    module_specs = [
        ("visual", getattr(backbone, "visual", None)),
        ("language_model", getattr(backbone, "language_model", None)),
        ("lm_head", getattr(model, "lm_head", None)),
        ("mask_decoder", getattr(model, "mask_decoder", None)),
    ]
    module_summaries: list[dict[str, Any]] = []
    for name, module in module_specs:
        if module is None:
            continue
        module_total, module_trainable = _count_parameters(module)
        module_summaries.append({
            "name": name,
            "class_name": module.__class__.__name__,
            "total_params": _format_count(module_total),
            "trainable_params": _format_count(module_trainable),
            "trainable_ratio": _format_ratio(module_trainable, module_total),
        })

    def _class_name(obj: Any) -> str:
        return obj.__class__.__name__ if obj is not None else type(None).__name__

    return {
        "model_class": model.__class__.__name__,
        "backbone_class": _class_name(backbone),
        "visual_class": _class_name(getattr(backbone, "visual", None)),
        "language_model_class": _class_name(getattr(backbone, "language_model", None)),
        "mask_decoder_class": _class_name(getattr(model, "mask_decoder", None)),
        "vision_config": {
            "hidden_size": getattr(vision_config, "hidden_size", None),
            "intermediate_size": getattr(vision_config, "intermediate_size", None),
            "num_heads": getattr(vision_config, "num_heads", None),
            "patch_size": getattr(vision_config, "patch_size", None),
            "spatial_merge_size": getattr(vision_config, "spatial_merge_size", None),
        },
        "text_config": {
            "hidden_size": getattr(text_config, "hidden_size", None),
            "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
            "num_attention_heads": getattr(text_config, "num_attention_heads", None),
            "num_key_value_heads": getattr(text_config, "num_key_value_heads", None),
            "head_dim": getattr(text_config, "head_dim", None),
        },
        "mask_decoder_config": decoder_config,
        "parameter_summary": {
            "total_params": _format_count(total_params),
            "trainable_params": _format_count(trainable_params),
            "frozen_params": _format_count(frozen_params),
            "trainable_ratio": _format_ratio(trainable_params, total_params),
        },
        "module_parameter_summary": module_summaries,
    }


def _build_training_summary(
    script_args: Any,
    training_args: Any,
    model_args: Any,
    trainer: Any,
    splits: dict[str, Any],
) -> dict[str, Any]:
    num_processes = trainer.accelerator.num_processes
    effective_train_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * num_processes
    )
    # transformers >= 5.15 removed TrainingArguments.warmup_ratio; read it
    # defensively so the preflight report works on both sides of that removal.
    # In 5.15 a fractional `warmup_steps` below 1 IS the ratio form (see
    # TrainingArguments.get_warmup_steps), so report it as a ratio instead of
    # truncating to zero steps.
    warmup_ratio = getattr(training_args, "warmup_ratio", 0.0)
    warmup_steps = getattr(training_args, "warmup_steps", 0) or 0
    if warmup_ratio:
        warmup = {"type": "ratio", "value": warmup_ratio}
    elif 0 < warmup_steps < 1:
        warmup = {"type": "ratio", "value": warmup_steps}
    elif warmup_steps:
        warmup = {"type": "steps", "value": int(warmup_steps)}
    else:
        warmup = {"type": "none", "value": 0}

    # `training_args.gradient_checkpointing` only reflects HF's Trainer-level
    # switch, which decoder-only training turns off (see
    # XiaomiLensTrainer._disable_unneeded_gradient_checkpointing). Report the
    # two custom modules' own recompute state too.
    unwrapped = getattr(trainer, "model", None)
    recompute = {
        attribute: bool(
            getattr(getattr(unwrapped, attribute, None), "gradient_checkpointing", False)
        )
        for attribute in ("mask_decoder", "mask_loss_fn")
    }

    return {
        "model_name_or_path": model_args.model_name_or_path,
        "output_dir": training_args.output_dir,
        "data_file_paths": script_args.data_file_paths,
        "image_folders": script_args.image_folders,
        "data_config": script_args.data_config,
        "min_pixels": script_args.min_pixels,
        "max_pixels": script_args.max_pixels,
        "model_max_length": script_args.model_max_length,
        "num_processes": num_processes,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": model_args.torch_dtype,
        "unfreeze_all_parameters": model_args.unfreeze_all_parameters,
        "unfreeze_vision_tower": model_args.unfreeze_vision_tower,
        "unfreeze_vision_projection": model_args.unfreeze_vision_projection,
        "unfreeze_language_model_backbone": model_args.unfreeze_language_model_backbone,
        "unfreeze_language_model_head": model_args.unfreeze_language_model_head,
        "unfreeze_embed_tokens": model_args.unfreeze_embed_tokens,
        "unfreeze_new_tokens": model_args.unfreeze_new_tokens,
        "unfreeze_mask_decoder": model_args.unfreeze_mask_decoder,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "effective_train_batch_size": effective_train_batch_size,
        "num_train_epochs": training_args.num_train_epochs,
        "max_steps": training_args.max_steps,
        "learning_rate": training_args.learning_rate,
        "lr_scheduler_type": _normalize_enum(training_args.lr_scheduler_type),
        "warmup": warmup,
        "weight_decay": training_args.weight_decay,
        "max_grad_norm": training_args.max_grad_norm,
        "gradient_checkpointing": training_args.gradient_checkpointing,
        "module_recompute": recompute,
        "bf16": training_args.bf16,
        "fp16": training_args.fp16,
        "mask_loss_chunk_size": training_args.mask_loss_chunk_size,
        "lm_loss_impl": training_args.lm_loss_impl,
        "logging_steps": training_args.logging_steps,
        "save_strategy": _normalize_enum(training_args.save_strategy),
        "save_steps": training_args.save_steps,
        "save_total_limit": training_args.save_total_limit,
        "save_safetensors": training_args.save_safetensors,
        "save_max_shard_size": training_args.save_max_shard_size,
        "group_by_modality_length": training_args.group_by_modality_length,
        "eval_strategy": _normalize_enum(training_args.eval_strategy),
        "eval_steps": training_args.eval_steps,
        "prediction_loss_only": training_args.prediction_loss_only,
        "report_to": training_args.report_to,
        "seed": training_args.seed,
        "train_split_size": len(splits["train"]),
        "validation_split_size": len(splits.get("validation", [])),
    }


def build_preflight_report(
    script_args: Any,
    training_args: Any,
    model_args: Any,
    trainer: Any,
    splits: dict[str, Any],
) -> str:
    sections: list[tuple[str, Any]] = [
        ("training", _build_training_summary(script_args, training_args, model_args, trainer, splits)),
        (
            "datasets",
            {
                split_name: _build_dataset_summary(dataset, split_name)
                for split_name, dataset in splits.items()
            },
        ),
        ("model", _build_model_summary(trainer.model)),
    ]

    lines = ["=" * 88, "Xiaomi-Lens-0 Preflight", "=" * 88]
    for section_name, payload in sections:
        lines.append(f"[{section_name}]")
        lines.extend(json.dumps(payload, indent=2, ensure_ascii=False).splitlines())
        lines.append("")
    return "\n".join(lines).rstrip()
