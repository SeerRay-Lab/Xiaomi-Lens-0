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

"""Command-line argument dataclasses for Xiaomi-Lens-0 training."""

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class XiaomiLensTrainingConfig(TrainingArguments):
    model_init_kwargs: Optional[dict] = field(default=None)
    remove_unused_columns: Optional[bool] = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    learning_rate: float = field(default=1e-5)
    save_safetensors: bool = field(default=True)
    save_max_shard_size: str = field(default="5GB")

    lm_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of the causal LM loss. A negative value disables it."},
    )
    mask_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of the mask loss. A negative value disables it."},
    )
    mask_dice_weight: float = field(default=1.0)
    mask_bce_weight: float = field(default=2.0)
    mask_loss_chunk_size: int = field(
        default=8,
        metadata={
            "help": "Maximum flattened (sample, mask-slot) rows per FP32 dice+BCE "
                    "chunk. With gradient checkpointing, each chunk is "
                    "checkpointed independently, bounding loss temporaries by "
                    "chunk_size*H*W instead of B*K*H*W. Values <= 0 use one "
                    "unchunked reference path."
        },
    )

    lm_loss_impl: str = field(
        default="fused_linear_ce",
        metadata={
            "help": "Causal-LM loss kernel. `fused_linear_ce` uses liger-kernel's "
                    "fused linear+cross-entropy and never materializes the "
                    "(B, S, vocab) logits. `flash_ce` uses flash-attn cross "
                    "entropy over BF16 logits. `hf` forces the FP32 reference "
                    "loss. Unavailable kernels warn once and fall back in that order."
        },
    )

    group_by_modality_length: bool = field(
        default=False,
        metadata={
            "help": "Group samples by (modality, text length) so each global batch contains "
                    "similar-length sequences from the same modality, reducing padding waste. "
                    "Applies only to the train_dataset."
        },
    )

    auto_resume: bool = field(
        default=False,
        metadata={
            "help": "On launch, resume from the highest-step `checkpoint-*` under "
                    "`output_dir`, or start fresh when none exists. When a "
                    "checkpoint is found, `ignore_data_skip` is also set to True "
                    "(unless passed explicitly) so skipped batches are not "
                    "re-encoded. An explicit `--resume_from_checkpoint` takes precedence."
        },
    )


@dataclass
class XiaomiLensScriptArguments:
    data_file_paths: Optional[str] = field(
        default=None,
        metadata={"help": "Colon-separated list of JSONL dataset files. Mutually exclusive with --data_config."},
    )
    image_folders: Optional[str] = field(
        default=None,
        metadata={"help": "Colon-separated image folders aligned with --data_file_paths."},
    )
    data_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a YAML data manifest. Train-only manifests "
                          "use top-level `datasets:`; training-time evaluation "
                          "uses `train:` and `eval:` sections. Mutually exclusive "
                          "with --data_file_paths."},
    )
    max_pixels: Optional[int] = field(default=12845056)
    min_pixels: Optional[int] = field(default=3136)
    model_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "Override tokenizer.model_max_length; None keeps the value loaded from the processor."},
    )


@dataclass
class XiaomiLensModelConfig:
    model_name_or_path: Optional[str] = field(default=None)
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    # `unfreeze_*` flags follow a freeze-everything-then-opt-in convention: the
    # trainer first sets `requires_grad=False` on every parameter, then each
    # True flag re-enables its module. See
    # `XiaomiLensTrainer._apply_param_trainability`.
    unfreeze_all_parameters: bool = field(
        default=False,
        metadata={
            "help": "Train the whole model. When True, all other unfreeze_* flags are ignored."
        },
    )
    unfreeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "Unfreeze the ViT blocks (`model.visual.*` except `merger.*`)."},
    )
    unfreeze_vision_projection: bool = field(
        default=False,
        metadata={"help": "Unfreeze the visual-to-LLM projection (`model.visual.merger.*`)."},
    )
    unfreeze_language_model_backbone: bool = field(
        default=False,
        metadata={
            "help": "Unfreeze `language_model.*` except `embed_tokens.*` "
                    "(decoder layers and final norm)."
        },
    )
    unfreeze_language_model_head: bool = field(
        default=False,
        metadata={
            "help": "Unfreeze `lm_head.*`. With tied embeddings (e.g. Qwen3.5-4B), "
                    "`lm_head.weight` is `embed_tokens.weight`, so this coincides "
                    "with `unfreeze_embed_tokens`."
        },
    )
    unfreeze_embed_tokens: bool = field(
        default=False,
        metadata={
            "help": "Unfreeze the entire `embed_tokens.weight`. Takes precedence "
                    "over `unfreeze_new_tokens`."
        },
    )
    unfreeze_new_tokens: bool = field(
        default=False,
        metadata={
            "help": "Train only the `<|mask|>` row of `embed_tokens.weight`. A "
                    "backward hook zeros every other row's gradient. Supported "
                    "with DDP and DeepSpeed ZeRO-1/2."
        },
    )
    unfreeze_mask_decoder: bool = field(
        default=False,
        metadata={"help": "Unfreeze the PixelMaskDecoder, which is trained from random init."},
    )
    torch_dtype: Optional[str] = field(default="bfloat16")
