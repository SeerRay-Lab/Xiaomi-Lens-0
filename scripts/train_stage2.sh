#!/bin/bash
# Stage 2: full-parameter SFT with LM loss + mask loss, starting from the
# Stage 1 output.
#
# Usage:
#   DATA_CONFIG=/path/to/stage2.yaml bash scripts/train_stage2.sh [extra train.py args...]
set -euo pipefail

REPO_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "${REPO_HOME}"

# ── Paths ──
MODEL_PATH="${MODEL_PATH:-${REPO_HOME}/checkpoints/xiaomi-lens-stage1}"
: "${DATA_CONFIG:?Set DATA_CONFIG to a YAML data manifest (see examples/train/README.md)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_HOME}/checkpoints/xiaomi-lens-stage2}"
# ZeRO-2 partitions the bf16 gradients across ranks. ZeRO-1 and ZeRO-2
# optimizer shards are not interchangeable: resume a checkpoint with the stage
# that wrote it.
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_HOME}/src/xiaomi_lens/train/configs/zero2.json}"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "MODEL_PATH=${MODEL_PATH} has no config.json; run scripts/train_stage1.sh first or set MODEL_PATH." >&2
    exit 1
fi

# ── Distributed ──
# Single node by default with all visible GPUs. For multi-node, set NNODES,
# NODE_RANK and MASTER_ADDR on each node (and optionally NPROC_PER_NODE,
# MASTER_PORT).
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l | tr -d ' ')}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-}"

if [[ "${NNODES}" -gt 1 && -z "${MASTER_ADDR}" ]]; then
    echo "MASTER_ADDR must be set when NNODES > 1" >&2
    exit 1
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

LAUNCHER_ARGS=(
    --nproc_per_node "${NPROC_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

# ── Launch ──
# --unfreeze_all_parameters trains everything; the per-module flags are kept
# explicit for readability. Keeping both --unfreeze_language_model_head and
# --unfreeze_embed_tokens true works for tied (4B) and untied (9B) variants.
# Keep gradient checkpointing in TrainingArguments only; enabling DeepSpeed
# `activation_checkpointing` as well causes recomputation shape mismatches.
python -m torch.distributed.run "${LAUNCHER_ARGS[@]}" train.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_PATH}" \
    --attn_implementation flash_attention_2 \
    --bf16 \
    --data_config "${DATA_CONFIG}" \
    --max_pixels 4194304 \
    --min_pixels 65536 \
    --model_max_length 8192 \
    --group_by_modality_length true \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers true \
    --dataloader_prefetch_factor 4 \
    --dataloader_drop_last true \
    --unfreeze_all_parameters true \
    --unfreeze_vision_tower true \
    --unfreeze_vision_projection true \
    --unfreeze_language_model_backbone true \
    --unfreeze_language_model_head true \
    --unfreeze_embed_tokens true \
    --unfreeze_new_tokens false \
    --unfreeze_mask_decoder true \
    --lm_loss_weight 1.0 \
    --mask_loss_weight 1.0 \
    --mask_loss_chunk_size 8 \
    --learning_rate 5e-6 \
    --weight_decay 0.0 \
    --warmup_steps 0.03 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --output_dir "${OUTPUT_DIR}" \
    --auto_resume true \
    --save_steps 2000 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --report_to tensorboard \
    --run_name "xiaomi-lens-stage2" \
    "$@"
