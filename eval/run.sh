#!/bin/bash
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

# Evaluate Xiaomi-Lens-0 on LensBench with the Hugging Face or vLLM backend.
#
# MODEL and DATASET accept Hugging Face IDs or local paths (a checkpoint
# directory; a test.parquet file or its directory).
#
# Usage:
#   bash eval/run.sh hf   [extra run.py args...]   # one HF shard per GPU in DEVICES
#   bash eval/run.sh vllm [extra run.py args...]   # text and mask engines on two GPUs
#
# Examples:
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" bash eval/run.sh hf
#   TEXT_DEVICE=cuda:0 MASK_DEVICE=cuda:1 bash eval/run.sh vllm --limit 16
#   bash eval/run.sh hf --resume
set -euo pipefail

BACKEND="${1:?usage: run.sh <hf|vllm> [extra run.py args...]}"
shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-SeerRay-Lab/Xiaomi-Lens-0}"
DATASET="${DATASET:-SeerRay-Lab/LensBench}"
# Unset MAX_NEW_TOKENS keeps run.py's default (DEFAULT_MAX_NEW_TOKENS).
COMMON_ARGS=()
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
    COMMON_ARGS=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi

case "${BACKEND}" in
    hf)
        # Space-separated devices; a single device runs without sharding.
        DEVICES="${DEVICES:-cuda:0}"
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/lensbench_hf}"
        read -r -a DEVICE_LIST <<< "${DEVICES}"
        if (( ${#DEVICE_LIST[@]} > 1 )); then
            DEVICE_ARGS=(--devices "${DEVICE_LIST[@]}")
        else
            DEVICE_ARGS=(--device "${DEVICE_LIST[0]}")
        fi
        exec python "${ROOT}/eval/run.py" \
            --backend hf \
            --model-path "${MODEL}" \
            --dataset-path "${DATASET}" \
            --output-dir "${OUTPUT_DIR}" \
            ${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"} \
            "${DEVICE_ARGS[@]}" \
            "$@"
        ;;
    vllm)
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/lensbench_vllm}"
        exec python "${ROOT}/eval/run.py" \
            --backend vllm \
            --model-path "${MODEL}" \
            --dataset-path "${DATASET}" \
            --output-dir "${OUTPUT_DIR}" \
            ${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"} \
            --vllm-text-device "${TEXT_DEVICE:-cuda:0}" \
            --vllm-mask-device "${MASK_DEVICE:-cuda:1}" \
            --vllm-max-model-len "${MAX_MODEL_LEN:-32768}" \
            --vllm-gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
            "$@"
        ;;
    *)
        echo "Unknown backend: ${BACKEND} (expected hf or vllm)" >&2
        exit 2
        ;;
esac
