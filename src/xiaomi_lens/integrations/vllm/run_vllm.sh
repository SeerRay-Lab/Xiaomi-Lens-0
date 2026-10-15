#!/bin/bash
# Launch vLLM to serve Xiaomi-Lens-0's TEXT phase (bboxes + <|mask|> tokens).
#
# vLLM's --runner is exclusive (one process is generative OR pooling, never both),
# so this generative serve handles text only; the MASK phase runs in a SECOND,
# same-venv (transformers 4.57) vLLM instance via run_vllm_mask.sh (--runner
# pooling). Together they replace the old vLLM-text + HF-mask_service(5.4) hybrid
# with a single-ENV, all-vLLM deployment (two processes).
#
# Serves the checkpoint as a vanilla Qwen3.5-VL, WITHOUT mutating it. A prep step
# (prepare_served.py) materializes a symlink "served view" that fixes two
# checkpoint quirks so stock vLLM loads it:
#   1) Architecture — `--hf-overrides` swaps the custom
#      XiaomiLensQwen3_5ForConditionalGeneration arch for the base
#      `Qwen3_5ForConditionalGeneration` that vLLM registers.
#   2) Tokenizer + weights — prepare_served.py corrects the transformers-5.4
#      tokenizer_config (AutoTokenizer can't load `TokenizersBackend` / a list
#      `extra_special_tokens`) and strips the `mask_decoder.*` tensors that vLLM's
#      strict loader would reject (the mask head runs in run_vllm_mask.sh, not here).
# `<|mask|>` and its delimiters survive — they live in tokenizer.json / the
# resized embedding, not the mask head.
#
# Usage:
#   bash run_vllm.sh SeerRay-Lab/Xiaomi-Lens-0 [extra vllm args...]   # or a local checkpoint dir
set -euo pipefail

CKPT="${1:?usage: run_vllm.sh <checkpoint_dir|hf_repo_id> [extra vllm args...]}"
shift || true

# Accept a Hugging Face repo ID (e.g. SeerRay-Lab/Xiaomi-Lens-0): download the
# snapshot, or reuse the cached one, and serve from the cache directory.
if [[ ! -d "${CKPT}" ]]; then
    CKPT="$(python3 -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))' "${CKPT}")"
fi

SERVED_NAME="${SERVED_NAME:-xiaomi-lens}"
PORT="${PORT:-8000}"
# By default vLLM reads max_pixels from the checkpoint's processor_config.json.
# Set MAX_PIXELS only for an intentional override; both serves must use the same
# value.
MAX_PIXELS="${MAX_PIXELS:-}"
MM_PROCESSOR_ARGS=()
if [[ -n "${MAX_PIXELS}" ]]; then
    MM_PROCESSOR_ARGS=(--mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS}}")
    echo "[run_vllm] max_pixels override: ${MAX_PIXELS}"
else
    echo "[run_vllm] max_pixels: checkpoint processor config"
fi

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_ROOT="$( cd "${HERE}/../../.." && pwd )"   # .../src, so `xiaomi_lens` is importable

# Build the served view once (idempotent; reuse across launches). Delete it to
# force a rebuild after a checkpoint change.
#
# The cache key includes a hash of the RESOLVED checkpoint path, not just its
# basename — different checkpoints often share a basename, so a basename-only key
# would make two different checkpoints collide on one served dir and silently
# serve the first one's weights.
CKPT_ABS="$( cd "${CKPT}" && pwd -P )"
CKPT_HASH="$(printf '%s' "${CKPT_ABS}" | cksum | cut -d' ' -f1)"
SERVED_DIR="${SERVED_DIR:-${TMPDIR:-/tmp}/xiaomi_lens_served_$(basename "${CKPT_ABS}")_${CKPT_HASH}}"
if [ ! -f "${SERVED_DIR}/model.safetensors.index.json" ]; then
    PYTHONPATH="${SRC_ROOT}" python3 -m xiaomi_lens.integrations.vllm.prepare_served \
        --checkpoint "${CKPT}" --out "${SERVED_DIR}"
fi
echo "[run_vllm] served view: ${SERVED_DIR}"

exec vllm serve "${SERVED_DIR}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --hf-overrides '{"architectures":["Qwen3_5ForConditionalGeneration"]}' \
    "${MM_PROCESSOR_ARGS[@]}" \
    --trust-remote-code \
    "$@"
