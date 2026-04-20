#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

CONFIG_NAME="${CONFIG_NAME:-pi0_fast_rlbench_pickplace_rand1_lora_cam_qbin64_all}"
EXP_NAME="${EXP_NAME:-qbin64_all_baseline_5k}"
STEP="${STEP:-5000}"
EVAL_REPO_ID="${EVAL_REPO_ID:-minyangli/pick_place_all}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/vis_results/${EXP_NAME}_step${STEP}}"
VIS_OUTPUT_TEMPLATE="${VIS_OUTPUT_TEMPLATE:-${OUT_DIR}/sample{i}.png}"

SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
SAMPLE_RANGE="${SAMPLE_RANGE:-0:15}"
INFER_TEMPERATURE="${INFER_TEMPERATURE:-}"
INFER_RNG_SEED="${INFER_RNG_SEED:-0}"
DEBUG_FAST_DECODE="${DEBUG_FAST_DECODE:-0}"
DEBUG_OBS="${DEBUG_OBS:-0}"
PRINT_IMAGE_PATHS="${PRINT_IMAGE_PATHS:-0}"
VIS_COMPARE_GT_PRED="${VIS_COMPARE_GT_PRED:-1}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/checkpoints/${CONFIG_NAME}/${EXP_NAME}/${STEP}}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/mnt/nas/minyangli}"
export HF_HOME="${HF_HOME:-/tmp/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets_cache}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}"

ARGS=(
  "${ROOT_DIR}/examples/inference_vis_rlbench.py"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --config-name "${CONFIG_NAME}"
  --eval-repo-id "${EVAL_REPO_ID}"
  --vis-output "${VIS_OUTPUT_TEMPLATE}"
  --infer-rng-seed "${INFER_RNG_SEED}"
)

if [[ -n "${SAMPLE_RANGE}" ]]; then
  ARGS+=(--sample-range "${SAMPLE_RANGE}")
else
  ARGS+=(--sample-index "${SAMPLE_INDEX}")
fi

if [[ -n "${INFER_TEMPERATURE}" ]]; then
  ARGS+=(--infer-temperature "${INFER_TEMPERATURE}")
fi

if [[ "${VIS_COMPARE_GT_PRED}" != "0" ]]; then
  ARGS+=(--vis-compare-gt-pred)
fi

if [[ "${DEBUG_FAST_DECODE}" != "0" ]]; then
  ARGS+=(--debug-fast-decode)
fi

if [[ "${DEBUG_OBS}" != "0" ]]; then
  ARGS+=(--debug-obs)
fi

if [[ "${PRINT_IMAGE_PATHS}" == "0" ]]; then
  ARGS+=(--no-print-image-paths)
fi

"${PYTHON_BIN}" "${ARGS[@]}"
