#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "${ROOT_DIR}/wandb/.wandb.env" ]]; then
  # Local-only secrets such as WANDB_API_KEY live here.
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/wandb/.wandb.env"
fi

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/mnt/nas/minyangli}"
export HF_HOME="${HF_HOME:-/tmp/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets_cache}"
export OPENPI_ENABLE_PROBE="${OPENPI_ENABLE_PROBE:-0}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"
cd "${ROOT_DIR}"

CONFIG_NAME="${CONFIG_NAME:-pickplace_all_qbin64}"
POST_TRAIN_EVAL="${POST_TRAIN_EVAL:-1}"
EVAL_REPO_ID="${EVAL_REPO_ID:-minyangli/pick_place_all_test}"
EVAL_METRIC_PREFIX="${EVAL_METRIC_PREFIX:-test}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_SEED="${EVAL_SEED:-0}"
export POST_TRAIN_EVAL EVAL_REPO_ID EVAL_METRIC_PREFIX EVAL_BATCH_SIZE EVAL_SEED

"${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/training/train.py" \
  "${CONFIG_NAME}" \
  "$@"
