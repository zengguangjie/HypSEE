#!/usr/bin/env bash
# Launch HypSEE W&B sweep in background (nohup).
#
# Usage:
#   ./scripts/run_sweep_nohup.sh
#   DATASET=IMDB-BINARY SWEEP_TAG=base_v0628 ./scripts/run_sweep_nohup.sh
#   DATASET=COLLAB BATCH_SIZE=32 SWEEP_ITERS=500 ./scripts/run_sweep_nohup.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

DATASET="${DATASET:-PROTEINS}"
SWEEP_TAG="${SWEEP_TAG:-base_v0628}"
SWEEP_NAME="${DATASET}_${SWEEP_TAG}"
SWEEP_PROFILE="${SWEEP_PROFILE:-base}"
SWEEP_ITERS="${SWEEP_ITERS:-500}"
RUNS="${RUNS:-5}"
EPOCHS="${EPOCHS:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-HypSEE}"
PYTHON="${PYTHON:-/home/zengguangjie/anaconda3/envs/HEAL/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-}"
EXTRA_ARGS=(${EXTRA_ARGS:-})

LOG_FILE="logs/sweep_${SWEEP_NAME}_$(date +%Y%m%d_%H%M%S).log"

CMD=(
  "${PYTHON}" -u main.py
  --sweep
  --data_name "${DATASET}"
  --sweep_name "${SWEEP_NAME}"
  --sweep_profile "${SWEEP_PROFILE}"
  --sweep_iters "${SWEEP_ITERS}"
  --runs "${RUNS}"
  --epochs "${EPOCHS}"
  --wandb_project "${WANDB_PROJECT}"
)

if [[ -n "${BATCH_SIZE}" ]]; then
  CMD+=(--batch_size "${BATCH_SIZE}")
fi

if ((${#EXTRA_ARGS[@]})); then
  CMD+=("${EXTRA_ARGS[@]}")
fi

nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &

SWEEP_PID=$!
echo "${SWEEP_PID}" > "logs/sweep_${SWEEP_NAME}.pid"
echo "$(date '+%Y-%m-%d %H:%M:%S')  pid=${SWEEP_PID}  sweep_name=${SWEEP_NAME}  log=${LOG_FILE}" >> logs/sweep_registry.log
echo "Sweep PID: ${SWEEP_PID} (sweep_name: ${SWEEP_NAME})"
echo "Log file: ${LOG_FILE}"
