#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=../config.env
source "${PACKAGE_DIR}/config.env"

DATASET_DIR="${PACKAGE_DIR}/dataset"
PIPER_DIR="${RUN_ROOT}/piper"
VENV_DIR="${RUN_ROOT}/.venv"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_FILE}"
TRAIN_DIR="${RUN_ROOT}/training_full"
SMOKE_DATASET_DIR="${RUN_ROOT}/dataset_smoke"
SMOKE_TRAIN_DIR="${RUN_ROOT}/training_smoke"
EXPORT_DIR="${RUN_ROOT}/export"

activate_venv() {
  if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "ERROR: falta el entorno ${VENV_DIR}. Ejecuta primero 01_setup.sh." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

latest_checkpoint() {
  local search_dir="$1"
  find "${search_dir}" -type f -name '*.ckpt' -printf '%T@ %p\n' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}
