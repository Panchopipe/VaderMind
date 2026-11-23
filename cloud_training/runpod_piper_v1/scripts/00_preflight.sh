#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "${SCRIPT_DIR}/common.sh"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

gpu_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if (( gpu_mib < 20000 )); then
  echo "ERROR: se requieren al menos 20 GB de VRAM; detectados ${gpu_mib} MiB." >&2
  exit 1
fi

echo "== Disco persistente =="
df -h "$(dirname "${RUN_ROOT}")"
free_gib="$(df -Pk "$(dirname "${RUN_ROOT}")" | awk 'NR==2 {print int($4/1024/1024)}')"
if (( free_gib < 20 )); then
  echo "ERROR: quedan menos de 20 GiB libres en el volumen persistente." >&2
  exit 1
fi

echo "== RAM =="
free -h

echo "== Dataset =="
python3 "${SCRIPT_DIR}/validate_dataset.py" --dataset-dir "${DATASET_DIR}"

echo "PREFLIGHT_OK"
