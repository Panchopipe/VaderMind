#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "${SCRIPT_DIR}/common.sh"
activate_venv

python "${SCRIPT_DIR}/validate_dataset.py" --dataset-dir "${DATASET_DIR}"

if [ -f "${TRAIN_DIR}/dataset.jsonl" ] && [ -f "${TRAIN_DIR}/config.json" ]; then
  echo "El dataset ya está preprocesado en ${TRAIN_DIR}"
else
  mkdir -p "${TRAIN_DIR}"
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m piper_train.preprocess \
    --language "${ESPEAK_VOICE}" \
    --input-dir "${DATASET_DIR}" \
    --output-dir "${TRAIN_DIR}" \
    --dataset-format ljspeech \
    --single-speaker \
    --sample-rate "${SAMPLE_RATE}" \
    --max-workers "${PREPROCESS_WORKERS}"
fi

processed="$(wc -l < "${TRAIN_DIR}/dataset.jsonl")"
if [ "${processed}" != "398" ]; then
  echo "ERROR: Piper preprocesó ${processed} filas; se esperaban 398." >&2
  exit 1
fi

touch "${RUN_ROOT}/.preprocess_ok"
echo "PREPROCESS_OK ${processed} clips"
