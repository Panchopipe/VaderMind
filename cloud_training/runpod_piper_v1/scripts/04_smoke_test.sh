#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "${SCRIPT_DIR}/common.sh"
activate_venv

[ -f "${CHECKPOINT_PATH}" ] || { echo "Ejecuta primero 02_download_checkpoint.sh" >&2; exit 1; }

rm -rf "${SMOKE_DATASET_DIR}" "${SMOKE_TRAIN_DIR}"
mkdir -p "${SMOKE_DATASET_DIR}"
head -n 50 "${DATASET_DIR}/metadata.csv" > "${SMOKE_DATASET_DIR}/metadata.csv"
ln -s "$(realpath "${DATASET_DIR}/wav")" "${SMOKE_DATASET_DIR}/wav"

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m piper_train.preprocess \
  --language "${ESPEAK_VOICE}" \
  --input-dir "${SMOKE_DATASET_DIR}" \
  --output-dir "${SMOKE_TRAIN_DIR}" \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate "${SAMPLE_RATE}" \
  --max-workers "${PREPROCESS_WORKERS}"

python -m piper_train \
  --dataset-dir "${SMOKE_TRAIN_DIR}" \
  --accelerator gpu \
  --devices 1 \
  --batch-size 16 \
  --validation-split 0.0 \
  --num-test-examples 0 \
  --max_epochs "${SMOKE_MAX_EPOCHS}" \
  --resume_from_checkpoint "${CHECKPOINT_PATH}" \
  --checkpoint-epochs 5 \
  --max-phoneme-ids "${MAX_PHONEME_IDS}" \
  --precision "${PRECISION}" \
  --log_every_n_steps 1 \
  2>&1 | tee "${SMOKE_TRAIN_DIR}/smoke_training.log"

smoke_ckpt="$(latest_checkpoint "${SMOKE_TRAIN_DIR}")"
if [ -z "${smoke_ckpt}" ] || [ ! -f "${smoke_ckpt}" ]; then
  echo "ERROR: el smoke test no produjo checkpoint." >&2
  exit 1
fi

mkdir -p "${SMOKE_TRAIN_DIR}/export" "${SMOKE_TRAIN_DIR}/samples"
python -m piper_train.export_onnx \
  "${smoke_ckpt}" \
  "${SMOKE_TRAIN_DIR}/export/${MODEL_BASENAME}-smoke.onnx"
cp "${SMOKE_TRAIN_DIR}/config.json" \
  "${SMOKE_TRAIN_DIR}/export/${MODEL_BASENAME}-smoke.onnx.json"
head -n 3 "${SMOKE_TRAIN_DIR}/dataset.jsonl" \
  | python -m piper_train.infer \
      --sample-rate "${SAMPLE_RATE}" \
      --checkpoint "${smoke_ckpt}" \
      --output-dir "${SMOKE_TRAIN_DIR}/samples"

touch "${RUN_ROOT}/.smoke_ok"
echo "SMOKE_OK"
echo "Escucha los WAV de ${SMOKE_TRAIN_DIR}/samples antes de iniciar 05_train_full.sh"
