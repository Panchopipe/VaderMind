#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "${SCRIPT_DIR}/common.sh"
activate_venv

[ -f "${RUN_ROOT}/.smoke_ok" ] || {
  echo "ERROR: falta un smoke test exitoso. Ejecuta y revisa 04_smoke_test.sh." >&2
  exit 1
}
[ -f "${RUN_ROOT}/.preprocess_ok" ] || {
  echo "ERROR: ejecuta primero 03_preprocess.sh." >&2
  exit 1
}

echo "Iniciando ajuste fino completo. El límite de seguridad es ${FULL_MAX_TIME}."
python -m piper_train \
  --dataset-dir "${TRAIN_DIR}" \
  --accelerator gpu \
  --devices 1 \
  --batch-size "${BATCH_SIZE}" \
  --validation-split "${VALIDATION_SPLIT}" \
  --num-test-examples 0 \
  --max_epochs "${FULL_MAX_EPOCHS}" \
  --max_time "${FULL_MAX_TIME}" \
  --resume_from_checkpoint "${CHECKPOINT_PATH}" \
  --checkpoint-epochs "${CHECKPOINT_EVERY_EPOCHS}" \
  --max-phoneme-ids "${MAX_PHONEME_IDS}" \
  --precision "${PRECISION}" \
  --log_every_n_steps 10 \
  2>&1 | tee "${TRAIN_DIR}/full_training.log"

full_ckpt="$(latest_checkpoint "${TRAIN_DIR}")"
if [ -z "${full_ckpt}" ] || [ ! -f "${full_ckpt}" ]; then
  echo "ERROR: no se encontró un checkpoint final." >&2
  exit 1
fi

touch "${RUN_ROOT}/.training_ok"
echo "TRAINING_OK ${full_ckpt}"
