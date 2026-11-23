#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "${SCRIPT_DIR}/common.sh"
activate_venv

[ -f "${RUN_ROOT}/.training_ok" ] || {
  echo "ERROR: no hay un entrenamiento completo marcado como exitoso." >&2
  exit 1
}

full_ckpt="$(latest_checkpoint "${TRAIN_DIR}")"
mkdir -p "${EXPORT_DIR}/samples"

python -m piper_train.export_onnx \
  "${full_ckpt}" \
  "${EXPORT_DIR}/${MODEL_BASENAME}.onnx"
cp "${TRAIN_DIR}/config.json" "${EXPORT_DIR}/${MODEL_BASENAME}.onnx.json"

python "${SCRIPT_DIR}/finalize_config.py" \
  --config "${EXPORT_DIR}/${MODEL_BASENAME}.onnx.json"

head -n 10 "${TRAIN_DIR}/dataset.jsonl" \
  | python -m piper_train.infer \
      --sample-rate "${SAMPLE_RATE}" \
      --checkpoint "${full_ckpt}" \
      --output-dir "${EXPORT_DIR}/samples"

sha256sum \
  "${EXPORT_DIR}/${MODEL_BASENAME}.onnx" \
  "${EXPORT_DIR}/${MODEL_BASENAME}.onnx.json" \
  > "${EXPORT_DIR}/SHA256SUMS"

tar -C "${EXPORT_DIR}" -czf "${RUN_ROOT}/${MODEL_BASENAME}.tar.gz" \
  "${MODEL_BASENAME}.onnx" \
  "${MODEL_BASENAME}.onnx.json" \
  "SHA256SUMS" \
  "samples"

echo "EXPORT_OK ${RUN_ROOT}/${MODEL_BASENAME}.tar.gz"
