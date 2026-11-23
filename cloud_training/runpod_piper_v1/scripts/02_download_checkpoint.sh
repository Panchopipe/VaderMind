#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "${SCRIPT_DIR}/common.sh"

mkdir -p "${CHECKPOINT_DIR}"
partial_path="${CHECKPOINT_PATH}.partial"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
  rm -f "${partial_path}"
  curl --fail --location --retry 5 --retry-delay 5 \
    --output "${partial_path}" "${CHECKPOINT_URL}"
  mv "${partial_path}" "${CHECKPOINT_PATH}"
fi

actual_size="$(stat -c '%s' "${CHECKPOINT_PATH}")"
if [ "${actual_size}" != "${CHECKPOINT_SIZE_BYTES}" ]; then
  echo "ERROR: tamaño de checkpoint incorrecto: ${actual_size}" >&2
  exit 1
fi

echo "${CHECKPOINT_SHA256}  ${CHECKPOINT_PATH}" | sha256sum --check --status
echo "CHECKPOINT_OK ${CHECKPOINT_PATH}"
