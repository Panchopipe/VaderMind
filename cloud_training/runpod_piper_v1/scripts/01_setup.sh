#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "${SCRIPT_DIR}/common.sh"

mkdir -p "${RUN_ROOT}"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  curl \
  espeak-ng \
  ffmpeg \
  git \
  libsndfile1 \
  ninja-build \
  python3-dev \
  python3-venv

if [ ! -d "${PIPER_DIR}/.git" ]; then
  git clone "${PIPER_REPO_URL}" "${PIPER_DIR}"
fi
git -C "${PIPER_DIR}" fetch --all --tags
git -C "${PIPER_DIR}" checkout --detach "${PIPER_REPO_COMMIT}"
actual_commit="$(git -C "${PIPER_DIR}" rev-parse HEAD)"
if [ "${actual_commit}" != "${PIPER_REPO_COMMIT}" ]; then
  echo "ERROR: commit Piper inesperado: ${actual_commit}" >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi
activate_venv

python -m pip install --upgrade "pip==24.0" "setuptools==69.5.1" wheel
python -m pip install \
  "Cython==0.29.36" \
  "librosa==0.9.2" \
  "numpy==1.23.5" \
  "onnx==1.14.1" \
  "onnxruntime==1.16.3" \
  "onnxsim==0.4.36" \
  "protobuf<4" \
  "pytorch-lightning==1.7.7" \
  "torchmetrics==0.11.4"
python -m pip install -e "${PIPER_DIR}/src/python"

pushd "${PIPER_DIR}/src/python" >/dev/null
bash ./build_monotonic_align.sh
popd >/dev/null

python "${SCRIPT_DIR}/verify_environment.py"

touch "${RUN_ROOT}/.setup_ok"
echo "SETUP_OK"
