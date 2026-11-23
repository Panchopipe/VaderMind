#!/usr/bin/env python3
"""Verify the pinned Piper 1.0 training runtime and CUDA device."""

import piper_train  # noqa: F401
import pytorch_lightning
import torch


print("torch:", torch.__version__)
print("pytorch_lightning:", pytorch_lightning.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch no detecta CUDA")
if int(torch.__version__.split(".", 1)[0]) >= 2:
    raise SystemExit(
        "ERROR: este paquete requiere PyTorch 1.x para el checkpoint Piper 1.0"
    )
print("gpu:", torch.cuda.get_device_name(0))
