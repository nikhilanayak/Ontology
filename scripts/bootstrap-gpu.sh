#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
. scripts/production-env.sh

if [[ ! -x .venv/bin/python ]]; then
  echo "Run ./scripts/bootstrap-production.sh first." >&2
  exit 1
fi

# CUDA 12.1 wheels run on the production host's CUDA 12.2-capable 535 driver.
.venv/bin/python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install 'transformers>=4.45,<5'

.venv/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch installed, but CUDA is unavailable")

left = torch.randn((1024, 1024), device="cuda")
right = torch.randn((1024, 1024), device="cuda")
result = left @ right
torch.cuda.synchronize()

properties = torch.cuda.get_device_properties(0)
print({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "vram_gib": round(properties.total_memory / 2**30, 1),
    "smoke_result": float(result[0, 0]),
})
PY
