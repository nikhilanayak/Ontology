#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
npm install

.venv/bin/python -m pytest -q

echo "Base environment is ready. Install the appropriate CUDA-enabled PyTorch wheel before GPU training."
