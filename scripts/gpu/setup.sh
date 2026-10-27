#!/usr/bin/env bash
# One-time setup of a rented GPU instance (Deep Learning Base AMI, Ubuntu 24.04, NVIDIA driver)
# for the offline jobs: query generation and judging (Ollama), embedding fine-tuning and
# comparison (PyTorch), ONNX export. The search service never runs here.
#   bash scripts/gpu/setup.sh            (from the repo copy on the instance)
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== Python 3.12 environment (same versions as the search server)"
sudo apt-get update -qq && sudo apt-get install -y -qq python3.12-venv python3-pip >/dev/null
[ -d ~/venv ] || python3.12 -m venv ~/venv
~/venv/bin/pip install -q --upgrade pip
~/venv/bin/pip install -q -r requirements.txt
~/venv/bin/pip install -q datasets==3.1.0 accelerate==1.1.1
~/venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "== Ollama, 4 requests in parallel"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_NUM_PARALLEL=4"\n' | sudo tee /etc/systemd/system/ollama.service.d/parallel.conf >/dev/null
sudo systemctl daemon-reload && sudo systemctl restart ollama
sleep 3
for m in gemma3:12b gemma3:27b qwen3:14b; do ollama pull "$m" >/dev/null && echo "pulled $m"; done
echo "== Setup done"
