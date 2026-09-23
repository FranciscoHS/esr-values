#!/bin/bash
#SBATCH --job-name=esr-steer
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm-%j.out
# Usage: sbatch scripts/run_steer.sbatch [LAYER] [LIMIT]   (defaults: 33 50)
# Llama-3.3-70B has 80 layers; 33 is the same relative depth as layer 13 of 32 in Llama-3.1-8B.
set -euo pipefail
cd ~/projects/esr-values
source .venv/bin/activate
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub} HF_HUB_OFFLINE=1 PYTHONPATH=. PYTHONUNBUFFERED=1 ESR_MODEL=meta-llama/Llama-3.3-70B-Instruct
LAYER=${1:-33}; LIMIT=${2:-50}
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo "== extract"; python -m esr.vectors extract --batch_size 8
echo "== sweep";   python -m esr.vectors sweep --vec caa --layer "$LAYER" --mults -2 0 2 4 6 8 12 --batch_size 8
echo "== generate"; python -m esr.generate --vec caa --layer "$LAYER" --mults 0 2 4 6 8 12 --limit "$LIMIT" --batch_size 25
echo "== DONE"
