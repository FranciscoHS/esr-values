#!/bin/bash
#SBATCH --job-name=esr-concept
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm-%j.out
# Positive control, step 1 (pilot): concept vectors + a strength sweep for one concept, 1 sample per prompt.
# Usage: sbatch scripts/run_concept.sbatch [LAYER]   (default 33, as in the ESR paper for Llama-3.3-70B)
set -euo pipefail
cd ~/projects/esr-values
source .venv/bin/activate
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub} HF_HUB_OFFLINE=1 PYTHONPATH=. PYTHONUNBUFFERED=1 ESR_MODEL=meta-llama/Llama-3.3-70B-Instruct
LAYER=${1:-33}
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo "== vectors";  python -m esr.concept vectors --batch_size 8
echo "== generate"; python -m esr.concept generate --layer "$LAYER" --concepts body_positions --alphas 0 0.4 0.6 0.8 1.0 1.2 --samples 1 --batch_size 38
echo "== DONE"
