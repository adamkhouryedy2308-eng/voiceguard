#!/bin/bash
# SLURM job: train our own SSL-AASIST on one GPU.
#
# One-time setup on the login node (rust), from the voiceguard folder:
#   1. Copy the code and the two dataset zips over, unzip them into data/
#      (same layout as on the laptop: data/asvspoof2019/LA, data/in_the_wild/...).
#   2. uv sync                                  # install Python packages
#   3. uv run python -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-300m')"
#      (downloads Meta's XLS-R now, in case the GPU nodes have no internet)
#   4. mkdir -p logs
#
# Submit:          sbatch cluster/train.sh
# Watch progress:  squeue -u $USER      and      tail -f logs/voiceguard-<job id>.out
# If the 3-day limit stops it, run sbatch again: --resume continues from the last epoch.

#SBATCH --job-name=voiceguard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
nvidia-smi
uv run python scripts/train_ssl_aasist.py --resume --workers 8 "$@"
