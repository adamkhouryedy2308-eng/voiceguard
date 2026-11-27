#!/bin/bash
# Download ML-ITW (Xue et al., 2026, arXiv:2603.05852) onto the cluster and turn it
# into normal audio files + a label list. Test only: never trained on, never shared.
#   24,529 clips (18,168 real, 6,361 fake) of 180 public figures, 14 languages,
#   7 platforms (YouTube, TikTok, X, Facebook, Bilibili, Douyin, Rednote). 3.3 GB.
#   The paper's own link (chlo1/ML-ITW) is private; this public copy has the same
#   24,529 clips and labels. No licence is stated: evaluation only, cite the paper.
#
# Run on the login node, in the background:
#   nohup bash cluster/download_ml_itw.sh > logs/download_ml_itw.log 2>&1 &
# Safe to run again: finished files are skipped.

set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/.local/bin/env"   # puts uv on the PATH

ROOT=data/ml_itw
mkdir -p "$ROOT/parquet"
HF=https://huggingface.co/datasets/nccm2p2/ML-ITW/resolve/main/data

for n in 0 1 2 3; do
    file="$ROOT/parquet/eval-0000$n-of-00004.parquet"
    if [ -f "$file.done" ]; then echo "already have $file"; continue; fi
    echo "$(date +%H:%M) downloading $file"
    curl -L --fail --retry 10 --retry-delay 30 -C - -o "$file" "$HF/eval-0000$n-of-00004.parquet"
    touch "$file.done"
done

echo "$(date +%H:%M) unpacking"
# pyarrow reads the parquet files; it's only needed here, so it isn't a project dependency.
uv run --with pyarrow python scripts/unpack_ml_itw.py "$ROOT"
echo "$(date +%H:%M) all done"
du -sh "$ROOT"
