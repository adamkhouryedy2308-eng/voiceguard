#!/bin/bash
# Download and unpack the extra datasets straight onto the cluster.
# The login node allows file transfers, so run it there, in the background,
# so a dropped SSH connection doesn't stop it:
#
#   nohup bash cluster/download_data.sh > logs/download.log 2>&1 &
#   tail -f logs/download.log        # watch (Ctrl+C stops watching, not the download)
#
# Safe to run again after an interruption: finished files are skipped and
# half-finished ones continue where they stopped.
#
#   ASVspoof 5 train (2024): 5 x 7.5 GB + protocols, ODC-By / CC BY 4.0
#       https://zenodo.org/records/14498691   -> Run 3 training data
#   ASVspoof 2021 DF eval, part 00 of 4: 8.6 GB + keys, ODbL / ODC-By
#       https://zenodo.org/record/4835108     -> the referee for choosing between runs

set -euo pipefail
cd "$(dirname "$0")/.."

get() {  # get URL FILE: download once, resume if interrupted
    if [ -f "$2.done" ]; then echo "already have $2"; return; fi
    echo "$(date +%H:%M) downloading $2"
    curl -L --fail --retry 10 --retry-delay 30 -C - -o "$2" "$1"
    touch "$2.done"
}

unpack() {  # unpack ARCHIVE DIR: once
    if [ -f "$1.unpacked" ]; then echo "already unpacked $1"; return; fi
    echo "$(date +%H:%M) unpacking $1"
    tar -xf "$1" -C "$2"
    touch "$1.unpacked"
}

A5=data/asvspoof5
DF=data/asvspoof2021_df
mkdir -p "$A5" "$DF"

Z5=https://zenodo.org/records/14498691/files
get "$Z5/ASVspoof5_protocols.tar.gz?download=1" "$A5/ASVspoof5_protocols.tar.gz"
for part in aa ab ac ad ae; do
    get "$Z5/flac_T_$part.tar?download=1" "$A5/flac_T_$part.tar"
done
get "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz?download=1" "$DF/ASVspoof2021_DF_eval_part00.tar.gz"
get "https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz" "$DF/DF-keys-full.tar.gz"

unpack "$A5/ASVspoof5_protocols.tar.gz" "$A5"
for part in aa ab ac ad ae; do
    unpack "$A5/flac_T_$part.tar" "$A5"
done
unpack "$DF/ASVspoof2021_DF_eval_part00.tar.gz" "$DF"
unpack "$DF/DF-keys-full.tar.gz" "$DF"

echo "$(date +%H:%M) all done"
du -sh "$A5" "$DF"
