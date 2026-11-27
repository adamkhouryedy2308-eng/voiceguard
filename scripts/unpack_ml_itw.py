"""Turn ML-ITW's parquet files into wav files + meta.csv (run by cluster/download_ml_itw.sh).

Each parquet row is one clip: file (the wav bytes), speaker, language, platform,
label ("bona-fide" or "spoof", like In-the-Wild). Writes
    <root>/audio/00000.wav, 00001.wav, ...   (numbered: speaker names have all sorts of letters)
    <root>/meta.csv                          (file,speaker,language,platform,label)

Usage:
    uv run --with pyarrow python scripts/unpack_ml_itw.py data/ml_itw
"""

import csv
import sys
from pathlib import Path

import pyarrow.parquet as pq

COLUMNS = ["speaker", "language", "platform", "label"]


def unpack(root):
    root = Path(root)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for part in sorted((root / "parquet").glob("*.parquet")):
        # 500 rows at a time, so memory stays small (the login node allows 8 GB)
        for batch in pq.ParquetFile(part).iter_batches(batch_size=500, columns=["file", *COLUMNS]):
            for row in batch.to_pylist():
                name = f"audio/{len(rows):05d}.wav"
                (root / name).write_bytes(row["file"]["bytes"])
                rows.append([name, *(row[c] for c in COLUMNS)])
        print(f"{part.name}: {len(rows)} clips so far", flush=True)
    with open(root / "meta.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", *COLUMNS])
        writer.writerows(rows)
    print(f"wrote {len(rows)} clips and {root / 'meta.csv'}")


if __name__ == "__main__":
    unpack(sys.argv[1])
