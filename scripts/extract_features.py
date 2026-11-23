"""Turn every labelled clip of one split into 1,024 XLS-R numbers and save them.

Usage:
    uv run python scripts/extract_features.py train
    uv run python scripts/extract_features.py dev --limit 20   # quick test
"""

import argparse
import time
from pathlib import Path

import numpy as np

from voiceguard.augment import augment
from voiceguard.dataset import read_in_the_wild, read_protocol
from voiceguard.features import describe, load_model

OUT_DIR = Path("data/features")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "dev", "eval", "in_the_wild"])
    parser.add_argument("--augment", action="store_true", help="add noise/phone/MP3 effects")
    parser.add_argument("--limit", type=int, default=None, help="only the first N clips")
    args = parser.parse_args()

    clips = read_in_the_wild() if args.split == "in_the_wild" else read_protocol(args.split)
    clips = clips[: args.limit]
    print(f"{args.split}: {len(clips)} clips")

    model = load_model()
    rng = np.random.default_rng(0)
    augment_fn = (lambda audio: augment(audio, rng)) if args.augment else None
    features = np.zeros((len(clips), 1024), dtype=np.float32)
    start = time.time()
    for i, (path, _, _) in enumerate(clips):
        features[i] = describe(model, path, augment_fn)
        if (i + 1) % 500 == 0 or i + 1 == len(clips):
            elapsed = time.time() - start
            left = elapsed / (i + 1) * (len(clips) - i - 1)
            print(f"{i + 1}/{len(clips)}  {elapsed / 60:.1f} min done, ~{left / 60:.1f} min left", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = args.split + ("_aug" if args.augment else "")
    if args.limit is not None:
        name += f"_first{args.limit}"
    out = OUT_DIR / f"{name}.npz"
    np.savez(
        out,
        features=features,
        labels=np.array([label for _, label, _ in clips]),
        attacks=np.array([attack for _, _, attack in clips]),
        file_ids=np.array([path.stem for path, _, _ in clips]),
    )
    print(f"saved {features.shape} to {out}")


if __name__ == "__main__":
    main()
