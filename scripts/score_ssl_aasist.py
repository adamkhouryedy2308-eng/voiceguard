"""Score a whole set with the authors' trained SSL-AASIST model.

Saves data/features/{split}_scores_{name}.npz, which evaluate.py then grades
(threshold frozen on dev, exactly like the baseline).

Usage:
    uv run python scripts/score_ssl_aasist.py dev --checkpoint LA_model
    uv run python scripts/score_ssl_aasist.py in_the_wild --checkpoint Best_LA_model_for_DF
    uv run python scripts/score_ssl_aasist.py dev --augment   # messy dev, for choosing the line
    uv run python scripts/score_ssl_aasist.py dev --checkpoint voiceguard --augment   # our Run 1
    uv run python scripts/score_ssl_aasist.py dev --checkpoint models/run2/best.pt   # any other run
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from voiceguard.audio import fix_length, load_clip
from voiceguard.augment import augment
from voiceguard.dataset import read_df21, read_in_the_wild, read_ml_itw, read_protocol
from voiceguard.features import DEVICE
from voiceguard.ssl_aasist import SSLAASIST, load_original_checkpoint

FEATURES = Path("data/features")
NAMES = {"LA_model": "ssl_aasist_la", "Best_LA_model_for_DF": "ssl_aasist_df", "voiceguard": "ssl_aasist_ours"}


def model_name(checkpoint):
    """Name used in the score file: ssl_aasist_df, ssl_aasist_ours (Run 1), ssl_aasist_run2 ..."""
    if checkpoint in NAMES:
        return NAMES[checkpoint]
    return "ssl_aasist_" + Path(checkpoint).parent.name  # models/run2/best.pt -> ssl_aasist_run2


def load_model(checkpoint):
    if checkpoint in ("LA_model", "Best_LA_model_for_DF"):
        return load_original_checkpoint(f"models/ssl_aasist/{checkpoint}.pth")
    # our own models, trained by scripts/train_ssl_aasist.py
    path = "models/voiceguard/best.pt" if checkpoint == "voiceguard" else checkpoint
    model = SSLAASIST()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


class Clips(Dataset):
    def __init__(self, clips, messy=False, echo=False):
        self.clips = clips
        self.messy = messy
        self.echo = echo

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        path, _, _ = self.clips[i]
        audio = load_clip(path)  # same preprocessing as the authors: 64,600 samples, repeat-padded
        if self.messy:
            # One random generator per clip (seeded by its position), so every
            # run makes exactly the same messy copy, whichever worker loads it.
            rng = np.random.default_rng(i)
            audio = fix_length(augment(audio, rng, echo=self.echo), len(audio)).astype(np.float32)
        return torch.tensor(audio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["dev", "eval", "in_the_wild", "df21", "ml_itw"],
                        help="df21 = the referee (ASVspoof 2021 DF, part 00) for choosing between runs; "
                             "ml_itw = second real-world final exam")
    parser.add_argument("--checkpoint", default="LA_model",
                        help="LA_model, Best_LA_model_for_DF, voiceguard (Run 1), or a path like models/run2/best.pt")
    parser.add_argument("--augment", action="store_true", help="score messy copies (noise/phone/MP3), e.g. of dev for calibration")
    parser.add_argument("--echo", action="store_true", help="with --augment: also add room echo (as in our training)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    name = model_name(args.checkpoint)
    split_name = args.split + ("_aug" if args.augment else "")

    if args.split == "in_the_wild":
        clips = read_in_the_wild()
    elif args.split == "df21":
        clips = read_df21()
    elif args.split == "ml_itw":
        clips = read_ml_itw()
    else:
        clips = read_protocol(args.split)
    clips = clips[: args.limit]
    print(f"{split_name}: {len(clips)} clips, model {name}")

    model = load_model(args.checkpoint).eval().to(DEVICE)
    loader = DataLoader(Clips(clips, messy=args.augment, echo=args.echo), batch_size=16, num_workers=4)

    scores = []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(DEVICE))
            # index 0 = spoof, so softmax[:, 0] is "chance of fake", like our baseline's score
            scores.append(torch.softmax(logits, dim=-1)[:, 0].cpu().numpy())
            done = sum(len(s) for s in scores)
            if done % 1600 == 0 or done == len(clips):
                elapsed = time.time() - start
                print(f"{done}/{len(clips)}  {elapsed / 60:.1f} min, ~{elapsed / done * (len(clips) - done) / 60:.1f} min left", flush=True)

    out = FEATURES / f"{split_name}_scores_{name}.npz"
    if args.limit is not None:
        out = out.with_name(out.stem + f"_first{args.limit}.npz")
    np.savez(
        out,
        scores=np.concatenate(scores),
        labels=np.array([label for _, label, _ in clips]),
        attacks=np.array([attack for _, _, attack in clips]),
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
