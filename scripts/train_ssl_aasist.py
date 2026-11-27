"""Train our own SSL-AASIST: Meta's XLS-R (unfrozen) + a fresh AASIST back end.

Trains on ASVspoof 2019 LA train. Each time a clip is loaded it has a 50% chance
of being a messy copy (noise, phone line, MP3, room echo), so the model sees
clean and messy versions. After every epoch it measures the EER on clean dev
and keeps the best epoch in models/voiceguard/best.pt. eval and In-the-Wild
are never touched here: they are tested once, at the very end.

Recipe from Tak et al. (2022): Adam, learning rate 0.000001, weight decay 0.0001,
weighted loss (fakes count 0.1, real clips 0.9, because train has ~9 fakes per real).

Usage:
    # tiny test on the laptop: a few clips, 1 epoch, just to see the code runs
    uv run python scripts/train_ssl_aasist.py --limit 16 --dev-limit 32 --epochs 1 --batch-size 2 --workers 2 --out models/voiceguard_test
    # Run 1, on the cluster (cluster/train.sh calls this)
    uv run python scripts/train_ssl_aasist.py --resume
    # Run 2: + RawBoost + random windows, saved in its own folder
    sbatch cluster/train.sh --out models/run2 --rawboost-prob 0.5 --random-windows
    # Run 3: Run 2 + ASVspoof 5 training data
    sbatch cluster/train.sh --out models/run3 --rawboost-prob 0.5 --random-windows --extra-train asvspoof5
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from voiceguard.audio import fix_length, load_audio, load_clip, random_window
from voiceguard.augment import augment
from voiceguard.dataset import read_asvspoof5_train, read_protocol
from voiceguard.features import DEVICE
from voiceguard.metrics import compute_eer
from voiceguard.rawboost import rawboost
from voiceguard.ssl_aasist import SSLAASIST


class TrainClips(Dataset):
    def __init__(self, clips, messy_prob=0.0, rawboost_prob=0.0, random_windows=False, seed=0):
        self.clips = clips
        self.messy_prob = messy_prob
        self.rawboost_prob = rawboost_prob
        self.random_windows = random_windows
        self.seed = seed
        self.epoch = 0  # updated before every epoch, so every epoch gets new messy copies

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        path, label, _ = self.clips[i]
        # One random generator per (seed, epoch, clip): a rerun makes exactly the same mess.
        rng = np.random.default_rng([self.seed, self.epoch, i])
        if self.random_windows:
            audio = random_window(load_audio(path), rng)  # a random ~4 s piece (Run 2)
        else:
            audio = load_clip(path)  # always the first ~4 s, repeat-padded (Run 1, and scoring)
        messy = rng.random() < self.messy_prob
        if self.rawboost_prob > 0 and rng.random() < self.rawboost_prob:
            audio = rawboost(audio, rng)  # recording-chain distortions (Run 2)
        if messy:
            audio = fix_length(augment(audio, rng, echo=True), len(audio)).astype(np.float32)
        # The model's output 0 means fake and 1 means real (the authors' order),
        # while our labels are 1 = fake, so flip it.
        return torch.tensor(audio), 1 - label


def pick(clips, limit, seed):
    """All clips, or a random handful of them (for quick tests)."""
    if limit is None:
        return clips
    order = np.random.default_rng(seed).permutation(len(clips))[:limit]
    return [clips[j] for j in order]


def dev_eer(model, loader):
    """Score every dev clip and return the dev EER (how well real and fake separate)."""
    model.eval()  # switch off dropout: we're measuring, not learning
    scores, labels = [], []
    with torch.no_grad():
        for audio, target in loader:
            logits = model(audio.to(DEVICE))
            scores.append(torch.softmax(logits, dim=-1)[:, 0].cpu().numpy())  # chance of fake
            labels.append(1 - target.numpy())  # back to our labels: 1 = fake
    eer, _ = compute_eer(np.concatenate(scores), np.concatenate(labels))
    return float(eer)  # a plain Python number, so last.pt loads back safely


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7, help="stop after this many epochs without a better dev EER")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--messy-prob", type=float, default=0.5, help="chance that a train clip is a messy copy")
    parser.add_argument("--rawboost-prob", type=float, default=0.0, help="chance of RawBoost distortion (Run 2: 0.5)")
    parser.add_argument("--random-windows", action="store_true", help="train on random ~4 s pieces, not always the first 4 s")
    parser.add_argument("--extra-train", choices=["asvspoof5"], default=None,
                        help="add another training set (Run 3: ASVspoof 5, 182,357 clips)")
    parser.add_argument("--workers", type=int, default=8, help="CPU helpers that load and mess up clips")
    parser.add_argument("--limit", type=int, default=None, help="use only this many train clips (testing)")
    parser.add_argument("--dev-limit", type=int, default=None, help="use only this many dev clips (testing)")
    parser.add_argument("--out", default="models/voiceguard")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt if it exists")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")  # faster maths on NVIDIA GPUs
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_clips = read_protocol("train")
    if args.extra_train == "asvspoof5":
        train_clips = train_clips + read_asvspoof5_train()
    train_set = TrainClips(pick(train_clips, args.limit, args.seed), args.messy_prob,
                           args.rawboost_prob, args.random_windows, args.seed)
    dev_set = TrainClips(pick(read_protocol("dev"), args.dev_limit, args.seed))
    shuffler = torch.Generator()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, generator=shuffler)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, num_workers=args.workers)
    print(f"device {DEVICE}: {len(train_set)} train clips, {len(dev_set)} dev clips, "
          f"{len(train_loader)} batches per epoch")

    model = SSLAASIST(pretrained=True).to(DEVICE)  # Meta's XLS-R + a fresh, untrained AASIST
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9], device=DEVICE))

    # Resume: the cluster stops jobs after 3 days, so we save after every epoch
    # and can pick up where we left off by submitting the job again.
    start_epoch, best_eer, bad_epochs = 0, float("inf"), 0
    last_path = out / "last.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_eer, bad_epochs = state["epoch"] + 1, state["best_eer"], state["bad_epochs"]
        print(f"resumed after epoch {state['epoch']} (best dev EER so far {best_eer:.2%})")

    log_path = out / "log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,train_loss,dev_eer,minutes\n")

    for epoch in range(start_epoch, args.epochs):
        train_set.epoch = epoch
        shuffler.manual_seed(args.seed * 1000 + epoch)  # a new (but repeatable) order each epoch
        model.train()  # dropout on: we're learning
        started = time.time()
        total_loss = 0.0

        for step, (audio, target) in enumerate(train_loader, 1):
            logits = model(audio.to(DEVICE))           # 1. forward: the model guesses
            loss = loss_fn(logits, target.to(DEVICE))  # 2. loss: how wrong was it?
            optimizer.zero_grad()                      #    (forget the last batch's blame)
            loss.backward()                            # 3. backward: which weights are to blame?
            optimizer.step()                           # 4. nudge every weight a tiny bit
            total_loss += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch}  batch {step}/{len(train_loader)}  "
                      f"loss {total_loss / step:.4f}  {(time.time() - started) / 60:.1f} min", flush=True)

        train_loss = total_loss / len(train_loader)
        eer = dev_eer(model, dev_loader)
        minutes = (time.time() - started) / 60

        # Keep the epoch with the best dev EER: later epochs may start memorising train.
        if eer < best_eer:
            best_eer, bad_epochs = eer, 0
            torch.save(model.state_dict(), out / "best.pt")
            note = "new best, saved best.pt"
        else:
            bad_epochs += 1
            note = f"no better for {bad_epochs} epoch(s)"
        print(f"epoch {epoch} done: train loss {train_loss:.4f}, dev EER {eer:.2%}, "
              f"{minutes:.1f} min ({note})", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{eer:.6f}", f"{minutes:.1f}"])

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "best_eer": best_eer, "bad_epochs": bad_epochs}, last_path)
        if bad_epochs >= args.patience:
            print(f"stopping early: dev EER hasn't improved for {args.patience} epochs")
            break

    print(f"best dev EER {best_eer:.2%}, saved in {out / 'best.pt'}")


if __name__ == "__main__":
    main()
