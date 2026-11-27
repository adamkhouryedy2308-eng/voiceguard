"""Grade the trained classifier honestly on a test set.

The threshold is frozen on dev (the dev EER threshold), then applied unchanged
to the test set. Reports test EER (hindsight) and HTER (honest), plus the EER
for each attack tool separately.

Usage:
    uv run python scripts/evaluate.py eval                        # baseline on eval
    uv run python scripts/evaluate.py in_the_wild --model logreg_aug
    uv run python scripts/evaluate.py in_the_wild --model ssl_aasist_df --calibrate-on dev_aug
"""

import argparse
from pathlib import Path

import joblib
import numpy as np

from voiceguard.metrics import compute_eer, compute_hter, rates_at

FEATURES = Path("data/features")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", nargs="?", default="eval")
    parser.add_argument("--model", default="logreg")
    parser.add_argument("--calibrate-on", default="dev", choices=["dev", "dev_aug"],
                        help="choose the line on clean dev or on messy dev copies")
    args = parser.parse_args()
    split = args.split
    print(f"model: {args.model}")

    # 1. Freeze the threshold on dev (clean, or messy copies made with augment.py).
    dev = np.load(FEATURES / f"{args.calibrate_on}_scores_{args.model}.npz")
    _, frozen = compute_eer(dev["scores"], dev["labels"])
    print(f"threshold frozen on {args.calibrate_on}: {frozen:.4f}")

    # 2. Score the test set. Our classifiers score from saved XLS-R features;
    #    models scored elsewhere (SSL-AASIST) already saved their scores.
    classifier_path = Path(f"models/{args.model}.joblib")
    if classifier_path.exists():
        classifier = joblib.load(classifier_path)
        test = np.load(FEATURES / f"{split}.npz")
        scores = classifier.predict_proba(test["features"])[:, 1]
    else:
        test = np.load(FEATURES / f"{split}_scores_{args.model}.npz")
        scores = test["scores"]
    labels = test["labels"]

    # 3. Hindsight vs honest.
    eer, best = compute_eer(scores, labels)
    fa, miss = rates_at(scores, labels, frozen)
    hter = compute_hter(scores, labels, frozen)
    print(f"{split} EER:  {eer:.2%}  (best threshold in hindsight: {best:.4f})")
    print(f"{split} HTER: {hter:.2%}  (false alarms {fa:.2%}, misses {miss:.2%})")

    # 4. EER per group. Fakes are grouped by attack tool or codec (real clips are "-"),
    #    so it's all real clips vs that group's fakes. ML-ITW groups real clips too
    #    (by language), so there it's that language's real vs that language's fakes.
    print("\nEER per group (attack tool, codec or language):")
    groups = test["attacks"]
    real = labels == 0
    real_groups = set(groups[real]) - {"-"}
    for group in sorted(set(groups) - {"-"}):
        keep = (groups == group) if group in real_groups else real | (groups == group)
        n_real, n_fake = int((labels[keep] == 0).sum()), int((labels[keep] == 1).sum())
        if n_real == 0 or n_fake == 0:
            print(f"  {group}: no {'real' if n_real == 0 else 'fake'} clips, skipped")
            continue
        group_eer, _ = compute_eer(scores[keep], labels[keep])
        print(f"  {group}: {group_eer:.2%}  ({n_real} real, {n_fake} fake)")

    if classifier_path.exists():
        np.savez(FEATURES / f"{split}_scores_{args.model}.npz", scores=scores, labels=labels, attacks=test["attacks"])


if __name__ == "__main__":
    main()
