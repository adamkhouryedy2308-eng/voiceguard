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

    # 4. EER per attack tool: all real clips vs the fakes from one tool.
    print("\nEER per attack tool:")
    real = labels == 0
    for attack in sorted(set(test["attacks"]) - {"-"}):
        keep = real | (test["attacks"] == attack)
        attack_eer, _ = compute_eer(scores[keep], labels[keep])
        print(f"  {attack}: {attack_eer:.2%}")

    if classifier_path.exists():
        np.savez(FEATURES / f"{split}_scores_{args.model}.npz", scores=scores, labels=labels, attacks=test["attacks"])


if __name__ == "__main__":
    main()
