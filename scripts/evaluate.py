"""Grade the trained classifier honestly on a test set.

The threshold is frozen on dev (the dev EER threshold), then applied unchanged
to the test set. Reports test EER (hindsight) and HTER (honest), plus the EER
for each attack tool separately.

Usage:
    uv run python scripts/evaluate.py          # eval set
    uv run python scripts/evaluate.py dev      # sanity check on dev
"""

import sys
from pathlib import Path

import joblib
import numpy as np

from voiceguard.metrics import compute_eer, compute_hter, rates_at

FEATURES = Path("data/features")


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "eval"
    classifier = joblib.load("models/logreg.joblib")

    # 1. Freeze the threshold on dev.
    dev = np.load(FEATURES / "dev_scores.npz")
    _, frozen = compute_eer(dev["scores"], dev["labels"])
    print(f"threshold frozen on dev: {frozen:.2f}")

    # 2. Score the test set with the trained classifier.
    test = np.load(FEATURES / f"{split}.npz")
    scores = classifier.predict_proba(test["features"])[:, 1]
    labels = test["labels"]

    # 3. Hindsight vs honest.
    eer, best = compute_eer(scores, labels)
    fa, miss = rates_at(scores, labels, frozen)
    hter = compute_hter(scores, labels, frozen)
    print(f"{split} EER:  {eer:.2%}  (best threshold in hindsight: {best:.2f})")
    print(f"{split} HTER: {hter:.2%}  (false alarms {fa:.2%}, misses {miss:.2%})")

    # 4. EER per attack tool: all real clips vs the fakes from one tool.
    print("\nEER per attack tool:")
    real = labels == 0
    for attack in sorted(set(test["attacks"]) - {"-"}):
        keep = real | (test["attacks"] == attack)
        attack_eer, _ = compute_eer(scores[keep], labels[keep])
        print(f"  {attack}: {attack_eer:.2%}")

    np.savez(FEATURES / f"{split}_scores.npz", scores=scores, labels=labels, attacks=test["attacks"])


if __name__ == "__main__":
    main()
