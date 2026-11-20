"""Train logistic regression on train features, grade it on dev with EER.

Usage:
    uv run python scripts/train_classifier.py
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from voiceguard.metrics import compute_eer

FEATURES = Path("data/features")
MODELS = Path("models")


def main():
    # 1. Load the saved XLS-R numbers and labels.
    train = np.load(FEATURES / "train.npz")
    dev = np.load(FEATURES / "dev.npz")

    # 2. Build the classifier.
    #    StandardScaler: puts every column on the same scale before the weights.
    #    class_weight="balanced": real clips count ~9x more, because train has ~9x more fakes.
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )

    # 3. Learn the weights from train only.
    classifier.fit(train["features"], train["labels"])

    # 4. Score every dev clip: column 1 of predict_proba is the chance of "fake".
    dev_scores = classifier.predict_proba(dev["features"])[:, 1]

    # 5. Grade it with our own EER code.
    eer, threshold = compute_eer(dev_scores, dev["labels"])
    print(f"dev EER: {eer:.2%} at threshold {threshold:.2f}")

    # 6. Save the trained classifier and the dev scores for later (HTER, demo).
    MODELS.mkdir(exist_ok=True)
    joblib.dump(classifier, MODELS / "logreg.joblib")
    np.savez(FEATURES / "dev_scores.npz", scores=dev_scores, labels=dev["labels"])
    print(f"saved classifier to {MODELS / 'logreg.joblib'}")


if __name__ == "__main__":
    main()
