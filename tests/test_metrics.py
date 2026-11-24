import numpy as np

from voiceguard.metrics import compute_eer, compute_hter


def test_eer_on_toy_example():
    scores = np.array([0.10, 0.35, 0.62, 0.91, 0.70, 0.40])
    labels = np.array([0, 0, 0, 1, 1, 1])
    eer, threshold = compute_eer(scores, labels)
    assert abs(eer - 1 / 3) < 1e-6


def test_perfect_model_has_zero_eer():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    eer, threshold = compute_eer(scores, labels)
    assert eer == 0

def test_eer_with_very_confident_scores():
    # All scores between 0.99 and 1.00, like SSL-AASIST. The old 101-step
    # version could not split these and reported a lopsided 50%.
    scores = np.array([0.9901, 0.9902, 0.9903, 0.9997, 0.9998, 0.9999])
    labels = np.array([0, 0, 0, 1, 1, 1])
    eer, threshold = compute_eer(scores, labels)
    assert eer == 0
    assert 0.9903 <= threshold < 0.9997


def test_hter_at_frozen_threshold():
    # Same toy clips as above. At threshold 0.5:
    # real 0.62 is a false alarm (1 of 3), fake 0.40 is a miss (1 of 3).
    scores = np.array([0.10, 0.35, 0.62, 0.91, 0.70, 0.40])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert abs(compute_hter(scores, labels, 0.5) - 1 / 3) < 1e-6
