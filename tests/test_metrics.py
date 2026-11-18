import numpy as np

from voiceguard.metrics import compute_eer


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