import numpy as np


def rates_at(scores, labels, threshold):
    predicted_fake = scores > threshold
    false_alarm_rate = predicted_fake[labels == 0].mean()
    miss_rate = (~predicted_fake)[labels == 1].mean()
    return false_alarm_rate, miss_rate


def compute_eer(scores, labels):
    best_gap = 1.0
    for t in np.linspace(0, 1, 101):
        fa, miss = rates_at(scores, labels, t)
        gap = abs(fa - miss)
        if gap < best_gap:
            best_gap = gap
            best_t = t
            best_eer = (fa + miss) / 2
    return best_eer, best_t
    