import numpy as np


def rates_at(scores, labels, threshold):
    predicted_fake = scores > threshold
    false_alarm_rate = predicted_fake[labels == 0].mean()
    miss_rate = (~predicted_fake)[labels == 1].mean()
    return false_alarm_rate, miss_rate


def compute_eer(scores, labels):
    """Try every score as a threshold and return the one where false alarms ~= misses.

    We used to try only 101 thresholds (0.00, 0.01 ... 1.00). That breaks for
    very confident models like SSL-AASIST, whose scores sit between 0.99 and
    0.9998: the balance point falls between two of the 101 marks.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    real = np.sort(scores[labels == 0])
    fake = np.sort(scores[labels == 1])
    thresholds = np.unique(scores)

    # Rule: score > threshold means "fake" (same as rates_at).
    # searchsorted counts, for every threshold at once, how many scores are <= it.
    real_at_or_below = np.searchsorted(real, thresholds, side="right")
    fake_at_or_below = np.searchsorted(fake, thresholds, side="right")
    false_alarm = 1 - real_at_or_below / len(real)  # real clips above the line
    miss = fake_at_or_below / len(fake)  # fake clips at or below the line

    best = np.argmin(np.abs(false_alarm - miss))
    return (false_alarm[best] + miss[best]) / 2, thresholds[best]


def compute_hter(scores, labels, threshold):
    fa, miss = rates_at(scores, labels, threshold)
    return (fa + miss) / 2
