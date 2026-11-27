"""RawBoost: data augmentation for raw audio (Tak et al., ICASSP 2022).

Ported from https://github.com/TakHemlata/SSL_Anti-spoofing (MIT licence,
Copyright (c) 2022 Hemlata Tak). Same maths and default settings as the
original; the only change is that every random draw comes from a numpy
Generator (rng), so each clip's distortion can be reproduced exactly.

It imitates what a recording chain does to a voice:
  1. LnL: the microphone and amplifier colour the sound (random filters),
     including slight distortion (non-linear: x, x^2, x^3 ... filtered).
  2. ISD: crackles and clicks that follow the loudness of the voice.
  3. SSI: steady background noise shaped by random filters.
The authors' LA model used 1 then 2 ("algo 5"); their DF model used 3 ("algo 3").
"""

import numpy as np
from scipy import signal

FS = 16000


def _norm(x, always):
    peak = np.max(np.abs(x))
    if always or peak > 1:
        x = x / (peak + 1e-12)
    return x


def _notch_filter(rng, n_bands=5, min_f=20, max_f=8000, min_bw=100, max_bw=1000,
                  min_coeff=10, max_coeff=100, min_g=0, max_g=0):
    """A random filter made of n_bands band-pass pieces (a random 'colour' of sound)."""
    b = np.array([1.0])
    for _ in range(n_bands):
        fc = rng.uniform(min_f, max_f)
        bw = rng.uniform(min_bw, max_bw)
        c = int(rng.uniform(min_coeff, max_coeff))
        if c % 2 == 0:
            c += 1
        f1 = max(fc - bw / 2, 1 / 1000)
        f2 = min(fc + bw / 2, FS / 2 - 1 / 1000)
        b = np.convolve(signal.firwin(c, [f1, f2], window="hamming", fs=FS), b)
    # The original draws between min_g and max_g even when min_g > max_g (e.g. -5 and -20);
    # the old numpy allowed that, the new Generator needs low <= high. Same range either way.
    gain = rng.uniform(min(min_g, max_g), max(min_g, max_g))
    _, h = signal.freqz(b, 1, fs=FS)
    return 10 ** (gain / 20) * b / np.max(np.abs(h))


def _apply_filter(x, b):
    n = b.shape[0] + 1
    y = signal.lfilter(b, 1, np.pad(x, (0, n)))
    return y[n // 2 : y.shape[0] - n // 2][: x.shape[0]]


def lnl_convolutive_noise(x, rng, n_f=5, min_bias=5, max_bias=20):
    """1. Linear and non-linear convolutive noise (mic/amplifier colouring)."""
    y = np.zeros_like(x, dtype=np.float64)
    min_g, max_g = 0, 0
    for i in range(n_f):
        if i == 1:
            min_g, max_g = min_g - min_bias, max_g - max_bias
        b = _notch_filter(rng, min_g=min_g, max_g=max_g)
        y = y + _apply_filter(np.power(x, i + 1), b)
    y = y - np.mean(y)
    return _norm(y, always=False)


def isd_additive_noise(x, rng, p=10, g_sd=2):
    """2. Impulsive signal-dependent noise (clicks that follow the voice's loudness)."""
    beta = rng.uniform(0, p)
    y = x.astype(np.float64).copy()
    n = int(len(x) * beta / 100)
    idx = rng.permutation(len(x))[:n]
    f_r = (2 * rng.random(n) - 1) * (2 * rng.random(n) - 1)
    y[idx] = x[idx] + g_sd * x[idx] * f_r
    return _norm(y, always=False)


def ssi_additive_noise(x, rng, snr_min=10, snr_max=40):
    """3. Stationary signal-independent noise (steady background, randomly coloured)."""
    noise = _apply_filter(rng.normal(0, 1, len(x)), _notch_filter(rng))
    noise = _norm(noise, always=True)
    snr = rng.uniform(snr_min, snr_max)
    noise = noise / (np.linalg.norm(noise) + 1e-12) * np.linalg.norm(x) / 10 ** (0.05 * snr)
    return x + noise


def rawboost(audio, rng):
    """Pick one of the authors' two recipes at random: LA's (1 then 2) or DF's (3)."""
    x = audio.astype(np.float64)
    if rng.random() < 0.5:
        x = isd_additive_noise(lnl_convolutive_noise(x, rng), rng)
    else:
        x = ssi_additive_noise(x, rng)
    return x.astype(np.float32)
