"""Make clean clips sound more like real-world audio.

Applied to train clips only, real and fake alike, so that noise, phone quality,
compression and room echo stop being clues for "fake".
"""

import io

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from voiceguard.audio import SAMPLE_RATE, fix_length


def add_noise(audio, rng):
    # Signal-to-noise ratio in decibels: 10 dB = loud noise, 30 dB = faint hiss.
    snr_db = rng.uniform(10, 30)
    signal_power = np.mean(audio**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)


def phone_quality(audio):
    # Phone calls carry only 8 kHz audio: go down to 8 kHz and back up.
    narrow = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=8000)
    return librosa.resample(narrow, orig_sr=8000, target_sr=SAMPLE_RATE)


def mp3_compress(audio, rng):
    # Squeeze through an MP3 encoder, like audio uploaded to the internet.
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="MP3", compression_level=rng.uniform(0.5, 0.95))
    buffer.seek(0)
    compressed, _ = sf.read(buffer, dtype="float32")
    return fix_length(compressed, len(audio))


def add_echo(audio, rng):
    # Room echo: the voice bounces off walls and reaches the mic a bit later,
    # quieter each time. We fake a "room" as a burst of fading noise and blend it in.
    seconds = rng.uniform(0.2, 0.8)  # how long the echo takes to die away (small room -> big hall)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    room = rng.normal(0, 1, len(t)) * np.exp(-6.9 * t / seconds)  # fades to 1/1000 at the end
    room[0] = 0
    direct_db = rng.uniform(0, 10)  # how much louder the straight-to-mic voice is than the echoes
    room *= np.sqrt(10 ** (-direct_db / 10) / np.sum(room**2))
    room[0] = 1  # the voice itself, straight to the mic
    wet = fftconvolve(audio, room)[: len(audio)]
    wet *= np.sqrt(np.mean(audio**2) / (np.mean(wet**2) + 1e-12))  # keep the same loudness
    return wet.astype(np.float32)


def augment(audio, rng, echo=False):
    """Apply each effect with some chance, so clips get different mixes.

    echo=False keeps the exact mix used for the messy-dev results so far;
    training our own model uses echo=True.
    """
    if echo and rng.random() < 0.3:
        audio = add_echo(audio, rng)  # first: the room, before the phone line or upload
    if rng.random() < 0.3:
        audio = phone_quality(audio)
    if rng.random() < 0.5:
        audio = mp3_compress(audio, rng)
    if rng.random() < 0.6:
        audio = add_noise(audio, rng)
    return audio
