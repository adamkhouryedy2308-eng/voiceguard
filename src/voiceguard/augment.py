"""Make clean clips sound more like real-world audio.

Applied to train clips only, real and fake alike, so that noise, phone quality
and compression stop being clues for "fake".
"""

import io

import librosa
import numpy as np
import soundfile as sf

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


def augment(audio, rng):
    """Apply each effect with some chance, so clips get different mixes."""
    if rng.random() < 0.3:
        audio = phone_quality(audio)
    if rng.random() < 0.5:
        audio = mp3_compress(audio, rng)
    if rng.random() < 0.6:
        audio = add_noise(audio, rng)
    return audio
