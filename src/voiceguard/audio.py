import librosa
import numpy as np

SAMPLE_RATE = 16000
LENGTH = 64600  # about 4 seconds at 16 kHz


def fix_length(audio, length=LENGTH):
    if len(audio) >= length:
        return audio[:length]
    repeats = length // len(audio) + 1
    return np.tile(audio, repeats)[:length]


def load_clip(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return fix_length(audio)