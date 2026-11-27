import librosa
import numpy as np

SAMPLE_RATE = 16000
LENGTH = 64600  # about 4 seconds at 16 kHz


def fix_length(audio, length=LENGTH):
    if len(audio) >= length:
        return audio[:length]
    repeats = length // len(audio) + 1
    return np.tile(audio, repeats)[:length]


def load_audio(path):
    """The whole clip at 16 kHz mono, any length."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


def load_clip(path):
    """The first ~4 seconds (repeat-padded if shorter). Used for scoring."""
    return fix_length(load_audio(path))


def random_window(audio, rng, length=LENGTH):
    """A random ~4-second piece of the clip, for training.

    Long clips: start somewhere random, so the model hears every part over
    the epochs, not only the beginning. Short clips: repeat-pad first, then
    start somewhere random in the repeats.
    """
    if len(audio) < length:
        audio = np.tile(audio, length // len(audio) + 2)
    start = int(rng.integers(0, len(audio) - length + 1))
    return audio[start : start + length]
