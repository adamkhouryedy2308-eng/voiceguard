import csv
from pathlib import Path

import numpy as np

LA_ROOT = Path("data/asvspoof2019/LA")

PROTOCOLS = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}


def read_protocol(split):
    """Read the label file for one split.

    Each line looks like: LA_0079 LA_T_1138215 - - bonafide
    (speaker, file, unused, attack, label).
    Returns a list of (audio_path, label, attack), label 0 = real, 1 = fake.
    """
    protocol = LA_ROOT / "ASVspoof2019_LA_cm_protocols" / PROTOCOLS[split]
    audio_dir = LA_ROOT / f"ASVspoof2019_LA_{split}" / "flac"
    clips = []
    for line in protocol.read_text().splitlines():
        speaker, file_id, _, attack, label = line.split()
        path = audio_dir / f"{file_id}.flac"
        clips.append((path, 1 if label == "spoof" else 0, attack))
    return clips


ITW_ROOT = Path("data/in_the_wild/release_in_the_wild")


def read_in_the_wild():
    """Read In-the-Wild's meta.csv (file,speaker,label; label is bona-fide or spoof).

    Returns the same (audio_path, label, attack) list as read_protocol.
    In-the-Wild doesn't name the fake-making tool, so every fake gets attack "ITW".
    """
    clips = []
    with open(ITW_ROOT / "meta.csv", newline="") as f:
        for row in csv.DictReader(f):
            fake = row["label"] == "spoof"
            clips.append((ITW_ROOT / row["file"], 1 if fake else 0, "ITW" if fake else "-"))
    return clips


ASVSPOOF5_ROOT = Path("data/asvspoof5")


def _find(root, pattern):
    """The first file under root matching pattern (the archives' folder names vary)."""
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"no {pattern} under {root}")
    return matches[0]


def read_asvspoof5_train():
    """ASVspoof 5 (2024) training set: 182,357 clips, crowdsourced, modern attacks.

    ASVspoof5.train.tsv is space-separated, 10 columns:
    SPEAKER_ID FLAC_FILE_NAME GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP
    KEY is "spoof" or "bonafide". Audio is in flac_T/<FLAC_FILE_NAME>.flac.
    Returns the same (audio_path, label, attack) list as read_protocol.
    """
    protocol = _find(ASVSPOOF5_ROOT, "ASVspoof5.train.tsv")
    audio_dir = ASVSPOOF5_ROOT / "flac_T"
    if not audio_dir.is_dir():
        audio_dir = _find(ASVSPOOF5_ROOT, "flac_T")
    clips = []
    for line in protocol.read_text().splitlines():
        fields = line.split()
        if len(fields) < 9 or fields[8] not in ("spoof", "bonafide"):
            continue  # header or blank line
        fake = fields[8] == "spoof"
        clips.append((audio_dir / f"{fields[1]}.flac", 1 if fake else 0, fields[7] if fake else "-"))
    return clips


DF21_ROOT = Path("data/asvspoof2021_df")


def read_df21(fakes=20000, seed=0):
    """The referee: ASVspoof 2021 DF eval, part 00 only (the part we downloaded).

    Different speakers, sources and compression (MP3, AAC, OGG ...) from all our
    training data, so it's a fair way to choose between runs without touching
    In-the-Wild. Keys line, e.g.:
      LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress ...
    (speaker, file, codec, source, attack, key, ...).
    To keep scoring quick: every real clip in part 00, plus a fixed random
    sample of `fakes` fake clips (same sample every time).
    """
    keys = _find(DF21_ROOT, "trial_metadata.txt")
    audio_dir = _find(DF21_ROOT, "flac")
    present = {p.stem for p in audio_dir.glob("*.flac")}
    real, fake = [], []
    for line in keys.read_text().splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[1] not in present:
            continue
        if fields[5] == "bonafide":
            real.append((audio_dir / f"{fields[1]}.flac", 0, "-"))
        elif fields[5] == "spoof":
            fake.append((audio_dir / f"{fields[1]}.flac", 1, fields[2]))  # group fakes by codec
    order = np.random.default_rng(seed).permutation(len(fake))[:fakes]
    return real + [fake[j] for j in sorted(order)]


ML_ITW_ROOT = Path("data/ml_itw")


def read_ml_itw():
    """ML-ITW (Xue et al., 2026): real-world clips of 180 public figures, 14 languages.

    Second real-world final exam, next to In-the-Wild: test only, never trained on.
    meta.csv (made by scripts/unpack_ml_itw.py): file,speaker,language,platform,label,
    label "bona-fide" or "spoof". Every clip, real or fake, is grouped by its language,
    so evaluate.py can give one EER per language (real vs fake of that language).
    """
    clips = []
    with open(ML_ITW_ROOT / "meta.csv", newline="") as f:
        for row in csv.DictReader(f):
            clips.append((ML_ITW_ROOT / row["file"], 1 if row["label"] == "spoof" else 0, row["language"]))
    return clips
