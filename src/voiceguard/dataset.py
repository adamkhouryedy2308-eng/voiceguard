from pathlib import Path

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
