"""Check the readers for the newer datasets on tiny made-up copies of their files."""

import voiceguard.dataset as dataset


def test_read_asvspoof5_train(tmp_path, monkeypatch):
    protocols = tmp_path / "ASVspoof5_protocols"
    protocols.mkdir()
    (protocols / "ASVspoof5.train.tsv").write_text(
        "T_4850 T_0000000000 F - - - - bonafide bonafide -\n"
        "T_4850 T_0000000001 F - - - AC1 A01 spoof -\n"
    )
    (tmp_path / "flac_T").mkdir()
    monkeypatch.setattr(dataset, "ASVSPOOF5_ROOT", tmp_path)

    clips = dataset.read_asvspoof5_train()
    assert clips == [
        (tmp_path / "flac_T" / "T_0000000000.flac", 0, "-"),
        (tmp_path / "flac_T" / "T_0000000001.flac", 1, "A01"),
    ]


def test_read_df21_keeps_only_downloaded_part(tmp_path, monkeypatch):
    keys = tmp_path / "keys" / "DF" / "CM"
    keys.mkdir(parents=True)
    (keys / "trial_metadata.txt").write_text(
        "LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress x - - - -\n"
        "LA_0023 DF_E_2000012 mp3m4a vcc2020 bonafide bonafide notrim eval x - - - -\n"
        "LA_0023 DF_E_2000013 ogg vcc2018 A07 spoof notrim eval x - - - -\n"
        "LA_0023 DF_E_9999999 nocodec asvspoof A14 spoof notrim eval x - - - -\n"  # not in part 00
    )
    flac = tmp_path / "ASVspoof2021_DF_eval" / "flac"
    flac.mkdir(parents=True)
    for name in ("DF_E_2000011", "DF_E_2000012", "DF_E_2000013"):
        (flac / f"{name}.flac").touch()
    monkeypatch.setattr(dataset, "DF21_ROOT", tmp_path)

    clips = dataset.read_df21(fakes=10)
    labels = sorted((path.stem, label, group) for path, label, group in clips)
    assert labels == [
        ("DF_E_2000011", 1, "nocodec"),
        ("DF_E_2000012", 0, "-"),
        ("DF_E_2000013", 1, "ogg"),
    ]
    # the fake sample is the same every time
    assert dataset.read_df21(fakes=1) == dataset.read_df21(fakes=1)
