import math

import numpy as np
import polars as pl
import pytest
import soundfile as sf

from bex.ingest import (
    dataset_audio_root,
    ingest_sne,
    list_datasets,
    parse_timestamp,
    read_dataset,
    recording_id_for,
    scan_folder,
    write_dataset,
)


def test_parse_timestamp():
    assert parse_timestamp("SNE_001_20180509_050002") == "2018-05-09T05:00:02"
    assert parse_timestamp("20240501_063000") == "2024-05-01T06:30:00"
    assert parse_timestamp("mysite-20240501-063000") == "2024-05-01T06:30:00"
    # Classic AudioMoth hex name: epoch seconds, so it IS UTC.
    assert parse_timestamp("5AFC0A30") == "2018-05-16T10:38:40Z"
    assert parse_timestamp("dawn_chorus_take2") == ""
    assert parse_timestamp("99999999_123456") == ""  # 8 digits, not a date


@pytest.fixture()
def wav_folder(tmp_path):
    """A messy real-world-ish folder: mixed rates, formats, nesting, a non-audio file."""
    rng = np.random.default_rng(0)
    sf.write(tmp_path / "20240501_063000.wav", rng.normal(0, 0.1, 44100 * 2), 44100)
    sf.write(tmp_path / "moth.flac", rng.normal(0, 0.1, 48000), 48000)
    (tmp_path / "siteB").mkdir()
    sf.write(tmp_path / "siteB" / "moth.flac", rng.normal(0, 0.1, 22050 * 3), 22050)
    (tmp_path / "notes.txt").write_text("not audio")
    return tmp_path


def test_scan_folder(wav_folder):
    df = scan_folder(wav_folder, site="glen", lat=56.9, lon=-4.8)
    assert len(df) == 3  # notes.txt ignored
    assert df["recording_id"].to_list() == ["20240501_063000", "moth", "siteB__moth"]
    by_id = {r["recording_id"]: r for r in df.to_dicts()}
    assert by_id["20240501_063000"]["start_time"] == "2024-05-01T06:30:00"
    assert by_id["20240501_063000"]["sample_rate"] == 44100
    assert by_id["moth"]["start_time"] == ""
    assert by_id["siteB__moth"]["path"] == "siteB/moth.flac"
    assert by_id["siteB__moth"]["duration_s"] == pytest.approx(3.0)
    assert (df["site"] == "glen").all()
    assert not df["labelled"].any()


def test_scan_failures(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        scan_folder(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="No audio files"):
        scan_folder(tmp_path)


def test_recording_id_nested():
    from pathlib import Path

    assert recording_id_for(Path("a.wav")) == "a"
    assert recording_id_for(Path("x/y/a.wav")) == "x__y__a"


def test_dataset_roundtrip(wav_folder, tmp_path):
    store = tmp_path / "store"
    df = scan_folder(wav_folder, site="glen")
    write_dataset(store, "glen", df, audio_root=wav_folder)
    assert list_datasets(store) == ["glen"]
    back, ann = read_dataset(store, "glen")
    assert back.equals(df)
    assert ann is None
    assert dataset_audio_root(store, "glen") == wav_folder

    # Simulate the unplugged-drive case: root recorded but gone.
    write_dataset(store, "gone", df, audio_root=wav_folder / "vanished")
    with pytest.raises(FileNotFoundError, match="audio_root"):
        dataset_audio_root(store, "gone")


def test_ingest_sne(sne_labels_dir, sne_audio_dir, tmp_path):
    """Stage 1 deliverable: the labelled pipeline produces the same manifest shape
    as the unlabelled one, plus a lossless annotations table."""
    recordings, ann = ingest_sne(sne_labels_dir, sne_audio_dir)
    assert len(recordings) == 33
    assert recordings["labelled"].all()
    assert recordings["sample_rate"].unique().to_list() == [32000]
    assert (recordings["site"] == "sierra-nevada").all()
    # Timestamps parsed from every SNE filename.
    assert (recordings["start_time"].str.len_chars() > 0).all()

    # 20,147 boxes in the CSV, minus the 2 zero-duration point annotations the
    # schema validator caught (they can never label a window).
    assert len(ann) == 20145
    assert ann["species_key"].n_unique() == 56
    assert set(ann["recording_id"]) <= set(recordings["recording_id"])

    # And it persists like any other dataset.
    d = write_dataset(tmp_path, "sne", recordings, ann, audio_root=sne_audio_dir)
    back_rec, back_ann = read_dataset(tmp_path, "sne")
    assert back_rec.equals(recordings) and back_ann.equals(ann)
    assert (d / "meta.json").exists()
