import numpy as np
import polars as pl
import pytest

from bex.schemas import (
    DETECTIONS_SCHEMA,
    RECORDINGS_SCHEMA,
    RunManifest,
    SchemaError,
    empty_detections,
    validate_detections,
    validate_recordings,
)


def make_recordings(**overrides) -> pl.DataFrame:
    base = {
        "recording_id": ["r1", "r2"],
        "path": ["a.flac", "b.wav"],
        "duration_s": [3600.0, 120.5],
        "sample_rate": [32000, 44100],
        "channels": [1, 2],
        "start_time": ["2018-05-09T05:00:02", ""],
        "site": ["sierra-nevada", "glen-affric"],
        "lat": [38.49, float("nan")],
        "lon": [-119.95, float("nan")],
        "labelled": [True, False],
    }
    base.update(overrides)
    return pl.DataFrame(base, schema=RECORDINGS_SCHEMA)


def make_detections(**overrides) -> pl.DataFrame:
    base = {
        "recording_id": ["r1", "r1"],
        "run_id": ["birdnet-abc123", "birdnet-abc123"],
        "start_s": [0.0, 3.0],
        "end_s": [3.0, 6.0],
        "species_key": ["Turdus merula", "Strepera graculina"],
        "score_raw": [0.91, 0.55],
        "occ_score": [0.8, float("nan")],
        "suppressed": [False, True],
        "rank_in_window": [1, 2],
    }
    base.update(overrides)
    return pl.DataFrame(base, schema=DETECTIONS_SCHEMA)


def test_valid_frames_pass():
    validate_recordings(make_recordings())
    validate_detections(make_detections())
    validate_detections(empty_detections())


def test_missing_column_named_in_error():
    with pytest.raises(SchemaError, match="missing columns.*lat"):
        validate_recordings(make_recordings().drop("lat"))


def test_extra_column_rejected():
    with pytest.raises(SchemaError, match="unexpected columns"):
        validate_detections(make_detections().with_columns(pl.lit(1).alias("stray")))


def test_wrong_dtype_named():
    df = make_detections().with_columns(pl.col("score_raw").cast(pl.Float64))
    with pytest.raises(SchemaError, match="score_raw.*expected"):
        validate_detections(df)


def test_duplicate_recording_ids_rejected():
    with pytest.raises(SchemaError, match="not unique"):
        validate_recordings(make_recordings(recording_id=["r1", "r1"]))


def test_score_out_of_range_rejected():
    with pytest.raises(SchemaError, match="outside \\[0, 1\\]"):
        validate_detections(make_detections(score_raw=[1.5, 0.5]))


def test_end_before_start_rejected():
    with pytest.raises(SchemaError, match="end_s <= start_s"):
        validate_detections(make_detections(end_s=[0.0, 6.0]))


def test_null_in_required_column_rejected():
    with pytest.raises(SchemaError, match="species_key.*null"):
        validate_detections(make_detections(species_key=["Turdus merula", None]))


def test_manifest_roundtrip(tmp_path):
    m = RunManifest(
        model_name="birdnet",
        model_version="2.4",
        window_s=3.0,
        hop_s=3.0,
        input_sr=48000,
        score_transform="sigmoid",
        geofilter={"lat": 38.49, "lon": -119.95, "week": 19, "sf_thresh": 0.03},
        profile="us-ca-sierra",
        packages={"birdnet": "0.1.0"},
    )
    assert m.run_id.startswith("birdnet-")
    assert m.created_utc  # stamped automatically

    path = m.save(tmp_path)
    assert path == tmp_path / "runs" / m.run_id / "manifest.json"
    loaded = RunManifest.load(tmp_path, m.run_id)
    assert loaded == m  # the executed config survives the round trip, exactly


def test_manifest_ids_unique():
    a = RunManifest("m", "1", 3.0, 3.0, 48000, "sigmoid")
    b = RunManifest("m", "1", 3.0, 3.0, 48000, "sigmoid")
    assert a.run_id != b.run_id
