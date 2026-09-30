import numpy as np
import polars as pl
import pytest

from bex.store import (
    ScoreMatrix,
    derive_detections,
    list_scores,
    read_detections,
    read_scores,
    write_detections,
    write_scores,
)


def make_matrix() -> ScoreMatrix:
    # 3 windows × 4 species, hand-checkable numbers.
    scores = np.array(
        [
            [0.90, 0.40, 0.005, 0.0],   # clear detection + a mid score
            [0.02, 0.03, 0.001, 0.0],   # everything sub-threshold except top-k
            [0.00, 0.00, 0.008, 0.6],   # implausible species on top
        ],
        dtype=np.float16,
    )
    return ScoreMatrix(
        recording_id="r1",
        run_id="test-run",
        scores=scores,
        start_s=np.array([0.0, 3.0, 6.0], dtype=np.float32),
        window_s=3.0,
        species_keys=["Turdus merula", "Erithacus rubecula", "Regulus regulus", "Strepera graculina"],
    )


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="species_keys"):
        ScoreMatrix("r", "run", np.zeros((2, 3), np.float16), np.zeros(2, np.float32), 3.0, ["a"])


def test_score_store_roundtrip(tmp_path):
    sm = make_matrix()
    write_scores(tmp_path, sm)
    assert list_scores(tmp_path, "test-run") == ["r1"]
    back = read_scores(tmp_path, "test-run", "r1")
    np.testing.assert_array_equal(back.scores, sm.scores)  # float16 in, float16 out
    np.testing.assert_array_equal(back.start_s, sm.start_s)
    assert back.species_keys == sm.species_keys
    assert (back.recording_id, back.run_id, back.window_s) == ("r1", "test-run", 3.0)


def test_derive_selection_and_ranks(tmp_path):
    sm = make_matrix()
    df = derive_detections(sm, top_k=2, min_score=0.01)

    # Window 0: top-2 (merula, rubecula) kept; 0.005 and 0.0 dropped.
    w0 = df.filter(pl.col("start_s") == 0.0)
    assert set(w0["species_key"]) == {"Turdus merula", "Erithacus rubecula"}
    top = w0.filter(pl.col("species_key") == "Turdus merula")
    assert top["rank_in_window"][0] == 1
    assert top["end_s"][0] == 3.0

    # Window 1: nothing clears min_score but top-2 are still kept (rank info matters).
    w1 = df.filter(pl.col("start_s") == 3.0)
    assert set(w1["species_key"]) == {"Turdus merula", "Erithacus rubecula"}

    # Window 2: the implausible species ranks first — exactly what we must not lose.
    w2 = df.filter(pl.col("start_s") == 6.0)
    assert w2.filter(pl.col("rank_in_window") == 1)["species_key"][0] == "Strepera graculina"

    # Geofilter columns default to "no information yet".
    assert df["occ_score"].is_nan().all()
    assert not df["suppressed"].any()


def test_always_keys_rescue_low_scores():
    sm = make_matrix()
    df = derive_detections(
        sm, top_k=2, min_score=0.01,
        always_keys={"Regulus regulus"}, always_min_score=1e-3,
    )
    # 0.005 and 0.008 are below min_score and outside top-2, but the profile key
    # keeps them; the 0.001 in window 1 sits at the always-floor and is kept too.
    regulus = df.filter(pl.col("species_key") == "Regulus regulus")
    assert len(regulus) == 3


def test_detections_parquet_roundtrip(tmp_path):
    sm = make_matrix()
    df = derive_detections(sm)
    write_detections(tmp_path, sm.run_id, df)
    assert read_detections(tmp_path, sm.run_id).equals(df)


def test_scores_are_float16_by_default(tmp_path):
    """D3's default, unchanged — the size decision the store was built on."""
    sm = make_matrix()
    write_scores(tmp_path, sm)
    assert read_scores(tmp_path, "test-run", "r1").scores.dtype == np.float16


def test_a_saturated_readout_can_be_stored_at_float32(tmp_path):
    """float16 steps by ~0.001 near 1.0, so a saturated score loses the very
    ranking AP is computed from. The fix has to survive the round trip."""
    saturated = np.array([[0.99951, 0.99958, 0.99963]], dtype=np.float32)
    sm = ScoreMatrix("r1", "sat-run", saturated, np.array([0.0], np.float32), 3.0,
                     ["a b", "c d", "e f"])
    write_scores(tmp_path, sm, dtype=np.float32)
    back = read_scores(tmp_path, "sat-run", "r1").scores
    assert back.dtype == np.float32
    assert len(np.unique(back)) == 3, "float32 must keep all three apart"
    # And the point of the exercise: float16 would not.
    assert len(np.unique(saturated.astype(np.float16))) < 3
