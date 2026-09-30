"""Alignment, on fixtures whose labels are obvious by inspection."""
import json
import pathlib

import numpy as np
import polars as pl
import pytest

from bex.ingest import write_dataset

REPO = pathlib.Path(__file__).resolve().parent.parent
from bex.schemas import ANNOTATIONS_SCHEMA, RECORDINGS_SCHEMA, RunManifest
from bex.store import ScoreMatrix, write_scores
from bex.truth import (
    SHARED_GRID,
    align_recording,
    align_run,
    eval_species,
    list_aligned,
    project_scores,
    read_aligned,
    write_aligned,
)
from bex.windows import GridParams

ROBIN = "Turdus migratorius"
THRUSH = "Catharus guttatus"
GHOST = "Strepera graculina"      # annotated nowhere, in no vocabulary


def boxes(*rows) -> pl.DataFrame:
    return pl.DataFrame(
        [{"recording_id": "r1", "start_s": a, "end_s": b, "low_hz": float("nan"),
          "high_hz": float("nan"), "species_key": k} for (a, b, k) in rows],
        schema=ANNOTATIONS_SCHEMA,
    )


def matrix(scores, start_s, window_s, keys, rid="r1", run="test-run") -> ScoreMatrix:
    return ScoreMatrix(rid, run, np.asarray(scores, dtype=np.float16),
                       np.asarray(start_s, dtype=np.float32), window_s, keys)


# --------------------------------------------------------------------------- #
# Grid projection
# --------------------------------------------------------------------------- #

def test_five_second_windows_spread_over_their_one_second_frames():
    """A 5 s window claiming 0.9 claims it for all five of its seconds."""
    scores = np.array([[1.0], [0.5], [0.25]], dtype=np.float32)
    out = project_scores(scores, np.array([0.0, 5.0, 10.0]), 5.0, np.arange(15.0), 1.0)
    np.testing.assert_allclose(out[:, 0], [1.0] * 5 + [0.5] * 5 + [0.25] * 5)


def test_overlapping_windows_take_the_max_not_the_mean():
    # 3 s windows -> 5 s frames: frame [0,5) is covered by the 0.2 and the 0.9 window.
    scores = np.array([[0.2], [0.9], [0.1]], dtype=np.float32)
    out = project_scores(scores, np.array([0.0, 3.0, 6.0]), 3.0, np.array([0.0, 5.0]), 5.0)
    np.testing.assert_allclose(out[:, 0], [0.9, 0.9])


def test_native_to_native_is_the_identity():
    scores = np.array([[0.3, 0.7], [0.1, 0.2]], dtype=np.float32)
    starts = np.array([0.0, 3.0])
    np.testing.assert_allclose(project_scores(scores, starts, 3.0, starts, 3.0), scores)


def test_frames_past_the_last_window_are_zero_not_missing():
    """The model produced nothing there, so a truth box in that tail is a miss."""
    out = project_scores(np.array([[0.8]], dtype=np.float32), np.array([0.0]), 5.0,
                         np.arange(8.0), 1.0)
    np.testing.assert_allclose(out[:, 0], [0.8] * 5 + [0.0] * 3)


def test_projection_of_nothing():
    assert project_scores(np.zeros((0, 2), np.float32), np.array([]), 5.0,
                          np.arange(3.0), 1.0).shape == (3, 2)


# --------------------------------------------------------------------------- #
# One recording
# --------------------------------------------------------------------------- #

def test_dense_rows_one_per_window_and_species():
    sm = matrix([[0.8, 0.1], [0.2, 0.3]], [0.0, 3.0], 3.0, [ROBIN, THRUSH])
    df = align_recording(sm, boxes((0.0, 2.0, ROBIN)), [ROBIN, THRUSH])
    assert df.height == 4                      # 2 windows x 2 species, zeros included
    assert df["start_s"].to_list() == [0.0, 0.0, 3.0, 3.0]
    assert df["end_s"].to_list() == [3.0, 3.0, 6.0, 6.0]


def test_truth_and_score_land_on_the_right_row():
    sm = matrix([[0.8, 0.1], [0.2, 0.3]], [0.0, 3.0], 3.0, [ROBIN, THRUSH])
    df = align_recording(sm, boxes((0.0, 2.0, ROBIN)), [ROBIN, THRUSH])
    got = {(r["start_s"], r["species_key"]): (r["y_true"], round(r["score"], 3))
           for r in df.iter_rows(named=True)}
    assert got[(0.0, ROBIN)] == (1, 0.8)       # the box is here
    assert got[(0.0, THRUSH)] == (0, 0.1)
    assert got[(3.0, ROBIN)] == (0, 0.2)       # and not here
    assert got[(3.0, THRUSH)] == (0, 0.3)


def test_the_label_floor_is_the_windows_rule():
    """min(0.5 s, box) — a 0.3 s chip must be fully inside the window to count."""
    sm = matrix([[0.5], [0.5]], [0.0, 3.0], 3.0, [ROBIN])
    inside = align_recording(sm, boxes((1.0, 1.3, ROBIN)), [ROBIN])
    assert inside["y_true"].to_list() == [1, 0]
    straddling = align_recording(sm, boxes((2.9, 3.2, ROBIN)), [ROBIN])
    assert straddling["y_true"].to_list() == [0, 0]


def test_species_the_model_cannot_name_score_zero_rather_than_vanishing():
    """Keeping it is what makes cmAP comparable across arms with different
    vocabularies — dropping it would score two models on two species sets."""
    sm = matrix([[0.8]], [0.0], 3.0, [ROBIN])
    df = align_recording(sm, boxes((0.0, 2.0, ROBIN), (0.0, 2.0, GHOST)),
                         [ROBIN, GHOST])
    ghost = df.filter(pl.col("species_key") == GHOST).row(0, named=True)
    assert (ghost["y_true"], ghost["score"]) == (1, 0.0)


def test_truth_uses_the_models_own_window_starts():
    """A model whose last window is short of the duration must not get labels on a
    window it never produced."""
    sm = matrix([[0.5], [0.5]], [0.0, 5.0], 5.0, [ROBIN])
    df = align_recording(sm, boxes((11.0, 13.0, ROBIN)), [ROBIN], duration_s=20.0)
    assert df.height == 2                      # not four windows' worth
    assert df["y_true"].to_list() == [0, 0]    # the box is past what the model scored


def test_shared_grid_relabels_truth_and_spreads_the_score():
    sm = matrix([[0.9], [0.1]], [0.0, 5.0], 5.0, [ROBIN])
    df = align_recording(sm, boxes((0.0, 2.0, ROBIN)), [ROBIN],
                         grid=SHARED_GRID, duration_s=10.0)
    assert df.height == 10                     # ten 1 s frames
    # abs=1e-3: the store keeps scores as float16, so 0.9 comes back as 0.89990.
    assert df["score"].to_list()[:5] == [pytest.approx(0.9, abs=1e-3)] * 5
    # The box covers [0,2): frames 0 and 1 clear the 0.5 s floor, frame 2 does not.
    assert df["y_true"].to_list() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def test_a_box_the_native_grid_blurs_is_sharp_on_the_shared_grid():
    """The reason the shared grid exists: a 1 s call inside a 5 s window labels the
    whole window natively, and only its own second on the shared grid."""
    sm = matrix([[0.9]], [0.0], 5.0, [ROBIN])
    native = align_recording(sm, boxes((3.0, 4.0, ROBIN)), [ROBIN])
    shared = align_recording(sm, boxes((3.0, 4.0, ROBIN)), [ROBIN],
                             grid=SHARED_GRID, duration_s=5.0)
    assert native["y_true"].sum() == 1 and native.height == 1
    assert shared["y_true"].to_list() == [0, 0, 0, 1, 0]


# --------------------------------------------------------------------------- #
# A whole run, through the store
# --------------------------------------------------------------------------- #

@pytest.fixture
def tiny_store(tmp_path):
    recs = pl.DataFrame(
        [{"recording_id": "r1", "path": "r1.flac", "duration_s": 10.0,
          "sample_rate": 32000, "channels": 1, "start_time": "", "site": "s",
          "lat": 0.0, "lon": 0.0, "labelled": True},
         {"recording_id": "r2", "path": "r2.flac", "duration_s": 10.0,
          "sample_rate": 32000, "channels": 1, "start_time": "", "site": "s",
          "lat": 0.0, "lon": 0.0, "labelled": True}],
        schema=RECORDINGS_SCHEMA,
    )
    ann = pl.concat([boxes((0.0, 2.0, ROBIN)),
                     boxes((6.0, 8.0, THRUSH)).with_columns(
                         pl.lit("r2").alias("recording_id"))])
    write_dataset(tmp_path, "tiny", recs, ann, audio_root=str(tmp_path))
    manifest = RunManifest(model_name="toy", model_version="1", window_s=5.0,
                           hop_s=5.0, input_sr=32000, score_transform="sigmoid",
                           dataset="tiny", run_id="toy-1")
    manifest.save(tmp_path)
    for rid, scores in (("r1", [[0.9, 0.1], [0.2, 0.3]]),
                        ("r2", [[0.1, 0.2], [0.3, 0.7]])):
        write_scores(tmp_path, matrix(scores, [0.0, 5.0], 5.0, [ROBIN, THRUSH], rid, "toy-1"))
    return tmp_path


def test_eval_species_is_every_annotated_species():
    ann = pl.concat([boxes((0.0, 1.0, THRUSH)), boxes((0.0, 1.0, ROBIN))])
    assert eval_species(ann) == sorted([ROBIN, THRUSH])


def test_align_run_covers_every_recording_window_and_species(tiny_store):
    aligned, meta = align_run(tiny_store, "tiny", "toy-1")
    assert aligned.height == 2 * 2 * 2            # 2 recordings x 2 windows x 2 species
    assert meta["n_recordings"] == 2 and meta["n_species"] == 2
    assert meta["n_windows"] == 4 and meta["n_positive"] == 2
    assert meta["grid"] == "native" and meta["model_name"] == "toy"
    assert meta["species_not_in_vocabulary"] == []


def test_align_run_puts_each_recordings_truth_in_its_own_recording(tiny_store):
    aligned, _ = align_run(tiny_store, "tiny", "toy-1")
    hits = aligned.filter(pl.col("y_true") == 1).select("recording_id", "species_key",
                                                        "start_s")
    assert sorted(hits.iter_rows()) == [("r1", ROBIN, 0.0), ("r2", THRUSH, 5.0)]


def test_align_run_on_the_shared_grid(tiny_store):
    aligned, meta = align_run(tiny_store, "tiny", "toy-1", grid=SHARED_GRID)
    assert meta["grid"] == "shared-1s" and meta["window_s"] == 1.0
    assert aligned.height == 2 * 10 * 2           # ten 1 s frames per recording
    assert aligned["y_true"].sum() == 4           # two 2 s boxes -> two frames each


def test_align_run_can_be_restricted_to_some_recordings(tiny_store):
    aligned, meta = align_run(tiny_store, "tiny", "toy-1", recordings=["r1"])
    assert meta["recordings"] == ["r1"] and aligned["recording_id"].unique().to_list() == ["r1"]


def test_align_run_reports_species_missing_from_the_vocabulary(tiny_store):
    aligned, meta = align_run(tiny_store, "tiny", "toy-1", species=[ROBIN, GHOST])
    assert meta["species_not_in_vocabulary"] == [GHOST]
    assert aligned.filter(pl.col("species_key") == GHOST)["score"].max() == 0.0


def test_align_run_needs_annotations(tmp_path):
    recs = pl.DataFrame([{"recording_id": "r1", "path": "r1.flac", "duration_s": 10.0,
                          "sample_rate": 32000, "channels": 1, "start_time": "",
                          "site": "s", "lat": 0.0, "lon": 0.0, "labelled": False}],
                        schema=RECORDINGS_SCHEMA)
    write_dataset(tmp_path, "bare", recs, None, audio_root=str(tmp_path))
    RunManifest(model_name="toy", model_version="1", window_s=5.0, hop_s=5.0,
                input_sr=32000, score_transform="sigmoid", run_id="toy-1").save(tmp_path)
    with pytest.raises(ValueError, match="no annotations"):
        align_run(tmp_path, "bare", "toy-1")


def test_align_run_with_nothing_scored(tiny_store):
    RunManifest(model_name="toy", model_version="1", window_s=5.0, hop_s=5.0,
                input_sr=32000, score_transform="sigmoid", run_id="toy-2").save(tiny_store)
    with pytest.raises(ValueError, match="no scored recordings"):
        align_run(tiny_store, "tiny", "toy-2")


def test_cache_roundtrip_keeps_the_frame_and_its_provenance(tiny_store):
    aligned, meta = align_run(tiny_store, "tiny", "toy-1")
    write_aligned(tiny_store, "tiny", "toy-1", aligned, meta)
    back, back_meta = read_aligned(tiny_store, "tiny", "toy-1")
    assert back.equals(aligned)
    assert back_meta["n_positive"] == meta["n_positive"]
    assert [m["run_id"] for m in list_aligned(tiny_store, "tiny")] == ["toy-1"]


def test_listing_alignments_of_a_dataset_with_none(tiny_store):
    assert list_aligned(tiny_store, "nothing-here") == []


def test_a_sigmoid_run_is_comparable_across_windows():
    m = RunManifest(model_name="birdnet", model_version="2.4", window_s=3.0, hop_s=3.0,
                    input_sr=48000, score_transform="sigmoid")
    ok, why = __import__("bex.truth", fromlist=["x"]).across_window_comparable(m)
    assert ok and why == ""


def test_a_softmax_run_is_flagged_because_ap_ranks_across_windows():
    """Softmax divides by a per-window sum, so the same bird scores lower in a
    busy window — which is precisely the comparison a per-species PR curve makes."""
    from bex.truth import across_window_comparable
    m = RunManifest(model_name="perch", model_version="v2", window_s=5.0, hop_s=5.0,
                    input_sr=32000, score_transform="softmax over raw logits",
                    run_id="perch-1")
    ok, why = across_window_comparable(m)
    assert ok is False
    assert "perch-1" in why and "sigmoid" in why


# --------------------------------------------------------------------------- #
# Cross-check against the real store, brute force (skips on a bare clone)
# --------------------------------------------------------------------------- #

def test_real_alignment_matches_a_naive_reimplementation():
    """The aligned frame, re-derived the slow obvious way and compared row for row.

    `label_starts` narrows with searchsorted and vectorises the overlap; this walks
    every (box, window) pair and applies the rule in plain Python. Same answer or
    the fast path is wrong — the discipline `scripts/crosscheck.py` established for
    the forensics numbers, applied to the labels everything is now scored against.
    """
    store_dir = REPO / "store"
    if not (store_dir / "truth" / "sne").exists():
        pytest.skip("no cached alignment — run scripts/evaluate.py first")
    from bex import ingest
    from bex.truth import read_aligned

    aligned, meta = read_aligned(store_dir, "sne", "birdnet-c102ff26", "native")
    _, ann = ingest.read_dataset(store_dir, "sne")
    rid = meta["recordings"][0]

    got = aligned.filter((pl.col("recording_id") == rid) & (pl.col("y_true") == 1))
    window_s = meta["window_s"]
    boxes_here = ann.filter(pl.col("recording_id") == rid)
    starts = sorted(aligned.filter(pl.col("recording_id") == rid)["start_s"].unique())

    expected = set()
    for b in boxes_here.iter_rows(named=True):
        floor = min(0.5, b["end_s"] - b["start_s"])
        for w in starts:
            overlap = min(b["end_s"], w + window_s) - max(b["start_s"], w)
            if overlap > 0 and overlap >= floor - 1e-9:
                expected.add((w, b["species_key"]))

    assert set(zip(got["start_s"], got["species_key"])) == expected
    assert len(expected) > 0, "fixture recording has no labels — pick another"


def test_both_arms_see_identical_truth_on_the_shared_grid():
    """Ground truth does not depend on the model, so on one shared grid the two
    arms' label columns must agree exactly. A projection bug that shifted one arm's
    windows would show up here and nowhere else."""
    store_dir = REPO / "store"
    from bex.truth import aligned_path, read_aligned
    runs = ["birdnet-c102ff26", "perch-495c1c32"]
    if not all(aligned_path(store_dir, "sne", r, "shared-1s").exists() for r in runs):
        pytest.skip("no cached shared-grid alignments — run evaluate.py --grid shared")

    a, _ = read_aligned(store_dir, "sne", runs[0], "shared-1s")
    b, _ = read_aligned(store_dir, "sne", runs[1], "shared-1s")
    key = ["recording_id", "start_s", "species_key"]
    assert a.sort(key)["y_true"].equals(b.sort(key)["y_true"])


def test_sigmoid_prose_mentioning_softmax_is_not_flagged_as_softmax():
    """Regression: the real sigmoid manifest says '...not comparable with the
    softmax run's', and a substring search for 'softmax' flagged it as one."""
    from bex.truth import across_window_comparable
    m = RunManifest(
        model_name="perch", model_version="v2", window_s=5.0, hop_s=5.0,
        input_sr=32000, run_id="perch-1",
        score_transform=("per-class sigmoid over raw logits (sensitivity 1.0) — θ "
                         "here is not comparable with the softmax run's."))
    assert across_window_comparable(m)[0] is True


# --------------------------------------------------------------------------- #
# Rank resolution — can the store still separate the top-ranked windows?
# --------------------------------------------------------------------------- #

def _resolution_frame(scores: np.ndarray, n_pos: int = 10) -> pl.DataFrame:
    """One species, `scores` over as many windows, the top `n_pos` labelled."""
    order = np.argsort(-scores)
    y = np.zeros(len(scores), dtype=np.uint8)
    y[order[:n_pos]] = 1
    return pl.DataFrame({
        "recording_id": ["r1"] * len(scores),
        "start_s": np.arange(len(scores), dtype=float),
        "end_s": np.arange(1, len(scores) + 1, dtype=float),
        "species_key": [ROBIN] * len(scores),
        "y_true": y,
        "score": scores.astype(np.float32),
    })


def test_well_resolved_scores_are_not_flagged():
    from bex.truth import rank_resolution
    scores = np.linspace(0.0, 0.5, 300)
    got = rank_resolution(_resolution_frame(scores))
    assert got["median_distinct_fraction"] == pytest.approx(1.0)
    assert got["damaged"] is False


def test_a_saturated_float16_score_is_flagged_as_damaged():
    """The real failure: sigmoid values crowded near 1.0, stored as float16, all
    collapse onto the same handful of representable values."""
    from bex.truth import rank_resolution
    saturated = np.linspace(0.9990, 0.9999, 300).astype(np.float16).astype(np.float32)
    got = rank_resolution(_resolution_frame(saturated))
    assert got["damaged"] is True
    assert got["median_distinct_fraction"] < 0.1
    # float32 over the same range keeps them apart — it is the storage, not the model.
    fine = np.linspace(0.9990, 0.9999, 300).astype(np.float32)
    assert rank_resolution(_resolution_frame(fine))["damaged"] is False


def test_species_with_too_few_labels_are_left_out():
    from bex.truth import rank_resolution
    got = rank_resolution(_resolution_frame(np.linspace(0, 1, 300), n_pos=3),
                          min_positives=10)
    assert got["n_species"] == 0 and np.isnan(got["median_distinct_fraction"])


def test_resolution_on_an_empty_frame():
    from bex.truth import rank_resolution
    empty = _resolution_frame(np.linspace(0, 1, 5)).head(0)
    assert rank_resolution(empty)["n_species"] == 0
