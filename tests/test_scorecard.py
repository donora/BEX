"""Per-species accounting, on fixtures small enough to count by hand."""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from bex.scorecard import (
    box_coverage,
    box_summary,
    by_recording,
    confusion_in_missed_windows,
    counts,
    species_index,
    timeline,
)
from bex.schemas import ANNOTATIONS_SCHEMA

REPO = Path(__file__).resolve().parent.parent
ROBIN = "Turdus migratorius"
THRUSH = "Catharus guttatus"


def aligned(*rows, species=ROBIN, rid="r1", window_s=3.0) -> pl.DataFrame:
    """(start_s, y_true, score) triples -> an aligned frame for one species."""
    return pl.DataFrame(
        [{"recording_id": rid, "start_s": float(s), "end_s": s + window_s,
          "species_key": species, "y_true": np.uint8(y), "score": np.float32(sc)}
         for (s, y, sc) in rows],
        schema={"recording_id": pl.Utf8, "start_s": pl.Float64, "end_s": pl.Float64,
                "species_key": pl.Utf8, "y_true": pl.UInt8, "score": pl.Float32},
    )


def boxes(*rows, species=ROBIN, rid="r1") -> pl.DataFrame:
    return pl.DataFrame(
        [{"recording_id": rid, "start_s": a, "end_s": b, "low_hz": float("nan"),
          "high_hz": float("nan"), "species_key": species} for (a, b) in rows],
        schema=ANNOTATIONS_SCHEMA,
    )


# --------------------------------------------------------------------------- #
# Counts
# --------------------------------------------------------------------------- #

def test_counts_are_the_four_outcomes():
    a = aligned((0, 1, 0.9),    # caught
                (3, 1, 0.1),    # missed
                (6, 0, 0.8),    # false alarm
                (9, 0, 0.0))    # correct silence
    got = counts(a, ROBIN, theta=0.5)
    assert (got["caught"], got["missed"], got["false_alarms"],
            got["correct_silence"]) == (1, 1, 1, 1)
    assert got["opportunity"] == 2
    assert got["precision"] == pytest.approx(0.5)
    assert got["recall"] == pytest.approx(0.5)


def test_opportunity_is_what_a_perfect_detector_would_have_reported():
    a = aligned((0, 1, 0.0), (3, 1, 0.0), (6, 0, 0.0))
    got = counts(a, ROBIN, theta=0.5)
    assert got["opportunity"] == 2 and got["caught"] == 0
    assert got["recall"] == pytest.approx(0.0)


def test_counts_for_a_species_with_no_rows():
    assert counts(aligned((0, 1, 0.9)), "Nothing here", 0.5)["windows"] == 0


def test_threshold_moves_the_counts_the_right_way():
    a = aligned((0, 1, 0.6), (3, 0, 0.7))
    strict = counts(a, ROBIN, 0.8)
    loose = counts(a, ROBIN, 0.5)
    assert (strict["caught"], strict["false_alarms"]) == (0, 0)
    assert (loose["caught"], loose["false_alarms"]) == (1, 1)


# --------------------------------------------------------------------------- #
# Per recording
# --------------------------------------------------------------------------- #

def test_by_recording_splits_the_accounting():
    a = pl.concat([aligned((0, 1, 0.9), (3, 1, 0.9), rid="r1"),
                   aligned((0, 1, 0.1), (3, 0, 0.9), rid="r2")])
    got = by_recording(a, ROBIN, 0.5).sort("recording_id")
    r1, r2 = got.row(0, named=True), got.row(1, named=True)
    assert (r1["caught"], r1["missed"], r1["false_alarms"]) == (2, 0, 0)
    assert r1["recall"] == pytest.approx(1.0)
    assert (r2["caught"], r2["missed"], r2["false_alarms"]) == (0, 1, 1)
    assert r2["recall"] == pytest.approx(0.0)


def test_recordings_without_the_species_still_report_false_alarms():
    """A false alarm in a recording the bird was never in is exactly as real as
    one anywhere else, so the row must not be dropped."""
    a = pl.concat([aligned((0, 1, 0.9), rid="r1"),
                   aligned((0, 0, 0.9), rid="r2")])
    got = by_recording(a, ROBIN, 0.5).filter(pl.col("recording_id") == "r2").row(0, named=True)
    assert got["opportunity"] == 0 and got["false_alarms"] == 1
    assert got["recall"] is None      # undefined, not zero


# --------------------------------------------------------------------------- #
# Boxes — the grid-free unit
# --------------------------------------------------------------------------- #

def test_a_long_box_spans_several_windows():
    """The user's arithmetic: a 9 s box on a 3 s grid is three windows."""
    a = aligned((0, 1, 0.9), (3, 1, 0.2), (6, 1, 0.2))
    cov = box_coverage(a, boxes((0.0, 9.0)), ROBIN, theta=0.5)
    row = cov.row(0, named=True)
    assert row["windows"] == 3
    assert row["caught"] == 1
    assert row["detected"] is True      # one window is enough to find the bird


def test_a_box_the_arm_was_silent_through_is_undetected():
    a = aligned((0, 1, 0.1), (3, 1, 0.1))
    cov = box_coverage(a, boxes((0.0, 6.0)), ROBIN, theta=0.5)
    assert cov.row(0, named=True)["detected"] is False
    assert cov.row(0, named=True)["caught"] == 0


def test_box_detection_is_more_forgiving_than_window_recall():
    """The distinction the module exists for: going quiet mid-song costs window
    recall but does not lose the bird."""
    a = aligned((0, 1, 0.9), (3, 1, 0.1), (6, 1, 0.1), (9, 1, 0.1))
    assert counts(a, ROBIN, 0.5)["recall"] == pytest.approx(0.25)
    assert box_summary(box_coverage(a, boxes((0.0, 12.0)), ROBIN, 0.5)
                       )["detection_rate"] == pytest.approx(1.0)


def test_box_summary_counts_vocalisations():
    a = aligned((0, 1, 0.9), (3, 0, 0.0), (6, 1, 0.1))
    cov = box_coverage(a, boxes((0.0, 3.0), (6.0, 9.0)), ROBIN, 0.5)
    got = box_summary(cov)
    assert (got["boxes"], got["detected"], got["missed"]) == (2, 1, 1)
    assert got["detection_rate"] == pytest.approx(0.5)


def test_short_box_must_be_inside_the_window_to_own_it():
    """The min(0.5 s, duration) floor, same as alignment."""
    a = aligned((0, 1, 0.9), (3, 0, 0.9))
    cov = box_coverage(a, boxes((2.9, 3.2)), ROBIN, 0.5)
    assert cov.row(0, named=True)["windows"] == 0


def test_box_coverage_with_nothing_to_cover():
    assert box_coverage(aligned((0, 1, 0.9)), boxes(), ROBIN, 0.5).is_empty()
    assert box_summary(pl.DataFrame())["boxes"] == 0


# --------------------------------------------------------------------------- #
# What it said instead
# --------------------------------------------------------------------------- #

def test_confusion_reports_what_fired_in_the_missed_windows():
    a = pl.concat([
        aligned((0, 1, 0.1), (3, 1, 0.1)),                      # robin missed twice
        aligned((0, 0, 0.9), (3, 0, 0.2), species=THRUSH),      # thrush fired once
    ])
    got = confusion_in_missed_windows(a, ROBIN, 0.5)
    assert got.row(0, named=True)["species_key"] == THRUSH
    assert got.row(0, named=True)["windows"] == 1
    assert got.row(0, named=True)["share"] == pytest.approx(0.5)


def test_confusion_is_empty_when_nothing_was_missed():
    a = aligned((0, 1, 0.9))
    assert confusion_in_missed_windows(a, ROBIN, 0.5).is_empty()


# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #

def test_timeline_names_each_window():
    a = aligned((0, 1, 0.9), (3, 1, 0.1), (6, 0, 0.8), (9, 0, 0.0))
    got = timeline(a, ROBIN, "r1", 0.5)
    assert got["outcome"].to_list() == ["caught", "missed", "false alarm",
                                        "correct silence"]


def test_species_index_ranks_by_how_much_truth_there_is():
    a = pl.concat([aligned((0, 1, 0.9), (3, 1, 0.9)),
                   aligned((0, 1, 0.9), species=THRUSH)])
    ann = pl.concat([boxes((0.0, 6.0)), boxes((0.0, 3.0), species=THRUSH)])
    got = species_index(a, ann)
    assert got["species_key"].to_list()[0] in (ROBIN, THRUSH)
    assert set(got.columns) >= {"species_key", "opportunity", "boxes", "seconds"}


# --------------------------------------------------------------------------- #
# Against the real store: box ownership must match the alignment exactly
# --------------------------------------------------------------------------- #

def test_boxes_own_exactly_the_windows_alignment_labelled():
    """box_coverage recomputes which windows a box is responsible for. If that ever
    drifts from `windows.label_starts`, the dashboard would report a model missing
    windows it was never asked about.

    Checked against a naive reimplementation: every (box, window) pair walked in
    plain Python, the union compared with the alignment's positives, both ways.
    """
    store_dir = REPO / "store"
    if not (store_dir / "truth" / "sne").exists():
        pytest.skip("no cached alignment — run scripts/evaluate.py first")
    from bex import ingest
    from bex.truth import read_aligned

    a, meta = read_aligned(store_dir, "sne", "birdnet-c102ff26", "native")
    _, ann = ingest.read_dataset(store_dir, "sne")
    window_s = meta["window_s"]

    for species in ("Corvus corax", "Oreortyx pictus"):
        sub = a.filter(pl.col("species_key") == species)
        positives = {(r["recording_id"], r["start_s"])
                     for r in sub.filter(pl.col("y_true") == 1).iter_rows(named=True)}

        owned = set()
        for rid, grp in sub.group_by("recording_id"):
            rid = rid[0] if isinstance(rid, tuple) else rid
            starts = grp["start_s"].to_list()
            for b in ann.filter((pl.col("species_key") == species)
                                & (pl.col("recording_id") == rid)).iter_rows(named=True):
                floor = min(0.5, b["end_s"] - b["start_s"])
                for w in starts:
                    overlap = min(b["end_s"], w + window_s) - max(b["start_s"], w)
                    if overlap > 0 and overlap >= floor - 1e-9:
                        owned.add((rid, w))

        assert owned == positives, (
            f"{species}: {len(owned - positives)} windows owned by a box but not "
            f"labelled, {len(positives - owned)} labelled but owned by no box")
        assert len(positives) > 0

        # And the counts box_coverage reports must add up to that union: with a
        # threshold below every score, every owned window counts as caught.
        cov = box_coverage(a, ann, species, theta=-1.0)
        assert int(cov["caught"].sum()) == int(cov["windows"].sum())
        assert int(cov["windows"].sum()) >= len(positives)   # boxes may overlap
        # Every box the grid *can* be asked about is found at this threshold; a
        # box owning no window is not a miss, it is unrepresentable on this grid.
        askable = cov.filter(pl.col("windows") > 0)
        assert bool(askable["detected"].all())


def test_a_box_no_window_can_own_is_not_counted_as_a_miss():
    """A 0.3 s chip straddling a window boundary clears no window's overlap floor,
    so the arm was never asked about it. Blaming the model would be blaming the
    grid — and how often this happens differs between a 3 s and a 5 s arm."""
    a = aligned((0, 1, 0.9), (3, 0, 0.9))
    got = box_summary(box_coverage(a, boxes((2.9, 3.2)), ROBIN, 0.5))
    assert got["boxes"] == 1
    assert got["unrepresentable"] == 1
    assert got["missed"] == 0
    assert np.isnan(got["detection_rate"])


# --------------------------------------------------------------------------- #
# The overview: every species at once
# --------------------------------------------------------------------------- #

def test_overview_rate_matches_the_single_species_path():
    """The fast all-species scan and the per-species one must agree exactly —
    the overview is what a reader picks a species *from*, so a discrepancy would
    send them to the wrong bird."""
    from bex.scorecard import detection_overview
    a = pl.concat([aligned((0, 1, 0.9), (3, 1, 0.1), (6, 1, 0.1)),
                   aligned((0, 1, 0.2), (3, 1, 0.9), species=THRUSH)])
    ann = pl.concat([boxes((0.0, 3.0), (6.0, 9.0)),
                     boxes((0.0, 3.0), species=THRUSH)])
    thetas = {ROBIN: 0.5, THRUSH: 0.5}
    over = detection_overview(a, ann, thetas)

    for spp in (ROBIN, THRUSH):
        want = box_summary(box_coverage(a, ann, spp, thetas[spp]))
        got = over.filter(pl.col("species_key") == spp).row(0, named=True)
        assert got["detected"] == want["detected"]
        assert got["askable"] == want["detected"] + want["missed"]
        if want["detection_rate"] == want["detection_rate"]:
            assert got["detection_rate"] == pytest.approx(want["detection_rate"])


def test_overview_agrees_with_the_single_species_path_on_real_data():
    from bex.scorecard import detection_overview
    store_dir = REPO / "store"
    if not (store_dir / "truth" / "sne").exists():
        pytest.skip("no cached alignment")
    from bex import ingest
    from bex.truth import read_aligned

    a, _ = read_aligned(store_dir, "sne", "birdnet-c102ff26", "native")
    _, ann = ingest.read_dataset(store_dir, "sne")
    species = ["Regulus satrapa", "Corvus corax", "Oreortyx pictus"]
    thetas = {s: 0.5 for s in ann["species_key"].unique().to_list()}
    over = detection_overview(a, ann, thetas)
    for spp in species:
        want = box_summary(box_coverage(a, ann, spp, 0.5))
        got = over.filter(pl.col("species_key") == spp).row(0, named=True)
        assert got["detected"] == want["detected"], spp
        assert got["not_asked"] == want["unrepresentable"], spp


def test_overview_marks_a_species_the_rule_cannot_reach():
    """NaN theta means no threshold satisfies the rule for this bird. That is a
    result, and it must not be drawn as a detection rate of zero."""
    from bex.scorecard import detection_overview
    a = aligned((0, 1, 0.9), (3, 1, 0.9))
    over = detection_overview(a, boxes((0.0, 6.0)), {ROBIN: float("nan")})
    row = over.row(0, named=True)
    assert row["reachable"] is False
    assert row["detection_rate"] is None


def test_overview_counts_boxes_recordings_and_unaskable_ones():
    from bex.scorecard import detection_overview
    a = pl.concat([aligned((0, 1, 0.9), rid="r1"), aligned((0, 1, 0.9), rid="r2")])
    ann = pl.concat([boxes((0.0, 3.0), rid="r1"), boxes((0.0, 3.0), rid="r2"),
                     boxes((900.0, 903.0), rid="r1")])   # past what was scored
    row = detection_overview(a, ann, {ROBIN: 0.5}).row(0, named=True)
    assert row["boxes"] == 3 and row["recordings"] == 2
    assert row["not_asked"] == 1 and row["askable"] == 2
    assert row["detected"] == 2


def test_overview_of_nothing():
    from bex.scorecard import detection_overview
    assert detection_overview(aligned((0, 0, 0.1)), boxes(), {}).is_empty()
