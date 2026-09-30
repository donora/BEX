"""The Geofilter page's numbers (bex.geofilter), on cases small enough to check by eye."""
import polars as pl
import pytest

from bex import geofilter as gf
from bex.benchmark import crowd
from bex.schemas import ANNOTATIONS_SCHEMA

ROBIN, WREN, KINGLET, FIRECREST = ("Turdus migratorius", "Troglodytes aedon",
                                   "Regulus satrapa", "Regulus ignicapilla")


def judged(*rows):
    """(start_s, species, score, above, implausible) in recording r1, 3 s windows."""
    return pl.DataFrame(
        [{"recording_id": "r1", "start_s": s, "end_s": s + 3.0, "species_key": k,
          "score_raw": sc, "occ_score": 0.5, "above": a, "implausible": i}
         for s, k, sc, a, i in rows])


def boxes(*rows):
    return pl.DataFrame(
        [{"recording_id": "r1", "start_s": a, "end_s": b, "low_hz": float("nan"),
          "high_hz": float("nan"), "species_key": k} for a, b, k in rows],
        schema=ANNOTATIONS_SCHEMA)


@pytest.fixture
def det():
    return judged(
        (0.0, ROBIN, 0.9, True, False),      # let through, annotated   -> correct
        (3.0, ROBIN, 0.8, True, False),      # let through, not there   -> wrong
        (0.0, FIRECREST, 0.85, True, True),  # hidden, not there        -> caught a mistake
        (6.0, WREN, 0.7, True, True),        # hidden, annotated        -> hid a real bird
        (9.0, ROBIN, 0.2, False, False),     # below its threshold: not reported at all
    )


@pytest.fixture
def ann():
    return boxes((0.0, 2.0, ROBIN), (6.5, 8.0, WREN))


def test_every_reported_detection_lands_in_one_of_the_four_outcomes(det, ann):
    assert gf.outcomes(gf.with_truth(det, ann)) == {
        "correct": 1, "wrong": 1, "filter caught a mistake": 1,
        "filter hid a real bird": 1}


def test_a_brush_of_an_annotation_is_not_a_match(det):
    # 0.3 s of overlap is under MIN_OVERLAP_S: the Robin window at 3 s stays wrong
    assert gf.outcomes(gf.with_truth(det, boxes((0.0, 2.0, ROBIN), (5.7, 8.0, ROBIN),
                                                (6.5, 8.0, WREN))))["wrong"] == 1


def test_without_annotations_only_the_filters_side_is_known(det):
    assert gf.outcomes(gf.with_truth(det, None)) == {"let through": 2, "hidden": 2}


def test_impossible_species_counts_recordings_and_real_birds(det, ann):
    got = gf.impossible_species(gf.with_truth(det, ann)).sort("species_key")
    assert got["species_key"].to_list() == [FIRECREST, WREN]
    assert got.filter(pl.col("species_key") == WREN)["real"][0] == 1
    assert got["recordings"].to_list() == [1, 1]


def test_muddy_windows_are_windows_with_a_reported_impossible_bird(det):
    m = gf.muddy_windows(det)
    assert (m["windows"], m["muddy_windows"]) == (3, 2)   # 0 s and 6 s, of 0/3/6 s


def test_unreportable_and_circular(det, ann):
    assert gf.unreportable(det, ann) == [WREN]            # every Wren row is ruled out
    assert not gf.circular(det, ann)
    plausible_only = det.with_columns(implausible=pl.col("species_key") == FIRECREST)
    assert gf.circular(plausible_only, ann)               # no annotated bird ruled out


def test_crowd_carries_mean_strength_for_the_nearness_chart():
    d = judged((0.0, KINGLET, 0.80, True, False),
               (0.0, FIRECREST, 0.75, True, True)).with_columns(
        strength=pl.Series([0.6, 0.4]))
    table, summary = crowd(d, KINGLET, delta=0.1)
    row = table.row(0, named=True)
    assert (row["strength"], row["focal_strength"]) == pytest.approx((0.4, 0.6))
    assert summary["focal_strength"] == pytest.approx(0.6)


def test_precision_with_and_without_the_filter(det, ann):
    # without: 2 right (Robin at 0 s, the hidden Wren) of 4; with: 1 of 2 shown
    before, after = gf.precision_with_and_without(gf.outcomes(gf.with_truth(det, ann)))
    assert (before, after) == pytest.approx((0.5, 0.5))
    assert gf.precision_with_and_without({"let through": 3, "hidden": 1}) != (0.5, 0.5)


def test_recall_with_and_without_the_filter():
    before, after = gf.recall_with_and_without(
        {"found": 6, "filter hid a real bird": 2, "missed": 2})
    assert (before, after) == pytest.approx((0.8, 0.6))
