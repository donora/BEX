"""The Compare dashboard's numbers (bex.benchmark), on cases small enough to check by eye."""
import math

import polars as pl
import pytest

from bex.benchmark import (
    coverage,
    false_detections_per_hour,
    label_windows,
    per_species,
    report_outcomes,
    summary,
    vocalisation_outcomes,
)
from bex.schemas import ANNOTATIONS_SCHEMA
from bex.truth import ALIGNED_SCHEMA

ROBIN, WREN = "Turdus migratorius", "Troglodytes aedon"


def aligned(*rows):
    """(start_s, species, y_true, score) on a 3 s grid in recording r1."""
    return pl.DataFrame(
        [{"recording_id": "r1", "start_s": s, "end_s": s + 3.0, "species_key": k,
          "y_true": y, "score": sc} for s, k, y, sc in rows],
        schema=ALIGNED_SCHEMA)


def boxes(*rows):
    return pl.DataFrame(
        [{"recording_id": "r1", "start_s": a, "end_s": b, "low_hz": float("nan"),
          "high_hz": float("nan"), "species_key": k} for a, b, k in rows],
        schema=ANNOTATIONS_SCHEMA)


@pytest.fixture
def lab():
    a = aligned(
        (0.0, ROBIN, 1, 0.9),    # correct
        (3.0, ROBIN, 1, 0.2),    # missed window
        (6.0, ROBIN, 0, 0.8),    # wrong …
        (9.0, ROBIN, 0, 0.8),    # … same event, next window
        (0.0, WREN, 1, 0.7),     # right, but the filter hides it
        (3.0, WREN, 0, 0.7),     # wrong, and the filter hides it
    )
    hidden = pl.DataFrame({"recording_id": ["r1", "r1"], "start_s": [0.0, 3.0],
                           "species_key": [WREN, WREN]})
    return label_windows(a, {ROBIN: 0.5, WREN: 0.5}, hidden)


def test_reported_windows_sort_into_the_navigators_outcomes(lab):
    assert report_outcomes(lab) == {"correct": 1, "filter hid a real bird": 1,
                                    "wrong": 2, "filter caught a mistake": 1}


def test_false_detections_are_counted_as_events_and_only_if_shown(lab):
    # two consecutive wrong Robin windows = one event; the hidden Wren is not seen
    assert false_detections_per_hour(lab, hours=0.5) == pytest.approx(2.0)


def test_vocalisations_found_hidden_missed_and_not_asked(lab):
    b = boxes((0.0, 2.0, ROBIN),     # owns window 0 (a visible hit)  -> found
              (3.5, 5.5, ROBIN),     # owns window 3 only (no hit)     -> missed
              (0.5, 2.5, WREN),      # only hit is hidden              -> hidden
              (30.0, 31.0, ROBIN))   # past every window               -> not asked
    got = vocalisation_outcomes(lab, b).sort("species_key", "outcome")
    assert sorted(got["outcome"].to_list()) == [
        "filter hid a real bird", "found", "missed", "not asked"]


def test_a_species_with_no_threshold_is_never_a_hit():
    lab = label_windows(aligned((0.0, ROBIN, 1, 0.99)), {ROBIN: float("nan")})
    assert not lab["hit"].any()


def test_per_species_and_summary(lab):
    b = boxes((0.0, 2.0, ROBIN), (3.5, 5.5, ROBIN), (0.5, 2.5, WREN))
    vb = vocalisation_outcomes(lab, b)
    ps = per_species(lab, vb).sort("species_key")
    robin = ps.filter(pl.col("species_key") == ROBIN).row(0, named=True)
    assert robin["found"] == pytest.approx(0.5)            # 1 of 2 songs
    assert robin["precision"] == pytest.approx(1 / 3)      # 1 right of 3 reported
    s = summary(lab, vb, hours=0.5)
    assert s["precision"] == pytest.approx(2 / 5)          # hidden-right counts as right
    assert s["found"] == pytest.approx(1 / 3)              # 1 of 3 askable songs
    assert s["singing"] == {"found": 1, "filter hid a real bird": 1, "missed": 1}


def test_coverage_groups():
    groups = coverage({"a": {"x", "y"}, "b": {"x", "z"}, "c": {"x"}},
                      ["x", "y", "z", "w"])
    assert groups["all models"] == ["x"]
    assert groups["only a"] == ["y"] and groups["only b"] == ["z"]
    assert groups["no model"] == ["w"]


def test_no_audio_means_no_rate():
    assert math.isnan(false_detections_per_hour(label_windows(aligned(), {}), 0.0))


def test_by_recording_keeps_recordings_with_only_false_detections(lab):
    from bex.benchmark import by_recording
    b = boxes((0.0, 2.0, ROBIN), (3.5, 5.5, ROBIN))
    t = by_recording(lab, vocalisation_outcomes(lab, b)).filter(
        pl.col("species_key") == ROBIN).row(0, named=True)
    assert (t["songs"], t["found_n"], t["false_events"]) == (2, 1, 1)


def test_mistaken_for_names_the_bird_reported_in_a_missed_window():
    from bex.benchmark import mistaken_for
    lab = label_windows(aligned((0.0, ROBIN, 1, 0.1),      # robin singing, missed
                                (0.0, WREN, 0, 0.9)),      # wren reported instead
                        {ROBIN: 0.5, WREN: 0.5})
    got = mistaken_for(lab, ROBIN)
    assert got["species_key"].to_list() == [WREN] and got["share"][0] == 1.0


def test_crowd_counts_companions_within_delta_at_their_own_thresholds():
    from bex.benchmark import crowd
    det = pl.DataFrame({
        "recording_id": ["r1"] * 4, "start_s": [0.0] * 4,
        "species_key": [ROBIN, WREN, "Imposter one", "Far below"],
        "score_raw": [0.80, 0.75, 0.78, 0.40],
        "above": [True, True, True, True],
        "implausible": [False, False, True, False],
    })
    table, s = crowd(det, ROBIN, delta=0.1)
    assert set(table["species_key"]) == {WREN, "Imposter one"}   # 0.40 is too far
    assert s["windows"] == 1 and s["companions"] == 2
    assert s["implausible_companions"] == 1


def test_species_lists_name_what_a_recording_would_be_said_to_hold():
    from bex.benchmark import species_list_summary, species_lists
    det = pl.DataFrame({
        "recording_id": ["r1"] * 6,
        "start_s": [0.0, 3.0, 0.0, 0.0, 6.0, 0.0],
        "species_key": [ROBIN, ROBIN, WREN, "Car", "Regulus ignicapilla", "Strix aluco"],
        "above": [True, True, True, True, True, True],
        "implausible": [False, False, True, False, False, True],
    })
    ann = boxes((0.0, 2.0, ROBIN), (0.0, 2.0, WREN), (0.0, 2.0, "Sitta pygmaea"))
    got = dict(species_lists(det, ann).select("species_key", "outcome").iter_rows())
    assert got == {ROBIN: "correct",                       # listed, annotated
                   WREN: "filter hid a real bird",         # only without the filter
                   "Sitta pygmaea": "missed",              # annotated, never listed
                   "Regulus ignicapilla": "wrong",         # listed, not there
                   "Strix aluco": "filter caught a mistake"}   # "Car" is no species
    s = species_list_summary(species_lists(det, ann))
    assert s["recall"] == pytest.approx(1 / 3) and s["precision"] == pytest.approx(1 / 2)
    assert s["listed_per_recording"] == 2 and s["annotated_per_recording"] == 3
    # needing two detections drops the one-off wrong species (and the wren)
    strict = dict(species_lists(det, ann, min_detections=2)
                  .select("species_key", "outcome").iter_rows())
    assert strict[ROBIN] == "correct" and "Regulus ignicapilla" not in strict
