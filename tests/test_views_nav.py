"""Navigator + overlay view functions (the Explorer's 'where should I look')."""
import polars as pl
import pytest

from bex.schemas import ANNOTATIONS_SCHEMA, DETECTIONS_SCHEMA
from bex.views import merge_runs, navigator_bins, species_spans


def det_rows(*rows) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"recording_id": "r1", "run_id": "run", "start_s": s, "end_s": s + 3.0,
             "species_key": k, "score_raw": sc, "occ_score": 0.5, "suppressed": sup,
             "rank_in_window": 1}
            for (s, k, sc, sup) in rows
        ],
        schema=DETECTIONS_SCHEMA,
    )


def ann_rows(*rows) -> pl.DataFrame:
    return pl.DataFrame(
        [{"recording_id": "r1", "start_s": s, "end_s": e, "low_hz": 1000.0,
          "high_hz": 5000.0, "species_key": k} for (s, e, k) in rows],
        schema=ANNOTATIONS_SCHEMA,
    )


def test_merge_runs_collapses_consecutive():
    blocks = det_rows(
        (0.0, "Turdus merula", 0.9, False),
        (3.0, "Turdus merula", 0.7, False),   # contiguous -> merges
        (6.0, "Erithacus rubecula", 0.6, False),
        (12.0, "Turdus merula", 0.8, False),  # gap -> separate span
    ).select("start_s", "end_s", "species_key", "score_raw")
    merged = merge_runs(blocks)
    assert merged["start_s"].to_list() == [0.0, 6.0, 12.0]
    assert merged["end_s"].to_list() == [6.0, 9.0, 15.0]
    # the run keeps its best score (Float32 in the schema, hence approx)
    assert merged["score_raw"][0] == pytest.approx(0.9, abs=1e-6)


def test_merge_runs_empty():
    empty = det_rows().select("start_s", "end_s", "species_key", "score_raw")
    assert merge_runs(empty).is_empty()


def test_species_spans_filters_layer():
    det = det_rows(
        (0.0, "Turdus merula", 0.9, False),
        (3.0, "Turdus merula", 0.8, False),
        (6.0, "Turdus merula", 0.05, False),   # below theta
        (9.0, "Turdus merula", 0.7, True),     # suppressed layer
        (0.0, "Erithacus rubecula", 0.9, False),
    )
    visible = species_spans(det, "r1", 0.0, 30.0, 0.1, "Turdus merula", suppressed=False)
    assert visible["start_s"].to_list() == [0.0]
    assert visible["end_s"].to_list() == [6.0]      # two windows merged
    hidden = species_spans(det, "r1", 0.0, 30.0, 0.1, "Turdus merula", suppressed=True)
    assert hidden["start_s"].to_list() == [9.0]
    both = species_spans(det, "r1", 0.0, 30.0, 0.1, "Turdus merula")
    assert len(both) == 2


def judged(det: pl.DataFrame, theta: float = 0.1) -> pl.DataFrame:
    """Thresholded and judged as the app hands frames to the navigator: the
    model's own geofilter as the judge, one θ for every species."""
    return det.with_columns(above=pl.col("score_raw") >= theta,
                            implausible=pl.col("suppressed"))


def test_navigator_sorts_species_into_outcomes():
    det = judged(det_rows(
        (0.0, "Turdus merula", 0.9, False),        # annotated, shown   -> correct
        (0.0, "Erithacus rubecula", 0.4, False),   # not annotated      -> wrong
        (0.0, "Sylvia atricapilla", 0.8, True),    # annotated, hidden  -> hid a real bird
        (3.0, "Strepera graculina", 0.8, True),    # not annotated, hidden -> caught
        (0.0, "Regulus regulus", 0.02, False),     # below θ, ignored
    ))
    ann = ann_rows((0.0, 5.0, "Turdus merula"), (0.0, 5.0, "Sylvia atricapilla"),
                   (0.0, 5.0, "Parus major"))
    nav = navigator_bins(det, ann, "r1", 120.0, n_bins=12)   # 10 s bins
    first = nav.row(0, named=True)
    assert first["n_correct"] == 1 and first["names_correct"] == "Turdus merula"
    assert first["n_wrong"] == 1
    assert first["n_filter hid a real bird"] == 1
    assert first["n_filter caught a mistake"] == 1
    assert first["names_missed"] == "Parus major"
    assert first["n_truth"] == 3
    # everything annotated is accounted for: found, hidden, or missed
    assert (first["n_correct"] + first["n_filter hid a real bird"]
            + first["n_missed"]) == first["n_truth"]
    assert nav["n_correct"][1:].sum() == 0


def test_navigator_counts_species_not_windows():
    """Two windows of one bird in a bin are one species — or a 3 s grid would
    out-count a 5 s grid for the same song."""
    det = judged(det_rows((0.0, "Turdus merula", 0.9, False),
                          (3.0, "Turdus merula", 0.9, False)))
    nav = navigator_bins(det, ann_rows((0.0, 6.0, "Turdus merula")), "r1", 60.0,
                         n_bins=6)
    assert nav["n_correct"][0] == 1


def test_navigator_correct_needs_the_label_rules_overlap():
    """A box grazing a window by less than min(0.5 s, box) does not make the
    detection correct — same rule as alignment, so the scorecard agrees."""
    det = judged(det_rows((0.0, "Turdus merula", 0.9, False)))
    nav = navigator_bins(det, ann_rows((2.8, 10.0, "Turdus merula")), "r1", 60.0,
                         n_bins=1)
    assert nav["n_correct"][0] == 0 and nav["n_wrong"][0] == 1


def test_navigator_without_annotations():
    det = judged(det_rows((0.0, "Turdus merula", 0.9, False),
                          (0.0, "Strepera graculina", 0.9, True)))
    nav = navigator_bins(det, None, "r1", 60.0, n_bins=2)
    assert nav["n_detected"][0] == 1 and nav["n_filter would hide"][0] == 1
    assert "n_missed" not in nav.columns


def test_navigator_focus_narrows_truth_too():
    det = judged(det_rows((0.0, "Turdus merula", 0.9, False)))
    ann = ann_rows((0.0, 30.0, "Turdus merula"), (30.0, 60.0, "Parus major"))
    nav = navigator_bins(det, ann, "r1", 60.0, n_bins=2, species_key="Turdus merula")
    assert nav["n_truth"].to_list() == [1, 0]
    assert nav["n_missed"].sum() == 0


def test_lane_lines_say_which_detections_are_right():
    from bex.views import lane_lines
    det = judged(det_rows((0.0, "Turdus merula", 0.9, False),
                          (30.0, "Turdus merula", 0.9, False))).with_columns(
        theta=pl.lit(0.1), strength=pl.lit(0.5))
    ann = ann_rows((0.0, 3.0, "Turdus merula"))
    got = lane_lines(det, "r1", 0.0, 60.0, ann).sort("start_s")
    # same species, same colour — only the second is wrong
    assert got["correct"].to_list() == [True, False]
    assert lane_lines(det, "r1", 0.0, 60.0)["correct"].null_count() == 2
