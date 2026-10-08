"""Recording-level intervals and paired differences (V1.2 E1–E2)."""
import math

import numpy as np
import polars as pl

from bex import uncertainty as U


def per_rec(rows):
    return pl.DataFrame([{"recording_id": f"r{i}", **dict(zip(U.COUNTS, map(float, r)))}
                         for i, r in enumerate(rows)])


def test_point_matches_pooled_ratios():
    p = per_rec([(10, 8, 20, 3, 4, 2, 1.0), (10, 2, 5, 1, 4, 6, 1.0)])
    s = U.point(p)
    assert s["precision"] == 0.5 and s["window_recall"] == 0.4
    assert s["found"] == 0.5 and s["false_per_hour"] == 4.0


def test_too_few_recordings_gives_no_interval():
    p = per_rec([(10, 8, 20, 3, 4, 2, 1.0)] * 4)
    assert all(math.isnan(lo) for lo, _ in U.bootstrap(p).values())


def test_interval_contains_the_value_and_narrows_with_more_recordings():
    rng = np.random.default_rng(0)
    def make(n):
        return per_rec([(10, int(rng.integers(3, 10)), 20, 3, 4, 2, 1.0) for _ in range(n)])
    small, big = make(8), make(80)
    lo, hi = U.bootstrap(small)["precision"]
    assert lo <= U.point(small)["precision"] <= hi
    assert U.width(big, "precision") < U.width(small, "precision")


def test_paired_difference_sees_a_consistent_gap_overlapping_intervals_hide():
    """Recordings vary a lot, but model a beats b by the same margin on every
    one: the separate intervals overlap, the paired one excludes zero."""
    rng = np.random.default_rng(1)
    base = rng.integers(20, 80, size=12)
    a = per_rec([(100, int(x) + 6, 100, 1, 1, 1, 1.0) for x in base])
    b = per_rec([(100, int(x), 100, 1, 1, 1, 1.0) for x in base])
    la, ha = U.bootstrap(a)["precision"]
    lb, hb = U.bootstrap(b)["precision"]
    assert la < hb                                  # the separate intervals overlap
    d, lo, hi = U.paired(a, b)["precision"]
    assert abs(d - 0.06) < 1e-9 and lo > 0
    assert U.verdict(d, lo, hi) == "higher"
    assert U.verdict(0.01, -0.02, 0.04) == "no clear difference"


def test_per_recording_counts_from_a_labelled_frame():
    lab = pl.DataFrame({
        "recording_id": ["r0"] * 4 + ["r1"] * 2,
        "start_s": [0.0, 3.0, 6.0, 0.0, 0.0, 3.0],
        "end_s": [3.0, 6.0, 9.0, 3.0, 3.0, 6.0],
        "species_key": ["a", "a", "a", "b", "a", "a"],
        "y_true": [1, 0, 0, 1, 0, 0], "hit": [True, True, True, False, False, True],
        "hidden": [False] * 6})
    boxes = pl.DataFrame({"species_key": ["a", "b"], "recording_id": ["r0", "r0"],
                          "outcome": ["found", "missed"]})
    p = U.per_recording(lab, boxes, {"r0": 1.0, "r1": 0.5})
    r0 = p.row(0, named=True)
    assert (r0["reported"], r0["right"], r0["positives"]) == (3, 1, 2)
    assert (r0["songs"], r0["songs_found"], r0["false_events"]) == (2, 1, 1)
    assert p.row(1, named=True)["false_events"] == 1
