"""The sweep, on fixtures small enough to check by hand.

The six-row example runs through most of the file. Its curve, written out:

    θ     tp fp   precision  recall
    0.9    1  0   1.000      0.333
    0.8    1  1   0.500      0.333
    0.7    2  1   0.667      0.667
    0.6    3  1   0.750      1.000
    0.5    3  2   0.600      1.000
    0.4    3  3   0.500      1.000
"""
import numpy as np
import polars as pl
import pytest

from bex.metrics import (
    MAX_F1,
    P95,
    PRCurve,
    Rule,
    apply_rule,
    average_precision,
    cmap,
    curve_frame,
    fbeta,
    per_species_table,
    pr_curve,
    species_curves,
    summary,
    thin,
)

Y_TRUE = np.array([1, 0, 1, 1, 0, 0])
Y_SCORE = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])


def curve() -> PRCurve:
    return pr_curve(Y_TRUE, Y_SCORE)


# --------------------------------------------------------------------------- #
# The curve itself
# --------------------------------------------------------------------------- #

def test_curve_matches_the_hand_computation():
    c = curve()
    assert len(c) == 6
    assert (c.n_pos, c.n_neg) == (3, 3)
    np.testing.assert_allclose(c.theta, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    np.testing.assert_allclose(c.tp, [1, 1, 2, 3, 3, 3])
    np.testing.assert_allclose(c.fp, [0, 1, 1, 1, 2, 3])
    np.testing.assert_allclose(c.fn, [2, 2, 1, 0, 0, 0])
    np.testing.assert_allclose(c.precision, [1, 0.5, 2 / 3, 0.75, 0.6, 0.5])
    np.testing.assert_allclose(c.recall, [1 / 3, 1 / 3, 2 / 3, 1, 1, 1])


def test_every_point_is_a_threshold_you_could_actually_set():
    """θ[i] must be a score the data contains, and `score >= θ[i]` must reproduce
    that point's counts — otherwise the app would report operating points no
    threshold can reach."""
    c = curve()
    for i, theta in enumerate(c.theta):
        predicted = Y_SCORE >= theta
        assert predicted.sum() == c.tp[i] + c.fp[i]
        assert (predicted & (Y_TRUE == 1)).sum() == c.tp[i]


def test_ties_share_one_point():
    c = pr_curve([1, 0, 1], [0.9, 0.9, 0.5])
    assert len(c) == 2                       # not 3: no θ can separate the tied pair
    np.testing.assert_allclose(c.theta, [0.9, 0.5])
    np.testing.assert_allclose(c.precision, [0.5, 2 / 3])


def test_average_precision_is_the_hand_sum():
    # (1/3)(1) + 0(0.5) + (1/3)(2/3) + (1/3)(0.75) + 0 + 0
    assert average_precision(curve()) == pytest.approx(1 / 3 + 2 / 9 + 0.25)


def test_average_precision_agrees_with_sklearn():
    """Cross-check against the reference implementation of the same convention."""
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 60)
        s = rng.random(60).round(2)          # rounding forces plenty of ties
        if y.sum() == 0:
            continue
        assert average_precision(pr_curve(y, s)) == pytest.approx(
            skm.average_precision_score(y, s))


def test_no_positives_gives_nan_not_zero():
    c = pr_curve([0, 0, 0], [0.9, 0.5, 0.1])
    assert c.n_pos == 0
    assert np.isnan(average_precision(c))


def test_empty_input():
    c = pr_curve([], [])
    assert len(c) == 0 and np.isnan(average_precision(c))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="differ"):
        pr_curve([1, 0], [0.5])


def test_prevalence_is_the_random_baseline():
    assert curve().prevalence == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Picking an operating point
# --------------------------------------------------------------------------- #

def test_fbeta_weights():
    p, r = np.array([0.5]), np.array([1.0])
    assert fbeta(p, r, 1.0)[0] == pytest.approx(2 / 3)
    assert fbeta(p, r, 2.0)[0] > fbeta(p, r, 1.0)[0]     # β=2 rewards the recall
    assert fbeta(p, r, 0.5)[0] < fbeta(p, r, 1.0)[0]


def test_max_f1_picks_the_best_point():
    op = apply_rule(curve(), Rule("fbeta", 1.0, min_support=1))
    assert op["feasible"]
    assert op["theta"] == pytest.approx(0.6)
    assert op["f1"] == pytest.approx(2 * 0.75 / 1.75)


def test_precision_floor_takes_the_deepest_qualifying_point():
    """Not the first point that clears the floor — the one with the most recall,
    which is the whole reason a researcher sets a precision floor."""
    op = apply_rule(curve(), Rule("precision", 0.7, min_support=1))
    assert op["theta"] == pytest.approx(0.6)
    assert op["precision"] == pytest.approx(0.75)
    assert op["recall"] == pytest.approx(1.0)


def test_precision_floor_that_only_the_top_point_clears():
    op = apply_rule(curve(), Rule("precision", 0.99, min_support=1))
    assert op["theta"] == pytest.approx(0.9)
    assert op["recall"] == pytest.approx(1 / 3)


def test_infeasible_floor_reports_why_rather_than_a_near_miss():
    op = apply_rule(curve(), Rule("precision", 1.01, min_support=1))
    assert op["feasible"] is False
    assert np.isnan(op["theta"])
    assert "never reaches" in op["reason"] and "1.000" in op["reason"]


def test_min_support_excludes_points_too_shallow_to_trust():
    """With a support floor of 4, the precision-1.0 point at θ=0.9 (one prediction)
    is out of the running, so the rule falls to the deeper, better-supported one."""
    op = apply_rule(curve(), Rule("precision", 0.7, min_support=4))
    assert op["theta"] == pytest.approx(0.6)
    assert apply_rule(curve(), Rule("precision", 0.99, min_support=4))["feasible"] is False


def test_recall_floor_is_the_mirror():
    op = apply_rule(curve(), Rule("recall", 0.9, min_support=1))
    assert op["theta"] == pytest.approx(0.6)      # best precision among R >= 0.9
    assert op["precision"] == pytest.approx(0.75)


def test_rule_with_no_positives_says_so():
    op = apply_rule(pr_curve([0, 0], [0.5, 0.1]), P95)
    assert op["feasible"] is False and "no positives" in op["reason"]


def test_rule_labels_and_slugs():
    assert P95.label == "precision ≥ 0.95" and P95.slug == "p0.95"
    assert MAX_F1.label == "max F1" and MAX_F1.slug == "f1"


def test_unknown_rule_kind_raises():
    with pytest.raises(ValueError, match="unknown rule"):
        apply_rule(curve(), Rule("vibes", 1.0))


def test_thin_keeps_the_ends():
    big = pr_curve(np.r_[np.ones(500), np.zeros(500)].astype(int),
                   np.random.default_rng(1).random(1000))
    small = thin(big, 50)
    assert len(small) <= 50
    assert small.theta[0] == big.theta[0] and small.theta[-1] == big.theta[-1]
    assert (small.n_pos, small.n_neg) == (big.n_pos, big.n_neg)


def test_curve_frame_carries_extras():
    df = curve_frame(curve(), arm="birdnet", species_key="Turdus merula")
    assert df.height == 6
    assert df["arm"].unique().to_list() == ["birdnet"]
    assert set(df.columns) >= {"theta", "precision", "recall", "f1", "tp", "fp", "fn"}


# --------------------------------------------------------------------------- #
# Per species, and the aggregate
# --------------------------------------------------------------------------- #

def aligned_frame() -> pl.DataFrame:
    """Two species: one the model ranks perfectly, one it ranks backwards."""
    rows = []
    for i, (y, s) in enumerate(zip([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])):
        rows.append({"recording_id": "r1", "start_s": float(i), "end_s": i + 1.0,
                     "species_key": "Good bird", "y_true": y, "score": s})
    for i, (y, s) in enumerate(zip([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])):
        rows.append({"recording_id": "r1", "start_s": float(i), "end_s": i + 1.0,
                     "species_key": "Bad bird", "y_true": y, "score": s})
    return pl.DataFrame(rows)


def test_species_curves_are_computed_per_species():
    curves = species_curves(aligned_frame())
    assert set(curves) == {"Good bird", "Bad bird"}
    assert average_precision(curves["Good bird"]) == pytest.approx(1.0)
    assert average_precision(curves["Bad bird"]) < 0.6


def test_per_species_table_has_a_theta_per_rule():
    t = per_species_table(aligned_frame(), rules=(Rule("precision", 0.95, min_support=1),
                                                  Rule("fbeta", 1.0, min_support=1)))
    assert t["species_key"].to_list() == ["Good bird", "Bad bird"]   # sorted by AP
    assert {"ap", "n_positive", "prevalence",
            "theta_p0.95", "precision_p0.95", "recall_p0.95", "feasible_p0.95",
            "theta_f1"} <= set(t.columns)
    good = t.filter(pl.col("species_key") == "Good bird").row(0, named=True)
    assert good["theta_p0.95"] == pytest.approx(0.8)   # both positives, no false ones
    assert good["recall_p0.95"] == pytest.approx(1.0)


def test_perfect_ranking_is_ap_one_and_backwards_is_near_the_floor():
    t = per_species_table(aligned_frame())
    aps = dict(zip(t["species_key"], t["ap"]))
    assert aps["Good bird"] == pytest.approx(1.0)
    # Backwards ranking: the two positives sit last, so the curve only reaches
    # P=1/3 at R=0.5 and P=0.5 at R=1 -> (0.5)(1/3) + (0.5)(0.5).
    assert aps["Bad bird"] == pytest.approx(0.5 * (1 / 3) + 0.5 * 0.5)


def test_cmap_averages_unweighted_and_counts_what_it_dropped():
    t = pl.DataFrame({"species_key": ["a", "b", "c"], "n_positive": [10, 10, 1],
                      "ap": [0.8, 0.4, 0.9]})
    assert cmap(t)["cmap"] == pytest.approx(0.7)
    strict = cmap(t, min_positives=5)
    assert strict["cmap"] == pytest.approx(0.6)
    assert (strict["n_species"], strict["n_excluded"]) == (2, 1)


def test_cmap_ignores_species_with_no_positives():
    t = pl.DataFrame({"species_key": ["a", "b"], "n_positive": [10, 0],
                      "ap": [0.8, float("nan")]})
    got = cmap(t)
    assert got["cmap"] == pytest.approx(0.8) and got["n_excluded"] == 1


def test_cmap_on_an_empty_table():
    assert np.isnan(cmap(pl.DataFrame())["cmap"])


def test_summary_reports_both_averages_and_the_global_thetas():
    s = summary(aligned_frame(), rules=(Rule("fbeta", 1.0, min_support=1),))
    assert s["n_species"] == 2
    assert s["n_windows"] == 4          # 4 windows × 2 species = 8 pairs
    assert s["n_pairs"] == 8
    assert 0.0 <= s["cmap"] <= 1.0 and 0.0 <= s["micro_ap"] <= 1.0
    assert "theta_f1" in s and "precision_f1" in s
