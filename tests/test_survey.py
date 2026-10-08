"""Survey protocol (bex.survey): detections → firm / check / not found, scored."""
import math

import numpy as np
import polars as pl
import pytest

from bex import survey
from bex import thresholds as th
from bex.schemas import ANNOTATIONS_SCHEMA
from bex.survey import Condition, Constraints, SurveyRule
from bex.truth import ALIGNED_SCHEMA

ROBIN, WREN, JAY, OWL = ("Turdus migratorius", "Troglodytes aedon",
                         "Cyanocitta stelleri", "Strix aluco")


def det(*rows):
    """(recording, start_s, species, score, implausible)."""
    return pl.DataFrame(
        [{"recording_id": r, "start_s": s, "species_key": k, "score_raw": sc,
          "implausible": imp} for r, s, k, sc, imp in rows],
        schema={"recording_id": pl.Utf8, "start_s": pl.Float64, "species_key": pl.Utf8,
                "score_raw": pl.Float32, "implausible": pl.Boolean})


def ann(*rows):
    return pl.DataFrame(
        [{"recording_id": r, "start_s": 0.0, "end_s": 3.0, "low_hz": float("nan"),
          "high_hz": float("nan"), "species_key": k} for r, k in rows],
        schema=ANNOTATIONS_SCHEMA)


def flat(theta: float = 0.5, **per_species) -> th.Resolved:
    """A bar: every species at `theta` unless named."""
    table = pl.DataFrame({"species_key": list(per_species),
                          "theta": [float(v) for v in per_species.values()],
                          "source": ["rule"] * len(per_species),
                          "reason": [""] * len(per_species)},
                         schema={"species_key": pl.Utf8, "theta": pl.Float64,
                                 "source": pl.Utf8, "reason": pl.Utf8})
    return th.Resolved("m", table, theta, "fallback")


@pytest.fixture
def case():
    # r1: robin sings 5 times (firm at k=3); the wren once (check); a jay —
    # not annotated, in the label set — 3 times (a firm error); an owl outside
    # the label set once (can't judge). r2: robin only behind the filter.
    d = det(*[("r1", t, ROBIN, 0.9, False) for t in (0, 3, 6, 9, 12)],
            ("r1", 30, WREN, 0.8, False),
            *[("r1", t, JAY, 0.7, False) for t in (0, 100, 200)],
            ("r1", 50, OWL, 0.9, False),
            ("r1", 60, "Car", 0.99, False),            # not a species: never listed
            *[("r2", t, ROBIN, 0.9, True) for t in (0, 3, 6)])
    a = ann(("r1", ROBIN), ("r1", WREN), ("r2", ROBIN), ("r2", JAY), ("r1", "Sitta pygmaea"))
    label_set = {ROBIN, WREN, JAY, "Sitta pygmaea"}
    return survey.species_only(d), a, label_set


def test_species_only_drops_sound_events(case):
    d, _, _ = case
    assert "Car" not in d["species_key"].to_list()


def test_evidence_counts_windows_above_the_bar_and_within_a_span(case):
    d, _, _ = case
    ev = survey.evidence(d, {"p0.95": flat(0.5), "p0.99": flat(0.85)}, spans=(10,))
    row = lambda bar, r, k: ev.filter((pl.col("bar") == bar) & (pl.col("recording_id") == r)
                                      & (pl.col("species_key") == k)).row(0, named=True)
    robin = row("p0.95", "r1", ROBIN)
    assert robin["n_all"] == 5 and robin["n_shown"] == 5
    assert robin["w10_shown"] == 4          # starts 0, 3, 6, 9 fall within 10 s
    jay = row("p0.95", "r1", JAY)
    assert jay["n_all"] == 3 and jay["w10_all"] == 1   # spread across the hour
    hidden = row("p0.95", "r2", ROBIN)
    assert hidden["n_all"] == 3 and hidden["n_shown"] == 0 and hidden["w10_shown"] == 0
    # the stricter bar loses the jay (0.7) and the wren (0.8)
    strict = ev.filter(pl.col("bar") == "p0.99")
    assert set(strict["species_key"]) == {ROBIN, OWL}


def test_a_species_with_no_threshold_is_never_above_it():
    d = survey.species_only(det(("r1", 0, ROBIN, 0.9, False)))
    assert survey.evidence(d, {"x": flat(float("nan"))}).is_empty()


def test_outcomes_three_tiers_against_the_annotations(case):
    d, a, label_set = case
    ev = survey.evidence(d, {"p0.95": flat(0.5)})
    rule = SurveyRule(check=Condition(0.95, 1), firm=Condition(0.95, 3))
    out = survey.outcomes(survey.tiers(ev, rule), a, ["r1", "r2"], label_set)
    got = {(r, k): o for r, k, o in out.select("recording_id", "species_key",
                                                "outcome").iter_rows()}
    assert got == {
        ("r1", ROBIN): "firm, right",
        ("r1", WREN): "check, real",
        ("r1", JAY): "firm, wrong",
        ("r1", OWL): "check, can't judge",
        ("r1", "Sitta pygmaea"): "missed",
        ("r2", ROBIN): "filter hid a real bird",
        ("r2", JAY): "missed",
    }
    s = survey.score(survey.per_recording(out, ["r1", "r2"]), {"r1": 1.0, "r2": 1.0})
    assert s["firm_precision"] == pytest.approx(1 / 2)           # robin right, jay wrong
    assert s["recall_reviewed"] == pytest.approx(2 / 5)          # robin + wren of 5
    assert s["firm_recall"] == pytest.approx(1 / 5)
    assert s["checks_per_recording"] == pytest.approx(1.0)       # wren, owl over 2 recs
    assert s["check_yield"] == pytest.approx(1.0)                # the owl isn't judged
    worse = survey.score(survey.per_recording(out, ["r1", "r2"]), {"r1": 1.0, "r2": 1.0},
                         unjudged_wrong=True)
    assert worse["check_yield"] == pytest.approx(1 / 2)


def test_a_span_condition_separates_a_burst_from_scattered_calls(case):
    d, a, label_set = case
    ev = survey.evidence(d, {"p0.95": flat(0.5)}, spans=(10,))
    rule = SurveyRule(check=Condition(0.95, 1), firm=Condition(0.95, 3, within_s=10))
    t = dict(survey.tiers(ev, rule, ["r1"]).select("species_key", "tier").iter_rows())
    assert t[ROBIN] == "firm" and t[JAY] == "check"    # 3 calls, but 100 s apart


def test_one_tier_rule_matches_compare_species_lists(case):
    """At firm = check the survey is Compare's species list (V1 C2), with
    out-of-label-set listings counted as wrong as C2 does."""
    from bex.benchmark import species_list_summary, species_lists
    d, a, label_set = case
    bar = flat(0.5)
    judged = th.attach(d.rename({}), bar).with_columns(
        above=pl.col("score_raw") >= 0.5)
    for k in (1, 2, 3):
        c2 = species_list_summary(species_lists(judged, a, k))
        ev = survey.evidence(d, {"p0.95": bar})
        one = SurveyRule(Condition(0.95, k), Condition(0.95, k))
        out = survey.outcomes(survey.tiers(ev, one), a, ["r1", "r2"], label_set)
        s = survey.score(survey.per_recording(out, ["r1", "r2"]), {"r1": 1, "r2": 1},
                         unjudged_wrong=True)
        assert s["firm_precision"] == pytest.approx(c2["precision"], nan_ok=True)
        assert s["firm_recall"] == pytest.approx(c2["recall"])
        assert s["counts"]["filter hid a real bird"] == c2["counts"]["filter hid a real bird"]


def test_sweep_cells_agree_with_scoring_the_rule_directly(case):
    d, a, label_set = case
    bars = {"p0.9": flat(0.5), "p0.95": flat(0.75)}
    ev = survey.evidence(d, bars)
    rule = SurveyRule(check=Condition(0.95, 1), firm=Condition(0.95, 3))
    hours = {"r1": 1.0, "r2": 1.0}
    sw = survey.sweep(ev, rule, a, ["r1", "r2"], hours, label_set,
                      floors=(0.9, 0.95), ks=(1, 2, 3, 5), check_floors=(0.9, 0.95))
    # the check bar is never stricter than the firm one: 3 floor pairs × 4 ks
    assert len(sw) == 12 and (sw["check_floor"] <= sw["floor"]).all()
    for row in sw.iter_rows(named=True):
        r = survey.swept(rule, row["floor"], row["k"], row["check_floor"])
        assert r.check.floor == row["check_floor"] and r.firm.floor == row["floor"]
        direct = survey.assess(ev, r, a, ["r1", "r2"], hours, label_set, n_boot=10)["score"]
        for key in ("firm_precision", "recall_reviewed", "checks_per_recording"):
            assert row[key] == pytest.approx(direct[key], nan_ok=True), (row, key)


def test_a_looser_check_bar_finds_birds_the_firm_bar_misses(case):
    """The wren scores 0.8: under a 0.85 bar it is missed, but a check tier at a
    0.5 bar catches it while the firm tier keeps its strict bar."""
    d, a, label_set = case
    ev = survey.evidence(d, {"p0.5": flat(0.5), "p0.95": flat(0.85)})
    hours = {"r1": 1.0, "r2": 1.0}
    one = survey.assess(ev, SurveyRule(Condition(0.95, 1), Condition(0.95, 3)), a,
                        ["r1"], hours, label_set, n_boot=0)["score"]
    split = survey.assess(ev, SurveyRule(Condition(0.5, 1), Condition(0.95, 3)), a,
                          ["r1"], hours, label_set, n_boot=0)["score"]
    assert split["recall_reviewed"] > one["recall_reviewed"]
    assert split["firm_precision"] == one["firm_precision"]


def test_best_respects_the_constraints():
    sw = pl.DataFrame({
        "floor": [0.9, 0.9, 0.95], "k": [1, 3, 1],
        "firm_precision": [0.80, 0.96, 0.99],
        "recall_reviewed": [0.9, 0.8, 0.6],
        "checks_per_recording": [0.0, 4.0, 1.0]})
    assert survey.best(sw, Constraints(0.95))["k"] == 3
    assert survey.best(sw, Constraints(0.95, 2.0))["floor"] == 0.95
    assert survey.best(sw, Constraints(0.999)) is None
    shared = survey.best_shared({"a": sw, "b": sw.with_columns(
        firm_precision=pl.Series([0.99, 0.90, 0.99]))}, Constraints(0.95))
    assert (shared["floor"], shared["k"]) == (0.95, 1)


def test_split_is_reproducible_and_complete():
    ids = [f"r{i}" for i in range(33)]
    tune, test = survey.split(ids, seed=7)
    assert survey.split(ids, seed=7) == (tune, test)
    assert sorted(tune + test) == sorted(ids) and not set(tune) & set(test)
    assert len(test) == 16 or len(test) == 17
    assert survey.split(ids, seed=8) != (tune, test)


def test_bootstrap_brackets_the_estimate(case):
    d, a, label_set = case
    ev = survey.evidence(d, {"p0.95": flat(0.5)})
    res = survey.assess(ev, SurveyRule(), a, ["r1", "r2"], {"r1": 1, "r2": 1},
                        label_set, n_boot=200)
    lo, hi = res["intervals"]["recall_reviewed"]
    assert lo <= res["score"]["recall_reviewed"] <= hi


def aligned_two_recordings():
    """r1 and r2, each with robin windows that score well when real."""
    rows = []
    for r in ("r1", "r2", "r3"):
        for i in range(40):
            y = int(i < 15)
            rows.append({"recording_id": r, "start_s": 3.0 * i, "end_s": 3.0 * i + 3,
                         "species_key": ROBIN, "y_true": y,
                         "score": (0.6 + 0.01 * i) if y else 0.02 * i})
    return pl.DataFrame(rows, schema=ALIGNED_SCHEMA)


def test_curve_source_refits_on_a_subset_exactly():
    from bex import metrics
    a = aligned_two_recordings()
    src = survey.CurveSource(a)
    curves, micro = src.fit(["r1", "r3"])
    sub = a.filter(pl.col("recording_id").is_in(["r1", "r3"]))
    want = metrics.pr_curve(sub["y_true"].to_numpy(), sub["score"].to_numpy())
    assert np.array_equal(curves[ROBIN].theta, want.theta)
    assert curves[ROBIN].n_pos == 30 and micro.n_pos == 30


def test_full_assessment_leaves_each_recording_out():
    a = aligned_two_recordings()
    d = survey.species_only(det(*[(r, 3.0 * i, ROBIN, (0.6 + 0.01 * i) if i < 15 else 0.02 * i,
                                   False) for r in ("r1", "r2", "r3") for i in range(40)]))
    annotations = ann(("r1", ROBIN), ("r2", ROBIN))
    m = survey.ModelInputs("m", "m v1 sigmoid", d, survey.CurveSource(a))
    spec = th.RuleSpec(min_support=1, min_labelled=1)
    seen = []
    res = survey.full_assessment([m], SurveyRule(), spec, annotations, ["r1", "r2", "r3"],
                                 {"r1": 1, "r2": 1, "r3": 1}, Constraints(0.5),
                                 n_boot=50, progress=lambda i, n, r: seen.append(r))
    assert seen == ["r1", "r2", "r3"] and res["n_rounds"] == 3
    own = res["models"]["m"]["own"]
    assert own["per_recording"]["recording_id"].to_list() == ["r1", "r2", "r3"]
    assert own["score"]["recall_reviewed"] == pytest.approx(1.0)
    stab = res["stability"]["m"]
    assert stab["rounds"] == 3 and 0 <= stab["same"] <= 3


def test_best_follows_the_objective_and_its_limits():
    sw = pl.DataFrame({
        "floor": [0.9, 0.9, 0.95], "k": [1, 3, 1],
        "firm_precision": [0.96, 0.99, 0.97],
        "recall_reviewed": [0.9, 0.8, 0.6],
        "firm_recall": [0.3, 0.2, 0.5],
        "checks_per_recording": [9.0, 4.0, 1.0]})
    pick = lambda **kw: (lambda r: (r["floor"], r["k"]))(survey.best(sw, Constraints(0.95, **kw)))
    assert pick() == (0.9, 1)                                        # most found after checking
    assert pick(objective="firm_recall") == (0.95, 1)
    assert pick(objective="checks_per_recording") == (0.95, 1)
    assert pick(objective="checks_per_recording", min_recall_reviewed=0.7) == (0.9, 3)
    assert pick(objective="firm_precision") == (0.9, 3)
    with pytest.raises(ValueError):
        Constraints(objective="vibes")


def test_paired_difference_on_the_same_recordings():
    """Model a always has one more firm-right call than b on every recording:
    the paired interval for the difference sits above zero."""
    import numpy as np
    from bex import survey as S
    rng = np.random.default_rng(3)
    rows_a, rows_b = [], []
    for i in range(10):
        base = {o: 0 for o in S.OUTCOMES}
        base["firm, right"], base["missed"] = int(rng.integers(2, 9)), 3
        rows_a.append({"recording_id": f"r{i}", **base,
                       "firm, right": base["firm, right"] + 1, "missed": 2})
        rows_b.append({"recording_id": f"r{i}", **base})
    hours = {f"r{i}": 1.0 for i in range(10)}
    a = {"per_recording": pl.DataFrame(rows_a), "hours_of": hours, "unjudged_wrong": False}
    b = {"per_recording": pl.DataFrame(rows_b), "hours_of": hours, "unjudged_wrong": False}
    d, lo, hi = S.paired(a, b)["firm_recall"]
    assert d > 0 and lo > 0 and hi >= d
