"""Threshold resolution, application and named sets (V1 S2 / H1)."""
import math

import numpy as np
import polars as pl
import pytest

from bex import metrics
from bex.thresholds import (
    LANE_BASELINE,
    RuleSpec,
    ThresholdSet,
    attach,
    build_set,
    list_sets,
    load_set,
    point_at,
    resolve,
    save_set,
    strength,
)

MODEL = "birdnet 2.4 sigmoid"


def curve(pos_scores, neg_scores):
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    return metrics.pr_curve(y, np.r_[pos_scores, neg_scores])


@pytest.fixture
def curves():
    # "easy": perfectly separable, max F1 at the lowest positive (0.6).
    # "rare": two positives only — the precision floor cannot be supported.
    return {
        "easy": curve([0.9, 0.8, 0.7, 0.6] * 5, [0.3, 0.2, 0.1] * 5),
        "rare": curve([0.5, 0.4], [0.45, 0.3, 0.2]),
    }


def micro(curves):
    return curve([0.9, 0.8, 0.7, 0.6] * 5 + [0.5, 0.4], [0.3, 0.2, 0.1] * 5 + [0.45])


def test_rule_fits_each_species(curves):
    r = resolve(MODEL, RuleSpec("max F1", min_support=1), curves, micro(curves))
    assert r.theta_of("easy") == pytest.approx(0.6)
    row = r.table.filter(pl.col("species_key") == "easy").row(0, named=True)
    assert row["source"] == "rule"


def test_infeasible_species_falls_back_and_says_why(curves):
    spec = RuleSpec("precision floor", precision_floor=0.95, min_support=10)
    r = resolve(MODEL, spec, curves, micro(curves))
    row = r.table.filter(pl.col("species_key") == "rare").row(0, named=True)
    assert row["source"] == "fallback"
    assert "→" in row["reason"]              # its own failure, then the fallback
    assert row["theta"] == pytest.approx(r.default_theta)


def test_unlisted_species_get_the_default(curves):
    r = resolve(MODEL, RuleSpec("max F1", min_support=1), curves, micro(curves))
    assert r.theta_of("never annotated") == r.default_theta
    assert not math.isnan(r.default_theta)


def test_fixed_rule_and_fixed_fallback(curves):
    spec = RuleSpec("fixed", fixed=((MODEL, 0.33),))
    r = resolve(MODEL, spec, curves, micro(curves))
    assert set(r.table["theta"].to_list()) == {0.33}
    assert r.default_theta == 0.33


def test_override_wins_over_everything(curves):
    r = resolve(MODEL, RuleSpec("max F1", min_support=1), curves, micro(curves),
                overrides={"easy": 0.77, "not in curves": 0.5})
    assert r.theta_of("easy") == 0.77
    assert r.theta_of("not in curves") == 0.5
    assert set(r.table.filter(pl.col("source") == "override")["species_key"]) == \
        {"easy", "not in curves"}


def test_snapshot_thresholds_unlabelled_data():
    r = resolve(MODEL, RuleSpec("max F1"), snapshot={"a": 0.4}, snapshot_default=0.2)
    assert r.theta_of("a") == 0.4 and r.theta_of("b") == 0.2
    assert r.table["source"][0] == "set"


def test_no_labels_and_no_set_means_no_threshold():
    r = resolve(MODEL, RuleSpec("max F1"))
    assert math.isnan(r.default_theta) and r.default_source == "none"


def det(*rows):
    return pl.DataFrame([{"recording_id": "r1", "start_s": 0.0, "end_s": 3.0,
                          "species_key": k, "score_raw": s} for k, s in rows],
                        schema={"recording_id": pl.Utf8, "start_s": pl.Float64,
                                "end_s": pl.Float64, "species_key": pl.Utf8,
                                "score_raw": pl.Float32})


def test_attach_uses_each_species_own_theta(curves):
    r = resolve(MODEL, RuleSpec("fixed", fixed=((MODEL, 0.5),)),
                overrides={"b": 0.9})
    out = attach(det(("a", 0.6), ("b", 0.6), ("c", 0.4)), r)
    assert out["above"].to_list() == [True, False, False]


def test_attach_never_passes_a_species_with_no_threshold():
    r = resolve(MODEL, RuleSpec("max F1"))          # nothing to fit on
    assert not attach(det(("a", 0.99)), r)["above"].any()


def test_strength_ranks_margin_over_own_threshold():
    r = resolve(MODEL, RuleSpec("fixed", fixed=((MODEL, 0.5),)),
                overrides={"low": 0.01})
    out = strength(attach(det(("a", 0.55), ("a", 0.99), ("low", 0.02), ("a", 0.1)), r))
    s = out["strength"].to_list()
    assert s[3] is None                             # below threshold: no mark
    assert s[1] == pytest.approx(1.0)               # strongest at the top
    assert min(v for v in s if v is not None) == pytest.approx(LANE_BASELINE)
    # 0.02 over a 0.01 threshold clears it by more (in log-odds) than 0.55 over 0.5
    assert s[2] > s[0]


def test_point_at_reads_any_theta(curves):
    c = curves["easy"]
    assert point_at(c, 0.6)["recall"] == pytest.approx(1.0)
    assert point_at(c, 0.65)["recall"] == pytest.approx(0.75)
    assert point_at(c, 0.95)["recall"] == 0.0


def test_sets_round_trip_and_report_stale_models(tmp_path, curves):
    spec = RuleSpec("max F1", min_support=1)
    r = resolve(MODEL, spec, curves, micro(curves), overrides={"easy": 0.7})
    ts = build_set("Mara — dawn survey", spec, [r], {(MODEL, "easy"): 0.7},
                   author="Mara", dataset="sne")
    save_set(tmp_path, ts)
    assert list_sets(tmp_path) == ["Mara — dawn survey"]

    back = load_set(tmp_path, "Mara — dawn survey")
    assert back.spec == spec
    assert back.overrides_for(MODEL) == {"easy": 0.7}
    snap, default = back.snapshot_for(MODEL)
    assert snap["easy"] == 0.7 and default == pytest.approx(r.default_theta)
    assert back.created_utc and back.updated_utc
    # A model that is no longer in the store is reported, not applied.
    assert back.stale({MODEL}) == {}
    assert back.stale({"birdnet 2.5 sigmoid"}) == {MODEL: 1 + len(r.table)}


def test_unreadable_files_are_not_offered(tmp_path):
    (tmp_path / "junk.json").write_text("{}")
    (tmp_path / "broken.json").write_text("not json")
    assert list_sets(tmp_path) == []
    with pytest.raises(ValueError):
        ThresholdSet.from_json("{}")


def test_too_few_labelled_windows_takes_the_fallback(curves):
    """Max F1 with a couple of labelled windows reaches as deep as it must to
    catch them, whatever that costs — so below `min_labelled` the rule is not
    fitted at all and the species falls back, saying why."""
    fitted = resolve(MODEL, RuleSpec("max F1", min_support=1, min_labelled=1),
                     curves, micro(curves))
    guarded = resolve(MODEL, RuleSpec("max F1", min_support=1, min_labelled=10),
                      curves, micro(curves))
    rare = lambda r: r.table.filter(pl.col("species_key") == "rare").row(0, named=True)
    assert rare(fitted)["source"] == "rule"
    assert rare(guarded)["source"] == "fallback"
    assert "only 2 labelled window(s)" in rare(guarded)["reason"]
    # a well-labelled species is untouched by the guard
    assert guarded.theta_of("easy") == fitted.theta_of("easy")


def test_older_sets_without_the_guard_load_with_the_default():
    spec = RuleSpec.from_dict({"kind": "max F1"})
    assert spec.min_labelled == 10


def test_every_rule_describes_itself():
    for kind in ("precision floor", "max F1", "max F2", "fixed"):
        text = RuleSpec(kind, precision_floor=0.9, min_labelled=12).describe()
        assert text.endswith(".") and len(text) > 40
    assert "90%" in RuleSpec("precision floor", precision_floor=0.9).describe()
    assert "fewer than 12" in RuleSpec("max F1", min_labelled=12).describe()
