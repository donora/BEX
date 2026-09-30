"""§4 metrics on hand-built fixtures whose answers are derivable by hand."""
import polars as pl
import pytest

from bex.profiles import Profile
from bex.schemas import ANNOTATIONS_SCHEMA, DETECTIONS_SCHEMA
from bex.stats import (
    MuddinessConfig,
    excluded_present_species,
    filter_errors,
    league_row,
    mark_implausible,
    muddiness_spread,
    muddy_window_rate,
    per_species_muddiness,
    shadowed_detections,
    shadowed_pairs,
    suppression_events,
    window_impostor_mass,
)

ROBIN = "Turdus migratorius"        # plausible, present
BLACKBIRD = "Turdus merula"         # implausible congener of the robin
CURRAWONG = "Strepera graculina"    # implausible, unrelated
THRUSH = "Catharus guttatus"        # plausible, present


def det_rows(*rows) -> pl.DataFrame:
    """(start_s, species, score, suppressed) -> a detections frame."""
    return pl.DataFrame(
        [{"recording_id": "r1", "run_id": "run", "start_s": s, "end_s": s + 3.0,
          "species_key": k, "score_raw": sc, "occ_score": occ, "suppressed": sup,
          "rank_in_window": 1}
         for (s, k, sc, occ, sup) in rows],
        schema=DETECTIONS_SCHEMA,
    )


def ann_rows(*rows) -> pl.DataFrame:
    return pl.DataFrame(
        [{"recording_id": "r1", "start_s": s, "end_s": e, "low_hz": 1000.0,
          "high_hz": 5000.0, "species_key": k} for (s, e, k) in rows],
        schema=ANNOTATIONS_SCHEMA,
    )


@pytest.fixture()
def profile() -> Profile:
    # Only the American species are plausible here; anything else is tier 3.
    return Profile(name="p", display_name="P", tiers={ROBIN: 1, THRUSH: 1})


@pytest.fixture()
def det(profile) -> pl.DataFrame:
    """Two windows, hand-checkable.

    window 0: robin 0.90, blackbird 0.80 (hidden), currawong 0.20 (hidden)
    window 3: thrush 0.95, robin 0.30
    """
    raw = det_rows(
        (0.0, ROBIN, 0.90, 0.65, False),
        (0.0, BLACKBIRD, 0.80, 0.001, True),
        (0.0, CURRAWONG, 0.20, 0.000, True),
        (3.0, THRUSH, 0.95, 0.50, False),
        (3.0, ROBIN, 0.30, 0.65, False),
    )
    return mark_implausible(raw, profile, mode="geofilter")


def test_mark_implausible_modes(profile):
    raw = det_rows((0.0, BLACKBIRD, 0.8, 0.001, True),
                   (0.0, CURRAWONG, 0.2, 0.0, False))  # profile says no, filter says yes
    by_filter = mark_implausible(raw, profile, mode="geofilter")
    assert by_filter["implausible"].to_list() == [True, False]
    by_profile = mark_implausible(raw, profile, mode="profile")
    assert by_profile["implausible"].to_list() == [True, True]
    either = mark_implausible(raw, profile, mode="either")
    assert either["implausible"].to_list() == [True, True]
    with pytest.raises(ValueError, match="unknown mode"):
        mark_implausible(raw, profile, mode="vibes")
    with pytest.raises(ValueError, match="needs a profile"):
        mark_implausible(raw, None, mode="profile")


def test_suppression_events_and_muddy_rate(det):
    events = suppression_events(det, theta=0.10)
    assert events["species_key"].to_list() == [BLACKBIRD, CURRAWONG]  # score order
    # Both windows hold a detection; only window 0 holds an implausible one.
    rate = muddy_window_rate(det, theta=0.10)
    assert rate == {"windows": 2, "muddy_windows": 1, "muddy_window_rate": 0.5}
    # Raise theta above the currawong and blackbird stays: still 1 of 2.
    assert muddy_window_rate(det, theta=0.85)["muddy_windows"] == 0


def test_window_impostor_mass(det):
    mass = window_impostor_mass(det, theta=0.10).sort("start_s")
    # window 0: (0.80 + 0.20) / (0.90 + 0.80 + 0.20) = 1.0 / 1.9
    assert mass["impostor_mass"][0] == pytest.approx(1.0 / 1.9, abs=1e-6)
    # window 3: nothing implausible
    assert mass["impostor_mass"][1] == pytest.approx(0.0)


def test_impostor_mass_counts_only_detections(det):
    # Raising theta above the currawong (0.20) must drop it from BOTH sums,
    # not just the numerator — the quantity is a share of confident mass.
    mass = window_impostor_mass(det, theta=0.50).sort("start_s")
    assert mass["impostor_mass"][0] == pytest.approx(0.80 / (0.90 + 0.80), abs=1e-6)


def test_per_species_muddiness(det):
    cfg = MuddinessConfig(theta=0.10, delta=0.15)
    per = per_species_muddiness(det, cfg).sort("species_key")
    by = {r["species_key"]: r for r in per.to_dicts()}

    # Robin at 0.90 in window 0: companions are species scoring >= 0.75 —
    # the blackbird at 0.80 only (the currawong at 0.20 is far below).
    # Robin at 0.30 in window 3: companions >= 0.15 — none (thrush 0.95 is
    # above, so it counts!). Careful: "within delta" means >= score - delta,
    # which includes anything scoring higher.
    assert by[ROBIN]["windows"] == 2
    assert by[ROBIN]["companions"] == pytest.approx((1 + 1) / 2)  # blackbird; thrush
    assert by[ROBIN]["implausible_companions"] == pytest.approx((1 + 0) / 2)

    # Thrush at 0.95 alone above 0.80 in window 3 -> no companions.
    assert by[THRUSH]["companions"] == pytest.approx(0.0)
    assert by[THRUSH]["impostor_mass"] == pytest.approx(0.0)


def test_muddiness_spread_needs_enough_windows():
    per = pl.DataFrame({
        "species_key": ["a", "b", "c"],
        "windows": [10, 10, 2],          # 'c' is too rare to characterise
        "companions": [1.0, 3.0, 99.0],
    })
    spread = muddiness_spread(per, min_windows=5)
    assert spread["species"] == 2
    assert spread["median"] == pytest.approx(2.0)
    assert muddiness_spread(per.head(0))["species"] == 0


def test_shadowed_detections_flags_congeners(det):
    cfg = MuddinessConfig(theta=0.10, delta=0.15)
    shadow = shadowed_detections(det, cfg)
    # Robin 0.90 shadowed by blackbird 0.80 (gap 0.10 <= delta). The currawong
    # at 0.20 is too far below to shadow anything.
    assert len(shadow) == 1
    row = shadow.to_dicts()[0]
    assert (row["species_key"], row["rival_key"]) == (ROBIN, BLACKBIRD)
    assert row["gap"] == pytest.approx(0.10, abs=1e-6)
    assert row["congeneric"] is True      # both Turdus — the blackbird case

    pairs = shadowed_pairs(shadow)
    assert pairs["windows"][0] == 1

    # A tighter delta breaks the pairing entirely.
    assert shadowed_detections(det, MuddinessConfig(theta=0.1, delta=0.05)).is_empty()


def test_filter_errors(det):
    # Truth: the robin sings in window 0; the blackbird is NOT present.
    ann = ann_rows((0.0, 3.0, ROBIN), (3.0, 6.0, THRUSH))
    errors = filter_errors(det, ann, MuddinessConfig(theta=0.10))
    # robin w0 and thrush w3 agree with truth. The robin detection in w3 does
    # NOT: its box ends at exactly 3.0 where that window begins, so they share
    # no audio — the same zero-overlap boundary rule the window grid uses.
    assert errors["true_positive_detections"] == 2
    assert errors["false_suppressions"] == 0         # nothing true was hidden
    # Blackbird and currawong appear nowhere in truth, but both were suppressed,
    # so the filter caught them: no false admissions.
    assert errors["false_admissions"] == 0

    # Now let the blackbird through the filter and it becomes a false admission.
    leaky = det.with_columns(
        implausible=pl.when(pl.col("species_key") == BLACKBIRD)
        .then(False).otherwise(pl.col("implausible"))
    )
    assert filter_errors(leaky, ann, MuddinessConfig(theta=0.10))["false_admissions"] == 1


def test_false_suppression_when_filter_hides_a_present_species():
    # The thrush IS annotated, and the filter suppressed it: the filter is wrong.
    det = mark_implausible(
        det_rows((0.0, THRUSH, 0.9, 0.01, True)), None, mode="geofilter")
    ann = ann_rows((0.0, 3.0, THRUSH))
    errors = filter_errors(det, ann, MuddinessConfig(theta=0.10))
    assert errors["false_suppressions"] == 1
    assert errors["false_suppression_rate"] == pytest.approx(1.0)
    assert errors["false_suppressed_species"] == [THRUSH]


def test_excluded_present_species(det):
    ann = ann_rows((0.0, 3.0, ROBIN), (3.0, 6.0, THRUSH))
    # With a 0.6 cutoff the thrush (occ 0.50) is structurally unreportable.
    excluded = excluded_present_species(det, ann, occ_threshold=0.60)
    assert excluded["species_key"].to_list() == [THRUSH]
    # With the real 0.03 cutoff, both present species clear it.
    assert excluded_present_species(det, ann, occ_threshold=0.03).is_empty()


def test_league_row(det):
    ann = ann_rows((0.0, 3.0, ROBIN))
    row = league_row(det, "birdnet", MuddinessConfig(theta=0.10), ann)
    assert row["arm"] == "birdnet"
    assert row["detection_windows"] == 2
    assert row["muddy_window_rate"] == pytest.approx(0.5)
    assert row["suppression_events"] == 2
    assert row["congeneric_shadowed"] == 1
    assert "false_suppressions" in row


def test_empty_inputs_do_not_crash(profile):
    empty = mark_implausible(det_rows(), profile, mode="geofilter")
    assert muddy_window_rate(empty, 0.1)["windows"] == 0
    assert window_impostor_mass(empty, 0.1).is_empty()
    assert per_species_muddiness(empty).is_empty()
    assert shadowed_detections(empty).is_empty()
    assert shadowed_pairs(shadowed_detections(empty)).is_empty()


def test_theta_for_detection_count(det):
    from bex.stats import theta_for_detection_count
    # Scores present: 0.95, 0.90, 0.80, 0.30, 0.20.
    assert theta_for_detection_count(det, 1) == pytest.approx(0.95, abs=1e-6)
    assert theta_for_detection_count(det, 3) == pytest.approx(0.80, abs=1e-6)
    # Asking for more detections than exist returns the lowest score, not an error.
    assert theta_for_detection_count(det, 99) == pytest.approx(0.20, abs=1e-6)


def test_matched_league_puts_arms_on_equal_footing(det, profile):
    """Two arms with wildly different score scales must still be comparable.

    `scaled` is `det` with every score divided by 10 — a stand-in for Perch's
    softmax, whose numbers are far smaller than BirdNET's sigmoids. At a shared
    theta the second arm would look like it detects nothing; matched on
    detection count, the two are directly comparable.
    """
    from bex.stats import matched_league

    scaled = mark_implausible(
        det.with_columns(score_raw=pl.col("score_raw") / 10),
        profile, mode="geofilter",
    )
    table = matched_league({"a": det, "b": scaled}, target_detections=3)
    assert table["detections"].to_list() == [3, 3]      # equal footing
    assert table["theta"][0] > table["theta"][1]        # different thresholds
    # Same ranking underneath, so the muddiness verdict agrees.
    assert table["muddy_window_rate"][0] == pytest.approx(table["muddy_window_rate"][1])


def test_filter_errors_flags_circular_judge():
    """A profile derived from the annotations cannot disagree with them.

    `us-ca-sierra` was generated from SNE's own species list, so in profile
    mode every annotated species is plausible and every plausible species is
    annotated: both filter-error rates are structurally zero. Reporting that as
    "0% false suppression" would be a circular claim, so the result says so.
    """
    ann = ann_rows((0.0, 3.0, ROBIN))
    circular = Profile(name="c", display_name="C", tiers={ROBIN: 1})
    det = mark_implausible(
        det_rows((0.0, ROBIN, 0.9, 0.5, False)), circular, mode="profile")
    errors = filter_errors(det, ann, MuddinessConfig(theta=0.1))
    assert errors["false_suppressions"] == 0
    assert errors["degenerate_false_suppression"] is True
    assert errors["degenerate_false_admission"] is True

    # A judge that CAN disagree: the thrush is annotated but implausible here.
    honest = Profile(name="h", display_name="H", tiers={ROBIN: 1})
    det2 = mark_implausible(
        det_rows((0.0, ROBIN, 0.9, 0.5, False), (0.0, THRUSH, 0.8, 0.5, False)),
        honest, mode="profile")
    ann2 = ann_rows((0.0, 3.0, ROBIN), (0.0, 3.0, THRUSH))
    errors2 = filter_errors(det2, ann2, MuddinessConfig(theta=0.1))
    assert errors2["degenerate_false_suppression"] is False
    assert errors2["false_suppressions"] == 1
