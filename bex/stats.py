"""The geofilter forensics — PLAN.md §4, as pure functions over the store.

Everything the app's forensics tab shows is computed here, so every number is
reproducible headlessly (`scripts/forensics.py`) and checkable against the raw
score matrices (`scripts/crosscheck.py`). Nothing in this module reads a file
or knows about Streamlit.

Two independent notions of "this bird should not be here" run through all of
it, and keeping them separate is the point:

- **geofilter** — the model's own verdict, replayed rather than applied: the
  `suppressed` column set by its occurrence meta-model. Only exists for models
  that ship a location prior.
- **profile** — our plausibility list for the site (`profiles/`), which applies
  identically to every model and so makes the comparison symmetric. A species
  absent from the profile is tier 3, implausible.

`mode="either"` is deliberately available but is not the default: counting a
detection as muddy because *either* judge objects mixes two different claims
into one number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .profiles import Profile
from .taxonomy import genus

WINDOW_KEYS = ["recording_id", "start_s"]


@dataclass(frozen=True)
class MuddinessConfig:
    """Knobs shared by every §4 metric, so a table can name its own settings.

    - `theta`: a score at or above this counts as a detection.
    - `delta`: the "scoring nearly as high" band. A companion/shadow must be
      within this of the focal score — 0.15 means a rival at 0.75 shadows a
      detection at 0.90.
    - `min_tier`: profile tier at or above which a species is implausible.
    """

    theta: float = 0.10
    delta: float = 0.15
    min_tier: int = 3


def mark_implausible(
    det: pl.DataFrame,
    profile: Profile | None = None,
    mode: str = "geofilter",
    min_tier: int = 3,
) -> pl.DataFrame:
    """Add an `implausible` column under the chosen judge (see module docstring)."""
    if mode == "geofilter":
        return det.with_columns(implausible=pl.col("suppressed"))
    if profile is None:
        raise ValueError(f"mode={mode!r} needs a profile")
    keys = det["species_key"].to_list()
    by_profile = pl.Series(
        [profile.tier_of(k) >= min_tier for k in keys], dtype=pl.Boolean
    )
    if mode == "profile":
        return det.with_columns(implausible=by_profile)
    if mode == "either":
        return det.with_columns(implausible=by_profile | pl.col("suppressed"))
    raise ValueError(f"unknown mode {mode!r} (geofilter | profile | either)")


def suppression_events(det: pl.DataFrame, theta: float) -> pl.DataFrame:
    """Every (window, species) the judge calls implausible yet the model scored
    at or above theta — the atomic unit of §4, one browsable row each."""
    return (
        det.filter(pl.col("implausible") & (pl.col("score_raw") >= theta))
        .sort("score_raw", descending=True)
    )


def detection_windows(det: pl.DataFrame, theta: float) -> pl.DataFrame:
    """Windows holding at least one detection — the denominator for rates."""
    return (
        det.filter(pl.col("score_raw") >= theta)
        .select(WINDOW_KEYS)
        .unique()
    )


def muddy_window_rate(det: pl.DataFrame, theta: float) -> dict:
    """Fraction of detection-positive windows carrying >=1 implausible detection.

    The headline per-model number: how often is a confident answer accompanied
    by a confident impossibility?
    """
    positive = detection_windows(det, theta)
    if positive.is_empty():
        return {"windows": 0, "muddy_windows": 0, "muddy_window_rate": float("nan")}
    muddy = (
        det.filter(pl.col("implausible") & (pl.col("score_raw") >= theta))
        .select(WINDOW_KEYS)
        .unique()
    )
    n_muddy = len(muddy.join(positive, on=WINDOW_KEYS, how="inner"))
    return {
        "windows": len(positive),
        "muddy_windows": n_muddy,
        "muddy_window_rate": n_muddy / len(positive),
    }


def window_impostor_mass(det: pl.DataFrame, theta: float) -> pl.DataFrame:
    """Per detection-positive window: what share of the *confident* score mass
    sits on implausible species.

    The continuous counterpart of the muddy-window rate: it weights a 0.9
    impossibility far above a 0.12 one.

    Both sums run over species scoring at or above theta, not over every stored
    row. That matters for more than tidiness — the detections index keeps every
    species scoring >= 0.01, so for any theta >= 0.01 this quantity is computable
    *exactly* from the index, with no dependence on the storage policy's top-k
    cut. Summing the whole class vector instead would put thousands of ~0.001
    scores in the denominator and make the number an artefact of how many
    classes the model happens to have. `scripts/crosscheck.py` verifies the
    exactness against the raw score matrices.
    """
    positive = detection_windows(det, theta)
    if positive.is_empty():
        return pl.DataFrame(schema={"recording_id": pl.Utf8, "start_s": pl.Float64,
                                    "impostor_mass": pl.Float64})
    return (
        det.filter(pl.col("score_raw") >= theta)
        .group_by(WINDOW_KEYS)
        .agg(
            total=pl.col("score_raw").cast(pl.Float64).sum(),
            impostor=pl.col("score_raw").cast(pl.Float64)
            .filter(pl.col("implausible")).sum(),
        )
        .with_columns(
            impostor_mass=pl.when(pl.col("total") > 0)
            .then(pl.col("impostor").fill_null(0.0) / pl.col("total"))
            .otherwise(0.0)
        )
        .select(*WINDOW_KEYS, "impostor_mass")
    )


def _window_groups(det: pl.DataFrame) -> dict[tuple, np.ndarray]:
    """(recording_id, start_s) -> that window's scores, descending."""
    grouped = det.group_by(WINDOW_KEYS).agg(
        pl.col("score_raw").cast(pl.Float64).sort(descending=True).alias("scores")
    )
    return {
        (r, s): np.asarray(scores)
        for r, s, scores in grouped.select(*WINDOW_KEYS, "scores").iter_rows()
    }


def _count_within_delta(
    focal: pl.DataFrame, groups: dict[tuple, np.ndarray], delta: float, floor: float
) -> np.ndarray:
    """For each focal row, how many scores in its window's group are at or above
    `max(score - delta, floor)`.

    The `floor` is load-bearing, not a tidy-up. Without it the count is
    confounded with the focal score: at delta 0.15 a detection scoring 0.10
    admits everything down to -0.05, so every stored row in the window counts
    and the weakest detections score as the "muddiest" species — an artefact of
    their own weakness. Requiring a companion to clear theta too means a
    companion is what the word implies: a rival answer the model actually gave.

    The count is *raw*: it includes the focal row itself whenever that row
    belongs to the group being counted. Callers subtract self explicitly,
    because a plausible species is not a member of the implausible group and
    must not have a self subtracted from it.

    Counted by binary search into the window's descending score array, so this
    stays linear-ish in the number of focal rows rather than quadratic in the
    window's species count.
    """
    out = np.zeros(len(focal), dtype=np.int32)
    rows = focal.select(*WINDOW_KEYS, "score_raw").iter_rows()
    for i, (rec, start, score) in enumerate(rows):
        scores = groups.get((rec, start))
        if scores is None:
            continue
        threshold = max(score - delta, floor)
        out[i] = int(np.searchsorted(-scores, -threshold, side="right"))
    return out


def per_species_muddiness(
    det: pl.DataFrame, cfg: MuddinessConfig = MuddinessConfig()
) -> pl.DataFrame:
    """How muddy is each species' detection, for this model?

    Over the windows where the model detects species *s* at or above theta:

    - `companions` — mean number of other species that are themselves detected
      (at or above theta) and score within delta of the focal species. A
      distinctive song sings alone; a confusable one always drags a crowd.
    - `implausible_companions` — the same count restricted to species the judge
      calls impossible: the geographic share of that confusion.
    - `impostor_mass` — mean impostor mass of those windows.

    This is the species-level answer to "which birds does this model actually
    know, and which does it merely rank first". Returned sorted by companions
    descending, so the muddiest species read first.
    """
    focal = det.filter(pl.col("score_raw") >= cfg.theta)
    if focal.is_empty():
        return pl.DataFrame(schema={
            "species_key": pl.Utf8, "windows": pl.UInt32, "mean_score": pl.Float64,
            "companions": pl.Float64, "implausible_companions": pl.Float64,
            "impostor_mass": pl.Float64,
        })

    all_groups = _window_groups(det)
    imp_groups = _window_groups(det.filter(pl.col("implausible")))
    # Every focal row is in the all-species group, so one self always comes off.
    # Only an implausible focal row is in the implausible group.
    focal = focal.with_columns(
        companions=pl.Series(
            _count_within_delta(focal, all_groups, cfg.delta, cfg.theta)) - 1,
        implausible_companions=pl.Series(
            _count_within_delta(focal, imp_groups, cfg.delta, cfg.theta))
        - pl.when(pl.col("implausible")).then(1).otherwise(0),
    )
    masses = window_impostor_mass(det, cfg.theta)
    focal = focal.join(masses, on=WINDOW_KEYS, how="left")

    return (
        focal.group_by("species_key")
        .agg(
            windows=pl.len(),
            mean_score=pl.col("score_raw").cast(pl.Float64).mean(),
            companions=pl.col("companions").cast(pl.Float64).mean(),
            implausible_companions=pl.col("implausible_companions").cast(pl.Float64).mean(),
            impostor_mass=pl.col("impostor_mass").mean(),
        )
        .sort("companions", descending=True)
    )


def muddiness_spread(per_species: pl.DataFrame, min_windows: int = 5) -> dict:
    """Median and IQR of per-species muddiness across species.

    The model-level summary: a low spread means uniformly clean (or uniformly
    muddy); a high one means the muddiness is concentrated in particular
    confusable groups. Species with very few detections are excluded — a mean
    over two windows is noise, not a property of the species.
    """
    kept = per_species.filter(pl.col("windows") >= min_windows)
    if kept.is_empty():
        return {"species": 0, "median": float("nan"), "iqr": float("nan"),
                "p25": float("nan"), "p75": float("nan")}
    values = kept["companions"].to_numpy()
    p25, p50, p75 = (float(np.percentile(values, q)) for q in (25, 50, 75))
    return {"species": len(kept), "median": p50, "p25": p25, "p75": p75,
            "iqr": p75 - p25}


def shadowed_detections(
    det: pl.DataFrame, cfg: MuddinessConfig = MuddinessConfig()
) -> pl.DataFrame:
    """Plausible detections with an implausible rival scoring nearly as high.

    These are the cases where the geofilter is doing *identification work the
    classifier failed to do* — the model could not separate the two birds, and
    the map broke the tie. `congeneric` marks the subset sharing a genus, which
    is the sharpest form of the problem (the European/Australasian blackbird
    case). Family would be sharper still, but the canonical table carries only
    genus, so that is what we claim.

    One row per (window, plausible species, implausible rival).
    """
    hits = det.filter(pl.col("score_raw") >= cfg.theta)
    plausible = hits.filter(~pl.col("implausible")).select(
        *WINDOW_KEYS, "species_key", "score_raw"
    )
    rivals = hits.filter(pl.col("implausible")).select(
        *WINDOW_KEYS,
        pl.col("species_key").alias("rival_key"),
        pl.col("score_raw").alias("rival_score"),
        pl.col("occ_score").alias("rival_occ"),
    )
    if plausible.is_empty() or rivals.is_empty():
        return pl.DataFrame(schema={
            "recording_id": pl.Utf8, "start_s": pl.Float64, "species_key": pl.Utf8,
            "score_raw": pl.Float32, "rival_key": pl.Utf8, "rival_score": pl.Float32,
            "rival_occ": pl.Float32, "gap": pl.Float64, "congeneric": pl.Boolean,
        })
    paired = plausible.join(rivals, on=WINDOW_KEYS, how="inner").filter(
        pl.col("rival_score") >= pl.col("score_raw") - cfg.delta
    )
    if paired.is_empty():
        return paired.with_columns(gap=pl.lit(0.0), congeneric=pl.lit(False))
    congeneric = pl.Series(
        [genus(a) == genus(b) for a, b in
         paired.select("species_key", "rival_key").iter_rows()],
        dtype=pl.Boolean,
    )
    return (
        paired.with_columns(
            gap=(pl.col("score_raw").cast(pl.Float64)
                 - pl.col("rival_score").cast(pl.Float64)),
            congeneric=congeneric,
        )
        .sort("rival_score", descending=True)
    )


def shadowed_pairs(shadowed: pl.DataFrame, top: int = 20) -> pl.DataFrame:
    """The offending species pairs, most frequent first."""
    if shadowed.is_empty():
        return pl.DataFrame(schema={"species_key": pl.Utf8, "rival_key": pl.Utf8,
                                    "windows": pl.UInt32, "mean_gap": pl.Float64,
                                    "congeneric": pl.Boolean})
    return (
        shadowed.group_by("species_key", "rival_key")
        .agg(windows=pl.len(), mean_gap=pl.col("gap").mean(),
             congeneric=pl.col("congeneric").first())
        .sort("windows", descending=True)
        .head(top)
    )


def filter_errors(
    det: pl.DataFrame, ann: pl.DataFrame, cfg: MuddinessConfig = MuddinessConfig()
) -> dict:
    """Score the geofilter itself against ground truth (labelled data only).

    - **false suppression** — the filter hid a detection of a species that is
      *annotated in that very window*. The filter was wrong and the model right.
    - **false admission** — it let through a confident detection of a species
      that appears nowhere in the labelled audio. Both the model and the filter
      were wrong, and the filter had its chance to catch it.

    Reported both as raw counts and as a rate over the relevant denominator, so
    "23 false suppressions" is readable as a share of what could have gone wrong.

    ⚠️ Both quantities can be **structurally zero** rather than good, and the
    result says which. If no annotated species is ever marked implausible, a
    false suppression is impossible by construction — which is exactly the case
    when the plausibility profile was *derived from* the annotations, as the
    `us-ca-sierra` profile was. `degenerate_*` flags that, because "0% false
    suppression" read as a result would be a circular claim.
    """
    hits = det.filter(pl.col("score_raw") >= cfg.theta)
    truth_windows = ann.select(
        "recording_id",
        pl.col("start_s").alias("box_start"),
        pl.col("end_s").alias("box_end"),
        "species_key",
    )
    # A detection "agrees with truth" when its window overlaps a box of the same
    # species in the same recording.
    agreeing = (
        hits.join(truth_windows, on=["recording_id", "species_key"], how="inner")
        .filter((pl.col("box_end") > pl.col("start_s"))
                & (pl.col("box_start") < pl.col("end_s")))
        .select(*WINDOW_KEYS, "species_key", "implausible")
        .unique()
    )
    false_suppressions = agreeing.filter(pl.col("implausible"))

    present = set(ann["species_key"].unique())
    admitted = hits.filter(~pl.col("implausible"))
    absent_admitted = admitted.filter(~pl.col("species_key").is_in(list(present)))

    # Can the judge disagree with truth at all, in either direction?
    implausible_keys = set(det.filter(pl.col("implausible"))["species_key"].unique())
    plausible_keys = set(det.filter(~pl.col("implausible"))["species_key"].unique())

    return {
        "degenerate_false_suppression": not (implausible_keys & present),
        "degenerate_false_admission": not (plausible_keys - present),
        "true_positive_detections": len(agreeing),
        "false_suppressions": len(false_suppressions),
        "false_suppression_rate": (len(false_suppressions) / len(agreeing)
                                   if len(agreeing) else float("nan")),
        "false_suppressed_species": sorted(
            set(false_suppressions["species_key"].unique())),
        "admitted_detections": len(admitted),
        "false_admissions": len(absent_admitted),
        "false_admission_rate": (len(absent_admitted) / len(admitted)
                                 if len(admitted) else float("nan")),
    }


def excluded_present_species(
    det: pl.DataFrame, ann: pl.DataFrame, occ_threshold: float
) -> pl.DataFrame:
    """Annotated species whose occurrence never clears the filter's threshold.

    Distinct from a false *suppression*, which needs the model to have detected
    the bird: this asks "which genuinely-present species is the geofilter
    structurally unable to report, whatever the audio says?" Those are the
    species for which the location prior has already decided the answer.
    """
    present = list(ann["species_key"].unique())
    return (
        det.filter(pl.col("species_key").is_in(present))
        .group_by("species_key")
        .agg(max_occ=pl.col("occ_score").max(),
             max_score=pl.col("score_raw").max())
        .filter(pl.col("max_occ") < occ_threshold)
        .sort("max_occ")
    )


def theta_for_detection_count(det: pl.DataFrame, target: int) -> float:
    """The threshold at which this arm yields `target` detections.

    Comparing two models at the same theta compares nothing: BirdNET's scores
    are per-class sigmoids and Perch's are a softmax over 14,795 classes, so
    0.1 is a different claim in each. Matching the *operating point* — the same
    number of detections — is the comparison that means something, and this is
    how each arm's theta is chosen to get there.
    """
    scores = det["score_raw"].cast(pl.Float64).sort(descending=True)
    if scores.is_empty():
        return 1.0
    if target >= len(scores):
        return float(scores[-1])
    return float(scores[max(0, target - 1)])


def matched_league(
    arms: dict[str, pl.DataFrame],
    target_detections: int | None = None,
    delta: float = 0.15,
    ann: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """League table with every arm at a matched operating point.

    `arms` maps a label to that arm's *marked* detections. The target defaults
    to the smallest arm's detection count at theta 0.1, so no arm is asked to
    produce more detections than it has. Each row records the theta that arm
    needed, because that number is itself informative.
    """
    if target_detections is None:
        target_detections = min(
            int((d["score_raw"] >= 0.1).sum()) for d in arms.values()
        ) or 1

    rows = []
    for label, det in arms.items():
        theta = theta_for_detection_count(det, target_detections)
        cfg = MuddinessConfig(theta=theta, delta=delta)
        row = league_row(det, label, cfg, ann)
        row["detections"] = int((det["score_raw"] >= theta).sum())
        rows.append(row)
    return pl.DataFrame(rows)


def league_row(
    det: pl.DataFrame,
    label: str,
    cfg: MuddinessConfig = MuddinessConfig(),
    ann: pl.DataFrame | None = None,
    *,
    rate: dict | None = None,
    masses: pl.DataFrame | None = None,
    species: pl.DataFrame | None = None,
    shadow: pl.DataFrame | None = None,
    events: pl.DataFrame | None = None,
) -> dict:
    """One model x profile row of the comparison table (PLAN.md §4).

    Detection quality sits beside the muddiness columns on purpose: a model can
    reach zero muddiness by detecting nothing, and the table has to make that
    visible rather than reward it.

    The keyword arguments let a caller that has already computed a component
    hand it over instead of paying for it twice — the per-species pass walks
    every stored row, so the app would otherwise recompute it for the table and
    again for this row.
    """
    rate = rate if rate is not None else muddy_window_rate(det, cfg.theta)
    masses = masses if masses is not None else window_impostor_mass(det, cfg.theta)
    species = species if species is not None else per_species_muddiness(det, cfg)
    spread = muddiness_spread(species)
    shadow = shadow if shadow is not None else shadowed_detections(det, cfg)
    events = events if events is not None else suppression_events(det, cfg.theta)

    row = {
        "arm": label,
        "theta": cfg.theta,
        "detection_windows": rate["windows"],
        "muddy_window_rate": rate["muddy_window_rate"],
        "median_impostor_mass": (float(masses["impostor_mass"].median())
                                 if len(masses) else float("nan")),
        "suppression_events": len(events),
        "shadowed_detections": len(shadow),
        "congeneric_shadowed": int(shadow["congeneric"].sum()) if len(shadow) else 0,
        "species_muddiness_median": spread["median"],
        "species_muddiness_iqr": spread["iqr"],
    }
    if ann is not None:
        row.update({k: v for k, v in filter_errors(det, ann, cfg).items()
                    if k in ("false_suppressions", "false_suppression_rate",
                             "false_admissions", "false_admission_rate")})
    return row
