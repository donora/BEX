"""What a model would give you, at the thresholds in force — the Compare dashboard.

`metrics.py` sweeps every threshold to say how good a model is in principle.
This module answers the question a researcher is actually deciding on: **if I
process my audio with this model, these thresholds and these settings, what
should I expect?**

- When it reports a bird, how often is it right? (window level, by outcome)
- When a bird is singing, how often does it find it? (vocalisation level)
- How many false detections should I expect per hour of audio?
- Which species does it handle best and worst?

Everything is on the evaluation label space — the annotated species — because
only there can a detection be judged. Outcomes use the navigator's vocabulary
(`views.OUTCOMES`), and a detection counts as *hidden* when the sidebar's judge
calls it implausible: the filter would have removed it before you saw it.

Two denominators, as in `scorecard.py`. "When it reports" counts **windows**,
because a window is what a model reports. "When a bird is singing" counts
**vocalisations** (annotated boxes), because a 3 s and a 5 s model cut the same
song into different numbers of windows, and only the song is comparable.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .scorecard import MIN_OVERLAP_S

REPORT_OUTCOMES = ("correct", "filter hid a real bird", "wrong", "filter caught a mistake")
SINGING_OUTCOMES = ("found", "filter hid a real bird", "missed")


def label_windows(
    aligned: pl.DataFrame,
    thetas: dict[str, float],
    hidden: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """The aligned frame with `theta`, `hit` and `hidden` added.

    `thetas` maps species to the θ in force (NaN: no operating point, never a
    hit). `hidden` lists the (recording_id, start_s, species_key) rows the judge
    calls implausible.
    """
    lab = aligned.join(
        pl.DataFrame({"species_key": list(thetas),
                      "theta": [float(v) for v in thetas.values()]},
                     schema={"species_key": pl.Utf8, "theta": pl.Float64}),
        on="species_key", how="left",
    ).with_columns(theta=pl.col("theta").fill_null(float("nan")))
    lab = lab.with_columns(
        hit=pl.col("theta").is_not_nan() & (pl.col("score") >= pl.col("theta")))
    if hidden is None or hidden.is_empty():
        return lab.with_columns(hidden=pl.lit(False))
    keys = hidden.select("recording_id", "start_s", "species_key").unique() \
                 .with_columns(hidden=pl.lit(True))
    return lab.join(keys, on=["recording_id", "start_s", "species_key"], how="left") \
              .with_columns(hidden=pl.col("hidden").fill_null(False))


def report_outcomes(lab: pl.DataFrame) -> dict[str, int]:
    """Every reported window, sorted into the navigator's four outcomes."""
    hits = lab.filter(pl.col("hit"))
    t, h = hits["y_true"] == 1, hits["hidden"]
    return {
        "correct": int((t & ~h).sum()),
        "filter hid a real bird": int((t & h).sum()),
        "wrong": int((~t & ~h).sum()),
        "filter caught a mistake": int((~t & h).sum()),
    }


def false_detections_per_hour(lab: pl.DataFrame, hours: float) -> float:
    """Wrong detections you would see, per hour of audio.

    Counted as **events**, not windows: consecutive wrong windows of one species
    are one false detection, as a listener would count them — otherwise a 3 s
    model would look worse than a 5 s one for the same mistake. Only what you
    would see: detections the filter hides are not shown to you.
    """
    if hours <= 0:
        return float("nan")
    ev = false_events(lab)
    return float(ev["events"].sum()) / hours if len(ev) else 0.0


def false_events(lab: pl.DataFrame) -> pl.DataFrame:
    """Shown wrong detections as events, per (recording, species).

    Consecutive wrong windows of one species are one event — one mistake a
    listener would notice, however many windows it spans.
    """
    wrong = (lab.filter(pl.col("hit") & (pl.col("y_true") == 0) & ~pl.col("hidden"))
                .select("recording_id", "species_key", "start_s", "end_s")
                .sort("recording_id", "species_key", "start_s"))
    if wrong.is_empty():
        return pl.DataFrame(schema={"recording_id": pl.Utf8, "species_key": pl.Utf8,
                                    "events": pl.UInt32})
    new_event = (
        (pl.col("recording_id") != pl.col("recording_id").shift(1))
        | (pl.col("species_key") != pl.col("species_key").shift(1))
        | (pl.col("start_s") > pl.col("end_s").shift(1) + 1e-6)
    ).fill_null(True)
    return (wrong.with_columns(new=new_event)
                 .group_by("recording_id", "species_key")
                 .agg(events=pl.col("new").sum().cast(pl.UInt32)))


def vocalisation_outcomes(
    lab: pl.DataFrame,
    annotations: pl.DataFrame,
    min_overlap_s: float = MIN_OVERLAP_S,
) -> pl.DataFrame:
    """One row per annotated vocalisation: found, hidden by the filter, missed.

    A box's windows are the ones alignment labelled positive for it (the same
    searchsorted narrowing and min(0.5 s, box) floor as `scorecard.box_coverage`).
    It is *found* if any of them is a visible hit, *filter hid a real bird* if
    the only hits are hidden, *missed* otherwise, and *not asked* if it owns no
    window at all (too short and straddling a boundary, or past the last window)
    — not the model's failure, so kept out of the rates.
    """
    pos = lab.filter(pl.col("y_true") == 1)
    if pos.is_empty() or annotations.is_empty():
        return pl.DataFrame(schema={"species_key": pl.Utf8, "recording_id": pl.Utf8,
                                    "outcome": pl.Utf8})
    window_s = float((pos["end_s"] - pos["start_s"]).max())
    lookup = {}
    for keys, grp in pos.sort("start_s").group_by(["species_key", "recording_id"],
                                                  maintain_order=True):
        lookup[(str(keys[0]), str(keys[1]))] = (
            grp["start_s"].to_numpy(),
            (grp["hit"] & ~grp["hidden"]).to_numpy(),
            (grp["hit"] & grp["hidden"]).to_numpy(),
        )
    rows = []
    eval_species = set(lab["species_key"].unique())
    for b in annotations.filter(pl.col("species_key").is_in(list(eval_species))) \
                        .iter_rows(named=True):
        spp, rid = b["species_key"], b["recording_id"]
        found = lookup.get((spp, rid))
        outcome = "not asked"
        if found is not None:
            starts, shown, hid = found
            lo, hi = b["start_s"], b["end_s"]
            floor = min(min_overlap_s, hi - lo)
            k_lo = int(np.searchsorted(starts, lo - window_s, side="right"))
            k_hi = int(np.searchsorted(starts, hi, side="left"))
            if k_hi > k_lo:
                k = np.arange(k_lo, k_hi)
                w = starts[k]
                overlap = np.minimum(hi, w + window_s) - np.maximum(lo, w)
                owned = k[(overlap > 0) & (overlap >= floor - 1e-9)]
                if len(owned):
                    outcome = ("found" if shown[owned].any()
                               else "filter hid a real bird" if hid[owned].any()
                               else "missed")
        rows.append({"species_key": spp, "recording_id": rid, "outcome": outcome})
    return pl.DataFrame(rows, schema={"species_key": pl.Utf8, "recording_id": pl.Utf8,
                                      "outcome": pl.Utf8})


def singing_outcomes(boxes: pl.DataFrame) -> dict[str, int]:
    """Vocalisation outcomes summed over species (the "not asked" left out)."""
    counts = dict(boxes.group_by("outcome").len().iter_rows()) if len(boxes) else {}
    return {o: int(counts.get(o, 0)) for o in SINGING_OUTCOMES}


def per_species(lab: pl.DataFrame, boxes: pl.DataFrame) -> pl.DataFrame:
    """One row per annotated species: how much truth, found rate, precision.

    `found` is the share of its askable vocalisations found (visible hits only).
    `precision` is the share of its reported windows that are right, hidden or
    not; null when the model never reports it.
    """
    win = lab.group_by("species_key").agg(
        n_positive=(pl.col("y_true") == 1).sum(),
        reported=pl.col("hit").sum(),
        reported_right=(pl.col("hit") & (pl.col("y_true") == 1)).sum(),
    ).with_columns(
        precision=pl.when(pl.col("reported") > 0)
                    .then(pl.col("reported_right") / pl.col("reported")))
    askable = boxes.filter(pl.col("outcome") != "not asked")
    voc = askable.group_by("species_key").agg(
        vocalisations=pl.len(),
        found_n=(pl.col("outcome") == "found").sum(),
    ).with_columns(found=pl.col("found_n") / pl.col("vocalisations"))
    return (win.join(voc, on="species_key", how="left")
               .with_columns(pl.col("vocalisations").fill_null(0),
                             pl.col("found_n").fill_null(0))
               .select("species_key", "n_positive", "vocalisations", "found_n",
                       "found", "reported", "precision")
               .sort("species_key"))


def by_recording(lab: pl.DataFrame, boxes: pl.DataFrame) -> pl.DataFrame:
    """Per recording and species: songs found, and false detections.

    Recordings where a species is *not* annotated but reported anyway are kept —
    that is exactly where false detections happen, and a table of only the
    annotated recordings would hide them.
    """
    askable = boxes.filter(pl.col("outcome") != "not asked")
    songs = askable.group_by("recording_id", "species_key").agg(
        songs=pl.len(), found_n=(pl.col("outcome") == "found").sum())
    return (songs.join(false_events(lab), on=["recording_id", "species_key"],
                       how="full", coalesce=True)
                 .with_columns(pl.col("songs").fill_null(0),
                               pl.col("found_n").fill_null(0),
                               pl.col("events").fill_null(0).alias("false_events"))
                 .with_columns(found=pl.when(pl.col("songs") > 0)
                                       .then(pl.col("found_n") / pl.col("songs")))
                 .select("recording_id", "species_key", "songs", "found_n", "found",
                         "false_events")
                 .sort("songs", "false_events", descending=True))


def mistaken_for(lab: pl.DataFrame, species_key: str, top: int = 8) -> pl.DataFrame:
    """Confusion: when this bird was singing and missed, which annotated bird the
    model reported in its place (each at its own threshold).

    Restricted to the evaluation label space, so it answers "of the birds we have
    truth for, which one did it prefer here". A congener near the top means the
    model heard something and named the wrong bird — a different failure from
    hearing nothing.
    """
    missed = lab.filter((pl.col("species_key") == species_key)
                        & (pl.col("y_true") == 1) & ~pl.col("hit"))
    if missed.is_empty():
        return pl.DataFrame(schema={"species_key": pl.Utf8, "windows": pl.UInt32,
                                    "also_annotated": pl.UInt32, "share": pl.Float64})
    keys = missed.select("recording_id", "start_s").unique()
    others = (lab.join(keys, on=["recording_id", "start_s"], how="inner")
                 .filter((pl.col("species_key") != species_key) & pl.col("hit")))
    if others.is_empty():
        return pl.DataFrame(schema={"species_key": pl.Utf8, "windows": pl.UInt32,
                                    "also_annotated": pl.UInt32, "share": pl.Float64})
    return (others.group_by("species_key")
                  .agg(windows=pl.len().cast(pl.UInt32),
                       also_annotated=(pl.col("y_true") == 1).sum().cast(pl.UInt32))
                  .with_columns(share=pl.col("windows") / len(keys))
                  .sort("windows", descending=True).head(top))


def crowd(det: pl.DataFrame, species_key: str, delta: float,
          top: int = 10) -> tuple[pl.DataFrame, dict]:
    """Muddiness, broken down: when the model reports this bird, who scores almost
    as high beside it?

    Over the windows where the model reports the bird (`above` its own θ), a
    *companion* is any other species also above its own θ that scores within
    `delta` of it — the same band the shadowing statistics use. A distinctive
    song sings alone; a confusable one drags a crowd, and an implausible member
    of that crowd means the map, not the model, is doing the identifying.

    `det` needs `above` (thresholds.attach) and `implausible`
    (stats.mark_implausible). Returns the companions — with each one's mean score
    and the focal bird's mean score in the windows they share, so the two can be
    drawn side by side, and the same in detection strength when `det` carries it
    — and a summary: windows reported, mean companions per
    window, mean implausible companions, and the bird's own mean score.
    """
    # Detection strength (thresholds.strength) rides along when present: raw
    # scores are on each model's own scale, strength is comparable between them.
    has_strength = "strength" in det.columns
    strength = pl.col("strength") if has_strength else pl.lit(None, dtype=pl.Float64)
    focal = det.filter((pl.col("species_key") == species_key) & pl.col("above")) \
               .select("recording_id", "start_s", pl.col("score_raw").alias("_focal"),
                       strength.cast(pl.Float64).alias("_focal_strength"))
    n = focal.select("recording_id", "start_s").unique().height
    empty = pl.DataFrame(schema={"species_key": pl.Utf8, "windows": pl.UInt32,
                                 "share": pl.Float64, "implausible": pl.Boolean,
                                 "score": pl.Float64, "focal_score": pl.Float64,
                                 "strength": pl.Float64, "focal_strength": pl.Float64})
    if n == 0:
        return empty, {"windows": 0, "companions": float("nan"),
                       "implausible_companions": float("nan"),
                       "focal_score": float("nan"), "focal_strength": float("nan")}
    comp = (det.filter((pl.col("species_key") != species_key) & pl.col("above"))
               .select("recording_id", "start_s", "species_key", "score_raw", "implausible",
                       strength.cast(pl.Float64).alias("_strength"))
               .join(focal, on=["recording_id", "start_s"], how="inner")
               .filter(pl.col("score_raw") >= pl.col("_focal") - delta))
    summary = {"windows": n, "companions": comp.height / n,
               "implausible_companions": int(comp["implausible"].sum()) / n,
               "focal_score": float(focal["_focal"].cast(pl.Float64).mean()),
               "focal_strength": (float(focal["_focal_strength"].mean())
                                  if has_strength else float("nan"))}
    if comp.is_empty():
        return empty, summary
    table = (comp.group_by("species_key")
                 .agg(windows=pl.len().cast(pl.UInt32),
                      implausible=pl.col("implausible").any(),
                      score=pl.col("score_raw").cast(pl.Float64).mean(),
                      focal_score=pl.col("_focal").cast(pl.Float64).mean(),
                      strength=pl.col("_strength").mean(),
                      focal_strength=pl.col("_focal_strength").mean())
                 .with_columns(share=pl.col("windows") / n)
                 .select("species_key", "windows", "share", "implausible",
                         "score", "focal_score", "strength", "focal_strength")
                 .sort("windows", descending=True).head(top))
    return table, summary


def coverage(found_by_model: dict[str, set[str]], species: list[str]) -> dict[str, list[str]]:
    """Which annotated species each model finds at all — all, only one, some, none.

    A species is *found* by a model when it finds at least one of its
    vocalisations. "Only <model>" is the interesting group: a real capability
    difference, confirmed by the annotations rather than by a threshold.
    """
    models = list(found_by_model)
    groups: dict[str, list[str]] = {"all models": []}
    for m in models:
        groups[f"only {m}"] = []
    if len(models) > 2:
        groups["some, not all"] = []
    groups["no model"] = []
    for sp in species:
        who = [m for m in models if sp in found_by_model[m]]
        if len(who) == len(models):
            groups["all models"].append(sp)
        elif len(who) == 1 and len(models) > 1:
            groups[f"only {who[0]}"].append(sp)
        elif not who:
            groups["no model"].append(sp)
        else:
            groups["some, not all"].append(sp)
    return groups


def summary(lab: pl.DataFrame, boxes: pl.DataFrame, hours: float) -> dict:
    """The headline card's numbers for one model."""
    rep = report_outcomes(lab)
    sing = singing_outcomes(boxes)
    reported = sum(rep.values())
    right = rep["correct"] + rep["filter hid a real bird"]
    positives = int((lab["y_true"] == 1).sum())
    return {
        "report": rep,
        "singing": sing,
        "precision": right / reported if reported else float("nan"),
        "window_recall": right / positives if positives else float("nan"),
        "found": (sing["found"] / sum(sing.values())) if sum(sing.values()) else float("nan"),
        "false_per_hour": false_detections_per_hour(lab, hours),
        "hours": hours,
    }
