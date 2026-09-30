"""What the location filter does, at the thresholds in force — the Geofilter page.

`stats.py` holds the PLAN.md §4 statistics at one θ per model; the headless
scripts and the matched-count league use it. This module asks the same
questions at **each species' own θ** — the sidebar's thresholds, the ones every
other page uses — and over every species the model can name, not only the
annotated ones: most of what a location filter removes is birds nobody
annotated, because they are not here.

Every function takes *judged* detections: `above` (`thresholds.attach`) and
`implausible` (`stats.mark_implausible`, under the sidebar's judge). A detection
the judge calls implausible is one the filter would **hide**; the rest it **lets
through**. Against the annotations each side splits in two, in the navigator's
words (`views.OUTCOMES`):

- let through — *correct* (a bird annotated in that window) or *wrong*;
- hidden — *filter hid a real bird* (annotated there) or *filter caught a mistake*.
"""
from __future__ import annotations

import polars as pl

from . import stats
from .scorecard import MIN_OVERLAP_S

LET_THROUGH = ("correct", "wrong")
HIDDEN = ("filter caught a mistake", "filter hid a real bird")
WINDOW = ["recording_id", "start_s"]


def reported(det: pl.DataFrame) -> pl.DataFrame:
    """The detections the model makes at its thresholds, before the filter acts."""
    return det.filter(pl.col("above"))


def with_truth(det: pl.DataFrame, ann: pl.DataFrame | None) -> pl.DataFrame:
    """Reported detections with `real`: an annotation of the same species overlaps
    the window by at least `MIN_OVERLAP_S` — the rule the alignment uses, so a
    window is right here exactly when it is right on the Compare page. Null
    throughout when there are no annotations."""
    hits = reported(det)
    if ann is None:
        return hits.with_columns(real=pl.lit(None, dtype=pl.Boolean))
    boxes = ann.select("recording_id", "species_key", pl.col("start_s").alias("_b0"),
                       pl.col("end_s").alias("_b1"))
    agree = (
        hits.select(*WINDOW, "end_s", "species_key")
        .join(boxes, on=["recording_id", "species_key"], how="inner")
        .filter(pl.min_horizontal("end_s", "_b1") - pl.max_horizontal("start_s", "_b0")
                >= MIN_OVERLAP_S)
        .select(*WINDOW, "species_key").unique()
        .with_columns(real=pl.lit(True))
    )
    return (hits.join(agree, on=[*WINDOW, "species_key"], how="left")
                .with_columns(pl.col("real").fill_null(False)))


def outcomes(judged: pl.DataFrame) -> dict[str, int]:
    """Reported detections by what the filter did and whether it was right.

    Takes `with_truth` output. Without annotations only the filter's side is
    known, so the counts come back as `let through` / `hidden`.
    """
    imp = pl.col("implausible")
    if judged.is_empty() or judged["real"].null_count() == len(judged):
        n_hidden = int(judged["implausible"].sum()) if len(judged) else 0
        return {"let through": len(judged) - n_hidden, "hidden": n_hidden}
    real = pl.col("real")
    row = judged.select(
        correct=(~imp & real).sum(), wrong=(~imp & ~real).sum(),
        caught=(imp & ~real).sum(), hid_real=(imp & real).sum()).row(0, named=True)
    return {"correct": row["correct"], "wrong": row["wrong"],
            "filter caught a mistake": row["caught"],
            "filter hid a real bird": row["hid_real"]}


def precision_with_and_without(o: dict[str, int]) -> tuple[float, float]:
    """Precision over every reported detection, before and after the filter.

    Without it you see everything, so every annotated detection counts as right —
    including the ones it would hide. With it you see only what it lets through.
    NaN where there is nothing to see, or no annotations to judge by."""
    if "correct" not in o:
        return float("nan"), float("nan")
    right_before = o["correct"] + o["filter hid a real bird"]
    total = right_before + o["wrong"] + o["filter caught a mistake"]
    shown = o["correct"] + o["wrong"]
    return (right_before / total if total else float("nan"),
            o["correct"] / shown if shown else float("nan"))


def recall_with_and_without(songs: dict[str, int]) -> tuple[float, float]:
    """Share of annotated songs found, before and after the filter.

    Takes `benchmark.singing_outcomes`: songs the model found, found only by
    detections the filter hides, and missed. Counted per song, as on the Compare
    page. The counterweight to precision: a filter that hides nearly everything
    can still lift precision, and only recall shows what it cost."""
    total = sum(songs.values())
    if not total:
        return float("nan"), float("nan")
    found = songs.get("found", 0)
    return (found + songs.get("filter hid a real bird", 0)) / total, found / total


def muddy_windows(det: pl.DataFrame) -> dict:
    """Of the windows where the model reports anything, how many also hold a
    bird the judge calls impossible (`stats.muddy_window_rate`, per-species θ)."""
    return stats.muddy_window_rate(reported(det), float("-inf"))


def impossible_species(judged: pl.DataFrame) -> pl.DataFrame:
    """Every species the model reports although the judge says it is not here.

    Takes `with_truth` output. Per species: detections, how many recordings they
    are spread over (one bad recording, or everywhere?), the strongest score, and
    `real` — detections that were an annotated bird after all, the ones the
    filter was wrong to hide. Most reported first.
    """
    imp = judged.filter(pl.col("implausible"))
    if imp.is_empty():
        return pl.DataFrame(schema={"species_key": pl.Utf8, "detections": pl.UInt32,
                                    "recordings": pl.UInt32, "peak": pl.Float64,
                                    "real": pl.UInt32})
    return (imp.group_by("species_key")
               .agg(detections=pl.len().cast(pl.UInt32),
                    recordings=pl.col("recording_id").n_unique().cast(pl.UInt32),
                    peak=pl.col("score_raw").cast(pl.Float64).max(),
                    real=pl.col("real").fill_null(False).sum().cast(pl.UInt32))
               .sort(["detections", "species_key"], descending=[True, False]))


def unreportable(det: pl.DataFrame, ann: pl.DataFrame | None) -> list[str]:
    """Annotated species the judge calls implausible every time the model scores
    them: the filter hides them whatever the audio says. Distinct from a hidden
    real bird, which needs the model to have detected it at its threshold."""
    if ann is None:
        return []
    present = ann["species_key"].unique().to_list()
    return sorted(det.filter(pl.col("species_key").is_in(present))
                     .group_by("species_key").agg(all=pl.col("implausible").all())
                     .filter(pl.col("all"))["species_key"].to_list())


def annotated_species(ann: pl.DataFrame | None) -> int:
    return 0 if ann is None else ann["species_key"].n_unique()


def circular(det: pl.DataFrame, ann: pl.DataFrame | None) -> bool:
    """True when no annotated species is ever implausible: the filter *cannot*
    hide a real bird, so a zero there is by construction, not merit — what
    happens when the plausibility profile was derived from these annotations."""
    if ann is None:
        return False
    present = set(ann["species_key"].unique().to_list())
    judged_impossible = set(det.filter(pl.col("implausible"))["species_key"].unique())
    return not (present & judged_impossible)
