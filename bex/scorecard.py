"""Per-species accounting at one operating point: what should have been found, what was.

`metrics.py` sweeps every threshold and reports how good an arm is overall.
This module answers the question a fieldworker actually asks, about one bird:

    Golden-crowned Kinglet is annotated 412 times across 33 recordings.
    On BirdNET's 3 s grid that is 5,402 windows it had the chance to report.
    At theta 0.86 it caught 3,748 of them, missed 1,654, and reported the
    species in 197 windows where it is not annotated at all.

Everything here derives from the aligned frame (`truth.align_run`) plus the
annotation boxes, so every number traces back to rows a reader can inspect.

**Two denominators, and the difference between them matters.** A 30 s box is ten
windows on BirdNET's 3 s grid and six on Perch's 5 s grid, so "windows it should
have reported" is not comparable across arms — it is a property of the grid as
much as of the bird. Window-level counts are still the right basis for choosing
theta, because theta is applied per window. But for *comparing* arms, and for the
question a survey actually asks, the honest unit is the **box**: of the
vocalisations a human marked, how many did the model detect at all? One window
inside a box is a detection of that bird; a box is missed only when the model was
silent across every window of it. Both are reported side by side, always.
"""
from __future__ import annotations

import numpy as np
import polars as pl

#: The label rule from `windows.label_starts`, repeated here because box coverage
#: has to reproduce exactly which windows a box was held responsible for.
MIN_OVERLAP_S = 0.5


def _species_frame(aligned: pl.DataFrame, species_key: str) -> pl.DataFrame:
    return aligned.filter(pl.col("species_key") == species_key)


def counts(aligned: pl.DataFrame, species_key: str, theta: float) -> dict:
    """TP / FP / FN / TN for one species at one threshold, plus the rates.

    `opportunity` is the window count a perfect detector would have reported —
    the y_true positives on *this arm's* grid. Reported alongside the counts
    because "caught 3,748" means nothing without "out of 5,402".
    """
    df = _species_frame(aligned, species_key)
    if df.is_empty():
        return {"opportunity": 0, "caught": 0, "missed": 0, "false_alarms": 0,
                "correct_silence": 0, "windows": 0, "precision": float("nan"),
                "recall": float("nan"), "theta": theta}

    hit = df["score"].to_numpy() >= theta
    pos = df["y_true"].to_numpy().astype(bool)
    tp = int((hit & pos).sum())
    fp = int((hit & ~pos).sum())
    fn = int((~hit & pos).sum())
    tn = int((~hit & ~pos).sum())
    return {
        "opportunity": tp + fn,
        "caught": tp,
        "missed": fn,
        "false_alarms": fp,
        "correct_silence": tn,
        "windows": len(df),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "theta": theta,
    }


def by_recording(aligned: pl.DataFrame, species_key: str, theta: float) -> pl.DataFrame:
    """The same accounting, split by recording — where does this arm fail?

    A species the model handles well overall can be invisible in the dawn hour or
    on one noisy recorder, and a single recall number hides that completely.
    Recordings where the species is not annotated are kept, because a false alarm
    there is exactly as real as one anywhere else.
    """
    df = _species_frame(aligned, species_key)
    if df.is_empty():
        return pl.DataFrame()
    hit = pl.col("score") >= theta
    pos = pl.col("y_true") == 1
    return (
        df.group_by("recording_id")
        .agg(
            opportunity=pos.sum(),
            caught=(hit & pos).sum(),
            missed=((~hit) & pos).sum(),
            false_alarms=(hit & (~pos)).sum(),
            windows=pl.len(),
            best_score=pl.col("score").max(),
        )
        .with_columns(
            recall=pl.when(pl.col("opportunity") > 0)
                     .then(pl.col("caught") / pl.col("opportunity"))
                     .otherwise(None),
            precision=pl.when(pl.col("caught") + pl.col("false_alarms") > 0)
                        .then(pl.col("caught")
                              / (pl.col("caught") + pl.col("false_alarms")))
                        .otherwise(None),
        )
        .sort("opportunity", descending=True)
    )


def box_coverage(
    aligned: pl.DataFrame,
    annotations: pl.DataFrame,
    species_key: str,
    theta: float,
    min_overlap_s: float = MIN_OVERLAP_S,
) -> pl.DataFrame:
    """One row per annotated vocalisation: how many windows it spans, how many fired.

    The grid-free unit. A box is `detected` when the arm cleared theta in at least
    one of the windows the label rule held it responsible for — which is what
    "did it find the bird" means to a surveyor, as against the stricter
    window-level recall that also punishes a model for going quiet in the middle
    of a long song.

    The window range per box is recomputed with the same searchsorted narrowing
    and the same `min(min_overlap_s, box duration)` floor as
    `windows.label_starts`, so a box's windows here are exactly the windows that
    were labelled positive for it during alignment.
    """
    df = _species_frame(aligned, species_key)
    boxes = annotations.filter(pl.col("species_key") == species_key)
    if df.is_empty() or boxes.is_empty():
        return pl.DataFrame()

    window_s = float((df["end_s"] - df["start_s"]).max())
    rows = []
    for rid, grp in df.sort("start_s").group_by("recording_id", maintain_order=True):
        rid = rid[0] if isinstance(rid, tuple) else rid
        starts = grp["start_s"].to_numpy()
        scores = grp["score"].to_numpy()
        for b in boxes.filter(pl.col("recording_id") == rid).iter_rows(named=True):
            lo, hi = b["start_s"], b["end_s"]
            floor = min(min_overlap_s, hi - lo)
            k_lo = int(np.searchsorted(starts, lo - window_s, side="right"))
            k_hi = int(np.searchsorted(starts, hi, side="left"))
            if k_hi <= k_lo:
                rows.append({"recording_id": rid, "start_s": lo, "end_s": hi,
                             "duration_s": hi - lo, "windows": 0, "caught": 0,
                             "detected": False, "best_score": float("nan")})
                continue
            k = np.arange(k_lo, k_hi)
            w = starts[k]
            overlap = np.minimum(hi, w + window_s) - np.maximum(lo, w)
            owned = k[(overlap > 0) & (overlap >= floor - 1e-9)]
            s = scores[owned]
            rows.append({
                "recording_id": rid, "start_s": lo, "end_s": hi,
                "duration_s": hi - lo,
                "windows": int(len(owned)),
                "caught": int((s >= theta).sum()),
                "detected": bool((s >= theta).any()),
                "best_score": float(s.max()) if len(s) else float("nan"),
            })
    return pl.DataFrame(rows).sort("start_s") if rows else pl.DataFrame()


def box_summary(coverage: pl.DataFrame) -> dict:
    """Box-level headline: how many vocalisations were detected at all."""
    if coverage.is_empty():
        return {"boxes": 0, "detected": 0, "missed": 0, "detection_rate": float("nan"),
                "windows": 0, "caught": 0, "median_duration_s": float("nan")}
    detected = int(coverage["detected"].sum())
    # A box can own no window at all: shorter than the label floor and straddling
    # a boundary, or past the last window the model scored. The arm was never
    # asked about it, so counting it as a miss would blame the model for the grid
    # — and the count differs per arm, which is itself worth seeing.
    unrepresentable = int((coverage["windows"] == 0).sum())
    askable = len(coverage) - unrepresentable
    return {
        "boxes": len(coverage),
        "detected": detected,
        "missed": askable - detected,
        "unrepresentable": unrepresentable,
        "detection_rate": detected / askable if askable else float("nan"),
        "windows": int(coverage["windows"].sum()),
        "caught": int(coverage["caught"].sum()),
        "median_duration_s": float(coverage["duration_s"].median()),
    }


def confusion_in_missed_windows(
    aligned: pl.DataFrame, species_key: str, theta: float, top: int = 10
) -> pl.DataFrame:
    """When the arm missed this bird, which annotated species did it report instead?

    Restricted to the evaluation label space, so this is "of the species we have
    truth for, what did it prefer here" — not a claim about the model's whole
    vocabulary. A congener at the top of this table is the interesting case: the
    model heard *something* and named the wrong bird, which is a different failure
    from hearing nothing at all, and only this view separates them.
    """
    df = _species_frame(aligned, species_key)
    missed = df.filter((pl.col("y_true") == 1) & (pl.col("score") < theta))
    if missed.is_empty():
        return pl.DataFrame()

    keys = missed.select("recording_id", "start_s").unique()
    others = (
        aligned.join(keys, on=["recording_id", "start_s"], how="inner")
        .filter((pl.col("species_key") != species_key) & (pl.col("score") >= theta))
    )
    if others.is_empty():
        return pl.DataFrame()
    return (
        others.group_by("species_key")
        .agg(windows=pl.len(),
             also_annotated=(pl.col("y_true") == 1).sum(),
             median_score=pl.col("score").median())
        .with_columns(share=pl.col("windows") / len(keys))
        .sort("windows", descending=True)
        .head(top)
    )


def timeline(
    aligned: pl.DataFrame, species_key: str, recording_id: str, theta: float
) -> pl.DataFrame:
    """Window-by-window verdicts for one recording: the strip the app draws.

    Four outcomes, named as a reader would name them rather than as a confusion
    matrix: `caught`, `missed`, `false alarm`, `correct silence`.
    """
    df = (_species_frame(aligned, species_key)
          .filter(pl.col("recording_id") == recording_id)
          .sort("start_s"))
    if df.is_empty():
        return df
    return df.with_columns(
        outcome=pl.when((pl.col("y_true") == 1) & (pl.col("score") >= theta))
                  .then(pl.lit("caught"))
                  .when((pl.col("y_true") == 1))
                  .then(pl.lit("missed"))
                  .when(pl.col("score") >= theta)
                  .then(pl.lit("false alarm"))
                  .otherwise(pl.lit("correct silence"))
    ).select("recording_id", "start_s", "end_s", "score", "y_true", "outcome")


def detection_overview(
    aligned: pl.DataFrame,
    annotations: pl.DataFrame,
    thetas: dict[str, float],
    min_overlap_s: float = MIN_OVERLAP_S,
) -> pl.DataFrame:
    """Box-level detection rate for **every** species at once — the tab's overview.

    `thetas` maps species_key to the threshold to judge that species at, because
    a per-species rule (95% precision on *this* bird) gives a different θ for
    every species, and that is the comparison worth drawing. A species missing
    from `thetas`, or mapped to NaN, is returned with a null rate and
    `reachable=False`: an arm that cannot reach the rule for a bird at any
    threshold has said something real, and plotting it at zero would say
    something else.

    Only the positive rows are scanned. The windows a box owns are exactly the
    windows alignment labelled positive for that species (asserted in
    `tests/test_scorecard.py` against a naive reimplementation), so restricting
    to `y_true == 1` is not an approximation — it is the same set, two orders of
    magnitude smaller.
    """
    pos = aligned.filter(pl.col("y_true") == 1)
    if pos.is_empty() or annotations.is_empty():
        return pl.DataFrame()
    window_s = float((pos["end_s"] - pos["start_s"]).max())

    lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for keys, grp in pos.sort("start_s").group_by(["species_key", "recording_id"],
                                                  maintain_order=True):
        lookup[(str(keys[0]), str(keys[1]))] = (grp["start_s"].to_numpy(),
                                                grp["score"].to_numpy())

    tally: dict[str, dict] = {}
    for b in annotations.iter_rows(named=True):
        spp, rid = b["species_key"], b["recording_id"]
        t = tally.setdefault(spp, {"boxes": 0, "detected": 0, "not_asked": 0,
                                   "recordings": set(), "seconds": 0.0})
        t["boxes"] += 1
        t["recordings"].add(rid)
        t["seconds"] += b["end_s"] - b["start_s"]

        starts_scores = lookup.get((spp, rid))
        if starts_scores is None:
            t["not_asked"] += 1
            continue
        starts, scores = starts_scores
        lo, hi = b["start_s"], b["end_s"]
        floor = min(min_overlap_s, hi - lo)
        k_lo = int(np.searchsorted(starts, lo - window_s, side="right"))
        k_hi = int(np.searchsorted(starts, hi, side="left"))
        if k_hi <= k_lo:
            t["not_asked"] += 1
            continue
        k = np.arange(k_lo, k_hi)
        w = starts[k]
        overlap = np.minimum(hi, w + window_s) - np.maximum(lo, w)
        owned = k[(overlap > 0) & (overlap >= floor - 1e-9)]
        if not len(owned):
            t["not_asked"] += 1
            continue
        theta = thetas.get(spp, float("nan"))
        if theta == theta and bool((scores[owned] >= theta).any()):
            t["detected"] += 1

    rows = []
    for spp, t in tally.items():
        theta = thetas.get(spp, float("nan"))
        askable = t["boxes"] - t["not_asked"]
        reachable = theta == theta
        rows.append({
            "species_key": spp,
            "boxes": t["boxes"],
            "askable": askable,
            "not_asked": t["not_asked"],
            "recordings": len(t["recordings"]),
            "seconds": t["seconds"],
            "theta": theta,
            "reachable": reachable,
            "detected": t["detected"] if reachable else 0,
            "detection_rate": (t["detected"] / askable
                               if reachable and askable else None),
        })
    return pl.DataFrame(rows).sort("detection_rate", descending=True, nulls_last=True)


def species_index(aligned: pl.DataFrame, annotations: pl.DataFrame) -> pl.DataFrame:
    """The picker's menu: every evaluation species with how much truth it has."""
    windows = (aligned.filter(pl.col("y_true") == 1)
               .group_by("species_key").agg(opportunity=pl.len()))
    boxes = (annotations.group_by("species_key")
             .agg(boxes=pl.len(),
                  seconds=(pl.col("end_s") - pl.col("start_s")).sum(),
                  recordings=pl.col("recording_id").n_unique()))
    return (aligned.select("species_key").unique()
            .join(windows, on="species_key", how="left")
            .join(boxes, on="species_key", how="left")
            .fill_null(0)
            .sort("boxes", descending=True))
