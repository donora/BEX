"""How far each headline number could be off (V1.2 E1–E3).

One method everywhere, the one V1.1's survey page already uses: **resample
recordings**. A recording's windows rise and fall together — a busy dawn chorus
makes every model look different from a quiet afternoon — so the recording,
not the window, is the independent unit. Every headline number here is a ratio
of counts that add up over recordings, so a resample is a sum of per-recording
counts, and 1,000 of them cost one matrix product.

**Comparing two models** (E2): both are scored on the same recordings, so their
errors are correlated, and "the two intervals overlap" says much less than it
seems to. `paired` resamples once and computes both models on every resample,
which gives an interval for the *difference* directly.

Below `MIN_UNITS` recordings the interval is not reported at all (NaN): with
three recordings a bootstrap interval is narrow by accident, and a narrow
interval that means nothing is worse than none.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .benchmark import false_events

#: Fewer recordings than this and no interval is drawn.
MIN_UNITS = 5

COUNTS = ("reported", "right", "positives", "songs_found", "songs", "false_events", "hours")

#: The Compare page's headline numbers, as ratios of the counts above.
STATS = {
    "precision": ("right", "reported"),
    "window_recall": ("right", "positives"),
    "found": ("songs_found", "songs"),
    "false_per_hour": ("false_events", "hours"),
}

STAT_NAMES = {"precision": "precision", "window_recall": "recall (windows)",
              "found": "songs found", "false_per_hour": "false detections / hour"}


def per_recording(lab: pl.DataFrame, boxes: pl.DataFrame,
                  hours_of: dict[str, float]) -> pl.DataFrame:
    """The counts behind every headline number, one row per recording.

    The same definitions as `benchmark.summary`: a reported window is right when
    the species is labelled there (shown or hidden by the filter); a song is
    found when a visible hit covers it; false detections are events.
    """
    rids = sorted(set(lab["recording_id"].unique().to_list()))
    hits = lab.filter(pl.col("hit"))
    rep = hits.group_by("recording_id").agg(
        reported=pl.len(), right=(pl.col("y_true") == 1).sum())
    pos = lab.filter(pl.col("y_true") == 1).group_by("recording_id").agg(positives=pl.len())
    sung = boxes.filter(pl.col("outcome") != "not asked")
    songs = sung.group_by("recording_id").agg(
        songs=pl.len(), songs_found=(pl.col("outcome") == "found").sum())
    fe = false_events(lab).group_by("recording_id").agg(false_events=pl.col("events").sum())
    base = pl.DataFrame({"recording_id": rids,
                         "hours": [float(hours_of.get(r, 0.0)) for r in rids]},
                        schema={"recording_id": pl.Utf8, "hours": pl.Float64})
    out = base
    for part in (rep, pos, songs, fe):
        out = out.join(part, on="recording_id", how="left")
    return out.with_columns([pl.col(c).fill_null(0).cast(pl.Float64)
                             for c in COUNTS if c != "hours"]).select("recording_id", *COUNTS)


def _ratios(sums: np.ndarray) -> dict[str, np.ndarray]:
    """sums: (..., len(COUNTS)) -> each statistic, NaN where undefined."""
    idx = {c: i for i, c in enumerate(COUNTS)}
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, (num, den) in STATS.items():
            n, d = sums[..., idx[num]], sums[..., idx[den]]
            out[name] = np.where(d > 0, n / np.where(d > 0, d, 1), np.nan)
    return out


def point(per_rec: pl.DataFrame) -> dict[str, float]:
    return {k: float(v) for k, v in _ratios(per_rec.select(COUNTS).to_numpy().sum(0)).items()}


def _draws(r: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, r, size=(n, r))


def _interval(v: np.ndarray, level: float) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan")
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def bootstrap(per_rec: pl.DataFrame, n: int = 1000, seed: int = 0,
              level: float = 0.95) -> dict[str, tuple[float, float]]:
    """A percentile interval for each headline number, resampling recordings."""
    r = len(per_rec)
    if r < MIN_UNITS:
        return {k: (float("nan"), float("nan")) for k in STATS}
    mat = per_rec.select(COUNTS).to_numpy()
    sums = mat[_draws(r, n, seed)].sum(axis=1)
    return {k: _interval(v, level) for k, v in _ratios(sums).items()}


def paired(a: pl.DataFrame, b: pl.DataFrame, n: int = 1000, seed: int = 0,
           level: float = 0.95) -> dict[str, tuple[float, float, float]]:
    """a − b for each headline number, with an interval from resampling the
    recordings both were scored on — the same recordings in every resample, so
    what is shared between the two models cancels."""
    common = sorted(set(a["recording_id"]) & set(b["recording_id"]))
    pa = a.filter(pl.col("recording_id").is_in(common)).sort("recording_id")
    pb = b.filter(pl.col("recording_id").is_in(common)).sort("recording_id")
    diff = {k: point(pa)[k] - point(pb)[k] for k in STATS}
    if len(common) < MIN_UNITS:
        return {k: (diff[k], float("nan"), float("nan")) for k in STATS}
    draws = _draws(len(common), n, seed)
    ra = _ratios(pa.select(COUNTS).to_numpy()[draws].sum(axis=1))
    rb = _ratios(pb.select(COUNTS).to_numpy()[draws].sum(axis=1))
    return {k: (diff[k], *_interval(ra[k] - rb[k], level)) for k in STATS}


def verdict(diff: float, lo: float, hi: float) -> str:
    """'higher', 'lower' or 'no clear difference' — whether the interval for a
    difference excludes zero."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "too few recordings to say"
    if lo > 0:
        return "higher"
    if hi < 0:
        return "lower"
    return "no clear difference"


def width(per_rec: pl.DataFrame, stat: str = "window_recall", **kw) -> float:
    """The width of one statistic's interval: what the labelling plan (L2)
    projects from."""
    lo, hi = bootstrap(per_rec, **kw)[stat]
    return hi - lo
