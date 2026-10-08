"""From detections to a species list — the rules a survey reads a model with (V1.1).

Compare's species lists (V1 C2) put a species on a recording's list after *k*
detections above its θ. This module makes that rule the thing under study, with
a middle tier for an expert to check:

- **firm**: reported without review;
- **check**: possibly there — an expert listens before it is reported;
- **not found**.

A `SurveyRule` is two `Condition`s, firm and check. Each says which score bar a
window must clear (the θ a precision floor gives each species, or the sidebar's
thresholds), how many windows must clear it, and optionally within how many
seconds of each other. A species takes the strictest tier whose condition holds.

**Honesty.** Choosing the best of many rules on the recordings it is then scored
on flatters it, and so does fitting θ on them. Everything here therefore takes
the recordings to fit on and the recordings to score on separately: the page
explores on *tuning* recordings and reports on *test* ones (P13), and the full
assessment (P15) leaves each recording out in turn, refitting θ and re-choosing
the rule without it.

Pure functions over polars frames; the app only caches and draws them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from . import metrics
from . import thresholds as th
from .taxonomy import is_species

#: The precision floors a score bar can be fitted at. The loose end is for the
#: check tier: an expert filters it, so a bar where one call in five is real
#: (precision ≥ 0.2) can still be worth listening to.
FLOORS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99)
#: The floors the sweep tries for each tier. Firm calls are never checked, so
#: their bar starts at 0.5; the check bar may never be stricter than the firm one.
FIRM_FLOORS = tuple(p for p in FLOORS if p >= 0.5)
CHECK_FLOORS = FLOORS
#: Firm counts the sweep tries.
KS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)
#: "Within" spans offered, in seconds; None is anywhere in the recording.
SPANS = (None, 10.0, 30.0, 60.0, 300.0)
#: The score bar that is the sidebar's own thresholds, whatever its rule.
SIDEBAR = "sidebar"

#: Outcome of one (recording, species) under a rule, in display order.
OUTCOMES = ("firm, right", "firm, wrong", "firm, can't judge",
            "check, real", "check, not there", "check, can't judge",
            "filter hid a real bird", "missed")
FIRM = OUTCOMES[:3]
CHECK = OUTCOMES[3:6]


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #

def bar_key(floor: float | None) -> str:
    """The evidence frame's name for a score bar."""
    return SIDEBAR if floor is None else f"p{floor:g}"


def bar_label(floor: float | None) -> str:
    return "the sidebar's thresholds" if floor is None else f"precision ≥ {floor:g}"


@dataclass(frozen=True)
class Condition:
    """At least `k` windows above the score bar, within `within_s` seconds of
    each other (counted by window start) or anywhere when None. `floor` None is
    the sidebar's thresholds."""

    floor: float | None = 0.95
    k: int = 1
    within_s: float | None = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1")

    @property
    def bar(self) -> str:
        return bar_key(self.floor)

    @property
    def count_col(self) -> str:
        return "n" if self.within_s is None else f"w{self.within_s:g}"

    def describe(self) -> str:
        n = f"{self.k} detection" + ("s" if self.k != 1 else "")
        where = ("" if self.within_s is None
                 else f" within {self.within_s:g} s")
        return f"{n}{where} at {bar_label(self.floor)}"


@dataclass(frozen=True)
class SurveyRule:
    check: Condition = Condition(0.95, 1)
    firm: Condition = Condition(0.95, 3)

    @property
    def one_tier(self) -> bool:
        return self.check == self.firm

    def describe(self) -> str:
        if self.one_tier:
            return f"listed after {self.firm.describe()}; no check tier"
        return (f"firm after {self.firm.describe()}; "
                f"checked after {self.check.describe()}")

    def to_dict(self) -> dict:
        c = lambda x: {"floor": x.floor, "k": x.k, "within_s": x.within_s}
        return {"check": c(self.check), "firm": c(self.firm)}


def swept(rule: SurveyRule, floor: float, k: int,
          check_floor: float | None = None) -> SurveyRule:
    """The rule a sweep cell stands for: firm at `floor` and `k`, the check tier
    at `check_floor` (`floor` when None), everything else as in `rule`."""
    cf = floor if check_floor is None else check_floor
    return SurveyRule(replace(rule.check, floor=cf, k=min(rule.check.k, k)),
                      replace(rule.firm, floor=floor, k=k))


# --------------------------------------------------------------------------- #
# Score bars: θ per species at each precision floor, fitted on chosen recordings
# --------------------------------------------------------------------------- #

class CurveSource:
    """An aligned (window × species) frame, split by species once, so θ can be
    refitted on any subset of recordings without re-grouping two million rows —
    the full assessment refits 33 times per model."""

    def __init__(self, aligned: pl.DataFrame):
        recs = aligned["recording_id"].to_numpy()
        self.recordings = sorted(set(recs.tolist()))
        code = {r: i for i, r in enumerate(self.recordings)}
        rc = np.fromiter((code[r] for r in recs), dtype=np.int32, count=len(recs))
        y = aligned["y_true"].to_numpy().astype(bool)
        s = aligned["score"].to_numpy().astype(np.float64)
        sp = aligned["species_key"].to_numpy()
        order = np.argsort(sp, kind="stable")
        sp_sorted = sp[order]
        cuts = np.r_[0, np.nonzero(sp_sorted[1:] != sp_sorted[:-1])[0] + 1, len(sp)]
        self.species = {str(sp_sorted[a]): order[a:b] for a, b in zip(cuts[:-1], cuts[1:])
                        if b > a}
        self._y, self._s, self._rc, self._code = y, s, rc, code

    def fit(self, recordings: list[str] | None = None
            ) -> tuple[dict[str, metrics.PRCurve], metrics.PRCurve]:
        """Per-species curves and the pooled curve, on `recordings` (all if None)."""
        if recordings is None:
            keep = np.ones(len(self._y), dtype=bool)
        else:
            want = np.zeros(len(self.recordings), dtype=bool)
            for r in recordings:
                if r in self._code:
                    want[self._code[r]] = True
            keep = want[self._rc]
        curves = {}
        for sp, idx in self.species.items():
            idx = idx[keep[idx]]
            curves[sp] = metrics.pr_curve(self._y[idx], self._s[idx])
        micro = metrics.pr_curve(self._y[keep], self._s[keep])
        return curves, micro


def fit_bars(source: CurveSource, model: str, spec: th.RuleSpec,
             recordings: list[str] | None = None,
             floors: tuple[float, ...] = FLOORS) -> dict[str, th.Resolved]:
    """One `Resolved` per precision floor, fitted on `recordings`.

    The sidebar's guards (minimum predicted / labelled windows) and its fallback
    are kept; only the rule becomes a precision floor. Hand overrides are not
    applied: a θ set by hand while looking at all the recordings would leak the
    test recordings into the tuning.
    """
    curves, micro = source.fit(recordings)
    base = replace(spec, kind="precision floor")
    return {bar_key(p): th.resolve(model, replace(base, precision_floor=p), curves, micro)
            for p in floors}


# --------------------------------------------------------------------------- #
# Evidence: what each model reported, per (recording, species, score bar)
# --------------------------------------------------------------------------- #

def species_only(det: pl.DataFrame) -> pl.DataFrame:
    """The detections a species list can contain: real species (not *Car* or
    *Engine*), with just the columns the survey reads."""
    keys = det["species_key"].unique()
    ok = [k for k in keys.to_list() if is_species(k)]
    return (det.filter(pl.col("species_key").is_in(ok))
               .select("recording_id", "start_s", "species_key", "score_raw",
                       "implausible"))


_BIG = 1e7   # seconds; larger than any recording, so group offsets never overlap


def _span_max(gid: np.ndarray, t: np.ndarray, span: float, n_groups: int) -> np.ndarray:
    """Per group, the most windows whose starts fall within `span` seconds of the
    first of them. `gid` ascending and `t` ascending within each group."""
    out = np.zeros(n_groups, dtype=np.int64)
    if not len(gid):
        return out
    key = gid.astype(np.float64) * _BIG + t
    cnt = np.searchsorted(key, key + span, side="left") - np.arange(len(key))
    np.maximum.at(out, gid, cnt)
    return out


def evidence(det: pl.DataFrame, bars: dict[str, th.Resolved],
             spans: tuple[float, ...] = ()) -> pl.DataFrame:
    """Per (recording, species, bar) with at least one window above the bar:
    `n_all` / `n_shown` windows above it (all, and those the judge would show),
    and for each span `w{span}_all` / `w{span}_shown`, the most of them within
    that many seconds.

    `det` is `species_only` output: species detections with the judge's
    `implausible` flag.
    """
    spans = tuple(sorted({float(s) for s in spans if s is not None}))
    frames = []
    for key, res in bars.items():
        table = res.table.select("species_key", "theta")
        default = res.default_theta
        above = (det.join(table, on="species_key", how="left")
                    .with_columns(pl.col("theta").fill_null(
                        float("nan") if default is None else default))
                    .filter(pl.col("theta").is_not_nan()
                            & (pl.col("score_raw") >= pl.col("theta")))
                    .sort("recording_id", "species_key", "start_s"))
        if above.is_empty():
            continue
        groups = (above.group_by("recording_id", "species_key", maintain_order=True)
                       .agg(n_all=pl.len().cast(pl.Int64),
                            n_shown=(~pl.col("implausible")).sum().cast(pl.Int64)))
        if spans:
            gid = (above.select(pl.struct("recording_id", "species_key")
                                .rle_id()).to_series().to_numpy())
            t = above["start_s"].to_numpy()
            shown = ~above["implausible"].to_numpy()
            n_groups = len(groups)
            cols = {}
            for s in spans:
                cols[f"w{s:g}_all"] = _span_max(gid, t, s, n_groups)
                cols[f"w{s:g}_shown"] = _span_max(gid[shown], t[shown], s, n_groups)
            groups = groups.with_columns(**{k: pl.Series(v) for k, v in cols.items()})
        frames.append(groups.with_columns(bar=pl.lit(key)))
    if not frames:
        return pl.DataFrame(schema={"recording_id": pl.Utf8, "species_key": pl.Utf8,
                                    "n_all": pl.Int64, "n_shown": pl.Int64,
                                    "bar": pl.Utf8,
                                    **{f"w{s:g}_{w}": pl.Int64 for s in spans
                                       for w in ("all", "shown")}})
    return pl.concat(frames, how="vertical")


# --------------------------------------------------------------------------- #
# Tiers and outcomes
# --------------------------------------------------------------------------- #

def _condition_counts(ev: pl.DataFrame, cond: Condition, name: str) -> pl.DataFrame:
    col = cond.count_col
    if f"{col}_all" not in ev.columns and col != "n":
        raise KeyError(f"evidence has no {cond.within_s:g} s span — pass it to evidence()")
    return (ev.filter(pl.col("bar") == cond.bar)
              .select("recording_id", "species_key",
                      pl.col(f"{col}_shown").alias(f"{name}_shown"),
                      pl.col(f"{col}_all").alias(f"{name}_all")))


def tiers(ev: pl.DataFrame, rule: SurveyRule,
          recordings: list[str] | None = None) -> pl.DataFrame:
    """Per (recording, species) the rule puts on any list: `tier` from what the
    judge would show (firm / check / not found), `tier_all` from everything the
    model reported (what it would be without the filter), and the counts."""
    if recordings is not None:
        ev = ev.filter(pl.col("recording_id").is_in(recordings))
    f = _condition_counts(ev, rule.firm, "firm")
    c = _condition_counts(ev, rule.check, "check")
    both = f.join(c, on=["recording_id", "species_key"], how="full", coalesce=True)
    both = both.with_columns(pl.col(x).fill_null(0) for x in
                             ("firm_shown", "firm_all", "check_shown", "check_all"))

    def tier(sfx: str) -> pl.Expr:
        return (pl.when(pl.col(f"firm_{sfx}") >= rule.firm.k).then(pl.lit("firm"))
                  .when(pl.col(f"check_{sfx}") >= rule.check.k).then(pl.lit("check"))
                  .otherwise(pl.lit("not found")))

    return (both.with_columns(tier=tier("shown"), tier_all=tier("all"))
                .filter(pl.col("tier_all") != "not found")
                .sort("recording_id", "species_key"))


def truth_pairs(ann: pl.DataFrame, recordings: list[str] | None = None) -> pl.DataFrame:
    """Every (recording, species) annotated — what a perfect list would hold."""
    t = ann.select("recording_id", "species_key").unique()
    if recordings is not None:
        t = t.filter(pl.col("recording_id").is_in(recordings))
    return t


def outcomes(tier_frame: pl.DataFrame, ann: pl.DataFrame,
             recordings: list[str], label_set: set[str] | None = None) -> pl.DataFrame:
    """Each (recording, species) listed or annotated, with its outcome.

    `label_set` is the species the annotations cover (all annotated species in
    the dataset by default). A listing of a species outside it cannot be judged
    right or wrong — nobody was asked to mark it (V1 F1) — so it is *can't
    judge*, not *wrong*.
    """
    label_set = set(ann["species_key"].unique()) if label_set is None else label_set
    tf = tier_frame.filter(pl.col("recording_id").is_in(recordings))
    real = truth_pairs(ann, recordings).with_columns(annotated=pl.lit(True))
    rows = (tf.join(real, on=["recording_id", "species_key"], how="full", coalesce=True)
              .with_columns(pl.col("annotated").fill_null(False),
                            pl.col("tier").fill_null("not found"),
                            pl.col("tier_all").fill_null("not found"),
                            judged=pl.col("species_key").is_in(sorted(label_set))))
    a, j = pl.col("annotated"), pl.col("judged")
    firm, check = pl.col("tier") == "firm", pl.col("tier") == "check"
    out = (pl.when(firm & a).then(pl.lit("firm, right"))
             .when(firm & j).then(pl.lit("firm, wrong"))
             .when(firm).then(pl.lit("firm, can't judge"))
             .when(check & a).then(pl.lit("check, real"))
             .when(check & j).then(pl.lit("check, not there"))
             .when(check).then(pl.lit("check, can't judge"))
             .when(a & (pl.col("tier_all") != "not found"))
             .then(pl.lit("filter hid a real bird"))
             .when(a).then(pl.lit("missed"))
             .otherwise(pl.lit(None, dtype=pl.Utf8)))
    return (rows.with_columns(outcome=out)
                .filter(pl.col("outcome").is_not_null())
                .sort("recording_id", "species_key"))


def per_recording(out: pl.DataFrame, recordings: list[str]) -> pl.DataFrame:
    """One row per recording (every one of `recordings`, even an empty one) with a
    count column per outcome."""
    counts = (out.group_by("recording_id", "outcome").len()
                 .pivot(on="outcome", index="recording_id", values="len")
              if len(out) else pl.DataFrame(schema={"recording_id": pl.Utf8}))
    base = pl.DataFrame({"recording_id": list(recordings)}, schema={"recording_id": pl.Utf8})
    counts = base.join(counts, on="recording_id", how="left")
    return counts.with_columns(
        [pl.col(o).fill_null(0).cast(pl.Int64) if o in counts.columns
         else pl.lit(0, dtype=pl.Int64).alias(o) for o in OUTCOMES]
    ).select("recording_id", *OUTCOMES)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

#: The headline statistics, in display order, with whether higher is better.
STATS = {
    "recall_reviewed": ("birds found after checking", True),
    "firm_errors_per_recording": ("firm errors per recording", False),
    "firm_precision": ("firm calls right", True),
    "firm_recall": ("birds found with no checking", True),
    "checks_per_recording": ("checks per recording", False),
    "checks_per_hour": ("checks per hour of audio", False),
    "check_yield": ("checks that are real", True),
}


def _stats(c: dict[str, np.ndarray], n_recs, hours, unjudged_wrong: bool) -> dict:
    """The headline numbers from outcome totals (arrays, so a bootstrap is one call)."""
    fr, fw, fu = c["firm, right"], c["firm, wrong"], c["firm, can't judge"]
    cr, cw, cu = c["check, real"], c["check, not there"], c["check, can't judge"]
    annotated = fr + cr + c["missed"] + c["filter hid a real bird"]
    firm_bad = fw + (fu if unjudged_wrong else 0)
    check_bad = cw + (cu if unjudged_wrong else 0)
    checks = cr + cw + cu
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "firm_precision": fr / (fr + firm_bad),
            "recall_reviewed": (fr + cr) / annotated,
            "firm_recall": fr / annotated,
            "checks_per_recording": checks / n_recs,
            "checks_per_hour": checks / hours,
            "check_yield": cr / (cr + check_bad),
            "firm_errors_per_recording": firm_bad / n_recs,
        }


def score(per_rec: pl.DataFrame, hours_of: dict[str, float],
          unjudged_wrong: bool = False) -> dict:
    """Pooled headline numbers over the recordings in `per_rec`, plus the totals."""
    totals = {o: np.float64(per_rec[o].sum()) for o in OUTCOMES}
    hours = sum(hours_of.get(r, 0.0) for r in per_rec["recording_id"])
    s = _stats(totals, max(1, len(per_rec)), hours or float("nan"), unjudged_wrong)
    return {**{k: float(v) for k, v in s.items()},
            "counts": {o: int(v) for o, v in totals.items()},
            "n_recordings": len(per_rec), "hours": hours,
            "annotated": int(totals["firm, right"] + totals["check, real"]
                             + totals["missed"] + totals["filter hid a real bird"])}


def bootstrap(per_rec: pl.DataFrame, hours_of: dict[str, float], n: int = 1000,
              seed: int = 0, unjudged_wrong: bool = False,
              level: float = 0.95) -> dict[str, tuple[float, float]]:
    """Percentile intervals for each headline number, resampling recordings.

    Recordings, not (recording, species) pairs, are the independent unit: a busy
    recording's species rise and fall together.
    """
    r = len(per_rec)
    if r < 2:
        return {k: (float("nan"), float("nan")) for k in STATS}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, r, size=(n, r))
    mat = per_rec.select(OUTCOMES).to_numpy().astype(np.float64)
    hrs = np.array([hours_of.get(x, 0.0) for x in per_rec["recording_id"]])
    sums = mat[idx].sum(axis=1)                      # n × outcomes
    c = {o: sums[:, i] for i, o in enumerate(OUTCOMES)}
    s = _stats(c, r, hrs[idx].sum(axis=1), unjudged_wrong)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {k: (float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi)))
            if np.isfinite(v).any() else (float("nan"), float("nan"))
            for k, v in s.items()}


def assess(ev: pl.DataFrame, rule: SurveyRule, ann: pl.DataFrame,
           recordings: list[str], hours_of: dict[str, float],
           label_set: set[str] | None = None, unjudged_wrong: bool = False,
           n_boot: int = 1000, seed: int = 0) -> dict:
    """One rule, scored over `recordings`: outcomes, per-recording counts, the
    headline numbers and their bootstrap intervals."""
    out = outcomes(tiers(ev, rule, recordings), ann, recordings, label_set)
    per_rec = per_recording(out, recordings)
    return {"rule": rule, "outcomes": out, "per_recording": per_rec,
            "score": score(per_rec, hours_of, unjudged_wrong),
            "intervals": bootstrap(per_rec, hours_of, n_boot, seed, unjudged_wrong),
            "hours_of": hours_of, "unjudged_wrong": unjudged_wrong}


def paired(a: dict, b: dict, n: int = 1000, seed: int = 0,
           level: float = 0.95) -> dict[str, tuple[float, float, float]]:
    """a − b for each headline number of two `assess` results, with an interval
    from resampling the recordings both were scored on **together** (V1.2 E2).

    A hard recording is hard for every model, so two models' own intervals
    overlap far more than their difference is uncertain; this asks the
    question a comparison actually needs answered.
    """
    pa, pb = a["per_recording"], b["per_recording"]
    common = sorted(set(pa["recording_id"]) & set(pb["recording_id"]))
    pa = pa.filter(pl.col("recording_id").is_in(common)).sort("recording_id")
    pb = pb.filter(pl.col("recording_id").is_in(common)).sort("recording_id")
    hours_of, uw = a["hours_of"], a["unjudged_wrong"]
    hrs = np.array([hours_of.get(x, 0.0) for x in common])
    ma = pa.select(OUTCOMES).to_numpy().astype(np.float64)
    mb = pb.select(OUTCOMES).to_numpy().astype(np.float64)
    r = len(common)

    def stats(mat, idx=None):
        if idx is None:
            sums, h = mat.sum(axis=0), hrs.sum()
            return _stats({o: sums[i] for i, o in enumerate(OUTCOMES)}, r, h, uw)
        sums = mat[idx].sum(axis=1)
        return _stats({o: sums[:, i] for i, o in enumerate(OUTCOMES)}, r,
                      hrs[idx].sum(axis=1), uw)

    sa, sb = stats(ma), stats(mb)
    diff = {k: float(sa[k] - sb[k]) for k in STATS}
    if r < 2:
        return {k: (diff[k], float("nan"), float("nan")) for k in STATS}
    idx = np.random.default_rng(seed).integers(0, r, size=(n, r))
    ra, rb = stats(ma, idx), stats(mb, idx)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = {}
    for k in STATS:
        d = np.asarray(ra[k] - rb[k], dtype=np.float64)
        d = d[np.isfinite(d)]
        out[k] = ((diff[k], float(np.percentile(d, lo)), float(np.percentile(d, hi)))
                  if len(d) else (diff[k], float("nan"), float("nan")))
    return out


# --------------------------------------------------------------------------- #
# The sweep: precision floor × firm k
# --------------------------------------------------------------------------- #

def sweep(ev: pl.DataFrame, rule: SurveyRule, ann: pl.DataFrame,
          recordings: list[str], hours_of: dict[str, float],
          label_set: set[str] | None = None, unjudged_wrong: bool = False,
          floors: tuple[float, ...] = FIRM_FLOORS, ks: tuple[int, ...] = KS,
          check_floors: tuple[float, ...] = CHECK_FLOORS) -> pl.DataFrame:
    """Every (check floor, firm floor, firm k) cell, with the check bar no
    stricter than the firm one: `rule` with its tiers moved to those floors and
    firm moved to that k (`swept`). One row per cell with the headline numbers,
    pooled over `recordings`.

    Vectorised: every (recording, species) that any bar lists or the truth holds
    is one row of a universe; each bar's counts are joined onto it once, and
    each cell is then array comparisons — all ks at once.
    """
    label_set = set(ann["species_key"].unique()) if label_set is None else label_set
    real = truth_pairs(ann, recordings).with_columns(annotated=pl.lit(True))
    hours = sum(hours_of.get(r, 0.0) for r in recordings) or float("nan")
    n_recs = max(1, len(recordings))
    evr = ev.filter(pl.col("recording_id").is_in(recordings))
    keys = ["recording_id", "species_key"]
    universe = (pl.concat([evr.select(keys), real.select(keys)]).unique()
                  .join(real, on=keys, how="left")
                  .with_columns(pl.col("annotated").fill_null(False),
                                judged=pl.col("species_key").is_in(sorted(label_set))))
    a = universe["annotated"].to_numpy()
    j = universe["judged"].to_numpy()
    wrong, unj = ~a & j, ~a & ~j

    def counts(floor: float, cond: Condition) -> tuple[np.ndarray, np.ndarray]:
        c = _condition_counts(evr, replace(cond, floor=floor), "c")
        got = universe.select(keys).join(c, on=keys, how="left", maintain_order="left")
        return (got["c_shown"].fill_null(0).to_numpy(),
                got["c_all"].fill_null(0).to_numpy())

    firm_c = {p: counts(p, rule.firm) for p in floors}
    check_c = {p: counts(p, rule.check) for p in check_floors}
    kv = np.asarray(ks)[:, None]
    kc = np.minimum(rule.check.k, kv)
    rows = []
    for pf in floors:
        fs, fa = firm_c[pf]
        firm = fs[None, :] >= kv                    # K × N
        firm_all = fa[None, :] >= kv
        fr, fw, fu = ((firm & m).sum(1) for m in (a, wrong, unj))
        for pc in check_floors:
            if pc > pf:
                continue
            cs, ca = check_c[pc]
            check = ~firm & (cs[None, :] >= kc)
            none = ~firm & ~check
            seen_all = firm_all | (ca[None, :] >= kc)
            c_ = {"firm, right": fr.astype(np.float64), "firm, wrong": fw.astype(np.float64),
                  "firm, can't judge": fu.astype(np.float64),
                  "check, real": (check & a).sum(1).astype(np.float64),
                  "check, not there": (check & wrong).sum(1).astype(np.float64),
                  "check, can't judge": (check & unj).sum(1).astype(np.float64),
                  "filter hid a real bird": (none & a & seen_all).sum(1).astype(np.float64),
                  "missed": (none & a & ~seen_all).sum(1).astype(np.float64)}
            st = _stats(c_, n_recs, hours, unjudged_wrong)
            for i, k in enumerate(ks):
                rows.append({"check_floor": pc, "floor": pf, "k": k,
                             **{x: float(v[i]) for x, v in st.items()}})
    return pl.DataFrame(rows)


#: What a rule can be chosen to optimise: stat -> (in words, higher is better).
OBJECTIVES = {
    "recall_reviewed": ("the most birds found after checking", True),
    "firm_recall": ("the most birds found with no checking", True),
    "checks_per_recording": ("the fewest checks", False),
    "firm_precision": ("the most trustworthy firm calls", True),
}


@dataclass(frozen=True)
class Constraints:
    """What "best" means: an objective to optimise, within limits — firm calls at
    least this precise, at most this many checks per recording, at least this
    share of birds found after checking (None = no limit)."""

    min_firm_precision: float = 0.95
    max_checks_per_recording: float | None = None
    objective: str = "recall_reviewed"
    min_recall_reviewed: float | None = None

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {self.objective!r}")

    def describe(self) -> str:
        s = f"{OBJECTIVES[self.objective][0]}, with firm calls ≥ {self.min_firm_precision:.0%} right"
        if self.max_checks_per_recording is not None:
            s += f", ≤ {self.max_checks_per_recording:g} checks per recording"
        if self.min_recall_reviewed is not None:
            s += f", ≥ {self.min_recall_reviewed:.0%} of birds found after checking"
        return s

    def feasible(self, frame: pl.DataFrame, prefix: str = "") -> pl.Expr:
        ok = pl.col(f"{prefix}firm_precision").fill_nan(None) >= self.min_firm_precision
        if self.max_checks_per_recording is not None:
            ok = ok & (pl.col(f"{prefix}checks_per_recording")
                       <= self.max_checks_per_recording)
        if self.min_recall_reviewed is not None:
            ok = ok & (pl.col(f"{prefix}recall_reviewed").fill_nan(None)
                       >= self.min_recall_reviewed)
        return ok.fill_null(False)

    def order(self) -> tuple[list[str], list[bool]]:
        """Sort keys for ranking cells: the objective, then more birds after
        checking, fewer checks, a stricter floor, a smaller k."""
        keys = [(self.objective, OBJECTIVES[self.objective][1]),
                ("recall_reviewed", True), ("checks_per_recording", False),
                ("floor", True), ("check_floor", True), ("k", False)]
        seen, cols, desc = set(), [], []
        for c, d in keys:
            if c not in seen:
                seen.add(c)
                cols.append(c)
                desc.append(d)
        return cols, desc


def best(sw: pl.DataFrame, cons: Constraints) -> dict | None:
    """The sweep cell meeting `cons`'s limits that does best on its objective
    (ties: more birds after checking, fewer checks, a stricter floor, a smaller
    k). None when no cell qualifies."""
    ok = sw.filter(cons.feasible(sw))
    if ok.is_empty():
        return None
    cols, desc = cons.order()
    keep = [i for i, c in enumerate(cols) if c in ok.columns]
    return ok.sort([cols[i] for i in keep], descending=[desc[i] for i in keep],
                   nulls_last=True).row(0, named=True)


def best_shared(sweeps: dict[str, pl.DataFrame], cons: Constraints) -> dict | None:
    """One cell for every model: it must meet `cons`'s limits for each of them,
    and among those, the one with the best mean objective."""
    if not sweeps:
        return None
    present = set.intersection(*(set(sw.columns) for sw in sweeps.values()))
    stats = sorted({"firm_precision", "checks_per_recording", "recall_reviewed",
                    "firm_recall", cons.objective} & present)
    cell = [c for c in ("check_floor", "floor", "k") if c in present]
    joined = None
    for i, sw in enumerate(sweeps.values()):
        part = sw.select(*cell, *[pl.col(c).alias(f"{i}_{c}") for c in stats])
        joined = part if joined is None else joined.join(part, on=cell)
    ok = pl.lit(True)
    for i in range(len(sweeps)):
        ok = ok & cons.feasible(joined, prefix=f"{i}_")
    joined = joined.filter(ok).with_columns(
        **{c: pl.mean_horizontal([f"{i}_{c}" for i in range(len(sweeps))]) for c in stats})
    if joined.is_empty():
        return None
    cols, desc = cons.order()
    keep = [i for i, c in enumerate(cols) if c in joined.columns]
    return (joined.sort([cols[i] for i in keep], descending=[desc[i] for i in keep],
                        nulls_last=True)
                  .select(*cell, *stats).row(0, named=True))


# --------------------------------------------------------------------------- #
# Tuning and test recordings (P13)
# --------------------------------------------------------------------------- #

def split(recordings: list[str], seed: int = 0, test_share: float = 0.5
          ) -> tuple[list[str], list[str]]:
    """A reproducible random split into (tuning, test). The same seed always
    draws the same split of the same recordings."""
    ids = sorted(recordings)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_test = int(round(len(ids) * test_share))
    test = sorted(ids[i] for i in perm[:n_test])
    tune = sorted(ids[i] for i in perm[n_test:])
    return tune, test


# --------------------------------------------------------------------------- #
# Full assessment (P15): leave each recording out in turn
# --------------------------------------------------------------------------- #

@dataclass
class ModelInputs:
    """One model, as the full assessment needs it."""

    label: str
    identity: str
    det: pl.DataFrame            # species_only(...) output
    source: CurveSource


def full_assessment(models: list[ModelInputs], rule: SurveyRule, spec: th.RuleSpec,
                    ann: pl.DataFrame, recordings: list[str],
                    hours_of: dict[str, float], cons: Constraints,
                    label_set: set[str] | None = None, unjudged_wrong: bool = False,
                    n_boot: int = 1000, seed: int = 0,
                    progress=None) -> dict:
    """Leave-one-recording-out over `recordings`.

    In each round θ is refitted at every floor on the other recordings, the sweep
    is run on them, and two rules are chosen: each model's own best, and one
    shared best for every model. Both are scored on the left-out recording only.
    The pooled held-out counts give honest numbers for *the way of choosing a
    rule*; the rule to actually use is the one chosen on every recording, also
    returned, with how often the rounds agreed with it.
    """
    spans = tuple(s for s in (rule.firm.within_s, rule.check.within_s) if s is not None)
    label_set = set(ann["species_key"].unique()) if label_set is None else label_set
    recordings = sorted(recordings)

    def choose(tune: list[str]):
        evs, sws = {}, {}
        for m in models:
            bars = fit_bars(m.source, m.identity, spec, tune)
            evs[m.label] = evidence(m.det, bars, spans)
            sws[m.label] = sweep(evs[m.label], rule, ann, tune, hours_of, label_set,
                                 unjudged_wrong)
        own = {m: best(sw, cons) for m, sw in sws.items()}
        return evs, own, best_shared(sws, cons)

    # The rule to use: chosen on everything.
    _, final_own, final_shared = choose(recordings)

    held = {"own": {m.label: [] for m in models}, "shared": {m.label: [] for m in models}}
    picks = {"own": {m.label: [] for m in models}, "shared": []}
    for i, r in enumerate(recordings):
        if progress is not None:
            progress(i, len(recordings), r)
        tune = [x for x in recordings if x != r]
        evs, own, shared = choose(tune)
        picks["shared"].append(shared)
        for m in models:
            for kind, cell in (("own", own[m.label]), ("shared", shared)):
                if kind == "own":
                    picks["own"][m.label].append(cell)
                if cell is None:
                    # No rule meets the constraints on the tuning recordings:
                    # nothing is reported, so every annotated bird is missed.
                    out = outcomes(pl.DataFrame(schema={
                        "recording_id": pl.Utf8, "species_key": pl.Utf8,
                        "tier": pl.Utf8, "tier_all": pl.Utf8}), ann, [r], label_set)
                else:
                    cell_rule = swept(rule, cell["floor"], cell["k"], cell.get("check_floor"))
                    out = outcomes(tiers(evs[m.label], cell_rule, [r]), ann, [r], label_set)
                held[kind][m.label].append(per_recording(out, [r]))

    result = {"final_own": final_own, "final_shared": final_shared, "models": {}}
    for kind in ("own", "shared"):
        for m in models:
            per_rec = pl.concat(held[kind][m.label])
            result["models"].setdefault(m.label, {})[kind] = {
                "per_recording": per_rec,
                "score": score(per_rec, hours_of, unjudged_wrong),
                "intervals": bootstrap(per_rec, hours_of, n_boot, seed, unjudged_wrong),
            }
    result["stability"] = {
        m.label: _stability(picks["own"][m.label], final_own[m.label]) for m in models}
    result["stability_shared"] = _stability(picks["shared"], final_shared)
    result["n_rounds"] = len(recordings)
    return result


def _stability(picks: list[dict | None], final: dict | None) -> dict:
    """How often the rounds chose the same cell as all the recordings did, and
    the spread of what they chose."""
    chosen = [p for p in picks if p is not None]
    cell = lambda p: (p.get("check_floor"), p["floor"], p["k"])
    same = sum(1 for p in chosen if final is not None and cell(p) == cell(final))
    spread = lambda key: ((min(v), max(v)) if (v := [p[key] for p in chosen
                                                     if p.get(key) is not None])
                          else (math.nan, math.nan))
    return {"rounds": len(picks), "none": len(picks) - len(chosen), "same": same,
            "floors": spread("floor"), "check_floors": spread("check_floor"),
            "ks": spread("k")}
