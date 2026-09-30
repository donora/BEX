"""Precision/recall against ground truth — the scoreboard every arm is judged on.

PLAN.md §3d asks for "per-window multi-label, cmAP + per-species AP". This module
is the arithmetic half of that; `truth.py` is the alignment half that produces the
`(y_true, score)` pairs it consumes. Everything here is a pure function over two
numpy arrays, so it can be tested without a store, a model, or a single byte of audio.

**Why a sweep and not a threshold.** Every number the app reported before this module
was conditional on an operating point, and the project's answer to "which θ?" was to
dodge it — match the arms on detection count instead (`stats.theta_for_detection_count`).
That is the right dodge for comparing muddiness, but it cannot say which θ a
researcher should actually *use*, and it cannot rank two models. A full sweep removes
the choice from the comparison: AP integrates over every operating point at once, and
the θ that a particular researcher wants falls out of the curve afterwards, by a rule
they pick and can see.

**Three rules, because there is no single sweet spot.** Max-F1 is the textbook answer
and is usually wrong for survey work: it treats a missed bird and a false alarm as
equally costly, which is a claim about the study, not about the model. `precision_floor`
("the highest recall I can get while staying 95% precise") is the question an ecologist
actually asks; `recall_floor` is its mirror for rare-species search, where a missed
detection is the expensive error. All three are computed from the same curve and
reported side by side, so the reader sees what the choice costs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class PRCurve:
    """Every achievable operating point, one per distinct score in the data.

    Each entry answers "if I set θ here, what do I get?", so `theta[i]` is a real
    score a window received and `precision[i]`/`recall[i]` are exactly what
    thresholding at `score >= theta[i]` would yield. Ties share one point — the
    alternative, splitting equal scores, invents operating points no threshold
    can reach.
    """

    theta: np.ndarray       # descending
    precision: np.ndarray
    recall: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    n_pos: int
    n_neg: int

    def __len__(self) -> int:
        return len(self.theta)

    @property
    def prevalence(self) -> float:
        """Positive rate — the precision a coin-flip ranker would average.

        The floor any curve must beat to have said anything, and the reason a
        per-species AP of 0.30 can be excellent for a rare bird and dismal for a
        common one.
        """
        total = self.n_pos + self.n_neg
        return self.n_pos / total if total else float("nan")


def pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> PRCurve:
    """Sweep θ over every distinct score, descending.

    Exact, not binned: with N rows there are at most N distinct thresholds and we
    evaluate all of them, so no operating point is missed between grid steps. A
    fixed-grid sweep (θ = 0.01, 0.02, …) would be cheaper and would misreport
    exactly where it matters — the top of the ranking, where scores bunch up.
    """
    yt = np.asarray(y_true).astype(bool)
    ys = np.asarray(y_score, dtype=np.float64)
    if yt.shape != ys.shape:
        raise ValueError(f"y_true {yt.shape} and y_score {ys.shape} differ")

    n_pos = int(yt.sum())
    n_neg = int(yt.size - n_pos)
    empty = np.zeros(0)
    if yt.size == 0:
        return PRCurve(empty, empty, empty, empty, empty, empty, 0, 0)

    order = np.argsort(-ys, kind="stable")
    ys, yt = ys[order], yt[order]

    # One point per run of equal scores: the index where each run ends.
    idx = np.r_[np.nonzero(np.diff(ys))[0], ys.size - 1]
    tp = np.cumsum(yt)[idx].astype(np.int64)
    fp = (idx + 1) - tp
    fn = n_pos - tp

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = tp / (tp + fp)
        recall = tp / n_pos if n_pos else np.full(len(idx), np.nan)
    return PRCurve(ys[idx], precision, recall, tp, fp, fn, n_pos, n_neg)


def average_precision(curve: PRCurve) -> float:
    """AP = Σ (Rₙ − Rₙ₋₁)·Pₙ — the area under the curve, no interpolation.

    The unsmoothed convention (scikit-learn's `average_precision_score`, and
    BirdCLEF's) rather than the 11-point or max-envelope interpolations, which
    flatter a model by replacing each precision with the best precision anywhere
    to its right.

    NaN when the species has no positives: a curve with an empty denominator is
    not a score of zero, and averaging it in as zero is how cmAP gets quietly
    dragged down by species the eval set never labelled.
    """
    if curve.n_pos == 0 or len(curve) == 0:
        return float("nan")
    return float(np.sum(np.diff(np.r_[0.0, curve.recall]) * curve.precision))


def fbeta(precision: np.ndarray, recall: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """F_β. β > 1 weights recall (β=2: a miss costs 4× a false alarm), β < 1 precision."""
    b2 = beta * beta
    num = (1 + b2) * precision * recall
    den = b2 * precision + recall
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)


@dataclass(frozen=True)
class Rule:
    """How to pick one operating point off a curve.

    - `fbeta`      — maximise F_β (value = β).
    - `precision`  — the highest *recall* whose precision is at least `value`.
    - `recall`     — the highest *precision* whose recall is at least `value`.

    `min_support` is a guard, not a tuning knob. Deep in a ranking a single lucky
    row can hit precision 1.0 on 1 prediction, and "θ for 95% precision" would
    then be chosen by noise; requiring a floor on predicted-positives makes the
    chosen point one the data can actually support.
    """

    kind: str
    value: float = 1.0
    min_support: int = 10

    @property
    def label(self) -> str:
        if self.kind == "fbeta":
            return f"max F{self.value:g}"
        if self.kind == "precision":
            return f"precision ≥ {self.value:g}"
        if self.kind == "recall":
            return f"recall ≥ {self.value:g}"
        raise ValueError(f"unknown rule kind: {self.kind}")

    @property
    def slug(self) -> str:
        return {"fbeta": f"f{self.value:g}",
                "precision": f"p{self.value:g}",
                "recall": f"r{self.value:g}"}[self.kind]


MAX_F1 = Rule("fbeta", 1.0)
MAX_F2 = Rule("fbeta", 2.0)
P95 = Rule("precision", 0.95)
DEFAULT_RULES = (P95, MAX_F1, MAX_F2)


def apply_rule(curve: PRCurve, rule: Rule) -> dict:
    """The operating point this rule selects, or NaNs with a reason if none exists.

    A rule failing is a result, not an error: "no θ makes this species 95% precise"
    is exactly what a researcher needs to know before trusting it in a survey, and
    is far more useful than a silently substituted nearest-miss.
    """
    miss = {"theta": float("nan"), "precision": float("nan"), "recall": float("nan"),
            "f1": float("nan"), "tp": 0, "fp": 0, "fn": curve.n_pos,
            "feasible": False}
    if len(curve) == 0 or curve.n_pos == 0:
        return {**miss, "reason": "no positives in the evaluation set"}

    eligible = (curve.tp + curve.fp) >= rule.min_support
    if rule.kind == "fbeta":
        scores = np.where(eligible, fbeta(curve.precision, curve.recall, rule.value), -1.0)
        if not np.any(eligible):
            return {**miss, "reason": f"no point predicts ≥ {rule.min_support} windows"}
        i = int(np.argmax(scores))
    elif rule.kind == "precision":
        ok = eligible & (curve.precision >= rule.value)
        if not np.any(ok):
            best = float(np.nanmax(curve.precision[eligible])) if np.any(eligible) else float("nan")
            return {**miss,
                    "reason": f"precision never reaches {rule.value:g} (best {best:.3f})"}
        # Recall rises with θ falling, so the deepest qualifying point is the one.
        i = int(np.nonzero(ok)[0][np.argmax(curve.recall[ok])])
    elif rule.kind == "recall":
        ok = eligible & (curve.recall >= rule.value)
        if not np.any(ok):
            best = float(np.nanmax(curve.recall[eligible])) if np.any(eligible) else float("nan")
            return {**miss,
                    "reason": f"recall never reaches {rule.value:g} (best {best:.3f})"}
        i = int(np.nonzero(ok)[0][np.argmax(curve.precision[ok])])
    else:
        raise ValueError(f"unknown rule kind: {rule.kind}")

    return {
        "theta": float(curve.theta[i]),
        "precision": float(curve.precision[i]),
        "recall": float(curve.recall[i]),
        "f1": float(fbeta(curve.precision[i], curve.recall[i], 1.0)),
        "tp": int(curve.tp[i]), "fp": int(curve.fp[i]), "fn": int(curve.fn[i]),
        "feasible": True, "reason": "",
    }


def thin(curve: PRCurve, n: int = 400) -> PRCurve:
    """Subsample for plotting. 24,000 points per species is a slow, identical picture."""
    if len(curve) <= n:
        return curve
    keep = np.unique(np.linspace(0, len(curve) - 1, n).astype(int))
    return PRCurve(curve.theta[keep], curve.precision[keep], curve.recall[keep],
                   curve.tp[keep], curve.fp[keep], curve.fn[keep],
                   curve.n_pos, curve.n_neg)


def curve_frame(curve: PRCurve, **extra) -> pl.DataFrame:
    """Curve as a frame, for charting and for export."""
    df = pl.DataFrame({
        "theta": curve.theta, "precision": curve.precision, "recall": curve.recall,
        "tp": curve.tp, "fp": curve.fp, "fn": curve.fn,
        "f1": fbeta(curve.precision, curve.recall, 1.0),
    })
    for k, v in extra.items():
        df = df.with_columns(pl.lit(v).alias(k))
    return df


# --------------------------------------------------------------------------- #
# Per-species and aggregate
# --------------------------------------------------------------------------- #

def species_curves(aligned: pl.DataFrame) -> dict[str, PRCurve]:
    """One curve per species, from the aligned (window × species) frame.

    Per-species is the unit that means something. A pooled curve over all species
    is dominated by whichever birds sing most — on SNE, a handful of them — so it
    measures the dawn chorus's composition as much as the model.
    """
    out = {}
    for (key,), grp in aligned.group_by(["species_key"], maintain_order=True):
        out[str(key)] = pr_curve(grp["y_true"].to_numpy(), grp["score"].to_numpy())
    return out


def per_species_table(
    aligned: pl.DataFrame,
    rules: tuple[Rule, ...] = DEFAULT_RULES,
    curves: dict[str, PRCurve] | None = None,
) -> pl.DataFrame:
    """The exportable scorecard: one row per species, AP plus each rule's θ.

    This is the table a researcher takes away — "for every species, the threshold
    that makes this model 95% precise, and what recall that leaves" — so the
    per-rule columns carry precision and recall alongside θ. A θ without the
    precision it actually achieved is an unfalsifiable number.
    """
    curves = curves if curves is not None else species_curves(aligned)
    rows = []
    for key, curve in curves.items():
        row = {
            "species_key": key,
            "n_positive": curve.n_pos,
            "n_windows": curve.n_pos + curve.n_neg,
            "prevalence": curve.prevalence,
            "ap": average_precision(curve),
        }
        for rule in rules:
            op = apply_rule(curve, rule)
            row[f"theta_{rule.slug}"] = op["theta"]
            row[f"precision_{rule.slug}"] = op["precision"]
            row[f"recall_{rule.slug}"] = op["recall"]
            # Every rule gets a feasibility flag, not just the precision floor: a
            # blank θ has two meanings — "no threshold satisfies this" and "too few
            # windows to say" — and a NaN alone in an exported CSV distinguishes
            # neither.
            row[f"feasible_{rule.slug}"] = op["feasible"]
        rows.append(row)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("ap", descending=True, nulls_last=True)


def cmap(per_species: pl.DataFrame, min_positives: int = 1) -> dict:
    """Class-mean average precision — AP averaged over species, unweighted.

    Unweighted is the point: cmAP asks "how well does this model do on a species,
    averaged over species", giving a scarce warbler the same vote as a robin that
    sings in a third of all windows. Micro-averaging answers a different and much
    easier question, and the two are reported side by side in `summary` so the gap
    is visible rather than a choice made silently.

    `min_positives` excludes species too rare in this eval set to have a stable AP,
    and the count of exclusions is returned — dropping species quietly is how a
    league table becomes a comparison between different species sets.
    """
    if per_species.is_empty():
        return {"cmap": float("nan"), "n_species": 0, "n_excluded": 0,
                "min_positives": min_positives}
    kept = per_species.filter(
        (pl.col("n_positive") >= min_positives) & pl.col("ap").is_not_nan())
    return {
        "cmap": float(kept["ap"].mean()) if len(kept) else float("nan"),
        "n_species": len(kept),
        "n_excluded": len(per_species) - len(kept),
        "min_positives": min_positives,
    }


def summary(
    aligned: pl.DataFrame,
    rules: tuple[Rule, ...] = DEFAULT_RULES,
    min_positives: int = 1,
    curves: dict[str, PRCurve] | None = None,
) -> dict:
    """One league row for one arm: cmAP, micro AP, and each rule's global θ.

    The global θ columns are what the app's threshold slider defaults to, and what
    a researcher gets if they refuse to tune per species. Comparing them with the
    per-species table is the argument for per-species thresholds, made in numbers.
    """
    curves = curves if curves is not None else species_curves(aligned)
    table = per_species_table(aligned, rules, curves)
    micro = pr_curve(aligned["y_true"].to_numpy(), aligned["score"].to_numpy())

    row = {**cmap(table, min_positives),
           "micro_ap": average_precision(micro),
           "n_windows": int(aligned.select(pl.struct("recording_id", "start_s")
                                           .n_unique()).item()),
           "n_pairs": len(aligned)}
    for rule in rules:
        op = apply_rule(micro, rule)
        row[f"theta_{rule.slug}"] = op["theta"]
        row[f"precision_{rule.slug}"] = op["precision"]
        row[f"recall_{rule.slug}"] = op["recall"]
    return row
