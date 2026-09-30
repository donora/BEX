"""Which θ each model uses for each species — the rule, its fallbacks, and overrides.

Every view that draws a detection needs a threshold, and a single θ shared by all
models is not one: BirdNET's per-class sigmoid and Perch's softmax put the same
number at completely different operating points. So the app never asks for "θ".
It asks for a **rule** (max F1, a precision floor, max F2, or a fixed value per
model) and resolves it here into one θ per (model, species), in a fixed order:

    hand override  ->  the rule (fitted here, or a saved set's snapshot)
                   ->  the fallback (model-wide θ for the rule, or a fixed θ)
                   ->  no threshold

Each resolved θ carries the layer that set it, so the app can always say *why* a
species is thresholded where it is.

**Model identity, not run id.** A saved threshold means something only for the
model that produced the scores, so sets key their entries by
`model_name model_version readout` (`identity`). A re-run of the same model picks
the thresholds up; a new version or a different readout does not, and its entries
are reported as stale rather than silently applied.

**Threshold sets** are named JSON files in the configured thresholds directory:
the rule, the hand overrides, and a snapshot of every resolved θ. The snapshot is
what lets a set fitted on a labelled dataset threshold an unlabelled one.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from . import metrics
from .schemas import RunManifest
from .views import readout_token

#: Rule kinds, in the order the sidebar offers them; the first is the default.
RULE_KINDS = ("precision floor", "max F1", "max F2", "fixed")
FALLBACKS = ("model-wide", "fixed")
DEFAULT_FIXED = 0.10
FORMAT = "bex-thresholds/1"

#: Where each species' θ came from — also the app's display vocabulary.
SOURCES = ("override", "rule", "set", "fallback", "none")


def identity(manifest: RunManifest) -> str:
    """The model a threshold belongs to: name, version and readout.

    The readout is part of it because two Perch runs, one softmax and one sigmoid,
    share a name and version and have nothing else in common as far as θ goes.
    """
    token = readout_token(manifest.score_transform) or "unknown-readout"
    return f"{manifest.model_name} {manifest.model_version} {token}"


@dataclass(frozen=True)
class RuleSpec:
    """The global threshold rule — what the sidebar sets.

    `fixed` maps model identity to θ; it is the whole rule when `kind == "fixed"`
    and the fallback when `fallback == "fixed"`.

    `min_labelled` is the fewest labelled windows a species needs before a rule
    is fitted to it at all. Below that the curve cannot support an operating
    point, and max-Fβ in particular fails badly: with one labelled window every
    threshold above it scores F1 = 0, so the maximum always reaches down to that
    one window however deep it ranks — on SNE, a θ of 0.0005 that bought one
    goldfinch and 16,000 false alarms. Such species take the fallback instead. `min_positives` is not used to
    pick θ; it is carried here because it is part of the same experiment
    definition (the species cmAP counts) and belongs in a saved set with it.
    """

    kind: str = "precision floor"
    precision_floor: float = 0.95
    min_support: int = 10
    min_positives: int = 1
    fallback: str = "model-wide"
    fixed: tuple[tuple[str, float], ...] = ()
    min_labelled: int = 10

    def __post_init__(self) -> None:
        if self.kind not in RULE_KINDS:
            raise ValueError(f"unknown rule {self.kind!r} (one of {RULE_KINDS})")
        if self.fallback not in FALLBACKS:
            raise ValueError(f"unknown fallback {self.fallback!r} (one of {FALLBACKS})")

    def fixed_for(self, model: str) -> float:
        return dict(self.fixed).get(model, DEFAULT_FIXED)

    def metric_rule(self) -> metrics.Rule | None:
        """The curve rule this spec picks θ with; None for a fixed θ."""
        return {
            "max F1": metrics.Rule("fbeta", 1.0, self.min_support),
            "max F2": metrics.Rule("fbeta", 2.0, self.min_support),
            "precision floor": metrics.Rule("precision", self.precision_floor,
                                            self.min_support),
        }.get(self.kind)

    @property
    def label(self) -> str:
        if self.kind == "precision floor":
            return f"precision ≥ {self.precision_floor:g}"
        if self.kind == "fixed":
            return "fixed θ"
        return self.kind

    def describe(self) -> str:
        """The rule in plain words, with its fallback — what a reader needs to know
        before reading any number that depends on it."""
        what = {
            "precision floor": (
                f"each species gets the threshold that finds the most birds while "
                f"keeping at least {self.precision_floor:.0%} of its detections "
                f"correct — few false alarms, more birds missed"),
            "max F1": (
                "each species gets the threshold that best balances missed birds "
                "against false alarms, counting both the same"),
            "max F2": (
                "each species gets the threshold that best balances missed birds "
                "against false alarms, counting a missed bird as four times worse — "
                "more birds found, more false alarms"),
            "fixed": "every species of a model is thresholded at the same fixed θ, "
                     "set per model",
        }[self.kind]
        if self.kind == "fixed":
            return what + "."
        fb = ("the model's fixed θ" if self.fallback == "fixed"
              else "one model-wide threshold for the same rule")
        return (f"{what}. Species with fewer than {self.min_labelled} labelled "
                f"windows, or that cannot meet the rule, use {fb} instead.")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fixed"] = {k: v for k, v in self.fixed}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RuleSpec":
        d = dict(d)
        d["fixed"] = tuple(sorted((str(k), float(v))
                                  for k, v in (d.get("fixed") or {}).items()))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Resolved:
    """One model's thresholds: a θ per listed species, and one for everyone else.

    `table` has species_key, theta, source, reason. `default_theta` applies to
    every species not in the table — in practice the model's thousands of
    unannotated classes — and is NaN when there is no fallback to give them.
    """

    model: str
    table: pl.DataFrame
    default_theta: float
    default_source: str

    def theta_of(self, species_key: str) -> float:
        hit = self.table.filter(pl.col("species_key") == species_key)
        return float(hit["theta"][0]) if len(hit) else self.default_theta

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.table["species_key"], self.table["theta"]))


def resolve(
    model: str,
    spec: RuleSpec,
    curves: dict[str, metrics.PRCurve] | None = None,
    micro: metrics.PRCurve | None = None,
    snapshot: dict[str, float] | None = None,
    snapshot_default: float | None = None,
    overrides: dict[str, float] | None = None,
) -> Resolved:
    """Resolve one model's thresholds.

    Pass `curves` (and `micro`, the pooled curve) to fit the rule on this dataset,
    or `snapshot` (and `snapshot_default`) to take θ from a saved set instead —
    the route for data with no annotations. Passing both uses the snapshot.
    """
    overrides = overrides or {}
    rule = spec.metric_rule()
    use_snapshot = snapshot is not None

    # ---- the fallback: what a species gets when its own rule has no answer ---- #
    if spec.fallback == "fixed" or spec.kind == "fixed":
        fb_theta, fb_reason = spec.fixed_for(model), "fixed θ for this model"
    elif use_snapshot and snapshot_default is not None:
        fb_theta, fb_reason = snapshot_default, "model-wide θ from the saved set"
    elif micro is not None and rule is not None:
        op = metrics.apply_rule(micro, rule)
        fb_theta = op["theta"]
        fb_reason = (f"model-wide θ for {spec.label}" if op["feasible"]
                     else f"model-wide {spec.label}: {op['reason']}")
    else:
        fb_theta, fb_reason = float("nan"), "no labelled data to fit a fallback on"

    species = set(overrides)
    species |= set(snapshot or {}) if use_snapshot else set(curves or {})

    rows = []
    for sp in sorted(species):
        theta, source, reason = float("nan"), "none", ""
        if spec.kind == "fixed":
            theta, source, reason = spec.fixed_for(model), "rule", "fixed θ for this model"
        elif use_snapshot and sp in snapshot:
            theta, source, reason = snapshot[sp], "set", "from the saved set"
        elif not use_snapshot and curves and sp in curves:
            n_pos = curves[sp].n_pos
            op = (metrics.apply_rule(curves[sp], rule) if n_pos >= spec.min_labelled
                  else {"feasible": False,
                        "reason": f"only {n_pos} labelled window(s), fewer than the "
                                  f"{spec.min_labelled} needed to fit a θ"})
            if op["feasible"]:
                theta, source, reason = op["theta"], "rule", spec.label
            else:
                reason = op["reason"]
        if _nan(theta) and not _nan(fb_theta):
            theta, source = fb_theta, "fallback"
            reason = f"{reason} → {fb_reason}" if reason else fb_reason
        if sp in overrides:
            theta, source, reason = float(overrides[sp]), "override", "set by hand"
        rows.append({"species_key": sp, "theta": theta, "source": source,
                     "reason": reason})

    table = (pl.DataFrame(rows, schema={"species_key": pl.Utf8, "theta": pl.Float64,
                                        "source": pl.Utf8, "reason": pl.Utf8})
             if rows else pl.DataFrame(schema={"species_key": pl.Utf8,
                                               "theta": pl.Float64,
                                               "source": pl.Utf8, "reason": pl.Utf8}))
    return Resolved(model, table, fb_theta,
                    "fallback" if not _nan(fb_theta) else "none")


def _nan(x: float | None) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# --------------------------------------------------------------------------- #
# Applying thresholds to detections
# --------------------------------------------------------------------------- #

#: Where every species' own threshold sits in a score lane (E5): a fraction of the
#: lane's height, with the strongest detections at the top.
LANE_BASELINE = 0.15


def attach(det: pl.DataFrame, resolved: Resolved) -> pl.DataFrame:
    """Add `theta`, `theta_source` and `above` to a detections frame.

    A species with no threshold (NaN) is never `above`: the model has not been
    given an operating point for it, and inventing one is exactly what this
    module exists not to do.
    """
    lookup = resolved.table.select(
        "species_key", pl.col("theta").alias("_theta"), pl.col("source").alias("_src"))
    out = det.join(lookup, on="species_key", how="left").with_columns(
        theta=pl.col("_theta").fill_null(resolved.default_theta),
        theta_source=pl.col("_src").fill_null(resolved.default_source),
    ).drop("_theta", "_src")
    return out.with_columns(
        above=pl.col("theta").is_not_nan() & (pl.col("score_raw") >= pl.col("theta")))


def strength(det: pl.DataFrame, eps: float = 1e-6) -> pl.DataFrame:
    """Add `strength` in [LANE_BASELINE, 1] to above-threshold rows (null elsewhere).

    How far a detection clears *its own* threshold, as a percentile among all of
    this model's above-threshold detections. The margin is taken in log-odds, so
    a softmax readout whose scores all sit near zero ranks the same way a sigmoid
    spread over [0, 1] does. It is deliberately pictographic: heights compare
    detections within one model and say nothing across models.

    Call it on the model's whole detections frame, not a view of it, or a line's
    height would change as the viewer pans.
    """
    if "above" not in det.columns:
        raise ValueError("call attach() first")
    s = pl.col("score_raw").cast(pl.Float64).clip(eps, 1 - eps)
    t = pl.col("theta").cast(pl.Float64).clip(eps, 1 - eps)
    margin = (s / (1 - s)).log() - (t / (1 - t)).log()
    above = det.filter(pl.col("above"))
    if above.is_empty():
        return det.with_columns(strength=pl.lit(None, dtype=pl.Float64))
    n = len(above)
    ranked = det.with_columns(
        _m=pl.when(pl.col("above")).then(margin).otherwise(None))
    pct = (pl.col("_m").rank("average") - 1) / max(n - 1, 1)
    return ranked.with_columns(
        strength=LANE_BASELINE + (1 - LANE_BASELINE) * pct.fill_nan(1.0)
    ).drop("_m")


# --------------------------------------------------------------------------- #
# Named threshold sets
# --------------------------------------------------------------------------- #

@dataclass
class ThresholdSet:
    """A saved, named, shareable set of thresholds.

    `overrides` and `snapshot` are lists of {"model", "species_key", "theta", ...}
    rows keyed by model identity; `snapshot_defaults` holds each model's
    fallback θ, for the species the snapshot does not list.
    """

    name: str
    rule: dict
    overrides: list[dict] = field(default_factory=list)
    snapshot: list[dict] = field(default_factory=list)
    snapshot_defaults: list[dict] = field(default_factory=list)
    author: str = ""
    dataset: str = ""
    created_utc: str = ""
    updated_utc: str = ""
    format: str = FORMAT

    @property
    def spec(self) -> RuleSpec:
        return RuleSpec.from_dict(self.rule)

    def overrides_for(self, model: str) -> dict[str, float]:
        return {r["species_key"]: float(r["theta"])
                for r in self.overrides if r["model"] == model}

    def snapshot_for(self, model: str) -> tuple[dict[str, float] | None, float | None]:
        rows = {r["species_key"]: float(r["theta"])
                for r in self.snapshot if r["model"] == model}
        default = next((float(r["theta"]) for r in self.snapshot_defaults
                        if r["model"] == model), None)
        return (rows or None), default

    def models(self) -> set[str]:
        return ({r["model"] for r in self.overrides}
                | {r["model"] for r in self.snapshot}
                | {r["model"] for r in self.snapshot_defaults})

    def stale(self, present: set[str]) -> dict[str, int]:
        """Models this set has entries for that are not in `present`, with the
        number of entries each — reported to the user, never applied."""
        out: dict[str, int] = {}
        for r in self.overrides + self.snapshot:
            if r["model"] not in present:
                out[r["model"]] = out.get(r["model"], 0) + 1
        for r in self.snapshot_defaults:
            if r["model"] not in present:
                out.setdefault(r["model"], 0)
        return out

    def to_json(self) -> str:
        d = asdict(self)
        for r in d["snapshot"] + d["snapshot_defaults"] + d["overrides"]:
            if isinstance(r.get("theta"), float) and math.isnan(r["theta"]):
                r["theta"] = None
        return json.dumps(d, indent=2, ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "ThresholdSet":
        d = json.loads(text)
        if d.get("format") != FORMAT:
            raise ValueError(f"not a BEX threshold set (format {d.get('format')!r})")
        for key in ("snapshot", "snapshot_defaults", "overrides"):
            for r in d.get(key, []):
                if r.get("theta") is None:
                    r["theta"] = float("nan")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name.strip()).strip("-").lower()
    if not s:
        raise ValueError("a threshold set needs a name")
    return s


def set_path(directory: str | Path, name: str) -> Path:
    return Path(directory) / f"{slug(name)}.json"


def list_sets(directory: str | Path) -> list[str]:
    """Names of every readable set in the directory, sorted."""
    d = Path(directory)
    if not d.exists():
        return []
    names = []
    for p in sorted(d.glob("*.json")):
        try:
            names.append(ThresholdSet.from_json(p.read_text()).name)
        except (ValueError, KeyError, TypeError):
            continue  # not ours, or damaged: never offered, never overwritten
    return sorted(names, key=str.lower)


def load_set(directory: str | Path, name: str) -> ThresholdSet:
    return ThresholdSet.from_json(set_path(directory, name).read_text())


def save_set(directory: str | Path, ts: ThresholdSet) -> Path:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ts.created_utc = ts.created_utc or now
    ts.updated_utc = now
    p = set_path(directory, ts.name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(ts.to_json())
    return p


def delete_set(directory: str | Path, name: str) -> None:
    set_path(directory, name).unlink(missing_ok=True)


def build_set(
    name: str,
    spec: RuleSpec,
    resolved: list[Resolved],
    overrides: dict[tuple[str, str], float],
    author: str = "",
    dataset: str = "",
    created_utc: str = "",
) -> ThresholdSet:
    """Freeze the current state — rule, overrides, every resolved θ — into a set."""
    snap, defaults = [], []
    for r in resolved:
        for row in r.table.iter_rows(named=True):
            snap.append({"model": r.model, "species_key": row["species_key"],
                         "theta": row["theta"], "source": row["source"]})
        defaults.append({"model": r.model, "theta": r.default_theta,
                         "source": r.default_source})
    ov = [{"model": m, "species_key": sp, "theta": float(t)}
          for (m, sp), t in sorted(overrides.items())]
    return ThresholdSet(name=name, rule=spec.to_dict(), overrides=ov, snapshot=snap,
                        snapshot_defaults=defaults, author=author, dataset=dataset,
                        created_utc=created_utc)


def point_at(curve: metrics.PRCurve, theta: float) -> dict:
    """Precision and recall a curve gives at an arbitrary θ (not only its own points)."""
    if len(curve) == 0 or _nan(theta):
        return {"precision": float("nan"), "recall": float("nan"), "tp": 0, "fp": 0}
    # theta is descending; the operating point is the last one still >= θ.
    k = int(np.searchsorted(-curve.theta, -theta, side="right")) - 1
    if k < 0:
        return {"precision": float("nan"), "recall": 0.0, "tp": 0, "fp": 0}
    return {"precision": float(curve.precision[k]), "recall": float(curve.recall[k]),
            "tp": int(curve.tp[k]), "fp": int(curve.fp[k])}
