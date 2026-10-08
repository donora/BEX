"""Survey setups — a rule you have measured, carried to the recordings nobody
labelled (V1.2 F4, V1.1 P8).

The point of labelling a sample is not the scores themselves: it is to choose a
way of reading a model's output on the sample, measure how well it does there,
and then run it over everything else. A **setup** is that choice, frozen:

- the model it reads (by identity, so any run of the same model and readout can
  use it, on this dataset or another);
- the survey rule (firm / check conditions);
- each species' θ at the score bars the rule uses, as fitted when it was saved —
  so applying it never refits, and never needs labels;
- the judge and profile that decide what the filter hides;
- **what it measured**: the headline numbers and their 95% intervals, on which
  recordings, against which labels. That is the honest claim attached to every
  row it later produces from unlabelled audio.

Setups are JSON files in `<store>/setups/`, readable without BEX.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from . import labels, survey
from . import thresholds as th


@dataclass
class Setup:
    name: str
    model: str                    # thresholds.identity of the run it was tuned on
    model_label: str
    rule: dict                    # SurveyRule.to_dict()
    bars: dict                    # bar key -> {"table": [{species_key, theta}], "default_theta"}
    judge: str
    profile: str
    fitted_on: dict               # dataset, truth_ref, recordings (tuning)
    performance: dict             # score, intervals, recordings, hours, on
    created: str = ""
    created_by: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def survey_rule(self) -> survey.SurveyRule:
        def c(d):
            return survey.Condition(d["floor"], int(d["k"]), d.get("within_s"))
        return survey.SurveyRule(check=c(self.rule["check"]), firm=c(self.rule["firm"]))

    def resolved_bars(self) -> dict[str, th.Resolved]:
        out = {}
        for key, b in self.bars.items():
            table = pl.DataFrame(b["table"], schema={"species_key": pl.Utf8,
                                                     "theta": pl.Float64})
            out[key] = th.Resolved(self.model, table.with_columns(
                source=pl.lit("setup"), reason=pl.lit("")), b.get("default_theta"),
                "setup")
        return out

    def spans(self) -> tuple[float, ...]:
        return tuple(sorted({c["within_s"] for c in self.rule.values()
                             if c.get("within_s") is not None}))

    def claim(self) -> str:
        """One line saying what this setup's output can be trusted to mean."""
        p = self.performance
        s, iv = p.get("score", {}), p.get("intervals", {})

        def f(k):
            v = s.get(k)
            lo, hi = iv.get(k, (None, None))
            if v is None:
                return "—"
            band = (f" ({lo:.0%}–{hi:.0%})" if lo is not None and hi is not None
                    and lo == lo and hi == hi else "")
            return f"{v:.0%}{band}"
        return (f"{self.model_label}, rule: {self.survey_rule().describe()}. Measured "
                f"on {p.get('recordings', 0)} {p.get('on', 'labelled')} recording(s) "
                f"({p.get('hours', 0):.1f} h) against "
                f"{labels.describe_ref(self.fitted_on.get('truth_ref', ''))} of "
                f"{self.fitted_on.get('dataset', '?')}: firm calls right "
                f"{f('firm_precision')}, birds found after checking "
                f"{f('recall_reviewed')}, 95% intervals in brackets. It holds for "
                "other recordings only as far as they resemble those.")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=float) + "\n"


def bars_payload(bars: dict[str, th.Resolved], keys: set[str]) -> dict:
    return {k: {"table": bars[k].table.select("species_key", "theta")
                .filter(pl.col("theta").is_not_nan()).to_dicts(),
                "default_theta": bars[k].default_theta}
            for k in sorted(keys) if k in bars}


def setups_dir(store_dir: str | Path) -> Path:
    return Path(store_dir) / "setups"


def save(store_dir: str | Path, s: Setup) -> Path:
    if not labels.valid_name(s.name):
        raise ValueError(f"{s.name!r} is not a usable name (letters, digits, - _ .)")
    s.created = s.created or datetime.now(timezone.utc).isoformat(timespec="seconds")
    d = setups_dir(store_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{s.name}.json"
    p.write_text(s.to_json())
    return p


def load(store_dir: str | Path, name: str) -> Setup:
    raw = json.loads((setups_dir(store_dir) / f"{name}.json").read_text())
    return Setup(**raw)


def list_setups(store_dir: str | Path) -> list[str]:
    d = setups_dir(store_dir)
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


def apply(s: Setup, det: pl.DataFrame, recordings: list[str],
          truth: labels.Truth | None = None) -> pl.DataFrame:
    """Run a setup over recordings: one row per (recording, species) on a list.

    `det` is `survey.species_only` output (species detections with the judge's
    `implausible` flag), from a run of the setup's model. Where `truth` labels a
    recording end to end, its list is the **labels** — an expert's list beats a
    model's — with the model's tier alongside; elsewhere the list is the model's
    output, plus any species labelled in that recording's closed chunks.
    """
    ev = survey.evidence(det.filter(pl.col("recording_id").is_in(recordings)),
                         s.resolved_bars(), s.spans())
    tiers = survey.tiers(ev, s.survey_rule(), recordings).filter(
        pl.col("tier") != "not found").select("recording_id", "species_key",
                                              model_tier="tier",
                                              firm_bar_detections="firm_all",
                                              check_bar_detections="check_all")
    complete = set(truth.complete) if truth is not None else set()
    labelled = (truth.annotations.select("recording_id", "species_key").unique()
                if truth is not None else
                pl.DataFrame(schema={"recording_id": pl.Utf8, "species_key": pl.Utf8}))
    out = (tiers.join(labelled.with_columns(labelled=pl.lit(True)),
                      on=["recording_id", "species_key"], how="full", coalesce=True)
           .with_columns(labelled=pl.col("labelled").fill_null(False),
                         model_tier=pl.col("model_tier").fill_null("not found"),
                         firm_bar_detections=pl.col("firm_bar_detections").fill_null(0),
                         check_bar_detections=pl.col("check_bar_detections").fill_null(0))
           .filter(pl.col("recording_id").is_in(recordings)))
    is_complete = pl.col("recording_id").is_in(list(complete))
    out = (out.filter(~is_complete | pl.col("labelled"))
           .with_columns(
               source=pl.when(is_complete).then(pl.lit("labelled"))
               .when(pl.col("labelled")).then(pl.lit("labelled (part of recording)"))
               .otherwise(pl.lit("model output")),
               listed=pl.when(pl.col("labelled")).then(pl.lit("present"))
               .otherwise(pl.col("model_tier"))))
    return (out.select("recording_id", "species_key", "listed", "source", "model_tier",
                       "firm_bar_detections", "check_bar_detections")
            .sort("recording_id", "source", "species_key"))
