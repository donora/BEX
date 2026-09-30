"""The data contracts (PLAN.md §3a) — typed schemas, validators, run manifests.

Everything downstream — adapters, stats, the app — meets through three shapes:

- **recordings**: one row per audio file, from either ingest pipeline (labelled or
  unlabelled). Labelled data is just unlabelled data plus an annotations table.
- **detections**: long-form, one row per (window, species) worth indexing — *derived*
  from the full score store, regenerable at any time (resolved D3).
- **run manifest**: the provenance record of one inference run. BirdGate's hard
  lesson (its SUMMARY §12): manifests must record the config that *actually ran*,
  so `RunManifest` is constructed by the runner from the values it is about to use,
  and every stats table traces back to a `run_id`.

Validators fail loudly and specifically: a frame that doesn't conform names every
violated rule, because "schema drift discovered three stages later" is the failure
mode this module exists to prevent.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# --------------------------------------------------------------------------- #
# Frame schemas
# --------------------------------------------------------------------------- #

RECORDINGS_SCHEMA: dict[str, pl.DataType] = {
    "recording_id": pl.Utf8,     # unique; by convention the filename stem
    "path": pl.Utf8,             # relative to the dataset's audio directory
    "duration_s": pl.Float64,
    "sample_rate": pl.Int32,
    "channels": pl.Int32,
    "start_time": pl.Utf8,       # ISO 8601 as parsed from the filename. Naive =
                                 # recorder-local clock; 'Z' suffix = UTC (AudioMoth
                                 # hex names); "" = unknown. Honest about what a
                                 # filename can actually tell us.
    "site": pl.Utf8,
    "lat": pl.Float64,           # NaN when unknown
    "lon": pl.Float64,
    "labelled": pl.Boolean,      # does an annotations table join to this recording?
}

ANNOTATIONS_SCHEMA: dict[str, pl.DataType] = {
    "recording_id": pl.Utf8,
    "start_s": pl.Float64,       # the ground-truth box, in recording time
    "end_s": pl.Float64,
    "low_hz": pl.Float64,        # frequency extent; NaN when the source has none
    "high_hz": pl.Float64,
    "species_key": pl.Utf8,      # canonical — mapped through taxonomy, never raw
}

DETECTIONS_SCHEMA: dict[str, pl.DataType] = {
    "recording_id": pl.Utf8,
    "run_id": pl.Utf8,           # joins to the RunManifest
    "start_s": pl.Float64,       # the model's *native* window (resolved D5)
    "end_s": pl.Float64,
    "species_key": pl.Utf8,      # canonical scientific name — see taxonomy.py
    "score_raw": pl.Float32,     # model confidence in [0, 1], BEFORE any geofilter.
                                 # Adapters emitting logits apply their model's
                                 # squashing (recorded as score_transform).
    "occ_score": pl.Float32,     # geofilter occurrence score; NaN if the model has none
    "suppressed": pl.Boolean,    # would the model's own default pipeline hide this row?
    "rank_in_window": pl.Int32,  # 1 = top species in this window by score_raw
}


class SchemaError(ValueError):
    """A frame violated its contract. The message lists every violation found."""


def _check_columns(df: pl.DataFrame, schema: dict[str, pl.DataType], name: str) -> list[str]:
    problems = []
    missing = [c for c in schema if c not in df.columns]
    extra = [c for c in df.columns if c not in schema]
    if missing:
        problems.append(f"{name}: missing columns {missing}")
    if extra:
        problems.append(f"{name}: unexpected columns {extra}")
    for col, want in schema.items():
        if col in df.columns and df.schema[col] != want:
            problems.append(f"{name}.{col}: dtype {df.schema[col]}, expected {want}")
    return problems


def _no_nulls(df: pl.DataFrame, cols: list[str], name: str) -> list[str]:
    problems = []
    for col in cols:
        if col in df.columns and df[col].null_count() > 0:
            problems.append(f"{name}.{col}: {df[col].null_count()} null(s); nulls are "
                            "not allowed (use NaN/'' for 'unknown' where the schema says so)")
    return problems


def _raise_if(problems: list[str]) -> None:
    if problems:
        raise SchemaError("; ".join(problems))


def validate_recordings(df: pl.DataFrame) -> pl.DataFrame:
    problems = _check_columns(df, RECORDINGS_SCHEMA, "recordings")
    if not problems:
        problems += _no_nulls(df, ["recording_id", "path", "duration_s", "labelled"], "recordings")
        if df["recording_id"].n_unique() != len(df):
            problems.append("recordings.recording_id: values are not unique")
        if len(df) and (df["duration_s"] <= 0).any():
            problems.append("recordings.duration_s: non-positive duration(s)")
    _raise_if(problems)
    return df


def validate_annotations(df: pl.DataFrame) -> pl.DataFrame:
    problems = _check_columns(df, ANNOTATIONS_SCHEMA, "annotations")
    if not problems:
        problems += _no_nulls(
            df, ["recording_id", "start_s", "end_s", "species_key"], "annotations"
        )
        if len(df) and (df["end_s"] <= df["start_s"]).any():
            problems.append("annotations: end_s <= start_s in some rows")
    _raise_if(problems)
    return df


def validate_detections(df: pl.DataFrame) -> pl.DataFrame:
    problems = _check_columns(df, DETECTIONS_SCHEMA, "detections")
    if not problems:
        problems += _no_nulls(
            df,
            ["recording_id", "run_id", "start_s", "end_s", "species_key",
             "score_raw", "suppressed", "rank_in_window"],
            "detections",
        )
        if len(df):
            if (df["end_s"] <= df["start_s"]).any():
                problems.append("detections: end_s <= start_s in some rows")
            bad = df.filter((pl.col("score_raw") < 0) | (pl.col("score_raw") > 1))
            if len(bad):
                problems.append(
                    f"detections.score_raw: {len(bad)} value(s) outside [0, 1] — "
                    "adapters must apply their model's score transform"
                )
            if (df["rank_in_window"] < 1).any():
                problems.append("detections.rank_in_window: ranks start at 1")
    _raise_if(problems)
    return df


def empty_detections() -> pl.DataFrame:
    return pl.DataFrame(schema=DETECTIONS_SCHEMA)


# --------------------------------------------------------------------------- #
# Run manifest
# --------------------------------------------------------------------------- #

@dataclass
class RunManifest:
    """Provenance of one inference run — the config that *executed*, not a default.

    `geofilter` holds whatever parameters the model's location prior consumed
    (lat, lon, week, thresholds) or {} for models without one. `score_transform`
    names how raw model output became the [0,1] `score_raw` (e.g. "sigmoid",
    "identity"). `packages` pins what ran, because runner envs drift independently.
    """

    model_name: str
    model_version: str
    window_s: float
    hop_s: float
    input_sr: int
    score_transform: str
    geofilter: dict = field(default_factory=dict)
    profile: str = ""
    dataset: str = ""
    packages: dict = field(default_factory=dict)
    created_utc: str = ""
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.run_id:
            self.run_id = f"{self.model_name}-{uuid.uuid4().hex[:8]}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RunManifest":
        return cls(**json.loads(text))

    def save(self, store_dir: str | Path) -> Path:
        """Write to `<store_dir>/runs/<run_id>/manifest.json` (dirs created)."""
        out = Path(store_dir) / "runs" / self.run_id / "manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json())
        return out

    @classmethod
    def load(cls, store_dir: str | Path, run_id: str) -> "RunManifest":
        p = Path(store_dir) / "runs" / run_id / "manifest.json"
        return cls.from_json(p.read_text())
