"""Bring your own model: import its detections as a BEX run.

Any model works, run however you like, as long as it writes one table (CSV or
Parquet) of what it heard. BEX turns that into a run like its own (the same
detections index, plus a manifest), so the guided path, the recall slider and
EcoLabel all treat it like Perch or BirdNET. `HOWTO` is the guide the app
shows; `bex import-detections` is the command-line way in.

(For a model BEX should run itself, every time, write a runner instead:
envs/README.md.)
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import polars as pl

from .schemas import DETECTIONS_SCHEMA, RunManifest
from .store import write_detections
from .taxonomy import is_species, normalise_key

#: Accepted spellings of each column (case-insensitive), first is the documented one.
COLUMNS = {
    "recording": ["recording", "recording_id", "file", "filename", "file_name", "path", "begin path"],
    "start_s": ["start_s", "start", "start (s)", "begin_time", "begin time (s)", "start_time"],
    "end_s": ["end_s", "end", "end (s)", "end_time", "end time (s)"],
    "species": ["species", "scientific_name", "scientific name", "species_key", "label", "class"],
    "score": ["score", "confidence", "probability", "prob"],
}
OPTIONAL = {"end_s"}

HOWTO = """
**Use any model** (your own, a colleague's, one from a paper) by importing what it
detected. BEX then treats it like Perch or BirdNET: the species list, checking,
the recall slider and EcoLabel all work the same.

**1. Run your model over the recordings**, however you normally do. Give it
short windows (3 or 5 seconds is typical; overlapping is fine).

**2. Save one table** (CSV, or Parquet for big ones) with **one row per window
per species**, and these columns:

| column | what it is | example |
|---|---|---|
| `recording` | the recording's file name (with or without `.wav`/`.flac`), as in step 1 | `SNE_004_20180509_080004.flac` |
| `start_s` | where the window starts, in seconds from the start of the file | `183.0` |
| `end_s` | where it ends (or set the window length when importing) | `186.0` |
| `species` | the **scientific name** | `Turdus migratorius` |
| `score` | how sure the model is, **from 0 to 1** | `0.87` |

- **Scores must be 0 to 1** and higher must mean surer. If your model gives
  logits, apply a sigmoid (one label per species) or a softmax (one label per
  window) first. They don't need to be calibrated probabilities: your
  checks in step 4 calibrate them.
- **Include more than the top species.** The guided path takes each window's
  top 3, so give at least that many per window (all species is fine too).
  Rows below about 0.01 can be left out to keep the file small.
- **Scientific names**, not codes: `Turdus migratorius`, not `amerob`. Common
  names go in the Species step. Non-species labels (`Noise`, `Dog`) are fine;
  they're kept but never sent to volunteers.
- Common column names from other tools are understood too: BirdNET-style
  `Begin Time (s)`, `Confidence`, `file`, `label`.

**3. Import it** here (or `bex import-detections <project> <file> --model <name>`).
BEX checks every row and tells you exactly what's wrong if something is.

**Licence**: you're responsible for being allowed to use the model's output.
For EcoLabel (a paid service) that means a model whose licence allows
commercial use.
"""


def template(recording_ids: list[str], n: int = 3) -> bytes:
    """A small example CSV in the right shape, using the project's own recordings."""
    rows = []
    for rec in recording_ids[:2]:
        for w in range(2):
            for sp, sc in (("Turdus migratorius", 0.91), ("Catharus guttatus", 0.12), ("Noise", 0.04))[:n]:
                rows.append({"recording": rec, "start_s": 3.0 * w, "end_s": 3.0 * w + 3, "species": sp,
                             "score": round(sc * (1 - 0.3 * w), 3)})
    return pl.DataFrame(rows).write_csv().encode()


def read_table(src) -> pl.DataFrame:
    """A CSV or Parquet file: a path, or bytes / a file-like object (an upload)."""
    if isinstance(src, (str, Path)):
        p = Path(src)
        return pl.read_parquet(p) if p.suffix.lower() == ".parquet" else pl.read_csv(p, infer_schema_length=10000)
    raw = src if isinstance(src, bytes) else src.read()
    if raw[:4] == b"PAR1":
        return pl.read_parquet(io.BytesIO(raw))
    return pl.read_csv(io.BytesIO(raw), infer_schema_length=10000)


class ImportProblem(ValueError):
    """Everything wrong with a table, said so a person can fix it."""


def _columns(df: pl.DataFrame) -> dict[str, str]:
    lower = {c.lower().strip(): c for c in df.columns}
    found, missing = {}, []
    for want, names in COLUMNS.items():
        hit = next((lower[n] for n in names if n in lower), None)
        if hit:
            found[want] = hit
        elif want not in OPTIONAL:
            missing.append(f"`{want}` (or one of: {', '.join(names[1:4])})")
    if missing:
        raise ImportProblem(f"Missing column{'s' if len(missing) > 1 else ''}: {'; '.join(missing)}. "
                            f"The table has: {', '.join(df.columns)}.")
    return found


def _stem(x: str) -> str:
    name = re.split(r"[\\/]", x.strip())[-1]
    return re.sub(r"\.(wav|flac|mp3|ogg|aiff?|m4a)$", "", name, flags=re.I)


def prepare(df: pl.DataFrame, recordings: pl.DataFrame, run_id: str, window_s: float | None = None) -> tuple[pl.DataFrame, dict]:
    """Check and convert a table to BEX detections. Raises ImportProblem with
    every problem found; returns (detections, a report)."""
    col = _columns(df)
    problems = []
    t = df.select(
        recording=pl.col(col["recording"]).cast(pl.Utf8),
        start_s=pl.col(col["start_s"]).cast(pl.Float64, strict=False),
        end_s=(pl.col(col["end_s"]).cast(pl.Float64, strict=False) if "end_s" in col
               else pl.col(col["start_s"]).cast(pl.Float64, strict=False) + (window_s or 0)),
        species=pl.col(col["species"]).cast(pl.Utf8),
        score=pl.col(col["score"]).cast(pl.Float64, strict=False))
    if "end_s" not in col and not window_s:
        problems.append("There's no end column: set the window length (seconds) when importing.")
    bad = t.filter(pl.any_horizontal(pl.all().is_null()))
    if bad.height:
        problems.append(f"{bad.height:,} rows have an empty or non-numeric value (first: {bad.row(0)}).")
        t = t.drop_nulls()
    # Recordings: by recording id, or file name with or without extension or folders.
    ids = recordings["recording_id"].to_list()
    by = {r: r for r in ids} | {_stem(p): r for r, p in zip(ids, recordings["path"])}
    t = t.with_columns(recording_id=pl.col("recording").map_elements(lambda x: by.get(_stem(x)), return_dtype=pl.Utf8))
    unknown = t.filter(pl.col("recording_id").is_null())["recording"].unique().to_list()
    if unknown:
        problems.append(f"{len(unknown)} recording name{'s' if len(unknown) > 1 else ''} not in this project, "
                        f"e.g. {', '.join(unknown[:3])}. Names should match the files (this project has, "
                        f"e.g., {', '.join(recordings['path'].head(2).to_list())}).")
    lo, hi = t["score"].min(), t["score"].max()
    if t.height and (lo < 0 or hi > 1):
        problems.append(f"Scores run from {lo:.3g} to {hi:.3g}, but must be 0 to 1. If they're logits, apply a "
                        "sigmoid (one label per species) or softmax (one label per window) first.")
    if t.height and ((t["end_s"] - t["start_s"]) <= 0).any():
        problems.append("Some rows end before they start (end_s must be after start_s).")
    dur = dict(zip(recordings["recording_id"], recordings["duration_s"]))
    over = t.filter(pl.col("recording_id").is_not_null()).filter(
        pl.col("start_s") > pl.col("recording_id").replace_strict(dur, default=None) + 1)
    if over.height:
        problems.append(f"{over.height:,} windows start after their recording ends (are times in seconds?).")
    if problems:
        raise ImportProblem("\n".join(f"- {p}" for p in problems))

    # Species: one spelling; "Genus species_Common Name" (BirdNET labels) keeps the first part.
    t = t.with_columns(species_key=pl.col("species").map_elements(
        lambda s: normalise_key(s.split("_")[0]) if is_species(normalise_key(s.split("_")[0])) else s.strip(),
        return_dtype=pl.Utf8))
    t = (t.group_by("recording_id", "start_s", "end_s", "species_key").agg(pl.col("score").max())
         .with_columns(rank_in_window=pl.col("score").rank("ordinal", descending=True)
                       .over("recording_id", "start_s").cast(pl.Int32)))
    det = t.select(
        "recording_id", run_id=pl.lit(run_id), start_s="start_s", end_s="end_s", species_key="species_key",
        score_raw=pl.col("score").cast(pl.Float32), occ_score=pl.lit(float("nan"), pl.Float32),
        suppressed=pl.lit(False), rank_in_window="rank_in_window",
    ).sort("recording_id", "start_s", "rank_in_window").select(list(DETECTIONS_SCHEMA))
    per_window = det.group_by("recording_id", "start_s").len()["len"]
    starts = np.sort(det.filter(pl.col("recording_id") == det["recording_id"][0])["start_s"].unique().to_numpy())
    species = det["species_key"].unique().to_list()
    report = {
        "rows": det.height, "recordings": det["recording_id"].n_unique(), "of_recordings": len(ids),
        "species": sum(is_species(s) for s in species), "other_labels": sum(not is_species(s) for s in species),
        "window_s": float((det["end_s"] - det["start_s"]).median()),
        "hop_s": float(np.median(np.diff(starts))) if len(starts) > 1 else 0.0,
        "per_window": float(per_window.median()),
        "warnings": [],
    }
    if report["per_window"] < 3:
        report["warnings"].append(f"Most windows have only {report['per_window']:.0f} species: the guided path "
                                  "takes each window's top 3, so export more per window if you can.")
    if report["recordings"] < len(ids):
        report["warnings"].append(f"Only {report['recordings']} of {len(ids)} recordings have detections.")
    if report["species"] == 0:
        report["warnings"].append("No scientific names recognised (e.g. `Turdus migratorius`): the species "
                                  "step needs them.")
    return det, report


def import_detections(store_dir, dataset: str, table: pl.DataFrame, model_name: str, model_version: str = "",
                      window_s: float | None = None, note: str = "") -> dict:
    """Import a table as a new run over `dataset`. Returns the report, with run_id."""
    from .ingest import read_dataset
    name = re.sub(r"[^a-z0-9]+", "", model_name.lower()) or "own"
    recs, _ = read_dataset(store_dir, dataset)
    man = RunManifest(model_name=name, model_version=model_version or "1", window_s=0.0, hop_s=0.0, input_sr=0,
                      score_transform="supplied with the imported table" + (f": {note}" if note else ""),
                      dataset=dataset, packages={"imported": True})
    det, report = prepare(table, recs, man.run_id, window_s)
    man.window_s, man.hop_s = report["window_s"], report["hop_s"]
    write_detections(store_dir, man.run_id, det)
    man.save(store_dir)
    return {**report, "run_id": man.run_id, "model_name": name}
