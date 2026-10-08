"""Label sets — ground truth you make yourself, stored the way SNE's is (V1.2 L6).

A **label set** is a named folder of labels for one dataset:

    store/datasets/<dataset>/labels/<set>/
        annotations.parquet   one row per box: ANNOTATIONS_SCHEMA (the same table
                              SNE's annotations.csv becomes) plus box_id,
                              labeller and labelled_at, which anything reading
                              only the standard columns ignores
        chunks.parquet        one row per 60 s chunk anyone has touched: open, or
                              closed (signed off: every bird heard is labelled)
        sample.parquet        the random sample of chunks to label, in order
        meta.json             derived_from, created, sample settings
        v1/, v2/ …            published versions: frozen copies of the above

**Only closed chunks are ground truth.** A box says a bird is there; a closed
chunk says every bird heard there has a box, which is what turns "no box" into
"no bird". Without it, an unlabelled stretch would score every detection in it
as wrong — PLAN.md Stage 6.5, and the reason SNE's format alone is not enough.

The dataset's own `annotations.parquet` (SNE's, imported) is the read-only
reference set **imported**: exhaustive, so every chunk of a labelled recording
counts as closed. Adding to it happens in a new set derived from it.

A set is referred to by a **ref**: `imported`, `<set>@v<N>` (a published
version), or `<set>@w<hash>` (the working copy, as it was when hashed). Every
cache built on truth is keyed on the ref, so a cache can never outlive the
labels it was built from.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from .schemas import ANNOTATIONS_SCHEMA, validate_annotations

#: The unit of sign-off (Stage 6.5). Chunks start at 0, 60, 120 … s; the last one
#: in a recording is shorter.
CHUNK_S = 60.0

#: The dataset's own annotations, read-only.
IMPORTED = "imported"

#: A bird heard but not identified (L5). It is kept in the set and counted, and
#: left out of species-level scoring: no model has a class for it.
UNKNOWN_BIRD = "unknown bird"

OPEN, CLOSED = "open", "closed"

BOXES_SCHEMA: dict[str, pl.DataType] = {
    **ANNOTATIONS_SCHEMA,
    "box_id": pl.Utf8,
    "labeller": pl.Utf8,
    "labelled_at": pl.Utf8,
    "assisted": pl.Boolean,      # the models' suggestions were viewed for this box
}

CHUNKS_SCHEMA: dict[str, pl.DataType] = {
    "recording_id": pl.Utf8,
    "start_s": pl.Float64,
    "end_s": pl.Float64,
    "state": pl.Utf8,          # open | closed
    "closed_by": pl.Utf8,
    "closed_at": pl.Utf8,
    "reopen_reason": pl.Utf8,  # why it was last reopened, if it was
}

SAMPLE_SCHEMA: dict[str, pl.DataType] = {
    "recording_id": pl.Utf8,
    "start_s": pl.Float64,
    "end_s": pl.Float64,
    "order": pl.Int32,         # the order to label in; any prefix is balanced
    "stratum": pl.Utf8,        # site · time of day
    "whole": pl.Boolean,       # part of a recording sampled whole (for P2 units)
}

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")


class StaleWrite(RuntimeError):
    """The set changed on disk since this copy was loaded (another tab, another
    labeller). The write is refused; reload and redo the change."""


class ClosedChunk(ValueError):
    """Boxes in a closed chunk cannot change until it is reopened."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty(schema: dict) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #

def chunk_starts(duration_s: float) -> np.ndarray:
    """Every chunk's start in a recording. A tail under 1 s is folded into the
    chunk before it rather than left as a chunk nobody could label."""
    n = max(1, math.ceil(max(duration_s - 1.0, 0.0) / CHUNK_S))
    return np.arange(n, dtype=np.float64) * CHUNK_S


def chunk_end(start_s: float, duration_s: float) -> float:
    starts = chunk_starts(duration_s)
    later = starts[starts > start_s + 1e-6]
    return float(later[0]) if len(later) else float(duration_s)


def chunk_of(t: float) -> float:
    """The start of the chunk containing time t."""
    return float(math.floor(max(t, 0.0) / CHUNK_S) * CHUNK_S)


def all_chunks(recordings: pl.DataFrame) -> pl.DataFrame:
    """Every chunk of every recording: recording_id, start_s, end_s, site, start_time."""
    rows = []
    for r in recordings.select("recording_id", "duration_s", "site",
                               "start_time").iter_rows(named=True):
        starts = chunk_starts(r["duration_s"])
        ends = np.append(starts[1:], r["duration_s"])
        for s, e in zip(starts, ends):
            rows.append({"recording_id": r["recording_id"], "start_s": float(s),
                         "end_s": float(e), "site": r["site"] or "",
                         "start_time": r["start_time"] or ""})
    return pl.DataFrame(rows, schema={"recording_id": pl.Utf8, "start_s": pl.Float64,
                                      "end_s": pl.Float64, "site": pl.Utf8,
                                      "start_time": pl.Utf8})


# --------------------------------------------------------------------------- #
# The set, in memory
# --------------------------------------------------------------------------- #

@dataclass
class LabelSet:
    dataset: str
    name: str
    boxes: pl.DataFrame
    chunks: pl.DataFrame
    sample: pl.DataFrame
    meta: dict = field(default_factory=dict)
    version: int | None = None   # None = the working copy
    stamp: str = ""              # content hash when loaded, for StaleWrite

    @property
    def read_only(self) -> bool:
        return self.version is not None

    def chunk_state(self, recording_id: str, start_s: float) -> str:
        hit = self.chunks.filter((pl.col("recording_id") == recording_id)
                                 & ((pl.col("start_s") - start_s).abs() < 1e-6))
        return hit["state"][0] if len(hit) else OPEN

    def chunk_boxes(self, recording_id: str, start_s: float, end_s: float) -> pl.DataFrame:
        """Every box that overlaps the chunk — including one that started in the
        chunk before and runs into this one."""
        return self.boxes.filter((pl.col("recording_id") == recording_id)
                                 & (pl.col("end_s") > start_s + 1e-6)
                                 & (pl.col("start_s") < end_s - 1e-6)).sort("start_s")


def chunks_touched(start_s: float, end_s: float) -> list[float]:
    """The starts of every chunk a box overlaps: a call can cross a minute."""
    first, last = chunk_of(start_s), chunk_of(max(start_s, end_s - 1e-6))
    return [float(c) for c in np.arange(first, last + 1e-6, CHUNK_S)]


def _check_open(ls: LabelSet, recording_id: str, t: float,
                end: float | None = None) -> None:
    """A box can only change while every chunk it touches is open, so editing
    one can never quietly change a minute someone has signed off."""
    if ls.read_only:
        raise ClosedChunk(f"{ls.name} v{ls.version} is a published version; "
                          "only the working copy can change")
    for c in chunks_touched(t, t if end is None else end):
        if ls.chunk_state(recording_id, c) == CLOSED:
            raise ClosedChunk(f"the chunk at {c:g} s of {recording_id} is closed; "
                              "reopen it to change a box that touches it")


def add_box(ls: LabelSet, recording_id: str, start_s: float, end_s: float,
            low_hz: float, high_hz: float, species_key: str, labeller: str,
            assisted: bool = False) -> str:
    """A new box. It may cross a minute boundary (a song does not stop for
    the chunk grid); every chunk it touches must be open."""
    _check_open(ls, recording_id, start_s, end_s)
    if end_s <= start_s:
        raise ValueError("a box needs end > start")
    if not species_key:
        raise ValueError("a box needs a species (or 'unknown bird')")
    lo, hi = sorted((float(low_hz), float(high_hz)))
    box_id = uuid.uuid4().hex[:12]
    row = pl.DataFrame([{"recording_id": recording_id, "start_s": float(start_s),
                         "end_s": float(end_s), "low_hz": lo, "high_hz": hi,
                         "species_key": species_key, "box_id": box_id,
                         "labeller": labeller, "labelled_at": _now(),
                         "assisted": bool(assisted)}],
                       schema=BOXES_SCHEMA)
    ls.boxes = pl.concat([ls.boxes, row])
    _touch_chunk(ls, recording_id, chunk_of(start_s))
    return box_id


def update_box(ls: LabelSet, box_id: str, labeller: str, **changes) -> None:
    """Change a box's species, times or frequencies. Moving it into another
    chunk needs both chunks open."""
    hit = ls.boxes.filter(pl.col("box_id") == box_id)
    if not len(hit):
        raise KeyError(box_id)
    old = hit.row(0, named=True)
    _check_open(ls, old["recording_id"], old["start_s"], old["end_s"])
    new = {**old, **{k: v for k, v in changes.items() if v is not None}}
    _check_open(ls, new["recording_id"], new["start_s"], new["end_s"])
    if new["end_s"] <= new["start_s"]:
        raise ValueError("a box needs end > start")
    new["low_hz"], new["high_hz"] = sorted((float(new["low_hz"]), float(new["high_hz"])))
    new["labeller"], new["labelled_at"] = labeller, _now()
    ls.boxes = pl.concat([ls.boxes.filter(pl.col("box_id") != box_id),
                          pl.DataFrame([new], schema=BOXES_SCHEMA)])


def delete_box(ls: LabelSet, box_id: str) -> None:
    hit = ls.boxes.filter(pl.col("box_id") == box_id)
    if not len(hit):
        raise KeyError(box_id)
    _check_open(ls, hit["recording_id"][0], hit["start_s"][0], hit["end_s"][0])
    ls.boxes = ls.boxes.filter(pl.col("box_id") != box_id)


def _touch_chunk(ls: LabelSet, recording_id: str, start_s: float,
                 end_s: float | None = None) -> None:
    """Make sure the chunk has a row (open) — it has been worked on."""
    exists = ls.chunks.filter((pl.col("recording_id") == recording_id)
                              & ((pl.col("start_s") - start_s).abs() < 1e-6))
    if len(exists):
        return
    row = pl.DataFrame([{"recording_id": recording_id, "start_s": start_s,
                         "end_s": end_s if end_s is not None else start_s + CHUNK_S,
                         "state": OPEN, "closed_by": "", "closed_at": "",
                         "reopen_reason": ""}], schema=CHUNKS_SCHEMA)
    ls.chunks = pl.concat([ls.chunks, row])


def _set_chunk(ls: LabelSet, recording_id: str, start_s: float, **values) -> None:
    ls.chunks = ls.chunks.with_columns([
        pl.when((pl.col("recording_id") == recording_id)
                & ((pl.col("start_s") - start_s).abs() < 1e-6))
          .then(pl.lit(v)).otherwise(pl.col(k)).alias(k)
        for k, v in values.items()])


def close_chunk(ls: LabelSet, recording_id: str, start_s: float, end_s: float,
                labeller: str) -> None:
    """Sign a chunk off: every bird heard in it is labelled. A closed chunk with
    no boxes means *listened, no birds* — which is truth too."""
    if ls.read_only:
        raise ClosedChunk("published versions cannot change")
    if not labeller:
        raise ValueError("closing a chunk needs the labeller's name")
    _touch_chunk(ls, recording_id, start_s, end_s)
    _set_chunk(ls, recording_id, start_s, state=CLOSED, closed_by=labeller,
               closed_at=_now(), end_s=float(end_s))


def reopen_chunk(ls: LabelSet, recording_id: str, start_s: float, reason: str = "") -> None:
    if ls.read_only:
        raise ClosedChunk("published versions cannot change")
    _set_chunk(ls, recording_id, start_s, state=OPEN, closed_by="", closed_at="",
               reopen_reason=reason or "reopened")


# --------------------------------------------------------------------------- #
# On disk
# --------------------------------------------------------------------------- #

def labels_dir(store_dir: str | Path, dataset: str) -> Path:
    return Path(store_dir) / "datasets" / dataset / "labels"


def set_dir(store_dir: str | Path, dataset: str, name: str) -> Path:
    return labels_dir(store_dir, dataset) / name


def has_imported(store_dir: str | Path, dataset: str) -> bool:
    return (Path(store_dir) / "datasets" / dataset / "annotations.parquet").exists()


def list_sets(store_dir: str | Path, dataset: str) -> list[str]:
    d = labels_dir(store_dir, dataset)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if (p / "meta.json").exists())


def versions(store_dir: str | Path, dataset: str, name: str) -> list[int]:
    d = set_dir(store_dir, dataset, name)
    return sorted(int(p.name[1:]) for p in d.glob("v*")
                  if p.is_dir() and p.name[1:].isdigit())


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name)) and name != IMPORTED


_FILES = ("annotations.parquet", "chunks.parquet", "sample.parquet")


def _stamp(d: Path) -> str:
    h = hashlib.sha1()
    for f in _FILES:
        p = d / f
        h.update(p.read_bytes() if p.exists() else b"-")
    return h.hexdigest()


def stamp(store_dir: str | Path, dataset: str, name: str) -> str:
    return _stamp(set_dir(store_dir, dataset, name))


def _read(d: Path, file: str, schema: dict) -> pl.DataFrame:
    p = d / file
    if not p.exists():
        return _empty(schema)
    df = pl.read_parquet(p)
    for col, dtype in schema.items():
        if col not in df.columns:
            # A column added since the set was written (e.g. `assisted`): false.
            df = df.with_columns(pl.lit(False if dtype == pl.Boolean else None,
                                        dtype=dtype).alias(col))
    return df.select(list(schema)).cast(schema)


def load_set(store_dir: str | Path, dataset: str, name: str,
             version: int | None = None) -> LabelSet:
    base = set_dir(store_dir, dataset, name)
    if not (base / "meta.json").exists():
        raise FileNotFoundError(f"no label set {name!r} in dataset {dataset!r}")
    d = base if version is None else base / f"v{version}"
    if not d.exists():
        raise FileNotFoundError(f"{name!r} has no version {version}")
    return LabelSet(dataset, name,
                    boxes=_read(d, "annotations.parquet", BOXES_SCHEMA),
                    chunks=_read(d, "chunks.parquet", CHUNKS_SCHEMA),
                    sample=_read(d, "sample.parquet", SAMPLE_SCHEMA),
                    meta=json.loads((base / "meta.json").read_text()),
                    version=version, stamp=_stamp(d))


def _atomic_write(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def save_set(store_dir: str | Path, ls: LabelSet) -> str:
    """Write the working copy. Refused (StaleWrite) when the files changed since
    `ls` was loaded — two tabs cannot overwrite each other. Returns the new stamp."""
    if ls.read_only:
        raise ClosedChunk("published versions cannot change")
    d = set_dir(store_dir, ls.dataset, ls.name)
    if ls.stamp and _stamp(d) != ls.stamp:
        raise StaleWrite(f"label set {ls.name!r} changed since it was loaded")
    validate_annotations(ls.boxes.select(list(ANNOTATIONS_SCHEMA)))
    _atomic_write(ls.boxes.sort("recording_id", "start_s"), d / "annotations.parquet")
    _atomic_write(ls.chunks.sort("recording_id", "start_s"), d / "chunks.parquet")
    _atomic_write(ls.sample.sort("order"), d / "sample.parquet")
    ls.stamp = _stamp(d)
    return ls.stamp


def create_set(store_dir: str | Path, dataset: str, name: str, created_by: str,
               derived_from: str | None = None) -> LabelSet:
    """A new working set: empty, or a copy of the imported annotations or of
    another set's working copy, recording what it came from."""
    if not valid_name(name):
        raise ValueError(f"{name!r} is not a usable set name (letters, digits, - _ .; "
                         f"not {IMPORTED!r})")
    d = set_dir(store_dir, dataset, name)
    if d.exists():
        raise FileExistsError(f"label set {name!r} already exists")
    boxes, chunks = _empty(BOXES_SCHEMA), _empty(CHUNKS_SCHEMA)
    sample = _empty(SAMPLE_SCHEMA)
    if derived_from == IMPORTED:
        boxes, chunks = _from_imported(store_dir, dataset)
    elif derived_from:
        src = load_set(store_dir, dataset, derived_from)
        boxes, chunks, sample = src.boxes, src.chunks, src.sample
    d.mkdir(parents=True)
    meta = {"dataset": dataset, "name": name, "created": _now(),
            "created_by": created_by, "derived_from": derived_from or ""}
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    ls = LabelSet(dataset, name, boxes, chunks, sample, meta)
    save_set(store_dir, ls)
    return ls


def _from_imported(store_dir, dataset) -> tuple[pl.DataFrame, pl.DataFrame]:
    from .ingest import read_dataset
    recs, ann = read_dataset(store_dir, dataset)
    ann = ann if ann is not None else _empty(ANNOTATIONS_SCHEMA)
    boxes = ann.with_columns(
        box_id=pl.Series([uuid.uuid4().hex[:12] for _ in range(len(ann))], dtype=pl.Utf8),
        labeller=pl.lit("imported"), labelled_at=pl.lit(""), assisted=pl.lit(False))
    whole = all_chunks(recs.filter(pl.col("labelled")))
    chunks = whole.select("recording_id", "start_s", "end_s").with_columns(
        state=pl.lit(CLOSED), closed_by=pl.lit("imported"), closed_at=pl.lit(""),
        reopen_reason=pl.lit(""))
    return boxes.select(list(BOXES_SCHEMA)).cast(BOXES_SCHEMA), chunks.cast(CHUNKS_SCHEMA)


def publish(store_dir: str | Path, ls: LabelSet) -> int:
    """Freeze the working copy as the next numbered version. Scores keyed on
    an earlier version never change because of it."""
    d = set_dir(store_dir, ls.dataset, ls.name)
    if ls.stamp and _stamp(d) != ls.stamp:
        raise StaleWrite(f"label set {ls.name!r} changed since it was loaded")
    n = (versions(store_dir, ls.dataset, ls.name) or [0])[-1] + 1
    v = d / f"v{n}"
    v.mkdir()
    for f in _FILES:
        if (d / f).exists():
            shutil.copy2(d / f, v / f)
    meta = json.loads((d / "meta.json").read_text())
    meta.setdefault("published", {})[str(n)] = _now()
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    ls.meta = meta
    return n


# --------------------------------------------------------------------------- #
# Refs, and the truth a ref stands for
# --------------------------------------------------------------------------- #

def working_ref(store_dir: str | Path, dataset: str, name: str) -> str:
    return f"{name}@w{stamp(store_dir, dataset, name)[:10]}"


def parse_ref(ref: str) -> tuple[str, int | None]:
    """'imported' -> ('imported', None); 'a@v2' -> ('a', 2); 'a@w…' -> ('a', None)."""
    if ref == IMPORTED:
        return IMPORTED, None
    name, _, tail = ref.partition("@")
    if tail.startswith("v") and tail[1:].isdigit():
        return name, int(tail[1:])
    return name, None


def describe_ref(ref: str) -> str:
    name, version = parse_ref(ref)
    if name == IMPORTED:
        return "the imported annotations"
    return f"{name} v{version}" if version is not None else f"{name} (working copy)"


@dataclass
class Truth:
    """What a ref says is really there, in the form the scoring code needs.

    `annotations` holds the boxes in closed chunks only (unknown birds left
    out), so a box in a chunk nobody finished never counts. `coverage` is the
    closed time, merged into spans; `complete` the recordings closed end to end,
    the only ones a whole-recording species list can be judged on.
    """

    ref: str
    annotations: pl.DataFrame          # ANNOTATIONS_SCHEMA
    coverage: pl.DataFrame             # recording_id, start_s, end_s (merged)
    complete: list[str]
    covered_s: dict[str, float]        # labelled seconds per recording
    exhaustive: bool                   # every recording complete (SNE)
    n_unknown: int = 0

    @property
    def label(self) -> str:
        return describe_ref(self.ref)

    @property
    def recordings(self) -> list[str]:
        return sorted(self.covered_s)

    @property
    def hours(self) -> float:
        return sum(self.covered_s.values()) / 3600


def merge_spans(chunks: pl.DataFrame) -> pl.DataFrame:
    """Adjacent closed chunks into one span each."""
    rows = []
    for (rid,), g in chunks.sort("recording_id", "start_s").group_by(
            ["recording_id"], maintain_order=True):
        cur = None
        for s, e in g.select("start_s", "end_s").iter_rows():
            if cur and s <= cur[1] + 1e-6:
                cur[1] = max(cur[1], e)
            else:
                if cur:
                    rows.append((rid, *cur))
                cur = [s, e]
        if cur:
            rows.append((rid, *cur))
    return pl.DataFrame(rows, schema={"recording_id": pl.Utf8, "start_s": pl.Float64,
                                      "end_s": pl.Float64}, orient="row")


def truth_from_set(ls: LabelSet, recordings: pl.DataFrame, ref: str) -> Truth:
    durations = dict(zip(recordings["recording_id"], recordings["duration_s"]))
    closed = ls.chunks.filter((pl.col("state") == CLOSED)
                              & pl.col("recording_id").is_in(list(durations)))
    coverage = merge_spans(closed)
    covered_s = {r: float(v) for r, v in closed.group_by("recording_id").agg(
        (pl.col("end_s") - pl.col("start_s")).sum()).iter_rows()}
    complete = sorted(r for r, s in covered_s.items()
                      if s >= durations[r] - 1.0)
    # A box is truth if any minute it touches is closed: a call that starts in
    # an open minute and runs into a closed one is still a bird in the closed one.
    first = (pl.col("start_s") / CHUNK_S).floor().cast(pl.Int64)
    last = ((pl.max_horizontal("start_s", pl.col("end_s") - 1e-6) / CHUNK_S)
            .floor().cast(pl.Int64))
    touched = (ls.boxes.with_columns(chunk=pl.int_ranges(first, last + 1))
               .explode("chunk", empty_as_null=False)
               .with_columns(chunk=pl.col("chunk") * CHUNK_S))
    hit = touched.join(closed.select("recording_id", chunk=pl.col("start_s")),
                       on=["recording_id", "chunk"], how="semi")["box_id"].unique()
    in_closed = ls.boxes.filter(pl.col("box_id").is_in(hit.to_list()))
    unknown = in_closed.filter(pl.col("species_key") == UNKNOWN_BIRD)
    ann = (in_closed.filter(pl.col("species_key") != UNKNOWN_BIRD)
           .select(list(ANNOTATIONS_SCHEMA)).sort("recording_id", "start_s"))
    return Truth(ref, ann, coverage, complete, covered_s,
                 exhaustive=bool(durations) and len(complete) == len(durations),
                 n_unknown=len(unknown))


def load_truth(store_dir: str | Path, dataset: str, ref: str) -> Truth | None:
    """The truth a ref stands for, or None for no ground truth at all."""
    from .ingest import read_dataset
    if not ref:
        return None
    recs, ann = read_dataset(store_dir, dataset)
    name, version = parse_ref(ref)
    if name == IMPORTED:
        if ann is None:
            return None
        labelled = recs.filter(pl.col("labelled"))
        covered = dict(zip(labelled["recording_id"], labelled["duration_s"]))
        return Truth(ref, ann.filter(pl.col("recording_id").is_in(list(covered))),
                     pl.DataFrame({"recording_id": list(covered), "start_s": 0.0,
                                   "end_s": list(covered.values())},
                                  schema={"recording_id": pl.Utf8, "start_s": pl.Float64,
                                          "end_s": pl.Float64}),
                     sorted(covered), {k: float(v) for k, v in covered.items()},
                     exhaustive=len(covered) == len(recs))
    return truth_from_set(load_set(store_dir, dataset, name, version), recs, ref)


def within(df: pl.DataFrame, coverage: pl.DataFrame) -> pl.DataFrame:
    """Rows (recording_id, start_s, end_s) lying wholly inside the covered time.

    A window that straddles a closed chunk and an open one is dropped: half of
    it was never listened to. The end of the last chunk is open-ended, so a
    model window that runs past the recording's last sample still counts.
    """
    if df.is_empty() or coverage.is_empty():
        return df.clear()
    cov = (coverage.sort("recording_id", "start_s")
           .rename({"start_s": "_cs", "end_s": "_ce"}))
    tagged = df.with_row_index("_i").sort("recording_id", "start_s")
    with warnings.catch_warnings():   # both sides are sorted just above
        warnings.simplefilter("ignore", UserWarning)
        joined = tagged.join_asof(cov, left_on="start_s", right_on="_cs",
                                  by="recording_id", strategy="backward")
    keep = joined.filter(pl.col("_ce").is_not_null()
                         & (pl.col("end_s") <= pl.col("_ce") + 1e-6))
    return keep.sort("_i").drop("_i", "_cs", "_ce")


def within_or_last(df: pl.DataFrame, coverage: pl.DataFrame,
                   durations: dict[str, float]) -> pl.DataFrame:
    """`within`, with each recording's final covered span extended past the
    recording's end: a model's last window may overhang the final sample."""
    if coverage.is_empty():
        return df.clear()
    ext = coverage.with_columns(
        end_s=pl.when(pl.col("end_s") >= pl.col("recording_id").replace_strict(
            durations, default=math.inf, return_dtype=pl.Float64) - 1e-6)
        .then(pl.lit(math.inf)).otherwise(pl.col("end_s")))
    return within(df, ext)


# --------------------------------------------------------------------------- #
# The sample: which chunks to label (L3)
# --------------------------------------------------------------------------- #

TIME_BANDS = (("night", 21, 4), ("dawn", 4, 8), ("day", 8, 17), ("dusk", 17, 21))


def time_band(start_time: str, offset_s: float) -> str:
    """Night / dawn / day / dusk for a moment in a recording, from its filename
    time; 'time unknown' when the filename said nothing."""
    if not start_time:
        return "time unknown"
    try:
        t = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    except ValueError:
        return "time unknown"
    hour = (t.hour + (t.minute * 60 + t.second + offset_s) / 3600) % 24
    for name, lo, hi in TIME_BANDS:
        if (lo <= hour < hi) if lo < hi else (hour >= lo or hour < hi):
            return name
    return "time unknown"


def with_strata(chunks: pl.DataFrame) -> pl.DataFrame:
    return chunks.with_columns(stratum=pl.struct("site", "start_time", "start_s").map_elements(
        lambda r: f"{r['site'] or 'no site'} · {time_band(r['start_time'], r['start_s'])}",
        return_dtype=pl.Utf8))


def draw_sample(recordings: pl.DataFrame, n_chunks: int, seed: int = 0,
                whole_recordings: int = 0,
                existing: pl.DataFrame | None = None) -> pl.DataFrame:
    """A stratified random sample of chunks, in labelling order.

    Strata are site × time of day. Each stratum gets chunks in proportion to its
    share of the audio (at least one while there are chunks to give), and
    within it the chunks are spread over as many recordings as possible, which
    is what makes the minutes nearly independent. The order interleaves the
    strata, so labelling any prefix of the sample — stopping early — still
    covers them evenly. `existing` is kept and extended, never redrawn.

    `whole_recordings` adds that many recordings in full, chosen across sites:
    a species list for a recording (V1.1 P2) can only be judged when every
    minute of it is labelled.
    """
    rng = np.random.default_rng(seed)
    pool = with_strata(all_chunks(recordings))
    taken = existing if existing is not None and len(existing) else _empty(SAMPLE_SCHEMA)
    if len(taken):
        pool = pool.join(taken.select("recording_id", "start_s"),
                         on=["recording_id", "start_s"], how="anti")
    next_order = int(taken["order"].max()) + 1 if len(taken) else 0
    out = []

    if whole_recordings > 0:
        have_whole = set(taken.filter(pl.col("whole"))["recording_id"])
        cands = (recordings.filter(~pl.col("recording_id").is_in(list(have_whole)))
                 .select("recording_id", "site").to_dicts())
        rng.shuffle(cands)
        by_site: dict[str, list[str]] = {}
        for c in cands:
            by_site.setdefault(c["site"] or "", []).append(c["recording_id"])
        picked = []
        while len(picked) < whole_recordings and any(by_site.values()):
            for site in sorted(by_site):
                if by_site[site] and len(picked) < whole_recordings:
                    picked.append(by_site[site].pop())
        whole = pool.filter(pl.col("recording_id").is_in(picked)).sort("recording_id", "start_s")
        for r in whole.iter_rows(named=True):
            out.append({**_srow(r), "order": next_order, "whole": True})
            next_order += 1
        pool = pool.filter(~pl.col("recording_id").is_in(picked))

    if n_chunks > 0 and len(pool):
        sizes = dict(pool.group_by("stratum").len().iter_rows())
        n = min(n_chunks, len(pool))
        total = sum(sizes.values())
        alloc = {s: (n * c) / total for s, c in sizes.items()}
        give = {s: min(sizes[s], max(1, int(math.floor(a)))) for s, a in alloc.items()}
        while sum(give.values()) > n:   # more strata than chunks: drop the smallest shares
            s = min((s for s in give if give[s] > 0), key=lambda s: (alloc[s], s))
            give[s] -= 1
        while sum(give.values()) < n:
            s = max((s for s in give if give[s] < sizes[s]),
                    key=lambda s: (alloc[s] - give[s], s))
            give[s] += 1
        per_stratum = []
        for s in sorted(give):
            g = pool.filter(pl.col("stratum") == s).to_dicts()
            rng.shuffle(g)
            # Spread over recordings: each recording's first random chunk, then
            # each one's second, and so on.
            seen: dict[str, int] = {}
            for r in g:
                seen[r["recording_id"]] = seen.get(r["recording_id"], 0) + 1
                r["_rank"] = seen[r["recording_id"]]
            g.sort(key=lambda r: r["_rank"])
            per_stratum.append(g[:give[s]])
        rank = 0
        while any(per_stratum):
            for g in per_stratum:
                if g:
                    out.append({**_srow(g.pop(0)), "order": next_order + rank,
                                "whole": False})
                    rank += 1
    new = pl.DataFrame(out, schema=SAMPLE_SCHEMA) if out else _empty(SAMPLE_SCHEMA)
    return pl.concat([taken.cast(SAMPLE_SCHEMA), new]).sort("order")


def _srow(r: dict) -> dict:
    return {"recording_id": r["recording_id"], "start_s": r["start_s"],
            "end_s": r["end_s"], "stratum": r["stratum"]}


def next_chunk(ls: LabelSet) -> dict | None:
    """The first sampled chunk not yet closed, in sample order."""
    if ls.sample.is_empty():
        return None
    closed = ls.chunks.filter(pl.col("state") == CLOSED).select("recording_id", "start_s")
    todo = ls.sample.join(closed, on=["recording_id", "start_s"], how="anti").sort("order")
    return todo.row(0, named=True) if len(todo) else None


# --------------------------------------------------------------------------- #
# Progress (L7)
# --------------------------------------------------------------------------- #

def progress(ls: LabelSet, recordings: pl.DataFrame) -> dict:
    """Coverage, what has been found, and the sample's state — the figures the
    Label page, the sidebar status panel and the home page all show."""
    closed = ls.chunks.filter(pl.col("state") == CLOSED)
    in_sample = (closed.join(ls.sample.select("recording_id", "start_s"),
                             on=["recording_id", "start_s"], how="semi")
                 if len(ls.sample) else closed.clear())
    minutes = float((closed["end_s"] - closed["start_s"]).sum() or 0) / 60
    t = truth_from_set(ls, recordings, "")
    sites = recordings.filter(pl.col("recording_id").is_in(t.recordings))["site"]
    species = (t.annotations.group_by("species_key")
               .agg(boxes=pl.len(), chunks=pl.struct(
                   "recording_id", (pl.col("start_s") / CHUNK_S).floor()).n_unique())
               .sort("boxes", descending=True))
    return {
        "closed_chunks": len(closed),
        "minutes": minutes,
        "recordings": len(t.recordings),
        "complete_recordings": len(t.complete),
        "sites": int(sites.n_unique()) if len(sites) else 0,
        "boxes": len(t.annotations),
        "assisted": int(ls.boxes["assisted"].fill_null(False).sum()),
        "all_boxes": len(ls.boxes),
        "unknown": t.n_unknown,
        "species": species,
        "n_species": len(species),
        "sample_size": len(ls.sample),
        "sample_closed": len(in_sample),
        "by_hand": len(closed) - len(in_sample),
        "open_with_boxes": int(ls.chunks.filter(pl.col("state") == OPEN).height),
    }


def accumulation(ls: LabelSet) -> pl.DataFrame:
    """Species found against minutes labelled, in the order chunks were closed.
    When the curve flattens, more labelling has stopped turning up new birds."""
    closed = (ls.chunks.filter(pl.col("state") == CLOSED)
              .sort("closed_at", "recording_id", "start_s"))
    if closed.is_empty():
        return pl.DataFrame(schema={"minutes": pl.Float64, "species": pl.Int64})
    boxes = (ls.boxes.filter(pl.col("species_key") != UNKNOWN_BIRD)
             .with_columns(chunk=(pl.col("start_s") / CHUNK_S).floor() * CHUNK_S))
    seen, mins, rows = set(), 0.0, [{"minutes": 0.0, "species": 0}]
    by_chunk: dict[tuple, set] = {}
    for rid, ch, sp in boxes.select("recording_id", "chunk", "species_key").iter_rows():
        by_chunk.setdefault((rid, ch), set()).add(sp)
    for rid, s, e in closed.select("recording_id", "start_s", "end_s").iter_rows():
        mins += (e - s) / 60
        seen |= by_chunk.get((rid, s), set())
        rows.append({"minutes": mins, "species": len(seen)})
    return pl.DataFrame(rows, schema={"minutes": pl.Float64, "species": pl.Int64})


def labelling_rate(ls: LabelSet, gap_min: float = 20.0) -> float | None:
    """Minutes of audio closed per hour of labelling, from the sign-off times.

    Closes more than `gap_min` apart start a new sitting; the first chunk of a
    sitting is charged the sitting's median time per chunk. None until there
    are a few closes to go on (or for imported chunks, which have no times).
    """
    c = (ls.chunks.filter((pl.col("state") == CLOSED) & (pl.col("closed_at") != "")
                          & (pl.col("closed_by") != "imported"))
         .sort("closed_at"))
    if len(c) < 3:
        return None
    times = [datetime.fromisoformat(t) for t in c["closed_at"]]
    audio = (c["end_s"] - c["start_s"]).to_numpy() / 60
    gaps = np.array([(b - a).total_seconds() / 60 for a, b in zip(times, times[1:])])
    within = gaps[gaps <= gap_min]
    if not len(within):
        return None
    per_chunk = float(np.median(within))
    spent = within.sum() + per_chunk * (1 + int((gaps > gap_min).sum()))
    return float(audio.sum() / (spent / 60)) if spent > 0 else None


# --------------------------------------------------------------------------- #
# How much to label (L2)
# --------------------------------------------------------------------------- #

def units_for_margin(margin: float, p: float = 0.8, z: float = 1.96) -> int:
    """Independent units for a ±margin interval on a proportion near p — the
    rule-of-thumb starting point, before anything is labelled."""
    return int(math.ceil(z * z * p * (1 - p) / (margin * margin)))


def project(width_now: float, n_now: int, width_target: float) -> tuple[int, int] | None:
    """How many recordings in all to bring an interval of `width_now` (from
    `n_now` recordings) down to `width_target`, as a range.

    Width shrinks about as 1/√n. The range is honest about that being an
    approximation: the low end assumes the next recordings look like the ones
    so far, the high end allows for them being more varied.
    """
    if not (width_now > 0 and n_now > 0 and width_target > 0) or not math.isfinite(width_now):
        return None
    n = n_now * (width_now / width_target) ** 2
    return max(n_now, int(math.ceil(n * 0.85))), max(n_now, int(math.ceil(n * 1.3)))


# --------------------------------------------------------------------------- #
# Out and in, in SNE's columns (L6, L8)
# --------------------------------------------------------------------------- #

SNE_COLUMNS = {"recording_id": "Filename", "start_s": "Start Time (s)",
               "end_s": "End Time (s)", "low_hz": "Low Freq (Hz)",
               "high_hz": "High Freq (Hz)", "species_key": "Species"}


def export_csv(ls: LabelSet, recordings: pl.DataFrame,
               names: dict[str, str] | None = None) -> str:
    """The boxes as SNE's annotations.csv columns (species as BEX's canonical
    key, the file name as on disk), plus provenance and whether the box's
    chunk is closed — the column that makes the file mean what it says."""
    file_of = {r: Path(p).name for r, p in recordings.select("recording_id", "path").iter_rows()}
    closed = ls.chunks.filter(pl.col("state") == CLOSED).select(
        "recording_id", chunk=pl.col("start_s"))
    out = (ls.boxes.with_columns(chunk=(pl.col("start_s") / CHUNK_S).floor() * CHUNK_S)
           .join(closed.with_columns(chunk_closed=pl.lit(True)),
                 on=["recording_id", "chunk"], how="left")
           .with_columns(chunk_closed=pl.col("chunk_closed").fill_null(False),
                         common_name=pl.col("species_key").replace_strict(
                             names or {}, default=pl.col("species_key")),
                         recording_id=pl.col("recording_id").replace_strict(
                             file_of, default=pl.col("recording_id")))
           .sort("recording_id", "start_s")
           .select(*SNE_COLUMNS, "common_name", "labeller", "labelled_at", "assisted",
                   "chunk_closed")
           .rename({**SNE_COLUMNS, "common_name": "Common Name", "labeller": "Labeller",
                    "labelled_at": "Labelled At", "assisted": "Suggestions Viewed",
                    "chunk_closed": "Chunk Closed"}))
    return out.write_csv()


def export_chunks_csv(ls: LabelSet) -> str:
    return ls.chunks.sort("recording_id", "start_s").write_csv()


def read_annotation_file(text: str, filename: str, recordings: pl.DataFrame,
                         species_lookup: dict[str, str]) -> tuple[pl.DataFrame, list[str]]:
    """Parse an annotations file into BOXES_SCHEMA rows. Returns the boxes and
    the names that could not be mapped to a species (never silently dropped:
    the caller refuses the import while any remain).

    Recognised: SNE / BEX CSV (Filename, Start Time (s), …, Species or Species
    eBird Code); Raven selection tables (tab-separated, Begin Time (s), …);
    Audacity label tracks (start<TAB>end<TAB>label, optional frequency lines).
    For Raven and Audacity, which hold one recording per file, the recording
    is the one whose file name matches `filename`'s stem.
    """
    import io
    by_file = {}
    for rid, p in recordings.select("recording_id", "path").iter_rows():
        by_file[rid] = rid
        by_file[Path(p).name] = rid
        by_file[Path(p).stem] = rid
    first = text.splitlines()[0] if text.strip() else ""
    rows: list[dict] = []
    if "Start Time (s)" in first or "Filename" in first:
        df = pl.read_csv(io.StringIO(text), infer_schema_length=0)
        sp_col = next(c for c in ("Species", "Species eBird Code", "Common Name")
                      if c in df.columns)
        for r in df.iter_rows(named=True):
            rows.append({"file": r["Filename"], "start_s": r["Start Time (s)"],
                         "end_s": r["End Time (s)"], "low_hz": r.get("Low Freq (Hz)"),
                         "high_hz": r.get("High Freq (Hz)"), "name": r[sp_col]})
    elif "Begin Time (s)" in first:
        df = pl.read_csv(io.StringIO(text), separator="\t", infer_schema_length=0)
        sp_col = next((c for c in ("Species", "Annotation", "Common Name", "Label")
                       if c in df.columns), None)
        if sp_col is None:
            raise ValueError("Raven table has no Species / Annotation column")
        file_col = next((c for c in ("Begin File", "Begin Path") if c in df.columns), None)
        for r in df.iter_rows(named=True):
            rows.append({"file": Path(r[file_col]).name if file_col else filename,
                         "start_s": r["Begin Time (s)"], "end_s": r["End Time (s)"],
                         "low_hz": r.get("Low Freq (Hz)"), "high_hz": r.get("High Freq (Hz)"),
                         "name": r[sp_col]})
    else:  # Audacity
        last = None
        for line in text.splitlines():
            parts = line.split("\t")
            if not line.strip():
                continue
            if parts[0] == "\\" and last is not None:
                last["low_hz"], last["high_hz"] = parts[1], parts[2]
                continue
            if len(parts) < 3:
                raise ValueError(f"not a recognised annotations file: {line[:60]!r}")
            last = {"file": filename, "start_s": parts[0], "end_s": parts[1],
                    "low_hz": None, "high_hz": None, "name": parts[2]}
            rows.append(last)

    out, unmapped = [], set()
    for r in rows:
        name = (r["name"] or "").strip()
        key = species_lookup.get(name) or species_lookup.get(name.lower())
        if key is None:
            unmapped.add(name)
            continue
        stem = Path(str(r["file"])).name
        # Raven names its tables "<recording>.Table.1.selections.txt".
        rid = (by_file.get(stem) or by_file.get(Path(stem).stem)
               or by_file.get(stem.split(".")[0]))
        if rid is None:
            raise ValueError(f"no recording in this dataset matches {r['file']!r}")
        f = lambda v: float(v) if v not in (None, "") else math.nan  # noqa: E731
        out.append({"recording_id": rid, "start_s": float(r["start_s"]),
                    "end_s": float(r["end_s"]), "low_hz": f(r["low_hz"]),
                    "high_hz": f(r["high_hz"]), "species_key": key,
                    "box_id": uuid.uuid4().hex[:12], "labeller": "imported",
                    "labelled_at": _now(), "assisted": False})
    boxes = pl.DataFrame(out, schema=BOXES_SCHEMA) if out else _empty(BOXES_SCHEMA)
    return boxes.filter(pl.col("end_s") > pl.col("start_s")), sorted(unmapped)


def import_boxes(store_dir: str | Path, dataset: str, name: str, boxes: pl.DataFrame,
                 recordings: pl.DataFrame, exhaustive: bool, created_by: str) -> LabelSet:
    """A new set from imported boxes. Exhaustive: every chunk of every recording
    the file covers is closed (the file claims every bird there is labelled).
    Not exhaustive: boxes only, every chunk open — presence, not absence."""
    ls = create_set(store_dir, dataset, name, created_by)
    ls.meta["imported"] = {"exhaustive": exhaustive, "at": _now()}
    (set_dir(store_dir, dataset, name) / "meta.json").write_text(
        json.dumps(ls.meta, indent=2) + "\n")
    ls.boxes = boxes.select(list(BOXES_SCHEMA)).cast(BOXES_SCHEMA)
    covered = recordings.filter(pl.col("recording_id").is_in(boxes["recording_id"].unique()))
    chunks = all_chunks(covered).select("recording_id", "start_s", "end_s")
    ls.chunks = chunks.with_columns(
        state=pl.lit(CLOSED if exhaustive else OPEN),
        closed_by=pl.lit("imported" if exhaustive else ""),
        closed_at=pl.lit(_now() if exhaustive else ""),
        reopen_reason=pl.lit("")).cast(CHUNKS_SCHEMA)
    save_set(store_dir, ls)
    return ls
