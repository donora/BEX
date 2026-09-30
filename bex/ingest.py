"""The two ingest pipelines (PLAN.md §2) — one manifest shape for both.

**Unlabelled** (`scan_folder`): point at a directory of field recordings — any
mix of formats and sample rates — and get a schema-valid recordings table. Site
metadata is supplied once per scan, because the geofilter needs lat/lon and
filenames rarely carry it. Timestamps are parsed from the filename patterns that
actually occur in the wild (AudioMoth's two conventions, SNE's, a generic
date_time) and left honestly blank otherwise.

**Labelled** (`ingest_sne`): the SNE test rig. The recordings table comes from the
same scan; the annotations CSV maps through the canonical taxonomy — an unmapped
species is an error here, because Stage 0's deliverable is that this join is
lossless.

A named **dataset** = one recordings table (+ optional annotations) persisted under
`store/datasets/<name>/`. Labelled data is just unlabelled data plus a table.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .audio import file_info
from .schemas import (
    ANNOTATIONS_SCHEMA,
    RECORDINGS_SCHEMA,
    validate_annotations,
    validate_recordings,
)
from .taxonomy import canonical_from_sne, map_ebird_codes

AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg"}

# Filename timestamp patterns, tried in order.
_SNE_RE = re.compile(r"^SNE_\d+_(\d{8})_(\d{6})$")          # SNE_001_20180509_050002
_GENERIC_RE = re.compile(r"(?:^|[^\d])(\d{8})[_\-T](\d{6})(?:[^\d]|$)")  # 20240501_063000
_AUDIOMOTH_HEX_RE = re.compile(r"^[0-9A-Fa-f]{8}$")          # 5AFC0A30 = epoch seconds


def parse_timestamp(stem: str) -> str:
    """Filename stem -> ISO 8601 string, or "" when the name says nothing.

    Naive result = the recorder's local clock (SNE, modern AudioMoth names);
    'Z'-suffixed = UTC (classic AudioMoth hex names are epoch seconds).
    """
    m = _SNE_RE.match(stem) or _GENERIC_RE.search(stem)
    if m:
        try:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            return dt.isoformat()
        except ValueError:
            pass  # eight digits that aren't a date — fall through
    if _AUDIOMOTH_HEX_RE.match(stem):
        dt = datetime.fromtimestamp(int(stem, 16), tz=timezone.utc)
        if 2010 <= dt.year <= 2040:  # plausible epoch, not an arbitrary hex name
            return dt.isoformat().replace("+00:00", "Z")
    return ""


def recording_id_for(rel_path: Path) -> str:
    """Stable id from the path relative to the scanned dir. Flat folders give the
    bare stem; nested ones stay unique by folding the directories in."""
    return str(rel_path.with_suffix("")).replace("/", "__")


def scan_folder(
    audio_dir: str | Path,
    site: str = "",
    lat: float = math.nan,
    lon: float = math.nan,
    labelled: bool = False,
) -> pl.DataFrame:
    """Walk a directory of audio files -> a validated recordings table.

    Unreadable files fail the scan loudly (a corrupt recording discovered during
    inference would be worse); an empty directory is an error, not an empty table.
    """
    root = Path(audio_dir)
    if not root.resolve().exists():
        raise FileNotFoundError(
            f"Audio directory not found: {root}. If it is a symlink to an external "
            "drive, plug the drive in."
        )
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not paths:
        raise FileNotFoundError(f"No audio files ({sorted(AUDIO_EXTS)}) under {root}")

    rows = []
    for p in paths:
        rel = p.relative_to(root)
        info = file_info(p)
        rows.append(
            {
                "recording_id": recording_id_for(rel),
                "path": str(rel),
                "duration_s": info.duration_s,
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "start_time": parse_timestamp(p.stem),
                "site": site,
                "lat": lat,
                "lon": lon,
                "labelled": labelled,
            }
        )
    return validate_recordings(pl.DataFrame(rows, schema=RECORDINGS_SCHEMA))


# --------------------------------------------------------------------------- #
# Labelled pipeline: the SNE test rig
# --------------------------------------------------------------------------- #

SNE_SITE = "sierra-nevada"
SNE_LAT, SNE_LON = 38.49, -119.95  # data/recording_location.txt


def ingest_sne(labels_dir: str | Path, audio_dir: str | Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """SNE audio + annotations.csv -> (recordings, annotations), both validated.

    The species join must be lossless (Stage 0 deliverable); an unmapped eBird
    code raises with the full report rather than dropping boxes.
    """
    recordings = scan_folder(audio_dir, site=SNE_SITE, lat=SNE_LAT, lon=SNE_LON, labelled=True)

    canonical = canonical_from_sne(labels_dir)
    raw = pl.read_csv(Path(labels_dir) / "annotations.csv").rename(
        {
            "Filename": "filename",
            "Start Time (s)": "start_s",
            "End Time (s)": "end_s",
            "Low Freq (Hz)": "low_hz",
            "High Freq (Hz)": "high_hz",
            "Species eBird Code": "code",
        }
    )
    report = map_ebird_codes(sorted(raw["code"].unique()), canonical)
    if not report.lossless:
        raise ValueError(f"SNE species join not lossless — {report.summary()}")

    # 2 of the 20,147 boxes are zero-duration points (zero bandwidth too — an
    # annotation-tool artefact). A zero-length box can never label any window
    # under the overlap rule, so dropping them loses nothing; the schema
    # validator is what surfaced them, and the ingest test pins the count.
    raw = raw.filter(pl.col("end_s") > pl.col("start_s"))

    ann = (
        raw.with_columns(
            recording_id=pl.col("filename").str.replace(r"\.flac$", ""),
            species_key=pl.col("code").replace_strict(report.mapping),
            low_hz=pl.col("low_hz").cast(pl.Float64),
            high_hz=pl.col("high_hz").cast(pl.Float64),
        )
        .select(list(ANNOTATIONS_SCHEMA))
        .sort("recording_id", "start_s")
    )
    validate_annotations(ann)

    orphans = set(ann["recording_id"].unique()) - set(recordings["recording_id"])
    if orphans:
        raise ValueError(f"annotations reference recordings not on disk: {sorted(orphans)[:5]}")
    return recordings, ann


# --------------------------------------------------------------------------- #
# Dataset persistence: store/datasets/<name>/
# --------------------------------------------------------------------------- #

def _dataset_dir(store_dir: str | Path, name: str) -> Path:
    return Path(store_dir) / "datasets" / name


def repairs_dir(store_dir: str | Path, dataset: str) -> Path:
    return Path(store_dir) / "repairs" / dataset


def resolve_audio(
    store_dir: str | Path,
    dataset: str,
    audio_root: str | Path,
    recording_id: str,
    rel_path: str,
) -> Path:
    """Where to actually read a recording's audio: a repaired copy when one
    exists (see repair.py), else the cold-tier original."""
    repaired = repairs_dir(store_dir, dataset) / f"{recording_id}.flac"
    return repaired if repaired.exists() else Path(audio_root) / rel_path


def write_dataset(
    store_dir: str | Path,
    name: str,
    recordings: pl.DataFrame,
    annotations: pl.DataFrame | None = None,
    audio_root: str | Path = "",
) -> Path:
    """`audio_root` is where the recordings' relative paths resolve — recorded in
    the dataset so later steps (mel cache, adapters) need no re-asking. It lives
    in the gitignored store, so an absolute local path is fine here."""
    d = _dataset_dir(store_dir, name)
    d.mkdir(parents=True, exist_ok=True)
    validate_recordings(recordings).write_parquet(d / "recordings.parquet")
    if annotations is not None:
        validate_annotations(annotations).write_parquet(d / "annotations.parquet")
    (d / "meta.json").write_text(json.dumps({"audio_root": str(audio_root)}, indent=2))
    return d


def dataset_audio_root(store_dir: str | Path, name: str) -> Path:
    """Where this dataset's audio lives. Fails loudly if it's gone (unplugged
    drive, moved folder) — with the file to edit if the move was deliberate."""
    meta_path = _dataset_dir(store_dir, name) / "meta.json"
    root = Path(json.loads(meta_path.read_text()).get("audio_root", ""))
    if not str(root) or not root.resolve().exists():
        raise FileNotFoundError(
            f"Audio root for dataset {name!r} not available: {root}. Plug the drive "
            f"in, or edit audio_root in {meta_path} if the audio moved."
        )
    return root


def read_dataset(store_dir: str | Path, name: str) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    d = _dataset_dir(store_dir, name)
    recordings = validate_recordings(pl.read_parquet(d / "recordings.parquet"))
    ann_path = d / "annotations.parquet"
    annotations = validate_annotations(pl.read_parquet(ann_path)) if ann_path.exists() else None
    return recordings, annotations


def list_datasets(store_dir: str | Path) -> list[str]:
    d = Path(store_dir) / "datasets"
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if (p / "recordings.parquet").exists())
