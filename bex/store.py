"""The score store — full matrices as ground truth, detections as a derived index.

Resolved D3 (PLAN.md §3a): we keep **everything** a model says. Per recording × run,
one compressed `.npz` holds the complete float16 score matrix (windows × classes),
the window start times, and the class → species_key vocabulary — self-contained, so
a file recovered on its own still means something. Every future metric is computable
retroactively from these files.

The long-form detections table the app queries is *derived* from the store
(`derive_detections`), so its selection policy can change freely — regenerating it
is minutes of group-bys, not a re-run of inference.

Layout under the configured store_dir:

    store/
    ├── runs/<run_id>/manifest.json            (schemas.RunManifest)
    ├── scores/<run_id>/<recording_id>.npz     (this module)
    └── detections/<run_id>.parquet            (derived index)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .schemas import DETECTIONS_SCHEMA, validate_detections


@dataclass
class ScoreMatrix:
    """One recording's complete output from one run."""

    recording_id: str
    run_id: str
    scores: np.ndarray        # float16, shape (n_windows, n_classes), in [0, 1]
    start_s: np.ndarray       # float32, shape (n_windows,)
    window_s: float
    species_keys: list[str]   # length n_classes; column j means species_keys[j]

    def __post_init__(self) -> None:
        w, c = self.scores.shape
        if len(self.start_s) != w:
            raise ValueError(f"start_s has {len(self.start_s)} entries for {w} windows")
        if len(self.species_keys) != c:
            raise ValueError(f"{len(self.species_keys)} species_keys for {c} score columns")


def _scores_path(store_dir: str | Path, run_id: str, recording_id: str) -> Path:
    return Path(store_dir) / "scores" / run_id / f"{recording_id}.npz"


def write_scores(store_dir: str | Path, sm: ScoreMatrix,
                 dtype: type = np.float16) -> Path:
    """Persist one recording's scores. float16 unless the readout needs more.

    D3 chose float16 for size, and for a score that spans orders of magnitude —
    a softmax — it is an excellent choice: float16 is a *floating* format, so its
    relative precision holds all the way down (spacing 6e-08 near 1e-4). It is a
    bad choice for a **saturated** score, where the informative range is crushed
    into the top of [0, 1] and float16 steps by 0.00098. Perch's per-class sigmoid
    is exactly that: stored as float16 it left 1.1% of top-ranked windows
    distinguishable, and any AP computed from it measured the format rather than
    the model (`truth.rank_resolution`).

    So precision is now the runner's call, recorded per run. BirdNET's sigmoid
    needs no help — its head is not saturated — which is why this is a parameter
    and not a rule about readouts.
    """
    out = _scores_path(store_dir, sm.run_id, sm.recording_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        scores=sm.scores.astype(dtype),
        start_s=sm.start_s.astype(np.float32),
        window_s=np.float64(sm.window_s),
        species_keys=np.array(sm.species_keys, dtype=np.str_),
        recording_id=np.str_(sm.recording_id),
        run_id=np.str_(sm.run_id),
    )
    return out


def read_scores(store_dir: str | Path, run_id: str, recording_id: str) -> ScoreMatrix:
    with np.load(_scores_path(store_dir, run_id, recording_id)) as z:
        return ScoreMatrix(
            recording_id=str(z["recording_id"]),
            run_id=str(z["run_id"]),
            scores=z["scores"],
            start_s=z["start_s"],
            window_s=float(z["window_s"]),
            species_keys=[str(s) for s in z["species_keys"]],
        )


def list_scores(store_dir: str | Path, run_id: str) -> list[str]:
    d = Path(store_dir) / "scores" / run_id
    return sorted(p.stem for p in d.glob("*.npz")) if d.exists() else []


def derive_detections(
    sm: ScoreMatrix,
    top_k: int = 20,
    min_score: float = 0.01,
    always_keys: set[str] | None = None,
    always_min_score: float = 1e-3,
) -> pl.DataFrame:
    """Full matrix -> the long-form detections index (PLAN.md §3a selection policy).

    A (window, species) row is kept when ANY of:
      - the species is in that window's top_k by score,
      - its score >= min_score,
      - it is in `always_keys` (the plausibility profile ∪ ground-truth species)
        and its score >= always_min_score.

    Geofilter columns (`occ_score`, `suppressed`) are filled by the caller once
    occurrence data exists (Stage 2's meta-model); here they default to
    NaN / False — "no geofilter information yet", which is exactly true.
    """
    scores = sm.scores.astype(np.float32)  # float16 upcast once, for stable sorting
    n_win, n_cls = scores.shape

    keep = scores >= min_score
    if top_k > 0 and top_k < n_cls:
        # Indices of the top_k columns per row (unordered — order comes from ranks).
        topk_idx = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
        rows = np.repeat(np.arange(n_win), top_k)
        keep[rows, topk_idx.ravel()] = True
    else:
        keep[:] = True
    if always_keys:
        always_cols = [j for j, k in enumerate(sm.species_keys) if k in always_keys]
        if always_cols:
            keep[:, always_cols] |= scores[:, always_cols] >= always_min_score

    # rank_in_window ranks *all* classes per window; we then read out the kept ones.
    # argsort of argsort = dense 0-based rank of each column, descending by score.
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(n_cls)[None, :].repeat(n_win, 0), axis=1)

    win_idx, cls_idx = np.nonzero(keep)
    keys = np.array(sm.species_keys)
    df = pl.DataFrame(
        {
            "recording_id": np.full(len(win_idx), sm.recording_id, dtype=object),
            "run_id": np.full(len(win_idx), sm.run_id, dtype=object),
            "start_s": sm.start_s[win_idx].astype(np.float64),
            "end_s": (sm.start_s[win_idx] + sm.window_s).astype(np.float64),
            "species_key": keys[cls_idx],
            "score_raw": scores[win_idx, cls_idx],
            "occ_score": np.full(len(win_idx), np.nan, dtype=np.float32),
            "suppressed": np.zeros(len(win_idx), dtype=bool),
            "rank_in_window": (ranks[win_idx, cls_idx] + 1).astype(np.int32),
        },
        schema=DETECTIONS_SCHEMA,
    )
    return validate_detections(df)


def write_detections(store_dir: str | Path, run_id: str, df: pl.DataFrame) -> Path:
    out = Path(store_dir) / "detections" / f"{run_id}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    validate_detections(df).write_parquet(out)
    return out


def read_detections(store_dir: str | Path, run_id: str) -> pl.DataFrame:
    return validate_detections(
        pl.read_parquet(Path(store_dir) / "detections" / f"{run_id}.parquet")
    )
