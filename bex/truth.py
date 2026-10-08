"""Align a run's scores with ground truth, window by window, species by species.

This is the missing foundation PLAN.md §3d specified and nothing built: until now
the only contact between a model's output and the annotations was set-level
("did this arm ever report this species?") and a loose any-overlap join in
`stats.filter_errors`. Neither can answer "was the model right *here*", which is
what a precision/recall curve integrates over.

What comes out is deliberately the dumbest possible shape — one row per
(recording, window, species) with a 0/1 truth and a score:

    recording_id  start_s  end_s  species_key          y_true  score
    SNE_001…      120.0    125.0  Turdus migratorius        1  0.8134
    SNE_001…      120.0    125.0  Catharus guttatus         0  0.0021

Dense, not sparse: the zeros are the negatives, and a precision denominator made
of rows that were filtered out before anyone looked is how evaluation code lies.
It is also directly inspectable — any number in the app can be traced to the rows
that produced it, which is the house rule.

**Three decisions that determine what the numbers mean.**

1. *The evaluation label space is the annotated species, not the model's.* SNE's
   annotations cover 56 species; Perch emits 14,795 and BirdNET 6,522. Scoring
   precision over a model's full vocabulary would count every unannotated species
   as a false positive, when the truth is that nobody labelled it — and would
   punish Perch for having a larger vocabulary, which is not a defect. Stage 5
   already caught one circular metric (`degenerate_false_suppression`); this is
   the same trap one level down. **This assumes the annotations are exhaustive for
   the species they cover** — true for SNE's fully-annotated soundscapes, and a
   thing to re-check before trusting these numbers on any other dataset.

2. *Every arm is scored on the same species set.* A species missing from a model's
   vocabulary is kept, scored 0 everywhere, and reported in the metadata — not
   dropped. Dropping it would let two arms be compared on different species
   without the table saying so, which makes cmAP meaningless.

3. *Truth is labelled on the window starts the model actually produced*, via
   `windows.label_starts`, not on a grid recomputed from the recording duration.
   See that function for why.

Native grids are the default and are the right basis for choosing an operating
point, because θ is applied to a model's own output. The shared 1 s grid
(`SHARED_GRID`) exists for cross-model comparison, where BirdNET's 3 s and Perch's
5 s windows otherwise give the two curves different denominators.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from . import ingest, labels, store
from .schemas import RunManifest
from .views import readout_token
from .windows import GridParams, label_starts, window_starts

#: The cross-model substrate of PLAN.md §3d. 1 s frames, each labelled by the
#: same min(0.5 s, box) rule, each scored by the max over the native windows
#: that cover it.
SHARED_GRID = GridParams(window_s=1.0, hop_s=1.0, min_overlap_s=0.5)

ALIGNED_SCHEMA: dict[str, pl.DataType] = {
    "recording_id": pl.Utf8,
    "start_s": pl.Float64,
    "end_s": pl.Float64,
    "species_key": pl.Utf8,
    "y_true": pl.UInt8,
    "score": pl.Float32,
}


def across_window_comparable(manifest: RunManifest) -> tuple[bool, str]:
    """Is this run's score readout safe for a per-species PR curve?

    A per-species curve ranks *windows* against each other, so the score must be
    monotone in the model's belief **across windows**. A per-class sigmoid is:
    it transforms each class's logit by a fixed function, so ranking windows by
    the score is ranking them by the logit. A **softmax is not** — it divides by
    a per-window sum over all classes, so a bird singing in a busy dawn-chorus
    window is scored lower than the same bird alone. Stage 5 chose softmax for
    Perch on the grounds that it is monotone in the logits; that is true *within*
    a window, which is all the muddiness statistics need, and false across them.

    **How much this actually costs, measured directly rather than argued
    (2026-09-11).** Perch was re-run with a per-class sigmoid at float32 and
    scored against the same truth. The readout does reorder windows substantially
    — across-window Spearman between the two readouts is only 0.69 (0.43-0.78
    over 28 well-labelled species) — but the reordering is close to unbiased with
    respect to truth, so the aggregate barely moves: **cmAP 0.3746 sigmoid against
    0.3629 softmax, and micro AP 0.6462 against 0.6463.** Individual species swing
    either way (-0.113 to +0.146).

    So this flag is a real caveat and a small one, and an earlier estimate of it
    here was badly wrong. That estimate came from renormalising *BirdNET's*
    per-class sigmoid probabilities within each window as a proxy, which cost
    0.168 micro AP — but dividing already-saturated probabilities by their sum is
    a far more violent and more biased transform than a softmax over logits, and
    it does not model Perch. **On SNE, BirdNET's lead over Perch at window level
    is real, not an artefact of the readout.**

    Keep the flag: on another dataset or head the normaliser may vary much more
    across windows. Do not assume the cost is ~0.01 anywhere else — measure it,
    which now costs one re-run and one script.
    """
    # The documented leading word, not a substring search: the sigmoid transform's
    # own prose mentions softmax ("not comparable with the softmax run's"), and a
    # naive `"softmax" in text` flagged the sigmoid run as softmax.
    if readout_token(manifest.score_transform) == "softmax":
        return False, (
            f"{manifest.run_id} stores softmax scores, which normalise within each "
            "window, while per-species AP ranks across them. Measured on SNE this "
            "is worth about +0.01 cmAP and nothing at all in micro AP, so the "
            "number is usable — but it is a caveat, and on another head the "
            "normaliser may vary far more. Compare against a sigmoid run to know."
        )
    return True, ""


def rank_resolution(aligned: pl.DataFrame, min_positives: int = 10) -> dict:
    """Can the stored scores still tell the top-ranked windows apart?

    AP is a ranking statistic, so it is only as good as the store's ability to
    separate two windows — and separation is decided by the storage format, not
    by the model. The score matrices are float16 (D3), whose *relative* precision
    is excellent, which makes it a fine home for a softmax (values spanning orders
    of magnitude, spacing 6e-08 near 1e-4) and a bad one for a **saturated**
    per-class sigmoid, whose informative range is crushed into the top of [0, 1]
    where float16 steps by 0.00098.

    Measured at the top of each species' ranking, which is the part AP integrates
    over — the mass of near-zero scores further down is legitimately tied and
    swamps any pooled count. On SNE the three arms separate completely: Perch's
    float16 sigmoid keeps **0.011** of its top-ranked windows distinguishable,
    against 0.941 for the softmax arm and 0.924 for BirdNET.

    Note BirdNET is a sigmoid arm too and is unaffected — its head is not
    saturated. The hazard is saturation meeting float16, not the readout's name,
    which is why this is measured rather than inferred from the manifest.
    """
    fracs, rows = [], []
    for (key,), grp in aligned.group_by(["species_key"], maintain_order=True):
        v = grp["score"].to_numpy()
        n_pos = int(grp["y_true"].sum())
        if n_pos < min_positives or not len(v):
            continue
        k = min(len(v), max(2 * n_pos, 100))
        top = np.sort(v)[::-1][:k]
        frac = len(np.unique(top)) / k
        fracs.append(frac)
        rows.append({"species_key": str(key), "n_positive": n_pos,
                     "top_k": int(k), "distinct_fraction": float(frac)})
    if not fracs:
        return {"median_distinct_fraction": float("nan"), "n_species": 0,
                "damaged": False, "per_species": pl.DataFrame()}
    median = float(np.median(fracs))
    return {
        "median_distinct_fraction": median,
        "n_species": len(fracs),
        "damaged": median < RANK_RESOLUTION_FLOOR,
        "per_species": pl.DataFrame(rows).sort("distinct_fraction"),
    }


#: Below this, more than half of the top-ranked windows share a score with another
#: and the arm's AP measures float16 as much as the model. The measured contrast on
#: SNE leaves an enormous margin either side (0.011 against 0.92-0.94), so the exact
#: value is not load-bearing.
RANK_RESOLUTION_FLOOR = 0.5


def project_scores(
    scores: np.ndarray,
    src_starts: np.ndarray,
    src_window_s: float,
    dst_starts: np.ndarray,
    dst_window_s: float,
) -> np.ndarray:
    """Move a score matrix from one window grid to another; overlaps take the max.

    PLAN.md §3d's rule, and the only defensible one for a *presence* score: if a
    5 s window says 0.9 for a robin, every second of that window is claimed to
    contain a robin, so each 1 s frame it covers inherits 0.9. Averaging would
    dilute a real detection by the silence either side of it.

    A destination frame no source window covers keeps 0 — the tail of a recording
    past the model's last whole window. That is the model genuinely saying nothing
    there, and a truth box in that region should count as a miss, so the zero is
    correct rather than missing data.
    """
    n_dst = len(dst_starts)
    out = np.zeros((n_dst, scores.shape[1]), dtype=np.float32)
    if n_dst == 0 or scores.shape[0] == 0:
        return out

    # Source window i overlaps destination frame j iff
    #   dst_starts[j] > src_starts[i] - dst_window_s  and
    #   dst_starts[j] < src_starts[i] + src_window_s.
    j_lo = np.searchsorted(dst_starts, src_starts - dst_window_s, side="right")
    j_hi = np.searchsorted(dst_starts, src_starts + src_window_s, side="left")
    counts = np.maximum(j_hi - j_lo, 0)
    if counts.sum() == 0:
        return out

    i_idx = np.repeat(np.arange(len(src_starts)), counts)
    # Ragged arange: each source window's run of destination indices.
    starts_of_runs = np.cumsum(counts) - counts
    j_idx = np.repeat(j_lo, counts) + (np.arange(counts.sum())
                                       - np.repeat(starts_of_runs, counts))
    np.maximum.at(out, j_idx, scores[i_idx])
    return out


def eval_species(annotations: pl.DataFrame, recording_ids: list[str] | None = None) -> list[str]:
    """The species the evaluation is over: every species the annotations name."""
    ann = annotations
    if recording_ids is not None:
        ann = ann.filter(pl.col("recording_id").is_in(recording_ids))
    return sorted(ann["species_key"].unique().to_list())


def align_recording(
    sm: store.ScoreMatrix,
    boxes: pl.DataFrame,
    species: list[str],
    grid: GridParams | None = None,
    min_overlap_s: float = 0.5,
    duration_s: float | None = None,
) -> pl.DataFrame:
    """One recording × one run -> the dense (window × species) truth/score frame.

    `grid=None` keeps the model's native windows. Pass `SHARED_GRID` (or any
    `GridParams`) to project onto a common grid first.
    """
    col_of = {k: j for j, k in enumerate(sm.species_keys)}
    native = np.zeros((sm.scores.shape[0], len(species)), dtype=np.float32)
    pairs = [(j, col_of[k]) for j, k in enumerate(species) if k in col_of]
    if pairs:
        where, cols = map(list, zip(*pairs))
        native[:, where] = sm.scores[:, cols].astype(np.float32)

    if grid is None:
        dst_starts = np.asarray(sm.start_s, dtype=np.float64)
        dst_window_s = sm.window_s
        dst_scores = native
    else:
        span = (duration_s if duration_s is not None
                else float(sm.start_s[-1] + sm.window_s) if len(sm.start_s) else 0.0)
        dst_starts = window_starts(span, grid).astype(np.float64)
        dst_window_s = grid.window_s
        dst_scores = project_scores(native, np.asarray(sm.start_s, dtype=np.float64),
                                    sm.window_s, dst_starts, dst_window_s)

    label_grid = GridParams(window_s=dst_window_s, hop_s=dst_window_s,
                            min_overlap_s=min_overlap_s)
    y = label_starts(boxes.filter(pl.col("species_key").is_in(species)),
                     dst_starts, label_grid, species)

    n_win, n_spp = y.shape
    return pl.DataFrame(
        {
            "recording_id": np.full(n_win * n_spp, sm.recording_id, dtype=object),
            "start_s": np.repeat(dst_starts, n_spp),
            "end_s": np.repeat(dst_starts + dst_window_s, n_spp),
            "species_key": np.tile(np.array(species, dtype=object), n_win),
            "y_true": y.ravel(),
            "score": dst_scores.ravel(),
        },
        schema=ALIGNED_SCHEMA,
    )


def align_run(
    store_dir: str | Path,
    dataset: str,
    run_id: str,
    grid: GridParams | None = None,
    species: list[str] | None = None,
    recordings: list[str] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    truth_ref: str = labels.IMPORTED,
) -> tuple[pl.DataFrame, dict]:
    """Align every scored recording of a run. Returns the frame and its provenance.

    Reads the **full score matrices**, not the derived detections index. The index
    keeps only the top-k and anything above a floor, so the rows it drops are
    precisely the low-scoring ones that populate the high-recall end of the curve;
    building a PR curve from it would truncate the curve and inflate AP. D3 stored
    the full matrices for exactly this kind of retroactive metric.

    `truth_ref` names the label set (`bex.labels`). Only time inside its
    closed chunks is scored: a window nobody has listened to is neither a hit
    nor a false alarm, so it is left out rather than counted as a negative.
    For an exhaustive set (SNE's imported annotations) that keeps everything.
    """
    store_dir = Path(store_dir)
    manifest = RunManifest.load(store_dir, run_id)
    recs, _ = ingest.read_dataset(store_dir, dataset)
    t = labels.load_truth(store_dir, dataset, truth_ref)
    if t is None or not t.recordings:
        raise ValueError(f"dataset {dataset!r} has no annotations in "
                         f"{labels.describe_ref(truth_ref)} (no closed chunks) — "
                         "nothing to score against")
    ann = t.annotations

    scored = store.list_scores(store_dir, run_id)
    wanted = set(t.recordings) & set(scored)
    if recordings is not None:
        wanted &= set(recordings)
    ids = sorted(wanted)
    if not ids:
        raise ValueError(f"run {run_id} has no scored recordings in dataset {dataset!r}")

    species = species if species is not None else eval_species(ann, ids)
    durations = dict(zip(recs["recording_id"], recs["duration_s"]))

    frames, missing_vocab = [], None
    for i, rid in enumerate(ids):
        if progress is not None:
            progress(rid, i, len(ids))
        sm = store.read_scores(store_dir, run_id, rid)
        if missing_vocab is None:
            missing_vocab = sorted(set(species) - set(sm.species_keys))
        frame = align_recording(
            sm,
            ann.filter(pl.col("recording_id") == rid),
            species,
            grid=grid,
            duration_s=durations.get(rid),
        )
        if rid not in t.complete:
            frame = labels.within_or_last(
                frame, t.coverage.filter(pl.col("recording_id") == rid), durations)
        frames.append(frame)

    aligned = pl.concat(frames)
    if aligned.is_empty():
        raise ValueError(f"no {run_id} window lies wholly inside a closed chunk")
    meta = {
        "run_id": run_id,
        "dataset": dataset,
        "model_name": manifest.model_name,
        "model_version": manifest.model_version,
        "grid": "native" if grid is None else f"shared-{grid.window_s:g}s",
        "window_s": manifest.window_s if grid is None else grid.window_s,
        "min_overlap_s": 0.5,
        "n_recordings": len(ids),
        "recordings": ids,
        "n_species": len(species),
        "species": species,
        "species_not_in_vocabulary": missing_vocab or [],
        "n_windows": int(aligned.select(pl.struct("recording_id", "start_s").n_unique()).item()),
        "n_pairs": len(aligned),
        "n_positive": int(aligned["y_true"].sum()),
        "truth_ref": truth_ref,
        "labelled_hours": sum(t.covered_s.get(r, 0.0) for r in ids) / 3600,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return aligned, meta


# --------------------------------------------------------------------------- #
# Cache — alignment is minutes of npz reads, the sweep on top of it is seconds
# --------------------------------------------------------------------------- #

def aligned_path(store_dir: str | Path, dataset: str, run_id: str,
                 grid_name: str = "native", truth_ref: str = labels.IMPORTED) -> Path:
    """The imported annotations keep the original layout (caches built before
    label sets existed stay valid); every other ref gets its own folder."""
    d = Path(store_dir) / "truth" / dataset
    if truth_ref != labels.IMPORTED:
        d = d / "sets" / truth_ref
    return d / f"{run_id}__{grid_name}.parquet"


def prune_working(store_dir: str | Path, dataset: str, name: str, keep: str) -> None:
    """Drop caches of a set's earlier working copies — each edit makes a new
    ref, and only the current one can be asked for again."""
    import shutil
    d = Path(store_dir) / "truth" / dataset / "sets"
    for p in d.glob(f"{name}@w*") if d.exists() else []:
        if p.name != keep:
            shutil.rmtree(p, ignore_errors=True)


def write_aligned(store_dir: str | Path, dataset: str, run_id: str,
                  aligned: pl.DataFrame, meta: dict) -> Path:
    out = aligned_path(store_dir, dataset, run_id, meta.get("grid", "native"),
                       meta.get("truth_ref", labels.IMPORTED))
    out.parent.mkdir(parents=True, exist_ok=True)
    aligned.write_parquet(out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return out


def read_aligned(store_dir: str | Path, dataset: str, run_id: str,
                 grid_name: str = "native",
                 truth_ref: str = labels.IMPORTED) -> tuple[pl.DataFrame, dict]:
    p = aligned_path(store_dir, dataset, run_id, grid_name, truth_ref)
    meta = json.loads(p.with_suffix(".json").read_text())
    return pl.read_parquet(p), meta


def list_aligned(store_dir: str | Path, dataset: str) -> list[dict]:
    """Every cached alignment for a dataset, newest first — what the app can score."""
    d = Path(store_dir) / "truth" / dataset
    if not d.exists():
        return []
    metas = [json.loads(p.read_text()) for p in d.glob("*.json")]
    return sorted(metas, key=lambda m: m.get("created_utc", ""), reverse=True)
