#!/usr/bin/env python
"""Cache Perch v2's per-window embeddings for an existing run (PLAN.md Stage 7).

The Stage 5 run scored every SNE recording but did not pass `--embeddings`, so
the 1,536-d vectors were never kept. This script fills that gap **without
re-scoring**: `run.py`'s resume path skips any recording that already has a
score matrix, which is correct for scores and useless here, and re-running the
classifier to get at the encoder would double the compute for nothing.

So: same model, same audio, same recordings, `encode` instead of `predict`, and
the vectors land under the *existing* run_id so they join to the scores and the
detections index with no new provenance to reconcile.

    .venv/bin/python embed.py --dataset sne --run-id perch-495c1c32
    .venv/bin/python embed.py --dataset sne --run-id perch-495c1c32 --limit 1

Idempotent and resumable: a recording whose .npz is already written is skipped,
so an interrupted run continues where it stopped.

**The window grid is asserted, not assumed.** An embedding is only useful here
because it lines up with a scored window — `embeddings[i]` must describe the
same audio as `scores[i]`. If `encode` ever returns a different number of
windows than `predict` did (a different segment size, a different tail-handling
rule), every downstream probe would train on quietly misaligned labels. That is
a silent, plausible-looking failure, so it is checked per recording and fails
the run rather than writing a file that looks fine.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

import birdnet
from bex import ingest, store
from bex.config import load_config
from bex.schemas import RunManifest




def default_workers() -> int:
    """Half the logical cores, leaving the machine usable while this runs.

    Not a performance tweak — a politeness one. See --workers.
    """
    return max(1, (os.cpu_count() or 4) // 2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-id", required=True, help="the existing Perch run to embed")
    ap.add_argument("--recordings", help="comma-separated recording_ids")
    ap.add_argument("--limit", type=int, help="process at most N unembedded recordings")
    ap.add_argument("--workers", type=int, default=None,
                    help="""workers the model may spawn. The package's default is one per
                         logical core, and each worker runs TensorFlow with its own
                         thread pool — on a 12-core machine that is 13 processes
                         fighting over 12 cores (observed load average ~80, and the
                         machine unusable). Half the cores leaves the box workable
                         and is not meaningfully slower, because the bottleneck is
                         already memory bandwidth rather than cores.""")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    workers = args.workers if args.workers is not None else default_workers()

    cfg = load_config(args.config)
    manifest = RunManifest.load(cfg.store_dir, args.run_id)
    if manifest.model_name != "perch":
        raise SystemExit(f"run {args.run_id} is a {manifest.model_name} run, not perch")

    recordings, _ = ingest.read_dataset(cfg.store_dir, args.dataset)
    audio_root = ingest.dataset_audio_root(cfg.store_dir, args.dataset)
    if args.recordings:
        wanted = set(args.recordings.split(","))
        recordings = recordings.filter(pl.col("recording_id").is_in(wanted))

    # Only recordings this run actually scored: an embedding without a score
    # matrix has nothing to align against, which is the whole point of the check.
    scored = set(store.list_scores(cfg.store_dir, args.run_id))
    out_dir = Path(cfg.store_dir) / "embeddings" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in out_dir.glob("*.npz")}

    todo = recordings.filter(
        pl.col("recording_id").is_in(list(scored))
        & ~pl.col("recording_id").is_in(list(done))
    )
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"run {args.run_id}: {len(scored)} scored, {len(done)} embedded, "
          f"{len(todo)} to process, {workers} workers", flush=True)
    if not len(todo):
        return 0

    print("loading Perch v2 (TensorFlow)...", flush=True)
    model = birdnet.load_perch_v2()
    window_s = float(model.get_segment_size_s())
    if abs(window_s - manifest.window_s) > 1e-6:
        raise SystemExit(
            f"loaded Perch has {window_s}s windows but run {args.run_id} was scored "
            f"at {manifest.window_s}s — these embeddings would not align with it"
        )

    dim = None
    for row in todo.iter_rows(named=True):
        rid = row["recording_id"]
        t0 = time.time()
        src = str(ingest.resolve_audio(cfg.store_dir, args.dataset, audio_root,
                                       rid, row["path"]))
        emb = np.asarray(
            model.encode(src, n_workers=workers, show_stats=None).embeddings)[0]

        sm = store.read_scores(cfg.store_dir, args.run_id, rid)
        if emb.shape[0] != sm.scores.shape[0]:
            raise SystemExit(
                f"{rid}: encode gave {emb.shape[0]} windows, the stored scores have "
                f"{sm.scores.shape[0]}. Embeddings and scores must index the same "
                "windows; refusing to write a misaligned file."
            )
        dim = int(emb.shape[1])

        np.savez_compressed(
            out_dir / f"{rid}.npz",
            embeddings=emb.astype(np.float16),
            start_s=sm.start_s.astype(np.float32),
            window_s=np.float64(window_s),
            recording_id=np.str_(rid),
            run_id=np.str_(args.run_id),
        )
        print(f"  {rid}: {emb.shape[0]} x {dim} in {time.time() - t0:.0f}s", flush=True)

    # Sidecar provenance: which model produced these vectors, on which grid.
    # The run manifest describes the *scores*; nothing in it records that an
    # encoder was run, or with what.
    (out_dir / "meta.json").write_text(json.dumps({
        "run_id": args.run_id,
        "dataset": args.dataset,
        "model_name": "perch",
        "model_version": str(model.get_version()),
        "window_s": window_s,
        "hop_s": window_s,
        "input_sr": model.get_sample_rate(),
        "dim": dim,
        "dtype": "float16",
        "aligned_with": "scores/<run_id>/<recording_id>.npz, window for window",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "packages": {
            "birdnet": importlib.metadata.version("birdnet"),
            "tensorflow": importlib.metadata.version("tensorflow"),
            "python": sys.version.split()[0],
        },
    }, indent=2, sort_keys=True) + "\n")
    print(f"{len(todo)} recording(s) embedded -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
