#!/usr/bin/env python
"""BirdNET v2.4 runner: raw all-class scores + geo occurrence -> the bex store.

The two halves of PLAN.md §3a, in one pass per recording:

- the acoustic model runs with NO confidence threshold and NO species filter —
  every one of the 6,522 class scores for every 3 s window lands in the score
  store, because the silenced layer is the thing this project studies;
- the geo meta-model is called separately per (lat, lon, week) with
  min_confidence=0, giving the full occurrence vector; occurrence scores and
  the would-BirdNET-have-suppressed-this flag are recorded per detection row,
  never applied destructively.

Run from this directory (or with BEX_CONFIG set):

    .venv/bin/python run.py --dataset sne                # everything
    .venv/bin/python run.py --dataset sne --limit 2      # smoke test
    .venv/bin/python run.py --dataset sne --run-id birdnet-xxxxxxxx   # resume

Resume skips recordings whose score matrices exist and rebuilds the detections
index for the whole run at the end, so an interrupted run converges to the same
result as an uninterrupted one.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

import birdnet
from bex import ingest, store
from bex.adapters import apply_occurrence, birdnet_week, scores_to_matrix, vocab_species_keys
from bex.config import load_config
from bex.profiles import load_profile
from bex.schemas import RunManifest

WINDOW_S = 3.0
SF_THRESH_DEFAULT = 0.03  # BirdNET-Analyzer's default species-list threshold


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--recordings", help="comma-separated recording_ids (default: all)")
    ap.add_argument("--limit", type=int, help="process at most N unprocessed recordings")
    ap.add_argument("--profile", help="plausibility profile (default: from config)")
    ap.add_argument("--sf-thresh", type=float, default=SF_THRESH_DEFAULT,
                    help="occurrence threshold reproducing BirdNET's own species filter")
    ap.add_argument("--run-id", default="", help="resume an existing run")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    recordings, annotations = ingest.read_dataset(cfg.store_dir, args.dataset)
    audio_root = ingest.dataset_audio_root(cfg.store_dir, args.dataset)
    if args.recordings:
        wanted = set(args.recordings.split(","))
        recordings = recordings.filter(pl.col("recording_id").is_in(wanted))

    profile_name = args.profile or cfg.default_profile
    profile = load_profile(cfg.profiles_dir / profile_name)
    always_keys = profile.plausible_keys(max_tier=2)
    if annotations is not None:
        always_keys |= set(annotations["species_key"].unique())

    print("loading BirdNET v2.4 (litert)...", flush=True)
    acoustic = birdnet.load("acoustic", "2.4", "tf", library="litert")
    geo = birdnet.load("geo", "2.4", "tf", library="litert")
    keys = vocab_species_keys(list(acoustic.species_list))
    geo_keys = vocab_species_keys(list(geo.species_list))

    if args.run_id:
        manifest = RunManifest.load(cfg.store_dir, args.run_id)
        assert manifest.model_name == "birdnet", manifest.model_name
    else:
        manifest = RunManifest(
            model_name="birdnet",
            model_version="2.4",
            window_s=WINDOW_S,
            hop_s=WINDOW_S,
            input_sr=acoustic.get_sample_rate(),
            score_transform="sigmoid",
            geofilter={
                "model": "geo-2.4",
                "sf_thresh": args.sf_thresh,
                "week_convention": "birdnet-48 from recording start_time; None -> year-round max",
                "applied": "recorded per row (occ_score, suppressed), never destructively",
            },
            profile=profile_name,
            dataset=args.dataset,
            packages={
                "birdnet": importlib.metadata.version("birdnet"),
                "python": sys.version.split()[0],
            },
        )
        manifest.save(cfg.store_dir)
    run_id = manifest.run_id
    # The model's original labels, index-aligned with the score columns — the
    # app derives display names (e.g. BirdNET's common names) from these.
    labels_path = Path(cfg.store_dir) / "runs" / run_id / "labels.txt"
    labels_path.write_text("\n".join(acoustic.species_list) + "\n")
    print(f"run {run_id} -> {cfg.store_dir}", flush=True)

    done = set(store.list_scores(cfg.store_dir, run_id))
    todo = recordings.filter(~pl.col("recording_id").is_in(done))
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"{len(done)} recording(s) already scored, {len(todo)} to process", flush=True)

    occ_cache: dict[tuple, dict[str, float] | None] = {}
    for row in todo.iter_rows(named=True):
        t0 = time.time()
        res = acoustic.predict(
            str(ingest.resolve_audio(cfg.store_dir, args.dataset, audio_root,
                                     row["recording_id"], row["path"])),
            top_k=None,
            default_confidence_threshold=None,
            show_stats=None,
        )
        probs = np.asarray(res.species_probs)[0]  # (windows, classes)
        assert res.hop_duration_s == res.segment_duration_s == WINDOW_S
        sm = store.ScoreMatrix(
            recording_id=row["recording_id"],
            run_id=run_id,
            scores=scores_to_matrix(probs),
            start_s=(np.arange(probs.shape[0]) * WINDOW_S).astype(np.float32),
            window_s=WINDOW_S,
            species_keys=keys,
        )
        store.write_scores(cfg.store_dir, sm)
        n_win = probs.shape[0]
        expected = math.floor(row["duration_s"] / WINDOW_S)
        if abs(n_win - expected) > 1:
            print(f"  WARNING {row['recording_id']}: {n_win} windows for "
                  f"{row['duration_s']:.0f}s audio (expected ~{expected})", flush=True)
        print(f"  {row['recording_id']}: {n_win} windows in {time.time() - t0:.0f}s", flush=True)

    # Detections index: rebuilt for the WHOLE run from the score store, so resumed
    # and clean runs converge (and reruns after a policy change are just this loop).
    frames = []
    rec_by_id = {r["recording_id"]: r for r in recordings.iter_rows(named=True)}
    for recording_id in store.list_scores(cfg.store_dir, run_id):
        sm = store.read_scores(cfg.store_dir, run_id, recording_id)
        row = rec_by_id.get(recording_id)
        if row is None:
            continue
        df = store.derive_detections(sm, always_keys=always_keys)
        occ_key = (row["lat"], row["lon"], birdnet_week(row["start_time"]))
        if occ_key not in occ_cache:
            if row["lat"] is None or math.isnan(row["lat"]):
                occ_cache[occ_key] = None
            else:
                gres = geo.predict(row["lat"], row["lon"], week=occ_key[2], min_confidence=0.0)
                occ_cache[occ_key] = dict(zip(geo_keys, np.asarray(gres.species_probs).tolist()))
        frames.append(apply_occurrence(df, occ_cache[occ_key], args.sf_thresh))

    detections = pl.concat(frames)
    out = store.write_detections(cfg.store_dir, run_id, detections)
    n_sup = int(detections.filter(pl.col("suppressed") & (pl.col("score_raw") >= 0.1)).height)
    print(f"{len(frames)} recordings, {len(detections)} detection rows -> {out}")
    print(f"({n_sup} rows with score >= 0.1 that BirdNET's own filter would have hidden)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
