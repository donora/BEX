#!/usr/bin/env python
"""Stage 4's cross-check (PLAN.md D6): do the app's numbers survive re-derivation?

The app and `forensics.py` both read the *detections index* — a selective
subset of each run's full score matrix. This script goes back to the matrices
themselves and checks three things:

1. **The index is faithful.** Re-derive it from the raw scores with the run's
   own selection policy and compare to the parquet on disk, row for row.
2. **The metrics are exact, not approximate.** Recompute the muddy-window rate
   and impostor mass from every one of the model's classes, and require them to
   match the index-based values. They should agree exactly for theta >= 0.01,
   because the index keeps every species scoring at or above 0.01 — this is the
   claim `stats.window_impostor_mass` makes, and here it is tested against data
   rather than asserted.
3. **The judge is reproducible.** Rebuild implausibility from the plausibility
   profile alone (which needs no stored occurrence) and confirm the same
   verdicts.

    .venv/bin/python scripts/crosscheck.py --run birdnet-c102ff26
    .venv/bin/python scripts/crosscheck.py --run <id> --all
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from bex import ingest, stats, store
from bex.config import load_config
from bex.profiles import load_profile
from bex.schemas import RunManifest


def full_matrix_metrics(
    sm: store.ScoreMatrix, implausible_mask: np.ndarray, theta: float
) -> dict:
    """Muddy-window rate and impostor mass over *every* class the model has."""
    scores = sm.scores.astype(np.float32)
    hits = scores >= theta
    positive = hits.any(axis=1)
    muddy = (hits & implausible_mask[None, :]).any(axis=1)

    confident = np.where(hits, scores, 0.0)
    total = confident.sum(axis=1)
    impostor = np.where(hits & implausible_mask[None, :], scores, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mass = np.where(total > 0, impostor / total, 0.0)

    return {
        "windows": int(positive.sum()),
        "muddy_windows": int((positive & muddy).sum()),
        "mass": mass[positive],
        "start_s": sm.start_s[positive],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--theta", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=3, help="recordings to check")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    manifest = RunManifest.load(cfg.store_dir, args.run)
    det = store.read_detections(cfg.store_dir, args.run)
    recordings, ann = ingest.read_dataset(cfg.store_dir, manifest.dataset)
    profile = load_profile(cfg.profiles_dir / manifest.profile)

    # The selection policy the runner used, reconstructed from the same inputs.
    always_keys = profile.plausible_keys(max_tier=2)
    if ann is not None:
        always_keys |= set(ann["species_key"].unique())

    ids = store.list_scores(cfg.store_dir, args.run)
    if not args.all:
        ids = ids[: args.limit]

    print(f"cross-checking run {args.run} against {len(ids)} raw score matrices")
    print(f"θ={args.theta} · profile {profile.name} · judge: profile tiers\n")

    failures = 0
    index_rows = mass_rows = 0
    worst_mass_delta = 0.0

    for recording_id in ids:
        sm = store.read_scores(cfg.store_dir, args.run, recording_id)

        # --- 1. the detections index re-derives exactly -------------------- #
        rebuilt = store.derive_detections(sm, always_keys=always_keys)
        stored = det.filter(pl.col("recording_id") == recording_id)
        cols = ["start_s", "species_key", "score_raw", "rank_in_window"]
        a = rebuilt.select(cols).sort(cols)
        b = stored.select(cols).sort(cols)
        index_ok = a.equals(b)
        index_rows += len(b)

        # --- 2 & 3. metrics from every class, judged by the profile -------- #
        mask = np.array([profile.tier_of(k) >= 3 for k in sm.species_keys])
        full = full_matrix_metrics(sm, mask, args.theta)

        marked = stats.mark_implausible(stored, profile, mode="profile")
        rate = stats.muddy_window_rate(marked, args.theta)
        idx_mass = (
            stats.window_impostor_mass(marked, args.theta)
            .sort("start_s")
        )

        rate_ok = (rate["windows"] == full["windows"]
                   and rate["muddy_windows"] == full["muddy_windows"])

        # Align the two impostor-mass series on window start and compare.
        full_mass = pl.DataFrame({
            "start_s": full["start_s"].astype(np.float64),
            "full_mass": full["mass"].astype(np.float64),
        })
        joined = idx_mass.join(full_mass, on="start_s", how="inner")
        delta = ((joined["impostor_mass"] - joined["full_mass"]).abs().max()
                 if len(joined) else 0.0)
        mass_ok = len(joined) == len(idx_mass) == full["windows"] and delta < 1e-6
        worst_mass_delta = max(worst_mass_delta, float(delta or 0.0))
        mass_rows += len(joined)

        ok = index_ok and rate_ok and mass_ok
        failures += not ok
        flag = "ok  " if ok else "FAIL"
        print(f"  [{flag}] {recording_id}: {len(b):,} index rows · "
              f"{rate['windows']:,} windows · {rate['muddy_windows']:,} muddy · "
              f"max mass Δ {float(delta or 0.0):.2e}")
        if not index_ok:
            print(f"         index mismatch: {len(a)} rebuilt vs {len(b)} stored")
        if not rate_ok:
            print(f"         rate mismatch: index {rate['windows']}/"
                  f"{rate['muddy_windows']} vs full {full['windows']}/"
                  f"{full['muddy_windows']}")

    print(f"\n{len(ids) - failures}/{len(ids)} recordings reproduce exactly")
    print(f"  {index_rows:,} index rows re-derived from raw scores")
    print(f"  {mass_rows:,} windows of impostor mass agree to "
          f"{worst_mass_delta:.2e} (float32 noise floor ~1e-7)")
    if failures:
        print("\nA mismatch means the app is showing something the raw scores do "
              "not support. Fix before trusting any table.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
