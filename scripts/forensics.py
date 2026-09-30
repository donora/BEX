#!/usr/bin/env python
"""Stage 4: what does BirdNET's geofilter actually do, scored against truth?

Runs the §4 metrics over a stored run and prints the tables that go into
PLAN.md. Headless by design — the app renders these same functions, so any
number on screen can be re-derived here (and `crosscheck.py` proves it).

    .venv/bin/python scripts/forensics.py --run birdnet-c102ff26
    .venv/bin/python scripts/forensics.py --run <id> --theta 0.5 --mode profile
"""
from __future__ import annotations

import argparse

import polars as pl

from bex import ingest, stats, store, views
from bex.config import load_config
from bex.profiles import load_profile
from bex.schemas import RunManifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--theta", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.15)
    ap.add_argument("--mode", default="geofilter",
                    choices=["geofilter", "profile", "either"])
    ap.add_argument("--profile", help="override the run's profile")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    manifest = RunManifest.load(cfg.store_dir, args.run)
    det = store.read_detections(cfg.store_dir, args.run)
    recordings, ann = ingest.read_dataset(cfg.store_dir, manifest.dataset)
    profile = load_profile(cfg.profiles_dir / (args.profile or manifest.profile))
    common = views.load_common_names(cfg.store_dir, args.run)
    name = lambda k: common.get(k, k)  # noqa: E731

    cfgm = stats.MuddinessConfig(theta=args.theta, delta=args.delta)
    marked = stats.mark_implausible(det, profile, mode=args.mode)
    sf_thresh = manifest.geofilter.get("sf_thresh", 0.03)

    print(f"run {args.run} · {manifest.model_name} {manifest.model_version} · "
          f"dataset {manifest.dataset} · profile {profile.name}")
    print(f"judge: {args.mode} · θ={args.theta} · Δ={args.delta} · "
          f"occurrence cutoff {sf_thresh}")
    print(f"{len(recordings)} recordings, {len(det):,} stored detection rows\n")

    # ---- 1. the headline rates ------------------------------------------- #
    rate = stats.muddy_window_rate(marked, cfgm.theta)
    masses = stats.window_impostor_mass(marked, cfgm.theta)
    events = stats.suppression_events(marked, cfgm.theta)
    print("== Muddiness")
    print(f"  detection-positive windows      {rate['windows']:,}")
    print(f"  of which carry an impossibility {rate['muddy_windows']:,} "
          f"({100 * rate['muddy_window_rate']:.1f}%)")
    print(f"  suppression events (rows)       {len(events):,}")
    if len(masses):
        print(f"  impostor mass  median {masses['impostor_mass'].median():.4f} · "
              f"mean {masses['impostor_mass'].mean():.4f} · "
              f"p90 {masses['impostor_mass'].quantile(0.9):.4f}")

    print("\n  most-suppressed species (windows ≥θ · max score · occurrence):")
    top_events = (
        events.group_by("species_key")
        .agg(windows=pl.len(), max_score=pl.col("score_raw").max(),
             occ=pl.col("occ_score").max())
        .sort("windows", descending=True)
        .head(args.top)
    )
    for r in top_events.iter_rows(named=True):
        print(f"    {r['windows']:6,}  {r['max_score']:.3f}  {r['occ']:.4f}  "
              f"{name(r['species_key'])}")

    # ---- 2. shadowed detections ------------------------------------------ #
    shadow = stats.shadowed_detections(marked, cfgm)
    n_cong = int(shadow["congeneric"].sum()) if len(shadow) else 0
    print(f"\n== Shadowed detections (a plausible bird with an impossible rival "
          f"within Δ={args.delta})")
    print(f"  {len(shadow):,} shadowed detections, {n_cong:,} congeneric "
          f"({100 * n_cong / len(shadow):.2f}%)" if len(shadow) else "  none")
    if len(shadow):
        print("\n  worst offending pairs (windows · mean gap · congeneric):")
        for r in stats.shadowed_pairs(shadow, top=args.top).iter_rows(named=True):
            mark = "same genus" if r["congeneric"] else ""
            print(f"    {r['windows']:5,}  {r['mean_gap']:+.3f}  "
                  f"{name(r['species_key'])}  ←  {name(r['rival_key'])}  {mark}")

    # ---- 3. per-species muddiness ---------------------------------------- #
    per = stats.per_species_muddiness(marked, cfgm)
    spread = stats.muddiness_spread(per)
    print(f"\n== Per-species muddiness  (species with ≥5 detected windows: "
          f"{spread['species']})")
    print(f"  companions per detection: median {spread['median']:.2f}, "
          f"IQR {spread['p25']:.2f}–{spread['p75']:.2f} ({spread['iqr']:.2f})")
    common_enough = per.filter(pl.col("windows") >= 5)
    print("\n  muddiest species (mean companions · of which impossible · windows):")
    for r in common_enough.head(args.top).iter_rows(named=True):
        print(f"    {r['companions']:5.2f}  {r['implausible_companions']:5.2f}  "
              f"{r['windows']:6,}  {name(r['species_key'])}")
    print("\n  cleanest species (the ones this model genuinely knows):")
    for r in common_enough.tail(args.top).reverse().iter_rows(named=True):
        print(f"    {r['companions']:5.2f}  {r['implausible_companions']:5.2f}  "
              f"{r['windows']:6,}  {name(r['species_key'])}")

    # ---- 4. score the filter against truth ------------------------------- #
    if ann is not None:
        errors = stats.filter_errors(marked, ann, cfgm)
        print("\n== The filter, scored against ground truth")
        if errors["degenerate_false_suppression"] or errors["degenerate_false_admission"]:
            print("  ⚠️  CIRCULAR: this judge cannot disagree with the annotations —")
            print("      the profile's plausible set was derived from them, so the")
            print("      rates below are zero by construction, not by merit.")
            print("      Use --mode geofilter to score a real location prior.")
        print(f"  detections agreeing with an annotation  "
              f"{errors['true_positive_detections']:,}")
        print(f"  of those, hidden by the filter          "
              f"{errors['false_suppressions']:,} "
              f"({100 * errors['false_suppression_rate']:.2f}% false suppression)")
        if errors["false_suppressed_species"]:
            print("    " + ", ".join(name(k) for k in
                                     errors["false_suppressed_species"][:10]))
        print(f"  detections the filter admitted          "
              f"{errors['admitted_detections']:,}")
        print(f"  of those, species never annotated       "
              f"{errors['false_admissions']:,} "
              f"({100 * errors['false_admission_rate']:.2f}% false admission)")

        excluded = stats.excluded_present_species(marked, ann, sf_thresh)
        print(f"\n  annotated species the filter can never report "
              f"(occurrence < {sf_thresh}): {len(excluded)}")
        for r in excluded.iter_rows(named=True):
            print(f"    occ {r['max_occ']:.4f}  peak score {r['max_score']:.3f}  "
                  f"{name(r['species_key'])}")

    # ---- 5. the league row ------------------------------------------------ #
    row = stats.league_row(marked, manifest.model_name, cfgm, ann)
    print("\n== League row")
    for k, v in row.items():
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
