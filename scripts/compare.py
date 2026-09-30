#!/usr/bin/env python
"""Stage 5: the cross-model league table, at matched operating points.

Compares every stored run for a dataset. The arms are judged by the *profile*,
not by each model's own geofilter, because only one of them has a geofilter —
applying the same plausibility list to all of them is what makes the numbers
mean the same thing.

Thresholds are chosen per arm to yield the same number of detections. Comparing
at a shared theta would compare nothing: BirdNET's scores are per-class
sigmoids, Perch's a softmax over 14,795 classes, so 0.1 is a different claim in
each. The theta each arm needed is reported, because that is itself a fact
about the model.

    .venv/bin/python scripts/compare.py --dataset sne
    .venv/bin/python scripts/compare.py --dataset sne --detections 50000
"""
from __future__ import annotations

import argparse

import polars as pl

from bex import ingest, stats, store, views
from bex.config import load_config
from bex.profiles import load_profile


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="sne")
    ap.add_argument("--profile", help="override each run's profile")
    ap.add_argument("--detections", type=int,
                    help="matched operating point (default: the smallest arm at θ=0.1)")
    ap.add_argument("--delta", type=float, default=0.15)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--sweep", action="store_true",
                    help="report every arm across a range of operating points")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    recordings, ann = ingest.read_dataset(cfg.store_dir, args.dataset)
    manifests = [m for m in views.list_runs(cfg.store_dir) if m.dataset == args.dataset]
    if not manifests:
        raise SystemExit(f"no runs for dataset {args.dataset!r}")

    profile_name = args.profile or manifests[0].profile
    profile = load_profile(cfg.profiles_dir / profile_name)

    label_of = views.arm_labels(manifests)
    arms: dict[str, pl.DataFrame] = {}
    names: dict[str, dict[str, str]] = {}
    for m in manifests:
        scored = store.list_scores(cfg.store_dir, m.run_id)
        if len(scored) < len(recordings):
            print(f"skipping {m.run_id}: only {len(scored)}/{len(recordings)} "
                  "recordings scored")
            continue
        det = store.read_detections(cfg.store_dir, m.run_id)
        label = label_of[m.run_id]
        arms[label] = stats.mark_implausible(det, profile, mode="profile")
        names[label] = views.load_common_names(cfg.store_dir, m.run_id)

    if len(arms) < 2:
        raise SystemExit(f"need at least two complete runs; have {list(arms)}")

    print(f"dataset {args.dataset} · profile {profile_name} · judge: profile tiers")
    print(f"arms: {', '.join(arms)}\n")

    table = stats.matched_league(arms, args.detections, args.delta, ann)
    keep = ["arm", "theta", "detections", "detection_windows", "muddy_window_rate",
            "median_impostor_mass", "shadowed_detections", "congeneric_shadowed",
            "species_muddiness_median", "species_muddiness_iqr",
            "false_suppressions", "false_admissions"]
    # The profile judge is the only one every arm shares, but if that profile
    # was derived from the annotations it cannot disagree with them.
    probe = stats.filter_errors(next(iter(arms.values())), ann,
                                stats.MuddinessConfig(delta=args.delta)) if ann is not None else {}
    if probe.get("degenerate_false_suppression") or probe.get("degenerate_false_admission"):
        print("⚠️  false_suppressions / false_admissions are structurally zero here:")
        print(f"    the '{profile_name}' profile's plausible set was derived from the")
        print("    annotations, so the profile judge cannot disagree with them. Those")
        print("    two columns carry no information in this table; the muddiness and")
        print("    shadowing columns do.\n")

    if args.sweep:
        # One matched point can mislead: matching a small budget drives the
        # sharper-scoring arm to a threshold near its ceiling, where it has
        # almost nothing left to be muddy with. Sweeping shows whether a
        # difference is a property of the model or of the operating point.
        #
        # The comparison is invariant to each arm's score transform: matching on
        # detection count selects the top-N ranked (window, species) pairs, and
        # both a sigmoid and a softmax are monotone in the logits, so the
        # selected set does not depend on which was applied.
        truth = set(ann["species_key"].unique()) if ann is not None else set()
        print("== Operating-point sweep\n")
        print(f"{'budget':>9} | {'arm':<14} {'θ':>7} {'muddy':>7} {'shadowed':>9} "
              f"{'congeneric':>11} {'species':>8} {'truth recall':>13}")
        for target in (5_000, 20_000, 60_000, 150_000):
            for label, det in arms.items():
                th = stats.theta_for_detection_count(det, target)
                cfgm = stats.MuddinessConfig(theta=th, delta=args.delta)
                rate = stats.muddy_window_rate(det, th)
                sh = stats.shadowed_detections(det, cfgm)
                rep = _top_species(det, th)
                recall = (f"{100 * len(rep & truth) / len(truth):.0f}%"
                          if truth else "—")
                print(f"{target:>9,} | {label:<14} {th:>7.4f} "
                      f"{100 * rate['muddy_window_rate']:>6.1f}% {len(sh):>9,} "
                      f"{int(sh['congeneric'].sum()) if len(sh) else 0:>11,} "
                      f"{len(rep):>8,} {recall:>13}")
            print()

    print("== League (matched detection count)")
    with pl.Config(tbl_rows=20, tbl_cols=20, float_precision=4,
                   tbl_formatting="ASCII_FULL_CONDENSED"):
        print(table.select([c for c in keep if c in table.columns]))

    # Where do the arms disagree, and does truth side with either?
    if ann is not None and len(arms) == 2:
        (la, da), (lb, db) = arms.items()
        ta = stats.theta_for_detection_count(da, int(table["detections"][0]))
        tb = stats.theta_for_detection_count(db, int(table["detections"][1]))
        top_a = _top_species(da, ta)
        top_b = _top_species(db, tb)
        truth = set(ann["species_key"].unique())

        print(f"\n== Agreement on species present (θ_{la}={ta:.4f}, θ_{lb}={tb:.4f})")
        for label, top in ((la, top_a), (lb, top_b)):
            hits = len(top & truth)
            print(f"  {label:14s} reports {len(top):4d} species, "
                  f"{hits:3d} annotated ({100 * hits / max(1, len(top)):.0f}% precision, "
                  f"{100 * hits / len(truth):.0f}% of the {len(truth)} truth species)")
        print(f"  both arms agree on {len(top_a & top_b)} species; "
              f"{len(top_a - top_b)} only {la}, {len(top_b - top_a)} only {lb}")

        for label, only in ((la, top_a - top_b), (lb, top_b - top_a)):
            real = sorted(only & truth)
            print(f"\n  species only {label} reports that ARE annotated ({len(real)}):")
            for k in real[:args.top]:
                print(f"    {names[label].get(k, k)}")
    return 0


def _top_species(det: pl.DataFrame, theta: float) -> set[str]:
    return set(det.filter(pl.col("score_raw") >= theta)["species_key"].unique())


if __name__ == "__main__":
    raise SystemExit(main())
