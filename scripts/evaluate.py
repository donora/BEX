#!/usr/bin/env python
"""Score every arm against ground truth: PR sweep, per-species AP, cmAP.

The half of PLAN.md §3d that Stage 5 left unbuilt. `compare.py` asks which arm is
muddier at a matched operating point; this asks which arm is *right*, at every
operating point at once, window by window and species by species.

Reads the full score matrices (not the detections index — see `truth.align_run`),
caches the aligned frame under `store/truth/`, and reports:

  * **cmAP** — per-species AP, averaged unweighted over species. The headline.
  * **micro AP** — pooled over all (window, species) pairs; easier, and dominated
    by whichever birds sing most. Reported beside cmAP so the gap is visible.
  * **θ under three rules** — 95% precision, max F1, max F2 — globally and, with
    `--export`, per species. A θ chosen by one rule is not the θ chosen by another,
    and the table says what each costs.

    .venv/bin/python scripts/evaluate.py --dataset sne
    .venv/bin/python scripts/evaluate.py --dataset sne --grid shared
    .venv/bin/python scripts/evaluate.py --dataset sne --export thresholds.csv
"""
from __future__ import annotations

import argparse
import time

import polars as pl

from bex import ingest, metrics, store, truth, views
from bex.config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="sne")
    ap.add_argument("--grid", choices=("native", "shared"), default="native",
                    help="native: each model's own windows (use this to pick θ). "
                         "shared: the 1 s grid, for comparing arms on one denominator.")
    ap.add_argument("--precision", type=float, default=0.95,
                    help="the precision floor rule (default 0.95)")
    ap.add_argument("--min-support", type=int, default=10,
                    help="a rule may only pick a point predicting at least this "
                         "many windows, so θ is not chosen by one lucky row")
    ap.add_argument("--min-positives", type=int, default=1,
                    help="species with fewer labelled windows are left out of cmAP")
    ap.add_argument("--export", help="write the per-species θ table here (CSV)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--rebuild", action="store_true", help="ignore the cached alignment")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    recordings, ann = ingest.read_dataset(cfg.store_dir, args.dataset)
    if ann is None:
        raise SystemExit(f"dataset {args.dataset!r} is unlabelled — nothing to score against")

    manifests = [m for m in views.list_runs(cfg.store_dir) if m.dataset == args.dataset]
    grid = None if args.grid == "native" else truth.SHARED_GRID
    grid_name = "native" if grid is None else "shared-1s"

    rules = (metrics.Rule("precision", args.precision, args.min_support),
             metrics.Rule("fbeta", 1.0, args.min_support),
             metrics.Rule("fbeta", 2.0, args.min_support))

    label_of = views.arm_labels(manifests)
    league, tables, metas, caveats, damaged = [], {}, {}, [], []
    for m in manifests:
        scored = store.list_scores(cfg.store_dir, m.run_id)
        if len(scored) < len(recordings):
            print(f"skipping {m.run_id}: only {len(scored)}/{len(recordings)} scored")
            continue
        label = label_of[m.run_id]

        cached = truth.aligned_path(cfg.store_dir, args.dataset, m.run_id, grid_name)
        if cached.exists() and not args.rebuild:
            aligned, meta = truth.read_aligned(cfg.store_dir, args.dataset, m.run_id, grid_name)
            print(f"{label}: cached alignment, {len(aligned):,} rows")
        else:
            t0 = time.time()
            print(f"{label}: aligning {len(scored)} recordings on the {grid_name} grid…",
                  flush=True)
            aligned, meta = truth.align_run(
                cfg.store_dir, args.dataset, m.run_id, grid=grid,
                progress=lambda rid, i, n: print(f"  [{i + 1}/{n}] {rid}", flush=True))
            truth.write_aligned(cfg.store_dir, args.dataset, m.run_id, aligned, meta)
            print(f"  {len(aligned):,} rows in {time.time() - t0:.0f}s")

        if meta["species_not_in_vocabulary"]:
            print(f"  ⚠️  {len(meta['species_not_in_vocabulary'])} annotated species are "
                  f"not in this model's vocabulary and score 0 everywhere: "
                  f"{', '.join(meta['species_not_in_vocabulary'][:5])}")

        ok, why = truth.across_window_comparable(m)
        if not ok:
            print(f"  ⚠️  {why}")
            caveats.append(label)

        res = truth.rank_resolution(aligned)
        if res["damaged"]:
            print(f"  ⚠️  {label}: only {res['median_distinct_fraction']:.1%} of the "
                  "top-ranked windows have distinguishable scores. This arm's AP "
                  "measures the float16 store as much as the model — see "
                  "truth.rank_resolution.")
            damaged.append(label)

        row = metrics.summary(aligned, rules, args.min_positives)
        row["arm"] = label
        league.append(row)
        tables[label] = metrics.per_species_table(aligned, rules)
        metas[label] = meta

    if not league:
        raise SystemExit("no complete runs to evaluate")

    first = next(iter(metas.values()))
    print(f"\ndataset {args.dataset} · {first['n_recordings']} recordings · "
          f"{first['n_species']} annotated species · {grid_name} grid")
    print("Evaluation label space is the annotated species only — scoring over a "
          "model's\nfull vocabulary would count every unlabelled species as a false "
          "positive.\n")

    if caveats:
        verb = "stores" if len(caveats) == 1 else "store"
        print(f"⚠️  {', '.join(caveats)} {verb} softmax scores. Per-species AP ranks "
              f"across\n    windows and softmax normalises within one, . Measured on SNE "
              f"against a\n    float32 sigmoid re-run, that is worth ~+0.01 cmAP and "
              f"nothing in micro AP — a\n    real caveat, and a small one. The "
              f"muddiness statistics in compare.py are\n    unaffected: they are "
              f"within-window.\n")

    if damaged:
        verb = "is" if len(damaged) == 1 else "are"
        print(f"⚠️  {', '.join(damaged)} {verb} QUANTISATION-DAMAGED: the stored "
              f"scores cannot\n    separate the top-ranked windows, so the AP and "
              f"cmAP below are lower bounds\n    on the store, not measurements of "
              f"the model. Re-run storing float32 scores.\n")

    print("== League")
    keep = ["arm", "cmap", "micro_ap", "n_species", "n_excluded", "n_windows"]
    keep += [c for r in rules for c in (f"theta_{r.slug}", f"precision_{r.slug}",
                                        f"recall_{r.slug}")]
    with pl.Config(tbl_rows=20, tbl_cols=30, float_precision=4,
                   tbl_formatting="ASCII_FULL_CONDENSED"):
        print(pl.DataFrame(league).select([c for c in keep if c in league[0]]))

    print(f"\nθ columns above are one global threshold per arm. Per species they vary "
          f"widely —\nthat spread is the argument for per-species thresholds:\n")
    floor = rules[0]
    for label, t in tables.items():
        feasible = t.filter(pl.col(f"feasible_{floor.slug}"))
        th = feasible[f"theta_{floor.slug}"]
        print(f"  {label:<14} {floor.label}: reachable for {len(feasible)}/{len(t)} "
              f"species" + (f", θ from {th.min():.4f} to {th.max():.4f} "
                            f"(median {th.median():.4f})" if len(feasible) else ""))

    names = views.merge_common_names(cfg.store_dir,
                                     [m["run_id"] for m in metas.values()])
    for label, t in tables.items():
        print(f"\n== {label}: best and worst species by AP")
        show = (t.with_columns(pl.col("species_key")
                               .map_elements(lambda k: names.get(k, k), return_dtype=pl.Utf8)
                               .alias("species"))
                 .select("species", "n_positive", "ap", f"theta_{floor.slug}",
                         f"recall_{floor.slug}", "theta_f1", "recall_f1"))
        with pl.Config(tbl_rows=2 * args.top, tbl_cols=20, float_precision=4,
                       tbl_formatting="ASCII_FULL_CONDENSED", fmt_str_lengths=28):
            print(pl.concat([show.head(args.top), show.tail(args.top)]))

    if args.export:
        out = pl.concat([
            t.with_columns(
                pl.lit(label).alias("arm"),
                pl.col("species_key")
                  .map_elements(lambda k: names.get(k, k), return_dtype=pl.Utf8)
                  .alias("common_name"),
                pl.lit(grid_name).alias("grid"))
            for label, t in tables.items()
        ])
        cols = ["arm", "grid", "species_key", "common_name"] + [
            c for c in out.columns if c not in ("arm", "grid", "species_key", "common_name")]
        out.select(cols).write_csv(args.export)
        print(f"\n{len(out)} rows -> {args.export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
