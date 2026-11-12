"""Measure how a model's scores typically turn into "is it real" (bex.ecolabel.TYPICAL).

    .venv/bin/python scripts/typical_calibration.py perch-495c1c32 birdnet-c102ff26

Needs the SNE test rig ingested as dataset `sne` (bex ingest-sne) and runs over
it. Truth: a window-species pair is real when an expert box of that species
overlaps the window by 0.5 s or more (an inner join: no box, not real).
Candidates: each window's top 3, on the Sierra Nevada species list (tier <= 2).
Prints a, b for p = sigmoid(a * logit(score) + b), with a check by score band.
"""
import sys

import numpy as np
import polars as pl

from bex import ecolabel as E
from bex.config import load_config


def main(runs: list[str]) -> None:
    cfg = load_config()
    ann = pl.read_parquet(f"{cfg.store_dir}/datasets/sne/annotations.parquet")
    tiers = dict(pl.read_csv("profiles/us-ca-sierra/tiers.csv").select("species_key", "tier").iter_rows())
    for run in runs:
        pool = E.candidate_pool(cfg.store_dir, "sne", run).filter(
            pl.col("species_key").is_in([k for k, v in tiers.items() if v <= 2]))
        hit = (pool.join(ann.select("recording_id", "species_key", b0="start_s", b1="end_s"),
                         on=["recording_id", "species_key"], how="inner")
               .filter(pl.min_horizontal("end_s", "b1") - pl.max_horizontal("start_s", "b0") >= 0.5)
               .select(E.KEY).unique().with_columns(y=pl.lit(1.0)))
        t = pool.join(hit, on=E.KEY, how="left").with_columns(pl.col("y").fill_null(0.0))
        a, b, _ = E._platt(E._logit(t["score"].to_numpy()), t["y"].to_numpy())
        by_rank = t.group_by("rank").agg(pl.col("y").mean()).sort("rank")["y"].round(2).to_list()
        print(f"{run}: a={a:.3f} b={b:.3f}  ({t.height:,} candidates, {t['y'].mean():.0%} real; by rank {by_rank})")
        for q in np.quantile(t["score"], [0.1, 0.3, 0.5, 0.7, 0.9]):
            m = np.abs(np.log(t["score"].to_numpy()) - np.log(q)) < 0.15
            fit = 1 / (1 + np.exp(-(a * E._logit(q) + b)))
            print(f"   score {q:.3f}: fitted {fit:.2f}, observed {t['y'].to_numpy()[m].mean():.2f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["perch-495c1c32", "birdnet-c102ff26"])
