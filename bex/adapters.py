"""Adapter-side helpers shared by the model runners (PLAN.md §3a-b).

Everything here is importable — and tested — without any model runtime, so the
main suite exercises the logic the runners depend on, while the runners
themselves (envs/*/run.py) stay thin wrappers around their model's API.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from .taxonomy import map_scientific_names


def birdnet_week(iso_timestamp: str) -> int | None:
    """ISO timestamp -> BirdNET's 48-week calendar, or None when unknown.

    BirdNET's geo model doesn't use ISO weeks: it divides each month into four
    quarters (weeks 1-4 of January are 1-4, of February 5-8, ...), 48 per year.
    None (timestamp unknown) tells the geo model to use year-round aggregation —
    honest, and recorded in the manifest.
    """
    if not iso_timestamp:
        return None
    dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    return (dt.month - 1) * 4 + min(4, (dt.day - 1) // 7 + 1)


def vocab_species_keys(labels: list[str], split_common: bool = True) -> list[str]:
    """A model's label list -> canonical species keys, aligned index-for-index.

    Labels *define* keys here (open mapping): BirdNET's 'Genus species_Common'
    rows become 'Genus species'; its non-species classes ('Engine_Engine',
    'Human vocal_Human vocal', ...) become keys too — they're real classifier
    outputs, and a profile simply treats them as tier 3. What is *not* tolerated
    is collision: two labels folding to one key would mis-align score columns,
    so that raises rather than guesses.

    `split_common=False` for models whose labels are already the key (Perch).
    """
    report = map_scientific_names(labels, canonical=None, split_common=split_common)
    if not report.lossless:
        raise ValueError(f"unparseable labels: {report.unmapped[:5]}")
    keys = [report.mapping[label] for label in labels]
    if len(set(keys)) != len(keys):
        seen: dict[str, str] = {}
        for label, key in zip(labels, keys):
            if key in seen:
                raise ValueError(
                    f"label collision on key {key!r}: {seen[key]!r} vs {label!r}"
                )
            seen[key] = label
    return keys


def apply_occurrence(
    detections: pl.DataFrame,
    occ: dict[str, float] | None,
    sf_thresh: float,
) -> pl.DataFrame:
    """Fill the geofilter columns from an occurrence map (species_key -> score).

    `suppressed` reproduces BirdNET's own default behaviour — occurrence below
    `sf_thresh` means the species would not be on the location's list — except
    we record the fact instead of deleting the row. `occ=None` (no location for
    this recording) leaves the columns at their "no geofilter information"
    defaults. A species the geo model doesn't know keeps occ_score=NaN and is
    *not* marked suppressed: absence of evidence, recorded as absence.
    """
    if occ is None:
        return detections
    df = detections.with_columns(
        occ_score=pl.col("species_key")
        .replace_strict(occ, default=float("nan"), return_dtype=pl.Float32)
    )
    return df.with_columns(
        suppressed=(pl.col("occ_score") < sf_thresh).fill_null(False) & pl.col("occ_score").is_not_nan()
    )


def scores_to_matrix(probs: np.ndarray, dtype: type = np.float16) -> np.ndarray:
    """A model's (windows, classes) float probabilities -> the store's array,
    clipped to [0, 1] so rounding can never violate the schema.

    float16 by default (D3). Pass float32 for a saturated readout, whose useful
    range sits where float16 is coarsest — see `store.write_scores`.
    """
    return np.clip(np.asarray(probs, dtype=np.float32), 0.0, 1.0).astype(dtype)
