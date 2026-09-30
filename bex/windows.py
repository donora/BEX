"""Project ground-truth boxes onto a model's window grid. From `birdsong/windows.py`.

Every model brings its own grid (BirdNET 3 s, Perch 5 s — resolved D5: we keep them
native), so the sibling's fixed-5-s windowing generalises to: *given* a grid, which
windows does each annotation box label?

The label rule is inherited unchanged, because it's the part that was hard-won:
a box labels a window when their temporal overlap is at least
`min(min_overlap_s, box duration)`. The `min` handles both extremes — a 0.3 s chip
labels a window only when fully inside it; a 100 s run of calls clears the floor
trivially. A fraction-of-box rule breaks on long boxes, fraction-of-window on short
ones. Frequency extents are ignored here: these are whole-window presence labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class GridParams:
    """One model's windowing: length, stride, and the label floor."""

    window_s: float
    hop_s: float
    min_overlap_s: float = 0.5


def n_windows(duration_s: float, g: GridParams) -> int:
    """Whole windows only — a trailing partial window is dropped, as the sibling
    does, so every window has exactly window_s of audio behind its score."""
    return max(0, int(math.floor((duration_s - g.window_s) / g.hop_s)) + 1)


def window_starts(duration_s: float, g: GridParams) -> np.ndarray:
    return np.arange(n_windows(duration_s, g)) * g.hop_s


def label_windows(
    boxes: pl.DataFrame,
    duration_s: float,
    g: GridParams,
    species_keys: list[str],
) -> np.ndarray:
    """Boxes -> multi-hot labels on this grid.

    `boxes` needs columns start_s, end_s, species_key. Returns uint8 array of
    shape (n_windows, len(species_keys)); column j means species_keys[j]. Boxes
    for species not in `species_keys` raise — silently dropping a species is the
    taxonomy module's cardinal sin, and it's no better here.
    """
    return label_starts(boxes, window_starts(duration_s, g), g, species_keys)


def label_starts(
    boxes: pl.DataFrame,
    starts: np.ndarray,
    g: GridParams,
    species_keys: list[str],
) -> np.ndarray:
    """The same rule, for an explicit (ascending) list of window start times.

    Evaluation needs this form rather than the duration-derived one: a run's
    stored score matrix has the window starts the model *actually* produced, and
    recomputing them from the recording duration can differ by a window at the
    tail (the runners warn about, but tolerate, an off-by-one). Labels derived
    from a recomputed grid would then be shifted against the scores they are
    meant to judge — every metric downstream would be quietly wrong, and nothing
    would look broken. So truth is labelled on the model's own starts.
    """
    key_to_col = {k: j for j, k in enumerate(species_keys)}
    unknown = set(boxes["species_key"].unique()) - set(species_keys) if len(boxes) else set()
    if unknown:
        raise ValueError(f"boxes contain species not in the vocabulary: {sorted(unknown)}")

    starts = np.asarray(starts, dtype=np.float64)
    if starts.size > 1 and np.any(np.diff(starts) < 0):
        raise ValueError("window starts must be ascending")

    y = np.zeros((len(starts), len(species_keys)), dtype=np.uint8)
    if len(starts) == 0:
        return y

    for start_s, end_s, key in boxes.select("start_s", "end_s", "species_key").iter_rows():
        box_dur = end_s - start_s
        # A box shorter than the floor must be fully contained to count.
        threshold = min(g.min_overlap_s, box_dur)

        # Narrow to windows that can touch the box: window k spans
        # [starts[k], starts[k] + window_s). It overlaps iff
        # starts[k] > start_s - window_s and starts[k] < end_s.
        k_lo = int(np.searchsorted(starts, start_s - g.window_s, side="right"))
        k_hi = int(np.searchsorted(starts, end_s, side="left"))
        if k_hi <= k_lo:
            continue

        k = np.arange(k_lo, k_hi)
        w_start = starts[k]
        overlap = np.minimum(end_s, w_start + g.window_s) - np.maximum(start_s, w_start)
        # 1e-9 absorbs float error so a box exactly meeting the floor counts;
        # `overlap > 0` keeps "any overlap" (floor 0) meaning actual shared audio,
        # not a window merely touching the box's endpoint.
        hit = (overlap > 0) & (overlap >= threshold - 1e-9)
        y[k[hit], key_to_col[key]] = 1
    return y
