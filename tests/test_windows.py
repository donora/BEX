import numpy as np
import polars as pl

from bex.windows import GridParams, label_windows, n_windows, window_starts

BIRDNET = GridParams(window_s=3.0, hop_s=3.0)
PERCH = GridParams(window_s=5.0, hop_s=5.0)
KEYS = ["Turdus merula", "Erithacus rubecula"]


def boxes(*rows):
    return pl.DataFrame(
        [{"start_s": a, "end_s": b, "species_key": k} for a, b, k in rows],
        schema={"start_s": pl.Float64, "end_s": pl.Float64, "species_key": pl.Utf8},
    )


def test_grid_counts():
    assert n_windows(60.0, BIRDNET) == 20
    assert n_windows(60.0, PERCH) == 12
    assert n_windows(2.0, BIRDNET) == 0        # shorter than one window
    assert window_starts(9.0, BIRDNET).tolist() == [0.0, 3.0, 6.0]


def test_short_box_needs_full_containment():
    # A 0.3 s chip: shorter than the 0.5 s floor, so only the window that fully
    # contains it is labelled.
    y = label_windows(boxes((2.9, 3.2, KEYS[0])), 9.0, BIRDNET, KEYS)
    # It straddles the 3.0 boundary — contained by neither window 0 nor 1.
    assert y.sum() == 0
    y = label_windows(boxes((3.5, 3.8, KEYS[0])), 9.0, BIRDNET, KEYS)
    assert y[:, 0].tolist() == [0, 1, 0]


def test_long_box_labels_every_window_it_covers():
    y = label_windows(boxes((0.0, 9.0, KEYS[0])), 9.0, BIRDNET, KEYS)
    assert y[:, 0].tolist() == [1, 1, 1]


def test_edge_clip_below_floor_does_not_label():
    # A long box overlapping the next window by only 0.2 s < 0.5 s floor.
    y = label_windows(boxes((0.0, 3.2, KEYS[0])), 9.0, BIRDNET, KEYS)
    assert y[:, 0].tolist() == [1, 0, 0]


def test_exact_floor_counts():
    y = label_windows(boxes((0.0, 3.5, KEYS[0])), 9.0, BIRDNET, KEYS)
    assert y[:, 0].tolist() == [1, 1, 0]


def test_same_boxes_different_grids():
    # The point of grid-agnosticism: one annotation, each model's own truth grid.
    b = boxes((4.0, 6.5, KEYS[1]))
    y_birdnet = label_windows(b, 15.0, BIRDNET, KEYS)
    y_perch = label_windows(b, 15.0, PERCH, KEYS)
    assert y_birdnet[:, 1].tolist() == [0, 1, 1, 0, 0]   # windows [3,6) and [6,9)
    assert y_perch[:, 1].tolist() == [1, 1, 0]           # windows [0,5) and [5,10)

    # A box ending exactly on a window boundary shares zero audio with the next
    # window — boundary semantics: it must NOT label it.
    y = label_windows(boxes((4.0, 6.0, KEYS[1])), 15.0, BIRDNET, KEYS)
    assert y[:, 1].tolist() == [0, 1, 0, 0, 0]


def test_unknown_species_raises():
    import pytest

    with pytest.raises(ValueError, match="not in the vocabulary"):
        label_windows(boxes((0.0, 1.0, "Imaginarius birdus")), 9.0, BIRDNET, KEYS)


def test_empty_boxes_ok():
    y = label_windows(boxes(), 9.0, BIRDNET, KEYS)
    assert y.shape == (3, 2) and y.sum() == 0
