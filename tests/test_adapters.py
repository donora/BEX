import numpy as np
import polars as pl
import pytest

from bex.adapters import apply_occurrence, birdnet_week, scores_to_matrix, vocab_species_keys
from bex.schemas import validate_detections
from bex.store import ScoreMatrix, derive_detections


def test_birdnet_week():
    assert birdnet_week("2018-05-09T05:00:02") == 18   # May, 2nd quarter
    assert birdnet_week("2018-05-23T08:00:04") == 20
    assert birdnet_week("2018-05-29T05:00:02") == 20   # day 29+ folds into week 4
    assert birdnet_week("2024-01-01T00:00:00") == 1
    assert birdnet_week("2024-12-31T23:59:59") == 48
    assert birdnet_week("2018-05-16T10:38:40Z") == 19
    assert birdnet_week("") is None


def test_vocab_species_keys():
    labels = [
        "Turdus merula_Eurasian Blackbird",
        "Strepera graculina_Pied Currawong",
        "Engine_Engine",  # BirdNET's non-species classes become keys too
    ]
    assert vocab_species_keys(labels) == ["Turdus merula", "Strepera graculina", "Engine"]

    with pytest.raises(ValueError, match="collision"):
        vocab_species_keys(["Turdus merula_Blackbird", "Turdus merula_Common Blackbird"])


def test_scores_to_matrix_clips():
    m = scores_to_matrix(np.array([[1.0000001, -1e-9, 0.5]]))
    assert m.dtype == np.float16
    assert m.max() <= 1.0 and m.min() >= 0.0


def make_detections() -> pl.DataFrame:
    sm = ScoreMatrix(
        recording_id="r1",
        run_id="run",
        scores=np.array([[0.9, 0.6, 0.2]], dtype=np.float16),
        start_s=np.array([0.0], dtype=np.float32),
        window_s=3.0,
        species_keys=["Turdus migratorius", "Strepera graculina", "Engine"],
    )
    return derive_detections(sm)


def test_apply_occurrence():
    df = make_detections()
    # Robin plausible at the site, currawong not; the geo model has never heard
    # of 'Engine' so it stays NaN and unsuppressed.
    occ = {"Turdus migratorius": 0.67, "Strepera graculina": 0.001}
    out = validate_detections(apply_occurrence(df, occ, sf_thresh=0.03))

    by_key = {r["species_key"]: r for r in out.to_dicts()}
    assert by_key["Turdus migratorius"]["occ_score"] == pytest.approx(0.67)
    assert not by_key["Turdus migratorius"]["suppressed"]
    assert by_key["Strepera graculina"]["suppressed"]  # the blackbird case, logged not lost
    assert by_key["Strepera graculina"]["score_raw"] == pytest.approx(0.6, abs=1e-3)
    assert np.isnan(by_key["Engine"]["occ_score"])
    assert not by_key["Engine"]["suppressed"]


def test_apply_occurrence_no_location():
    df = make_detections()
    out = apply_occurrence(df, None, sf_thresh=0.03)
    assert out["occ_score"].is_nan().all()
    assert not out["suppressed"].any()


def test_vocab_species_keys_perch_style():
    """Perch labels are already keys: bare scientific names plus AudioSet event
    classes whose underscores are NOT a common-name separator. Splitting them
    collapses 'Bass_drum' and 'Bass_guitar' onto one key and misaligns every
    score column after it — so the split is per-model, never guessed."""
    labels = ["Turdus merula", "Bass_drum", "Bass_guitar", "Abavorana luctuosa"]
    keys = vocab_species_keys(labels, split_common=False)
    assert keys == ["Turdus merula", "Bass_drum", "Bass_guitar", "Abavorana luctuosa"]
    with pytest.raises(ValueError, match="collision on key 'Bass'"):
        vocab_species_keys(labels, split_common=True)
