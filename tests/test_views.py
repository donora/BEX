from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parent.parent

from bex.profiles import Profile
from bex.schemas import DETECTIONS_SCHEMA
from bex.views import (
    CATEGORICAL,
    OTHER_COLOR,
    click_to_time,
    first_detection,
    lane_blocks,
    pack_rows,
    species_palette,
    window_grid,
    window_table,
)


def det_rows(*rows) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "recording_id": "r1", "run_id": "run",
                "start_s": s, "end_s": s + 3.0, "species_key": k,
                "score_raw": sc, "occ_score": occ, "suppressed": sup,
                "rank_in_window": 1,
            }
            for (s, k, sc, occ, sup) in rows
        ],
        schema=DETECTIONS_SCHEMA,
    )


def test_lane_blocks_keeps_co_detections():
    det = det_rows(
        (0.0, "Turdus migratorius", 0.9, 0.6, False),
        (0.0, "Catharus guttatus", 0.7, 0.5, False),     # a real co-detection
        (0.0, "Hylocichla mustelina", 0.8, 0.0, True),   # the hidden layer
        (3.0, "Catharus guttatus", 0.4, 0.5, False),
        (3.0, "Turdus migratorius", 0.05, 0.6, False),   # below theta
        (9.0, "Catharus guttatus", 0.9, 0.5, False),     # outside span
    )
    visible, suppressed = lane_blocks(det, "r1", 0.0, 9.0, theta=0.1)
    # Both species in window 0 survive, best score first; window 3 keeps its one.
    assert visible["species_key"].to_list() == [
        "Turdus migratorius", "Catharus guttatus", "Catharus guttatus"]
    assert visible["start_s"].to_list() == [0.0, 0.0, 3.0]
    assert suppressed["species_key"].to_list() == ["Hylocichla mustelina"]


def test_lane_blocks_per_window_cap():
    det = det_rows(*[(0.0, f"Species {c}", s, 0.5, False)
                     for c, s in zip("abcd", [0.9, 0.8, 0.7, 0.6])])
    top1, _ = lane_blocks(det, "r1", 0.0, 3.0, theta=0.1, per_window=1)
    assert top1["species_key"].to_list() == ["Species a"]
    top2, _ = lane_blocks(det, "r1", 0.0, 3.0, theta=0.1, per_window=2)
    assert top2["species_key"].to_list() == ["Species a", "Species b"]


def test_species_palette_fixed_order_and_other():
    frames = [det_rows(*[(3.0 * i, f"Species {chr(97 + i)}", 0.5, 0.5, False)
                         for i in range(10)])]
    # Species a..j, one block each -> ties broken alphabetically; first 8 get
    # categorical slots in order, the rest are Other.
    pal = species_palette(*frames)
    assert [pal[f"Species {c}"] for c in "abcdefgh"] == CATEGORICAL
    assert pal["Species i"] == pal["Species j"] == OTHER_COLOR


def test_window_table_columns():
    det = det_rows(
        (0.0, "Turdus migratorius", 0.9, 0.6, False),
        (0.0, "Strepera graculina", 0.6, 0.001, True),
    )
    profile = Profile(name="p", display_name="P", tiers={"Turdus migratorius": 1})
    t = window_table(det, "r1", 1.5, profile, {"Turdus migratorius": "American Robin"})
    rows = t.to_dicts()
    assert rows[0]["species"] == "American Robin"
    assert rows[0]["tier"] == "regular"
    assert rows[1]["species"] == "Strepera graculina"  # no common name known
    assert rows[1]["tier"] == "implausible"            # absent from profile
    assert rows[1]["suppressed"] is True
    assert rows[1]["xeno_canto"] == "https://xeno-canto.org/species/Strepera-graculina"


def test_window_grid_snaps_to_model_windows():
    # The inspector can only address windows the model actually scored.
    assert window_grid(0.0, 12.0, 3.0) == [0.0, 3.0, 6.0, 9.0]
    # Off-grid viewport: every window that overlaps the view is selectable,
    # including the partly-visible one at each edge.
    assert window_grid(7.0, 16.0, 3.0) == [6.0, 9.0, 12.0, 15.0]
    # A view shorter than one window still overlaps that window.
    assert window_grid(0.0, 2.0, 3.0) == [0.0]


def test_first_detection_skips_empty_windows():
    det = det_rows(
        (0.0, "Turdus migratorius", 0.05, 0.6, False),   # quiet window
        (6.0, "Turdus migratorius", 0.95, 0.6, False),   # the first real one
        (9.0, "Catharus guttatus", 0.99, 0.5, False),
    )
    assert first_detection(det, "r1", 0.0, 30.0, theta=0.9) == 6.0
    assert first_detection(det, "r1", 0.0, 30.0, theta=0.01) == 0.0
    assert first_detection(det, "r1", 0.0, 30.0, theta=0.999) is None


def test_pack_rows_preserves_caller_order_within_window():
    # Best score must land on row 0, or which species sits on top of the lane
    # would depend on sort stability.
    blocks = det_rows(
        (0.0, "Turdus migratorius", 0.9, 0.6, False),
        (0.0, "Catharus guttatus", 0.7, 0.5, False),
        (3.0, "Catharus guttatus", 0.8, 0.5, False),
    ).select("start_s", "end_s", "species_key", "score_raw")
    packed = pack_rows(blocks)
    by_key = {(r["species_key"], r["start_s"]): r["row"] for r in packed.to_dicts()}
    assert by_key[("Turdus migratorius", 0.0)] == 0
    assert by_key[("Catharus guttatus", 0.0)] == 1
    assert by_key[("Catharus guttatus", 3.0)] == 0   # next window reuses row 0


def test_click_to_time_maps_image_fraction_to_time():
    # Plotting area occupies the middle 80% of the image; the left margin is
    # the lane labels, which must not select anything.
    call = lambda frac: click_to_time(frac, 0.1, 0.9, 0.0, 12.0)
    assert call(0.1) == pytest.approx(0.0)
    assert call(0.9) == pytest.approx(12.0)
    assert call(0.5) == pytest.approx(6.0)
    assert call(0.05) is None
    assert call(0.95) is None


def test_click_to_time_degenerate_inputs():
    assert click_to_time(0.5, 0.9, 0.9, 0.0, 12.0) is None
    assert click_to_time(0.5, 0.1, 0.9, 5.0, 5.0) is None


def test_window_at_uses_each_models_own_grid():
    from bex.views import window_at
    det = det_rows((0.0, "a", 0.5, 0.1, False), (3.0, "a", 0.5, 0.1, False))
    assert window_at(det, "r1", 4.0) == (3.0, 6.0)
    assert window_at(det, "r1", 3.0) == (3.0, 6.0)   # a boundary opens the next
    assert window_at(det, "r1", 7.0) is None


# --------------------------------------------------------------------------- #
# Arm labels — unique, or a comparison silently loses an arm
# --------------------------------------------------------------------------- #

def _manifest(run_id: str, transform: str, name: str = "perch", version: str = "v2"):
    from bex.schemas import RunManifest
    return RunManifest(model_name=name, model_version=version, window_s=5.0,
                       hop_s=5.0, input_sr=32000, score_transform=transform,
                       run_id=run_id)


def test_readout_token_reads_the_documented_prefix():
    from bex.views import readout_token
    assert readout_token("softmax over raw logits — sigmoid saturates") == "softmax"
    assert readout_token("per-class sigmoid over raw logits (sensitivity 1.0)") == "sigmoid"
    assert readout_token("sigmoid") == "sigmoid"
    assert readout_token("something we have never seen") == ""


def test_distinct_models_keep_their_plain_labels():
    from bex.views import arm_labels
    got = arm_labels([_manifest("perch-1", "softmax over raw logits"),
                      _manifest("bn-1", "sigmoid", name="birdnet", version="2.4")])
    assert got == {"perch-1": "perch-v2", "bn-1": "birdnet-2.4"}


def test_two_readouts_of_one_model_are_told_apart():
    """The bug this exists to prevent: both runs are 'perch-v2', so a dict keyed on
    the label keeps one and drops the other with no error."""
    from bex.views import arm_labels
    got = arm_labels([_manifest("perch-1", "softmax over raw logits"),
                      _manifest("perch-2", "per-class sigmoid over raw logits")])
    assert got["perch-1"] == "perch-v2 (softmax)"
    assert got["perch-2"] == "perch-v2 (sigmoid)"
    assert len(set(got.values())) == 2


def test_identical_model_and_readout_falls_back_to_the_run_id():
    from bex.views import arm_labels
    got = arm_labels([_manifest("perch-aaaaaaaa", "softmax over raw logits"),
                      _manifest("perch-bbbbbbbb", "softmax over raw logits")])
    assert len(set(got.values())) == 2
    assert all(v.startswith("perch-v2 (softmax) [") for v in got.values())


def test_real_runs_in_the_store_all_get_distinct_labels():
    from bex.views import arm_labels, list_runs
    store_dir = REPO / "store"
    if not (store_dir / "runs").exists():
        pytest.skip("no local store")
    labels = arm_labels(list_runs(store_dir))
    assert len(set(labels.values())) == len(labels), labels


# --------------------------------------------------------------------------- #
# Navigator focus — detections and truth must narrow together
# --------------------------------------------------------------------------- #

def _nav_det(*rows) -> pl.DataFrame:
    """(start_s, species, score) -> a detections frame for one recording."""
    return pl.DataFrame(
        [{"recording_id": "r1", "run_id": "run", "start_s": s, "end_s": s + 3.0,
          "species_key": k, "score_raw": sc, "occ_score": float("nan"),
          "suppressed": False, "rank_in_window": 1} for (s, k, sc) in rows],
        schema=DETECTIONS_SCHEMA,
    )


def _nav_ann(*rows) -> pl.DataFrame:
    from bex.schemas import ANNOTATIONS_SCHEMA
    return pl.DataFrame(
        [{"recording_id": "r1", "start_s": a, "end_s": b, "low_hz": float("nan"),
          "high_hz": float("nan"), "species_key": k} for (a, b, k) in rows],
        schema=ANNOTATIONS_SCHEMA,
    )


ROBIN_K = "Turdus migratorius"
THRUSH_K = "Catharus guttatus"


# --------------------------------------------------------------------------- #
# Focus options — the picker must not be narrowed by θ or by which arm is primary
# --------------------------------------------------------------------------- #

def test_focus_options_union_truth_and_every_arm():
    from bex.views import focus_options
    det = {"a": _nav_det((0.0, ROBIN_K, 0.9)),
           "b": _nav_det((0.0, "Strepera graculina", 0.9))}
    ann = _nav_ann((0.0, 10.0, THRUSH_K))
    got = focus_options(det, "r1", 0.5, ann)
    assert set(got["species_key"]) == {ROBIN_K, THRUSH_K, "Strepera graculina"}


def test_an_annotated_species_survives_any_threshold():
    """The bug: raising θ removed annotated birds from the picker, so the species
    a model was failing on was the one you could not select."""
    from bex.views import focus_options
    det = {"a": _nav_det((0.0, ROBIN_K, 0.2))}
    ann = _nav_ann((0.0, 10.0, ROBIN_K))
    for theta in (0.1, 0.5, 0.99):
        got = focus_options(det, "r1", theta, ann)
        assert ROBIN_K in got["species_key"].to_list(), f"lost at θ={theta}"
    # ...and it is marked as detected by nobody once θ passes its score.
    strict = focus_options(det, "r1", 0.99, ann).filter(
        pl.col("species_key") == ROBIN_K).row(0, named=True)
    assert strict["in_truth"] is True and strict["n_models"] == 0


def test_focus_options_count_how_many_arms_detect_each_species():
    from bex.views import focus_options
    det = {"a": _nav_det((0.0, ROBIN_K, 0.9)), "b": _nav_det((0.0, ROBIN_K, 0.9)),
           "c": _nav_det((0.0, THRUSH_K, 0.9))}
    got = focus_options(det, "r1", 0.5, None)
    by_key = {r["species_key"]: r for r in got.iter_rows(named=True)}
    assert by_key[ROBIN_K]["n_models"] == 2
    assert by_key[THRUSH_K]["n_models"] == 1
    assert by_key[ROBIN_K]["in_truth"] is False      # no annotations passed


def test_focus_options_puts_annotated_species_first():
    from bex.views import focus_options
    det = {"a": _nav_det((0.0, "Strepera graculina", 0.9))}
    got = focus_options(det, "r1", 0.5, _nav_ann((0.0, 10.0, ROBIN_K)))
    assert got["species_key"].to_list()[0] == ROBIN_K
    assert got["in_truth"].to_list() == [True, False]


def test_focus_options_ignores_other_recordings():
    from bex.views import focus_options
    det = {"a": _nav_det((0.0, ROBIN_K, 0.9))}
    assert focus_options(det, "other", 0.5, _nav_ann((0.0, 1.0, THRUSH_K))).is_empty()


def test_palette_colours_annotated_species_first():
    """A bird that is annotated but rarely detected must not be the one greyed out."""
    loud = pl.DataFrame({"species_key": [f"loud{i}" for i in range(9) for _ in range(5)]})
    ann = pl.DataFrame({"species_key": ["quiet"]})
    by_detections = species_palette(loud, pl.DataFrame({"species_key": ["quiet"]}))
    assert by_detections["quiet"] == OTHER_COLOR           # the old failure
    pal = species_palette(loud, priority=ann)
    assert pal["quiet"] == CATEGORICAL[0]
    assert sum(1 for v in pal.values() if v != OTHER_COLOR) == len(CATEGORICAL)


def test_navigator_acts_on_the_chart_that_was_clicked():
    """The bug: a truth-chart click stayed selected and shadowed every later
    click on a model chart, so the view never moved."""
    from bex.views import new_click
    names = ["pick0", "pick1"]
    t, seen = new_click({"pick0": [{"bin_start_s": 100.0}]}, names, {})
    assert t == 100.0
    # the same selection on the next rerun is not a new click
    t, seen = new_click({"pick0": [{"bin_start_s": 100.0}]}, names, seen)
    assert t is None
    # clicking the model chart: truth's stale selection is still there
    t, seen = new_click({"pick0": [{"bin_start_s": 100.0}],
                         "pick1": [{"bin_start_s": 900.0}]}, names, seen)
    assert t == 900.0
    # and a new truth click after that still works
    t, _ = new_click({"pick0": [{"bin_start_s": 40.0}],
                      "pick1": [{"bin_start_s": 900.0}]}, names, seen)
    assert t == 40.0


def test_one_common_name_whichever_model_reported_it(tmp_path):
    """Perch's label file is scientific names only; its rows must still read
    'American Robin', from any other run's labels or the dataset's species list."""
    import json
    from bex.views import common_name_index, display_name

    def run(rid, labels):
        d = tmp_path / "store" / "runs" / rid
        d.mkdir(parents=True)
        (d / "labels.txt").write_text("\n".join(labels))
        (d / "manifest.json").write_text(json.dumps({
            "run_id": rid, "model_name": rid, "model_version": "1", "window_s": 3.0,
            "hop_s": 3.0, "input_sr": 32000, "score_transform": "sigmoid",
            "created_utc": "2026-01-01"}))

    run("birdnet", ["Turdus migratorius_American Robin"])
    run("perch", ["Turdus migratorius", "Keys_jangling", "Regulus satrapa"])
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "species.csv").write_text(
        "Species eBird Code,Scientific Name,Common Name\n"
        "gockin,Regulus satrapa,Golden-crowned Kinglet\n")

    names = common_name_index(tmp_path / "store", tmp_path / "labels")
    assert names["Turdus migratorius"] == "American Robin"
    assert names["Regulus satrapa"] == "Golden-crowned Kinglet"     # species list
    assert "Keys" not in names                                     # no split junk
    assert display_name("Keys_jangling", names) == "Keys jangling"
    assert display_name("Genus unknown", names) == "Genus unknown"


def test_a_model_keeps_its_colour_when_a_newer_run_arrives():
    from types import SimpleNamespace as M
    from bex.views import model_styles
    old = [M(run_id="birdnet", created_utc="2026-01-01"), M(run_id="perch", created_utc="2026-02-01")]
    before = model_styles(old)
    after = model_styles(old + [M(run_id="mine", created_utc="2026-03-01")])
    assert all(after[r]["colour"] == before[r]["colour"] for r in before)
    assert len({s["colour"] for s in after.values()}) == 3
