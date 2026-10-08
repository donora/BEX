"""Where a dataset is in the flow, for the home page (V1.2 F0, F1)."""
import math

import polars as pl

from bex import flow
from bex import labels as L
from bex.ingest import write_dataset
from bex.schemas import ANNOTATIONS_SCHEMA, RECORDINGS_SCHEMA


def recs(labelled=False):
    return pl.DataFrame([{"recording_id": f"r{i}", "path": f"r{i}.wav", "duration_s": 120.0,
                          "sample_rate": 32000, "channels": 1, "start_time": "",
                          "site": "s", "lat": math.nan, "lon": math.nan,
                          "labelled": labelled} for i in range(2)], schema=RECORDINGS_SCHEMA)


def test_a_new_dataset_needs_its_models_first(tmp_path):
    write_dataset(tmp_path, "new", recs())
    p = flow.progress(tmp_path, tmp_path / "cache", "new")
    assert p["steps"]["models"][0] == "todo" and p["next"] == "models"
    assert p["has_labels"] is False
    assert p["steps"]["explored"][0] == "waiting"


def test_without_labels_labelling_is_not_started_and_scoring_waits(tmp_path):
    """Nothing is ever skipped: no view closes off the other."""
    write_dataset(tmp_path, "new", recs())
    p = flow.progress(tmp_path, tmp_path / "cache", "new")
    assert p["steps"]["labelled"] == ("todo", "not started yet")
    assert p["steps"]["analysed"] == ("waiting", "needs labels")


def test_explored_is_ticked_only_once_visited(tmp_path):
    write_dataset(tmp_path, "new", recs())
    flow.set_dataset_meta(tmp_path, "new", explored=True)
    assert flow.dataset_meta(tmp_path, "new")["explored"] is True
    assert flow.dataset_meta(tmp_path, "new")["audio_root"] == ""   # meta kept


def test_label_sample_progress_and_imported_datasets(tmp_path):
    write_dataset(tmp_path, "new", recs())
    ls = L.create_set(tmp_path, "new", "a", "me")
    ls.sample = L.draw_sample(recs(), 2, seed=0)
    first = L.next_chunk(ls)
    L.close_chunk(ls, first["recording_id"], first["start_s"], first["end_s"], "me")
    L.save_set(tmp_path, ls)
    p = flow.progress(tmp_path, tmp_path / "cache", "new")
    assert p["steps"]["labelled"] == ("partial", "a: 1/2 sampled chunks")
    assert p["has_labels"] is True

    ann = pl.DataFrame(schema=ANNOTATIONS_SCHEMA)
    write_dataset(tmp_path, "sne", recs(labelled=True), ann)
    assert flow.progress(tmp_path, tmp_path / "cache", "sne")["steps"]["labelled"][0] == "done"
