"""Label sets: chunks, sign-off, versions, truth and the sample (V1.2 L3–L8)."""
import math

import polars as pl
import pytest

from bex import labels as L
from bex.ingest import write_dataset
from bex.schemas import ANNOTATIONS_SCHEMA, RECORDINGS_SCHEMA

ROBIN = "Turdus migratorius"
WREN = "Troglodytes troglodytes"


def recordings(n=3, dur=300.0, labelled=False, sites=("a",)) -> pl.DataFrame:
    return pl.DataFrame([{
        "recording_id": f"r{i}", "path": f"r{i}.flac", "duration_s": dur,
        "sample_rate": 48000, "channels": 1,
        "start_time": f"2024-05-01T0{4 + i % 5}:00:00", "site": sites[i % len(sites)],
        "lat": math.nan, "lon": math.nan, "labelled": labelled} for i in range(n)],
        schema=RECORDINGS_SCHEMA)


@pytest.fixture
def store(tmp_path):
    write_dataset(tmp_path, "site", recordings())
    return tmp_path


def test_chunks_are_sixty_seconds_and_a_tiny_tail_folds_in():
    assert list(L.chunk_starts(300.0)) == [0, 60, 120, 180, 240]
    assert list(L.chunk_starts(300.5)) == [0, 60, 120, 180, 240]   # 0.5 s tail folded
    assert list(L.chunk_starts(330.0)) == [0, 60, 120, 180, 240, 300]
    assert L.chunk_end(240.0, 300.5) == 300.5


def test_boxes_only_change_in_open_chunks(store):
    ls = L.create_set(store, "site", "mine", "ann")
    bid = L.add_box(ls, "r0", 10, 12, 2000, 4000, ROBIN, "ann")
    L.close_chunk(ls, "r0", 0.0, 60.0, "ann")
    with pytest.raises(L.ClosedChunk):
        L.add_box(ls, "r0", 30, 31, 1000, 2000, WREN, "ann")
    with pytest.raises(L.ClosedChunk):
        L.delete_box(ls, bid)
    L.reopen_chunk(ls, "r0", 0.0, "missed a wren")
    L.add_box(ls, "r0", 30, 31, 1000, 2000, WREN, "ann")
    assert len(ls.boxes) == 2
    assert ls.chunk_state("r0", 0.0) == L.OPEN


def test_only_closed_chunks_are_truth(store):
    ls = L.create_set(store, "site", "mine", "ann")
    L.add_box(ls, "r0", 10, 12, 2000, 4000, ROBIN, "ann")     # chunk 0, closed below
    L.add_box(ls, "r0", 70, 72, 2000, 4000, WREN, "ann")      # chunk 60, left open
    L.add_box(ls, "r0", 20, 21, 2000, 4000, L.UNKNOWN_BIRD, "ann")
    L.close_chunk(ls, "r0", 0.0, 60.0, "ann")
    L.close_chunk(ls, "r1", 120.0, 180.0, "ann")              # listened: no birds
    t = L.truth_from_set(ls, recordings(), "mine@wx")
    assert t.annotations["species_key"].to_list() == [ROBIN]
    assert t.n_unknown == 1
    assert t.recordings == ["r0", "r1"]
    assert t.covered_s == {"r0": 60.0, "r1": 60.0}
    assert t.complete == [] and not t.exhaustive


def test_whole_recording_closed_is_complete(store):
    ls = L.create_set(store, "site", "mine", "ann")
    for s in L.chunk_starts(300.0):
        L.close_chunk(ls, "r2", float(s), float(s) + 60, "ann")
    t = L.truth_from_set(ls, recordings(), "x")
    assert t.complete == ["r2"]
    assert t.coverage.to_dicts() == [{"recording_id": "r2", "start_s": 0.0, "end_s": 300.0}]


def test_save_refuses_a_stale_copy(store):
    a = L.create_set(store, "site", "mine", "ann")
    b = L.load_set(store, "site", "mine")
    L.add_box(a, "r0", 1, 2, 1, 2, ROBIN, "ann")
    L.save_set(store, a)
    L.add_box(b, "r0", 3, 4, 1, 2, WREN, "bob")
    with pytest.raises(L.StaleWrite):
        L.save_set(store, b)
    assert L.load_set(store, "site", "mine").boxes["species_key"].to_list() == [ROBIN]


def test_published_versions_are_frozen(store):
    ls = L.create_set(store, "site", "mine", "ann")
    L.add_box(ls, "r0", 1, 2, 1, 2, ROBIN, "ann")
    L.save_set(store, ls)
    assert L.publish(store, ls) == 1
    L.add_box(ls, "r0", 3, 4, 1, 2, WREN, "ann")
    L.save_set(store, ls)
    v1 = L.load_set(store, "site", "mine", version=1)
    assert v1.boxes["species_key"].to_list() == [ROBIN]
    assert v1.read_only
    with pytest.raises(L.ClosedChunk):
        L.add_box(v1, "r0", 5, 6, 1, 2, WREN, "ann")
    assert L.versions(store, "site", "mine") == [1]
    assert L.parse_ref("mine@v1") == ("mine", 1)
    assert L.parse_ref(L.working_ref(store, "site", "mine")) == ("mine", None)


def test_working_ref_changes_with_the_labels(store):
    ls = L.create_set(store, "site", "mine", "ann")
    before = L.working_ref(store, "site", "mine")
    L.add_box(ls, "r0", 1, 2, 1, 2, ROBIN, "ann")
    L.save_set(store, ls)
    assert L.working_ref(store, "site", "mine") != before


def test_imported_is_exhaustive_and_derived_sets_copy_it(tmp_path):
    ann = pl.DataFrame([{"recording_id": "r0", "start_s": 5.0, "end_s": 6.0,
                         "low_hz": 1.0, "high_hz": 2.0, "species_key": ROBIN}],
                       schema=ANNOTATIONS_SCHEMA)
    write_dataset(tmp_path, "sne", recordings(labelled=True), ann)
    t = L.load_truth(tmp_path, "sne", L.IMPORTED)
    assert t.exhaustive and t.complete == ["r0", "r1", "r2"]
    ls = L.create_set(tmp_path, "sne", "plus", "ann", derived_from=L.IMPORTED)
    assert ls.boxes["species_key"].to_list() == [ROBIN]
    assert set(ls.chunks["state"]) == {L.CLOSED}
    assert L.truth_from_set(ls, recordings(labelled=True), "x").exhaustive


def test_within_keeps_only_windows_wholly_inside_coverage():
    cov = pl.DataFrame({"recording_id": ["r0"], "start_s": [60.0], "end_s": [120.0]})
    w = pl.DataFrame({"recording_id": ["r0"] * 4 + ["r1"],
                      "start_s": [57.0, 60.0, 117.0, 118.0, 60.0],
                      "end_s": [60.0, 63.0, 120.0, 121.0, 63.0]})
    assert L.within(w, cov)["start_s"].to_list() == [60.0, 117.0]
    # The recording's last span is open-ended: a window overhanging the end counts.
    assert L.within_or_last(w, cov, {"r0": 120.0})["start_s"].to_list() == [60.0, 117.0, 118.0]


def test_sample_spreads_over_strata_and_recordings_and_extends():
    recs = recordings(n=6, dur=600.0, sites=("a", "b"))
    s = L.draw_sample(recs, 12, seed=1)
    assert len(s) == 12 and s["order"].to_list() == list(range(12))
    assert s["recording_id"].n_unique() == 6           # every recording before seconds
    assert s.select(pl.struct("recording_id", "start_s").n_unique()).item() == 12
    first = s.head(4)
    assert first["stratum"].n_unique() >= 2            # a prefix is already spread
    more = L.draw_sample(recs, 6, seed=2, existing=s)
    assert more.head(12).equals(s)                     # never redrawn
    assert len(more) == 18
    assert more.select(pl.struct("recording_id", "start_s").n_unique()).item() == 18


def test_sample_can_take_whole_recordings():
    s = L.draw_sample(recordings(n=4), 4, seed=0, whole_recordings=1)
    whole = s.filter(pl.col("whole"))
    assert whole["recording_id"].n_unique() == 1 and len(whole) == 5
    assert len(s) == 9


def test_time_bands():
    assert L.time_band("2024-05-01T05:30:00", 0) == "dawn"
    assert L.time_band("2024-05-01T07:59:00", 120) == "day"
    assert L.time_band("2024-05-01T23:00:00", 0) == "night"
    assert L.time_band("", 0) == "time unknown"


def test_next_chunk_and_progress(store):
    recs = recordings()
    ls = L.create_set(store, "site", "mine", "ann")
    ls.sample = L.draw_sample(recs, 3, seed=0)
    first = L.next_chunk(ls)
    L.add_box(ls, first["recording_id"], first["start_s"] + 1, first["start_s"] + 2,
              1000, 2000, ROBIN, "ann")
    L.close_chunk(ls, first["recording_id"], first["start_s"], first["end_s"], "ann")
    L.close_chunk(ls, "r0", 240.0, 300.0, "ann") if (first["recording_id"], first["start_s"]) != ("r0", 240.0) else None
    assert L.next_chunk(ls)["order"] == 1
    p = L.progress(ls, recs)
    assert p["sample_closed"] == 1 and p["n_species"] == 1
    assert p["closed_chunks"] == 2 and p["by_hand"] == 1
    acc = L.accumulation(ls)
    assert acc["species"].to_list()[-1] == 1


def test_projection_shrinks_as_root_n():
    assert L.units_for_margin(0.10) == 62
    assert L.units_for_margin(0.05) == 246
    lo, hi = L.project(0.2, 10, 0.1)       # half the width -> about four times the units
    assert lo <= 40 <= hi


def test_csv_round_trip_in_sne_columns(store):
    recs = recordings()
    ls = L.create_set(store, "site", "mine", "ann")
    L.add_box(ls, "r1", 10, 12, 2000, 4000, ROBIN, "ann")
    L.close_chunk(ls, "r1", 0.0, 60.0, "ann")
    text = L.export_csv(ls, recs, {ROBIN: "American Robin"})
    assert text.splitlines()[0].startswith(
        "Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species")
    assert "r1.flac" in text and "true" in text.lower()
    boxes, unmapped = L.read_annotation_file(text, "x.csv", recs, {ROBIN: ROBIN})
    assert unmapped == [] and boxes["species_key"].to_list() == [ROBIN]
    assert boxes["recording_id"].to_list() == ["r1"]


def test_raven_and_audacity_imports_name_what_they_cannot_map(store):
    recs = recordings()
    raven = ("Selection\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tSpecies\n"
             "1\t1.0\t2.5\t1000\t3000\tAmerican Robin\n2\t4\t5\t1\t2\tDodo\n")
    boxes, unmapped = L.read_annotation_file(raven, "r2.Table.1.selections.txt", recs,
                                             {"american robin": ROBIN})
    assert unmapped == ["Dodo"]
    aud = "1.0\t2.0\tAmerican Robin\n\\\t1500\t3500\n"
    boxes, unmapped = L.read_annotation_file(aud, "r2.txt", recs, {"american robin": ROBIN})
    assert boxes.row(0, named=True)["high_hz"] == 3500 and boxes["recording_id"][0] == "r2"


def test_set_names_are_checked(store):
    with pytest.raises(ValueError):
        L.create_set(store, "site", "imported", "ann")
    with pytest.raises(ValueError):
        L.create_set(store, "site", "../evil", "ann")


def test_alignment_scores_only_closed_time(store):
    """A partly labelled recording: windows outside closed chunks are neither
    hits nor false alarms — they are not in the frame at all."""
    import numpy as np
    from bex import truth
    from bex.schemas import RunManifest
    from bex.store import ScoreMatrix, write_scores
    RunManifest(model_name="toy", model_version="1", window_s=5.0, hop_s=5.0,
                input_sr=32000, score_transform="sigmoid", run_id="toy-1").save(store)
    starts = np.arange(0, 300, 5.0)
    write_scores(store, ScoreMatrix("r0", "toy-1", np.full((len(starts), 1), 0.5),
                                    starts, 5.0, [ROBIN]))
    ls = L.create_set(store, "site", "mine", "ann")
    L.add_box(ls, "r0", 62, 64, 1000, 2000, ROBIN, "ann")
    L.add_box(ls, "r0", 200, 202, 1000, 2000, ROBIN, "ann")   # chunk left open
    L.close_chunk(ls, "r0", 60.0, 120.0, "ann")
    L.save_set(store, ls)
    ref = L.working_ref(store, "site", "mine")
    aligned, meta = truth.align_run(store, "site", "toy-1", truth_ref=ref)
    assert aligned["start_s"].min() == 60.0 and aligned["end_s"].max() == 120.0
    assert len(aligned) == 12 and int(aligned["y_true"].sum()) == 1
    assert meta["truth_ref"] == ref and meta["labelled_hours"] == 60 / 3600
    truth.write_aligned(store, "site", "toy-1", aligned, meta)
    assert truth.aligned_path(store, "site", "toy-1", "native", ref).exists()
    assert not truth.aligned_path(store, "site", "toy-1", "native").exists()


def test_a_box_can_cross_a_minute_and_counts_where_any_of_it_is_closed(store):
    ls = L.create_set(store, "site", "mine", "ann")
    bid = L.add_box(ls, "r0", 57, 64, 1000, 2000, ROBIN, "ann")      # 0:57–1:04
    assert L.chunks_touched(57, 64) == [0.0, 60.0]
    assert ls.chunk_boxes("r0", 60, 120)["box_id"].to_list() == [bid]  # shown in both
    L.close_chunk(ls, "r0", 60.0, 120.0, "ann")                         # only the second
    t = L.truth_from_set(ls, recordings(), "x")
    assert t.annotations["species_key"].to_list() == [ROBIN]            # still a bird there
    with pytest.raises(L.ClosedChunk):                                  # and frozen with it
        L.delete_box(ls, bid)
    with pytest.raises(L.ClosedChunk):
        L.add_box(ls, "r0", 50, 61, 1, 2, WREN, "ann")                  # reaches a closed minute


def test_boxes_record_whether_suggestions_were_viewed(store):
    ls = L.create_set(store, "site", "mine", "ann")
    L.add_box(ls, "r0", 1, 2, 1000, 2000, ROBIN, "ann", assisted=True)
    L.add_box(ls, "r0", 3, 4, 1000, 2000, WREN, "ann")
    L.save_set(store, ls)
    back = L.load_set(store, "site", "mine")
    assert back.boxes.sort("start_s")["assisted"].to_list() == [True, False]
    assert "Suggestions Viewed" in L.export_csv(back, recordings())
    # A set written before the column existed reads as "not viewed".
    p = L.set_dir(store, "site", "mine") / "annotations.parquet"
    pl.read_parquet(p).drop("assisted").write_parquet(p)
    assert L.load_set(store, "site", "mine").boxes["assisted"].to_list() == [False, False]


def test_practice_feedback_against_the_experts():
    def box(t0, t1, sp, rid="r0"):
        return {"recording_id": rid, "start_s": float(t0), "end_s": float(t1),
                "low_hz": 1000.0, "high_hz": 2000.0, "species_key": sp}
    ref = pl.DataFrame([box(5, 7, ROBIN), box(20, 22, WREN), box(40, 41, ROBIN),
                        box(65, 66, WREN)])                     # last one: next minute
    mine = pl.DataFrame([box(5.2, 6.8, ROBIN), box(20.5, 21.5, ROBIN),
                         box(50, 52, "Strix varia")])
    fb = L.compare_to_reference(mine, ref, "r0", 0.0, 60.0)
    assert [c["outcome"] for c in fb["calls"]] == ["found", "named differently", "missed"]
    assert fb["calls"][1]["yours"] == ROBIN
    assert fb["n_found"] == 1 and fb["n_calls"] == 3
    assert fb["missed_species"] == [WREN] and fb["found_species"] == [ROBIN]
    assert fb["extra_species"] == ["Strix varia"]


def test_a_practice_set_is_marked_and_samples_labelled_recordings(tmp_path):
    ann = pl.DataFrame([{"recording_id": "r0", "start_s": 5.0, "end_s": 6.0,
                         "low_hz": 1.0, "high_hz": 2.0, "species_key": ROBIN}],
                       schema=ANNOTATIONS_SCHEMA)
    write_dataset(tmp_path, "sne", recordings(labelled=True), ann)
    ls = L.create_practice(tmp_path, "sne", "Mara D", recordings(labelled=True), 3)
    assert ls.name == "practice-mara-d" and len(ls.sample) == 3
    assert L.is_practice(tmp_path, "sne", ls.name)
    again = L.create_practice(tmp_path, "sne", "Mara D", recordings(labelled=True), 3)
    assert again.name == ls.name                                   # reused, not duplicated
