"""Bring your own model (bex/byom.py): a table of detections becomes a run the
guided path can use, and a bad table says exactly what's wrong."""
import polars as pl
import pytest

from bex import byom
from bex import ecolabel as E
from tests.test_ecolabel import project  # noqa: F401  (the synthetic two-recording project)


def table(**over):
    rows = []
    for rec in ("REC_A_20260501_050000.wav", "REC_B_20260502_050000"):
        for w in range(10):
            for sp, sc in (("Turdus migratorius", 0.9), ("catharus  GUTTATUS", 0.3), ("Noise", 0.05)):
                rows.append({"recording": rec, "start_s": 3.0 * w, "end_s": 3.0 * w + 3, "species": sp, "score": sc})
    return pl.DataFrame(rows).with_columns(**over) if over else pl.DataFrame(rows)


def test_a_good_table_becomes_a_run(project):  # noqa: F811
    r = byom.import_detections(project.store_dir, "demo", table(), "My Net!", "2")
    assert r["rows"] == 60 and r["recordings"] == 2 and r["species"] == 2 and r["other_labels"] == 1
    assert r["window_s"] == 3.0 and r["hop_s"] == 3.0 and not r["warnings"]
    run = next(x for x in E.runs_for(project.store_dir, "demo") if x["run_id"] == r["run_id"])
    assert run["model_name"] == "mynet" and run["model_version"] == "2"
    pool = E.candidate_pool(project.store_dir, "demo", r["run_id"])
    assert set(pool["species_key"]) == {"Turdus migratorius", "Catharus guttatus", "Noise"}
    assert pool.filter(pl.col("rank") == 1)["species_key"].unique().to_list() == ["Turdus migratorius"]


def test_other_tools_column_names_work(project):  # noqa: F811
    t = table().rename({"recording": "File", "start_s": "Begin Time (s)", "end_s": "End Time (s)",
                        "species": "label", "score": "Confidence"})
    t = t.with_columns(pl.col("label").str.replace("Turdus migratorius", "Turdus migratorius_American Robin"))
    r = byom.import_detections(project.store_dir, "demo", t, "birdnetlike")
    assert r["species"] == 2


def test_a_bad_table_says_what_to_fix(project):  # noqa: F811
    with pytest.raises(byom.ImportProblem, match="Missing column.*score"):
        byom.import_detections(project.store_dir, "demo", table().drop("score"), "x")
    with pytest.raises(byom.ImportProblem, match="sigmoid"):
        byom.import_detections(project.store_dir, "demo", table(score=pl.col("score") * 10 - 3), "x")
    with pytest.raises(byom.ImportProblem, match="not in this project"):
        byom.import_detections(project.store_dir, "demo", table(recording=pl.lit("SOMETHING_ELSE.wav")), "x")
    with pytest.raises(byom.ImportProblem, match="window length"):
        byom.import_detections(project.store_dir, "demo", table().drop("end_s"), "x")
    r = byom.import_detections(project.store_dir, "demo", table().drop("end_s"), "x", window_s=3.0)
    assert r["window_s"] == 3.0


def test_the_template_imports_cleanly(project):  # noqa: F811
    from bex import ingest
    recs, _ = ingest.read_dataset(project.store_dir, "demo")
    t = byom.read_table(byom.template(recs["path"].to_list()))
    assert byom.import_detections(project.store_dir, "demo", t, "example")["rows"] == t.height
