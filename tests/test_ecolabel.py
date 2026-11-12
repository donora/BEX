"""The EcoLabel path (bex/ecolabel.py): pool, guided check, calibration,
recall, protocol estimates, the bundle, and the resumable upload — on a tiny
synthetic project, so no store or audio of your own is needed."""
from __future__ import annotations

import hashlib
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import soundfile as sf

from bex import ecolabel as E
from bex import ingest
from bex.config import load_config

SR = 16000
RUN = "perch-test0001"


@pytest.fixture()
def project(tmp_path: Path):
    """Two 60 s recordings, a model run with two species: Robin (right when it
    scores high) and Wren (mostly wrong)."""
    audio = tmp_path / "audio"
    audio.mkdir()
    rng = np.random.default_rng(0)
    for name in ("REC_A_20260501_050000", "REC_B_20260502_050000"):
        sf.write(audio / f"{name}.wav", (rng.normal(0, 0.05, SR * 60)).astype("float32"), SR)
    (tmp_path / "bex.toml").write_text(
        '[paths]\naudio_dir = "audio"\nlabels_dir = "."\nstore_dir = "store"\ncache_dir = "cache"\n'
        'profiles_dir = "profiles"\nthresholds_dir = "thresholds"\n[site]\nname = "test"\n')
    cfg = load_config(tmp_path / "bex.toml")
    recs = ingest.scan_folder(str(audio), site="test", lat=57.0, lon=-5.0)
    ingest.write_dataset(cfg.store_dir, "demo", recs, audio_root=str(audio))
    rows = []
    for rec in recs["recording_id"]:
        for w in range(19):
            s = w * 3.0
            rows += [
                {"recording_id": rec, "run_id": RUN, "start_s": s, "end_s": s + 3, "species_key": "Turdus migratorius",
                 "score_raw": float(rng.uniform(0.05, 0.99)), "occ_score": float("nan"), "suppressed": False, "rank_in_window": 1},
                {"recording_id": rec, "run_id": RUN, "start_s": s, "end_s": s + 3, "species_key": "Troglodytes troglodytes",
                 "score_raw": float(rng.uniform(0.01, 0.4)), "occ_score": float("nan"), "suppressed": False, "rank_in_window": 2},
            ]
    store = Path(cfg.store_dir)
    (store / "detections").mkdir(parents=True)
    pl.DataFrame(rows).with_columns(pl.col("score_raw", "occ_score").cast(pl.Float32),
                                    pl.col("rank_in_window").cast(pl.Int32)).write_parquet(store / "detections" / f"{RUN}.parquet")
    (store / "runs" / RUN).mkdir(parents=True)
    (store / "runs" / RUN / "manifest.json").write_text(json.dumps(
        {"run_id": RUN, "dataset": "demo", "model_name": "perch", "model_version": "v2", "window_s": 3.0,
         "score_transform": "softmax"}))
    return cfg


def truth(row: dict) -> bool:
    """The synthetic world: Robin is there when it scores above 0.5; Wren never."""
    return row["species_key"] == "Turdus migratorius" and row["score"] > 0.5


def answer_all(cfg, n: int = 30) -> None:
    pool = E.candidate_pool(cfg.store_dir, "demo", RUN)
    for k in ("Turdus migratorius", "Troglodytes troglodytes"):
        for _ in range(n):
            c = E.next_to_check(pool, E.load_checks(cfg.store_dir, "demo"), k)
            if c is None:
                break
            E.add_check(cfg.store_dir, "demo", c["recording_id"], c["start_s"], k, "yes" if truth(c) else "no")


def test_pool_is_each_windows_top_species(project):
    pool = E.candidate_pool(project.store_dir, "demo", RUN)
    assert pool.height == 2 * 19 * 2 and set(pool["rank"]) == {1, 2}
    assert not E.saturated(pool)


def test_species_need_a_name_and_follow_the_choice(project):
    pool = E.candidate_pool(project.store_dir, "demo", RUN)
    sp = E.species_table(pool, None, {"Turdus migratorius": "American Robin"}, {"included": None, "names": {}})
    assert set(sp.filter("include")["species_key"]) == {"Turdus migratorius", "Troglodytes troglodytes"}
    s = {"included": ["Turdus migratorius"], "names": {"Troglodytes troglodytes": "Wren"}}
    sp = E.species_table(pool, None, {}, s)
    assert sp.filter("include")["species_key"].to_list() == ["Turdus migratorius"]
    assert dict(zip(sp["species_key"], sp["name"]))["Troglodytes troglodytes"] == "Wren"


def test_guided_check_fills_yes_then_no(project):
    answer_all(project)
    r = E.readiness(E.load_checks(project.store_dir, "demo"), ["Turdus migratorius", "Troglodytes troglodytes"])
    robin = r.filter(pl.col("species_key") == "Turdus migratorius").row(0, named=True)
    assert robin["known_yes"] >= E.MIN_YES and robin["known_no"] >= E.MIN_NO
    assert robin["ready"] and E.progress(robin) == 1.0
    wren = r.filter(pl.col("species_key") == "Troglodytes troglodytes").row(0, named=True)
    assert wren["known_yes"] == 0 and not wren["ready"]   # never right: can't be taught to volunteers


def test_recall_estimate_tracks_the_truth(project):
    assert E.calibrate(E.candidate_pool(project.store_dir, "demo", RUN), E.load_checks(project.store_dir, "demo")) is None
    answer_all(project)
    pool = E.candidate_pool(project.store_dir, "demo", RUN)
    p = E.calibrate(pool, E.load_checks(project.store_dir, "demo"))
    assert p is not None and p.shape == (pool.height,)
    tab = E.recall_table(pool, p)
    assert tab["recall"].is_sorted() and tab["kept"].is_sorted()
    row = E.floor_for_recall(tab, 0.8)
    kept = pool.filter(pl.col("score") >= row["floor"])
    real = [truth(r) for r in kept.iter_rows(named=True)]
    all_real = sum(truth(r) for r in pool.iter_rows(named=True))
    assert sum(real) / all_real >= 0.7                         # aimed for 80%
    assert abs(100 * (1 - np.mean(real)) - row["false_per_100"]) < 25


def test_light_protocols_cost_less_than_every_call(project):
    answer_all(project)
    pool = E.candidate_pool(project.store_dir, "demo", RUN)
    p = E.calibrate(pool, E.load_checks(project.store_dir, "demo"))
    every = E.estimate_protocol(pool, p, {"mode": "every_call"})
    presence = E.estimate_protocol(pool, p, {"mode": "presence", "confirmations": 1, "give_up_after": 5})
    activity = E.estimate_protocol(pool, p, {"mode": "activity", "block_s": 30, "confirmations": 1, "give_up_after": 3})
    assert presence["checks"] < activity["checks"] < every["checks"] == pool.height
    assert presence["units"] == 4                              # 2 recordings × 2 species


def test_bundle_has_what_ecolabel_needs(project, tmp_path):
    answer_all(project)
    s = E.load_settings(project.store_dir, "demo")
    s.update(run_id=RUN, included=["Turdus migratorius", "Troglodytes troglodytes"],
             names={"Turdus migratorius": "American Robin", "Troglodytes troglodytes": "Wren"}, floor=0.3)
    s["about"]["project"]["title"] = "Test dawn"
    E.save_settings(project.store_dir, "demo", s)
    r = E.build_bundle(project, "demo", tmp_path / "out")
    b = Path(r["folder"])
    for f in ("manifest.json", "species.csv", "recordings.csv", "candidates.parquet", "labels.parquet",
              "checked.parquet", "references.csv"):
        assert (b / f).exists(), f
    cand = pl.read_parquet(b / "candidates.parquet")
    assert cand["candidate_id"].to_list() == list(range(cand.height))
    assert all((b / c).exists() for c in cand["clip"].unique())
    checked = pl.read_parquet(b / "checked.parquet")
    assert checked.height == r["known_answers"] > 0 and set(checked["candidate_id"]) <= set(cand["candidate_id"])
    refs = pl.read_csv(b / "references.csv")
    yes = checked.filter(pl.col("verdict") == "yes")
    # every confirmed Yes is a possible example, tied to its question
    assert refs.height == yes.height >= E.MIN_YES and all((b / f).exists() for f in refs["file"])
    assert set(refs["candidate_id"]) == set(yes["candidate_id"])
    m = json.loads((b / "manifest.json").read_text())
    assert m["format"] == "ecolabel-bundle" and m["about"]["project"]["title"] == "Test dawn"
    assert m["protocol"]["mode"] == "presence" and "lat" not in json.dumps(m)   # no coordinates leave
    assert "library" not in m                                  # nothing shared without consent
    s["library"] = {"share": True, "terms": E.LIBRARY_TERMS, "credit": "Test group", "agreed": "2026-11-12"}
    E.save_settings(project.store_dir, "demo", s)
    E.build_bundle(project, "demo", tmp_path / "out")
    m = json.loads((b / "manifest.json").read_text())
    assert m["library"]["share"] and m["library"]["credit"] == "Test group" and m["library"]["terms"] == E.LIBRARY_TERMS
    z = E.zip_bundle(b)
    assert zipfile.ZipFile(z).testzip() is None


def test_the_library_lightens_the_check(project):
    checks = E.load_checks(project.store_dir, "demo")
    lib = {"Turdus migratorius": {"yes": 9, "no": 4}, "Troglodytes troglodytes": {"yes": 2, "no": 0}}
    r = E.readiness(checks, ["Turdus migratorius", "Troglodytes troglodytes"], lib)
    robin, wren = r.row(0, named=True), r.row(1, named=True)
    assert robin["ready"] and robin["covered"] and robin["need_yes"] == robin["need_no"] == 0
    assert not wren["ready"] and wren["need_yes"] == E.MIN_YES - 2 and wren["need_no"] == E.MIN_NO
    pool = E.candidate_pool(project.store_dir, "demo", RUN)
    # a covered species goes straight to the score walk (calibration), not "best first"
    c = E.next_to_check(pool, checks, "Turdus migratorius", lib)
    assert c is not None
    assert E.library_coverage("http://127.0.0.1:9") == {}      # no server: check everything here


class _Stub(BaseHTTPRequestHandler):
    """The three upload endpoints, as the EcoLabel server answers them."""
    state: dict = {}
    fail_after: int | None = None

    def log_message(self, *a):
        pass

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.headers.get("authorization") != "Bearer good":
            return self._send(401, {"detail": "unknown access key"})
        if self.path == "/api/uploads":
            meta = json.loads(body)
            self.state.setdefault("data", b"")
            self.state["meta"] = meta
            return self._send(200, {"upload_id": "u1", "received": len(self.state["data"])})
        if self.path.endswith("/complete"):
            ok = hashlib.sha256(self.state["data"]).hexdigest() == self.state["meta"]["sha256"]
            return self._send(200 if ok else 400, {"project": "p1", "status": "draft"} if ok else {"detail": "bad"})

    def do_PUT(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        off = int(self.path.split("offset=")[1])
        if _Stub.fail_after is not None and off >= _Stub.fail_after:
            return self._send(500, {"detail": "connection lost"})
        assert off == len(self.state["data"])
        self.state["data"] += body
        return self._send(200, {"received": len(self.state["data"])})


def test_upload_resumes_where_it_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "CHUNK", 1000)
    f = tmp_path / "x.ecolabel.zip"
    f.write_bytes(np.random.default_rng(1).bytes(5500))
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    _Stub.state, _Stub.fail_after = {}, None
    with pytest.raises(RuntimeError, match="401"):
        E.upload(f, url, "bad")
    _Stub.fail_after = 3000
    with pytest.raises(RuntimeError, match="500"):
        E.upload(f, url, "good")
    assert len(_Stub.state["data"]) == 3000                    # stopped part-way…
    _Stub.fail_after = None
    seen = []
    r = E.upload(f, url, "good", progress=lambda frac, msg: seen.append(frac))
    assert r == {"project": "p1", "status": "draft"} and seen[0] > 0.5   # …and carried on, intact
    srv.shutdown()


def test_separated_answers_cannot_run_away(project):
    """All the Nos below all the Yeses (what the guided check tends to give)
    used to send the fit to infinity, calling every detection real."""
    pool = E.candidate_pool(project.store_dir, "demo", RUN)
    srt = pool.sort("score")
    for r in srt.head(8).iter_rows(named=True):
        E.add_check(project.store_dir, "demo", r["recording_id"], r["start_s"], r["species_key"], "no")
    for r in srt.tail(16).iter_rows(named=True):
        E.add_check(project.store_dir, "demo", r["recording_id"], r["start_s"], r["species_key"], "yes")
    checks = E.load_checks(project.store_dir, "demo")
    for typical in (None, E.typical_for(project.store_dir, RUN)):
        cal = E.calibration(pool, checks, typical)
        assert np.isfinite(cal["p"]).all() and 0.05 < cal["p"].mean() < 0.95


def test_a_known_model_has_an_estimate_before_any_answers(project):
    pool = E.candidate_pool(project.store_dir, "demo", RUN)
    typical = E.typical_for(project.store_dir, RUN)
    assert typical == E.TYPICAL["perch-softmax"]                 # the fixture is Perch, softmax readout
    none = E.load_checks(project.store_dir, "demo")
    assert E.calibration(pool, none, None) is None                # unknown model: answers first
    cal = E.calibration(pool, none, typical)
    assert cal["answers"] == 0 and cal["prior_answers"] == E.PRIOR_ANSWERS
    band = E.recall_band(pool, cal, np.array([0.5, 0.8, 0.95]))
    assert (band["kept_lo"] <= band["kept_mid"]).all() and (band["kept_mid"] <= band["kept_hi"]).all()
    assert band["kept_mid"].is_sorted()                           # more recall, more to check
    lo, hi = E.volunteer_hours(1000)
    assert 0 < lo < hi
