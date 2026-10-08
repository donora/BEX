"""A finished runner must read as finished (the zombie bug, 2026-10-08)."""
import json
import subprocess
import sys
import time

from bex import runners


def test_a_finished_child_is_not_running_and_its_exit_code_is_kept(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    jobs = tmp_path / "datasets" / "d" / "jobs"
    jobs.mkdir(parents=True)
    (jobs / "toy.json").write_text(json.dumps({"pid": p.pid, "log": ""}))
    time.sleep(0.5)                     # exited, but not yet reaped: a zombie
    j = runners.job(tmp_path, "d", "toy")
    assert j["running"] is False and j["exit_code"] == 3
    assert runners.job(tmp_path, "d", "toy")["exit_code"] == 3   # remembered


def test_a_live_child_is_running():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        assert runners._alive(p.pid) == (True, None)
    finally:
        p.kill(); p.wait()
