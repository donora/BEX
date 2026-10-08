"""Start a model run from the app (V1.2 F1: add a dataset, run the models).

Each model runs in its own Python environment under `envs/<model>/` (PLAN.md
§3b) — BirdNET and Perch need libraries that cannot share one install, and
none of them may enter the app's. So the app does not import a model: it
starts the runner's own interpreter as a separate process, writes its output
to a log file, and reads progress back from the store (scores written so far).

A job is recorded in `store/datasets/<dataset>/jobs/<model>.json` with its
process id, so a page reload still knows it is running.

NB: runners import this package; editing BEX's modules while one runs breaks
it, so the app says so while a job is live.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ENVS = Path(__file__).resolve().parent.parent / "envs"

#: Extra arguments each runner gets from the app.
DEFAULT_ARGS = {"perch": ["--readout", "sigmoid"]}


def available() -> dict[str, Path]:
    """Runners whose environment is installed: {model: its python}."""
    out = {}
    if ENVS.exists():
        for d in sorted(p for p in ENVS.iterdir() if (p / "run.py").exists()):
            py = d / ".venv" / "bin" / "python"
            if py.exists():
                out[d.name] = py
    return out


def known() -> list[str]:
    return sorted(p.name for p in ENVS.iterdir() if (p / "run.py").exists()) \
        if ENVS.exists() else []


def _job_path(store_dir, dataset, model) -> Path:
    return Path(store_dir) / "datasets" / dataset / "jobs" / f"{model}.json"


def _alive(pid: int) -> tuple[bool, int | None]:
    """(still running, exit code if we saw it finish).

    A runner the app started is its child, and a finished child stays in the
    process table as a zombie until it is reaped — `kill(pid, 0)` succeeds on a
    zombie, so a finished run used to read as running forever. Reap it first;
    only a process that is not our child (the app was restarted) falls back to
    the signal check.
    """
    try:
        done, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        try:
            os.kill(pid, 0)
        except (OSError, ValueError):
            return False, None
        return True, None
    except (OSError, ValueError):
        return False, None
    if done == 0:
        return True, None
    return False, os.waitstatus_to_exitcode(status)


def job(store_dir, dataset: str, model: str) -> dict | None:
    """The last job for this model on this dataset, with `running` (and, once
    seen, `exit_code`) filled in."""
    p = _job_path(store_dir, dataset, model)
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    if j.get("exit_code") is not None:
        j["running"] = False
        return j
    running, code = _alive(int(j.get("pid", -1)))
    j["running"] = running
    if code is not None:
        j["exit_code"] = code
        p.write_text(json.dumps({k: v for k, v in j.items() if k != "running"}, indent=2))
    return j


def running_jobs(store_dir, dataset: str) -> list[str]:
    return [m for m in known() if (j := job(store_dir, dataset, m)) and j["running"]]


def launch(store_dir, dataset: str, model: str, config_path: str | Path,
           extra: list[str] | None = None) -> dict:
    """Start `envs/<model>/run.py --dataset <dataset>` in the background."""
    py = available().get(model)
    if py is None:
        raise FileNotFoundError(f"the {model} environment is not installed "
                                f"(envs/{model}/README.md)")
    current = job(store_dir, dataset, model)
    if current and current["running"]:
        return current
    log = Path(store_dir) / "datasets" / dataset / "jobs" / f"{model}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    profile = _dataset_profile(store_dir, dataset)
    args = [str(py), "run.py", "--dataset", dataset, "--config", str(config_path),
            *(["--profile", profile] if profile else []),
            *DEFAULT_ARGS.get(model, []), *(extra or [])]
    with open(log, "ab") as fh:
        proc = subprocess.Popen(args, cwd=ENVS / model, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
    j = {"pid": proc.pid, "args": args, "log": str(log),
         "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _job_path(store_dir, dataset, model).write_text(json.dumps(j, indent=2))
    return {**j, "running": True}


def _dataset_profile(store_dir, dataset: str) -> str:
    p = Path(store_dir) / "datasets" / dataset / "meta.json"
    return json.loads(p.read_text()).get("profile", "") if p.exists() else ""


def tail(path: str | Path, lines: int = 8) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return "\n".join(p.read_text(errors="replace").splitlines()[-lines:])
