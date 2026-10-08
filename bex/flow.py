"""Where each dataset is in the flow (V1.2 F0, F1).

    Add ─► Run models ─► 1 · Label ─► 2 · Explore ─► 3 · Analyse ─► 4 · Survey protocol ─► Apply

Step names are verbs, not past tenses: a step is something you can do, and
come back to, not a box that is closed for good.

Nothing here is a commitment. A dataset can be looked at two ways at any time
— its labelled data, scored, or all its recordings, unscored (the sidebar's
view switch) — so labelling is never *skipped*, only *not started yet*, and
the steps that need labels say so rather than looking blocked.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import labels, setups, store
from .ingest import read_dataset
from .melcache import mel_path
from .views import list_runs

STEPS = {
    "added": "Add",
    "models": "Run models",
    "labelled": "1 · Label",
    "explored": "2 · Explore",
    "analysed": "3 · Analyse",
    "survey": "4 · Survey protocol",
    "applied": "Apply",
}

#: Where the "continue" button goes for a step.
PAGE_OF = {
    "models": "ui/pages/models.py",
    "labelled": "ui/pages/label.py",
    "explored": "ui/pages/explorer.py",
    "analysed": "ui/pages/compare.py",
    "survey": "ui/pages/survey.py",
    "applied": "ui/pages/survey.py",
}


def _meta_path(store_dir, name) -> Path:
    return Path(store_dir) / "datasets" / name / "meta.json"


def dataset_meta(store_dir: str | Path, name: str) -> dict:
    p = _meta_path(store_dir, name)
    return json.loads(p.read_text()) if p.exists() else {}


def set_dataset_meta(store_dir: str | Path, name: str, **values) -> None:
    meta = dataset_meta(store_dir, name)
    meta.update(values)
    _meta_path(store_dir, name).write_text(json.dumps(meta, indent=2) + "\n")


def real_sets(store_dir: str | Path, name: str) -> list[str]:
    """Label sets that are ground truth candidates — not practice sets."""
    return [s for s in labels.list_sets(store_dir, name)
            if not labels.is_practice(store_dir, name, s)]


def has_labels(store_dir: str | Path, name: str) -> bool:
    return labels.has_imported(store_dir, name) or bool(real_sets(store_dir, name))


def progress(store_dir: str | Path, cache_dir: str | Path, name: str) -> dict:
    """Each step's state ('done', 'partial', 'todo', 'skipped', 'waiting') and a
    one-line detail, plus the step to continue with."""
    recs, _ = read_dataset(store_dir, name)
    n = len(recs)
    labelled_any = has_labels(store_dir, name)
    steps: dict[str, tuple[str, str]] = {"added": ("done", f"{n} recordings")}

    runs = [m for m in list_runs(store_dir) if m.dataset == name]
    full = [m for m in runs if len(store.list_scores(store_dir, m.run_id)) >= n]
    mels = sum(mel_path(cache_dir, name, r).exists() for r in recs["recording_id"])
    from .runners import running_jobs
    busy = running_jobs(store_dir, name)
    if busy:
        steps["models"] = ("partial", f"running: {', '.join(busy)}; {len(full)} of "
                                      f"{len(runs)} model run(s) complete")
    elif full and mels == n:
        steps["models"] = ("done", f"{len(full)} model run(s); spectrograms built")
    elif runs or mels:
        steps["models"] = ("partial", f"spectrograms {mels}/{n}; "
                                      + (f"{len(full)} of {len(runs)} model run(s) complete"
                                         if runs else "no model run yet"))
    else:
        steps["models"] = ("todo", "no model has been run yet")

    sets = real_sets(store_dir, name)
    if labels.has_imported(store_dir, name) and not sets:
        steps["labelled"] = ("done", "came with annotations")
    elif sets:
        newest = max(sets, key=lambda x: (labels.set_dir(store_dir, name, x) / "meta.json")
                     .stat().st_mtime)
        ls = labels.load_set(store_dir, name, newest)
        closed = ls.chunks.filter(ls.chunks["state"] == labels.CLOSED)
        if ls.sample.is_empty():
            steps["labelled"] = (("partial" if len(closed) else "todo"),
                                 f"{newest}: {len(closed)} chunk(s) closed, no sample drawn")
        else:
            done = ls.sample.join(closed, on=["recording_id", "start_s"], how="semi").height
            state = "done" if done >= len(ls.sample) else "partial" if done else "todo"
            steps["labelled"] = (state, f"{newest}: {done}/{len(ls.sample)} sampled chunks")
    else:
        steps["labelled"] = ("todo", "not started yet")

    # Ticked once someone has opened the Explorer on it, not merely when it could be.
    if not full:
        steps["explored"] = ("waiting", "needs a complete model run")
    elif dataset_meta(store_dir, name).get("explored"):
        steps["explored"] = ("done", "explored")
    else:
        steps["explored"] = ("todo", "ready to explore")

    truth_dir = Path(store_dir) / "truth" / name
    aligned = truth_dir.exists() and any(truth_dir.rglob("*.parquet"))
    if not labelled_any:
        steps["analysed"] = ("waiting", "needs labels")
    else:
        steps["analysed"] = (("done", "models scored against labels") if aligned
                             else ("todo", "score the models on Compare"))

    mine = [x for x in setups.list_setups(store_dir)
            if setups.load(store_dir, x).fitted_on.get("dataset") == name]
    steps["survey"] = (("done", f"{len(mine)} setup(s) saved") if mine
                       else ("todo", "choose and save a survey setup"))
    results = Path(store_dir) / "datasets" / name / "results"
    applied = sorted(p.stem for p in results.glob("*.csv")) if results.exists() else []
    steps["applied"] = (("done", f"lists from {', '.join(applied)}") if applied
                        else ("todo", "apply a setup to every recording"))

    order = ["models", "labelled", "explored", "analysed", "survey", "applied"]
    nxt = next((k for k in order if steps[k][0] in ("todo", "partial")), None)
    if steps["models"][0] != "done":
        nxt = "models"
    return {"steps": steps, "next": nxt, "has_labels": labelled_any}
