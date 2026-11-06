"""Small things the app remembers between visits — e.g. that the welcome has
been seen. Streamlit's own state is lost on every browser refresh, so these
live in one JSON file in the store, next to the data they are about.
"""
from __future__ import annotations

import json
from pathlib import Path


def _path(store_dir: str | Path) -> Path:
    return Path(store_dir) / "ui_prefs.json"


def get(store_dir: str | Path, key: str, default=None):
    p = _path(store_dir)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text()).get(key, default)
    except (OSError, ValueError):
        return default            # a damaged prefs file never breaks the app


def starred(store_dir: str | Path) -> list[str]:
    """The datasets someone has starred on the Home page."""
    return list(get(store_dir, "starred_datasets", []))


def toggle_star(store_dir: str | Path, dataset: str) -> None:
    stars = starred(store_dir)
    put(store_dir, "starred_datasets",
        [d for d in stars if d != dataset] if dataset in stars else [*stars, dataset])


def stars_first(datasets: list[str], stars: list[str]) -> list[str]:
    """Starred datasets first; each group keeps the order it came in."""
    return [d for d in datasets if d in stars] + [d for d in datasets if d not in stars]


def put(store_dir: str | Path, key: str, value) -> None:
    p = _path(store_dir)
    try:
        data = json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        data = {}
    data[key] = value
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")
