"""Project extensions: code a project keeps to itself, without changing BEX.

A project's `bex.toml` can name a Python file:

    [extensions]
    panel = "extensions/my_panel.py"

and BEX calls that file's `panel(dataset, recording_id, t, page, store_dir)` under the
figure on the Explorer and Label pages, where it can draw whatever it likes with
Streamlit. `t` is the moment being looked at, in seconds into the recording, and `page`
is "explorer" or "label". This runs the file as ordinary Python, so name only your own.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable


def load_panel(path: str | Path) -> Callable:
    """The `panel` function in a project's file. Says what is wrong if it cannot."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[extensions] panel: no file at {path}")
    spec = importlib.util.spec_from_file_location(f"bex_project_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    panel = getattr(module, "panel", None)
    if not callable(panel):
        raise AttributeError(f"[extensions] panel: {path} has no panel() function")
    return panel
