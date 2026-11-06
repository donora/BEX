"""A project's own code, named in its bex.toml."""
import pytest

from bex.config import load_config
from bex.extensions import load_panel


def test_a_panel_is_found_relative_to_the_config(tmp_path):
    (tmp_path / "ext").mkdir()
    (tmp_path / "ext" / "mine.py").write_text("def panel(**kw):\n    return kw['page']\n")
    p = tmp_path / "bex.toml"
    p.write_text('[extensions]\npanel = "ext/mine.py"\n')
    cfg = load_config(p)
    assert cfg.panel_file == tmp_path / "ext" / "mine.py"
    assert load_panel(cfg.panel_file)(page="label") == "label"


def test_no_panel_by_default(tmp_path):
    p = tmp_path / "bex.toml"
    p.write_text("")
    assert load_config(p).panel_file is None


def test_a_missing_file_or_function_says_what_is_wrong(tmp_path):
    with pytest.raises(FileNotFoundError, match="no file"):
        load_panel(tmp_path / "gone.py")
    (tmp_path / "empty.py").write_text("x = 1\n")
    with pytest.raises(AttributeError, match="no panel"):
        load_panel(tmp_path / "empty.py")
