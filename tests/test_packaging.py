"""What a release contains: anything left in a packaged folder would ship publicly."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_assets_are_packaged_file_by_file():
    hatch = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["build"]
    forced = hatch["targets"]["wheel"]["force-include"]
    assert "assets" not in forced, "a whole folder here ships whatever is dropped into it"
    assert "assets" not in hatch["targets"]["sdist"]["include"]
    for source in forced:
        assert (ROOT / source).exists(), source


def test_the_packaged_assets_are_the_ones_in_the_folder():
    forced = tomllib.loads((ROOT / "pyproject.toml").read_text())[
        "tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    named = {Path(k).name for k in forced if k.startswith("assets/")}
    assert {p.name for p in (ROOT / "assets").iterdir()} == named, \
        "an unexpected file is in assets/: package it by name, or keep it elsewhere"
