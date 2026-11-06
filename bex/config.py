"""Load `bex.toml` — the one place paths and site defaults live.

Open-source rule 2 (PLAN.md §8): no absolute paths anywhere in code. Everything
resolves through a config file the user copies from `bex.example.toml`. Relative
paths in the file resolve against the file's own directory, so a cloned repo works
wherever it lands.

Lookup order: an explicit path argument, then `$BEX_CONFIG`, then `bex.toml`
found by walking up from the current directory, then the in-repo example (so tests
and a fresh clone run with zero setup).
"""
from __future__ import annotations

import os
import os.path
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Files the wheel carries inside the package (pyproject force-include); in a
# clone the same files live at the repo root.
_PACKAGE_DATA = Path(__file__).resolve().parent / "data"


def bundled(name: str) -> Path:
    """A file or folder that ships with BEX: the plausibility profiles, the
    example config. Inside the installed package, or at the root of a clone."""
    packaged = _PACKAGE_DATA / name
    return packaged if packaged.exists() else _REPO_ROOT / name


@dataclass(frozen=True)
class Config:
    """Resolved configuration. All paths are absolute by the time you see them."""

    source: Path  # the toml file this came from
    audio_dir: Path
    labels_dir: Path
    store_dir: Path
    cache_dir: Path
    profiles_dir: Path
    thresholds_dir: Path
    site_name: str
    site_lat: float | None
    site_lon: float | None
    default_profile: str
    xc_api_key: str
    # [display] overrides for the app's spectrogram (melcache.display_params).
    display: dict = field(default_factory=dict)
    # [defaults] threshold_set: the saved threshold set the app opens with.
    default_threshold_set: str = ""
    # [defaults] dataset: the dataset the app opens on.
    default_dataset: str = ""
    # [defaults] species_first: the species that take the first colours, in order.
    species_first: tuple = ()
    # [terms]: the app's words for other taxa, e.g. birdsong = "frog call".
    terms: dict = field(default_factory=dict)
    # [extensions] panel: a project's own panel for the Explorer and Label pages.
    panel_file: Path | None = None
    # [xeno_canto] links: species names link to Xeno-Canto, which is birds and a few others.
    xc_links: bool = True


def _find_config() -> Path:
    env = os.environ.get("BEX_CONFIG")
    if env:
        return Path(env)
    d = Path.cwd()
    for candidate in [d, *d.parents]:
        p = candidate / "bex.toml"
        if p.exists():
            return p
    example = _REPO_ROOT / "bex.example.toml"
    if example.exists():
        return example
    raise FileNotFoundError(
        "No bex.toml found (looked in $BEX_CONFIG, the working directory and "
        "its parents). Run `bex init` in the folder you want BEX to keep its "
        "data in, or copy bex.example.toml to bex.toml and edit the paths."
    )


def load_config(path: str | Path | None = None) -> Config:
    toml_path = Path(path) if path is not None else _find_config()
    if not toml_path.exists():
        raise FileNotFoundError(f"Config file not found: {toml_path}")
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    base = toml_path.resolve().parent

    def resolve(section: str, key: str, default: str) -> Path:
        value = raw.get(section, {}).get(key, default)
        p = Path(value)
        if p.is_absolute():
            return p
        # normpath, not resolve(): '..' collapses but symlinks are preserved —
        # data/audio must keep pointing *through* the link, not at today's target.
        return Path(os.path.normpath(base / p))

    site = raw.get("site", {})
    return Config(
        source=toml_path.resolve(),
        audio_dir=resolve("paths", "audio_dir", "../data/audio"),
        labels_dir=resolve("paths", "labels_dir", "../data"),
        store_dir=resolve("paths", "store_dir", "store"),
        cache_dir=resolve("paths", "cache_dir", "cache"),
        profiles_dir=_profiles(resolve("paths", "profiles_dir", "profiles")),
        thresholds_dir=resolve("paths", "thresholds_dir", "thresholds"),
        site_name=site.get("name", ""),
        site_lat=site.get("lat"),
        site_lon=site.get("lon"),
        default_profile=raw.get("defaults", {}).get("profile", ""),
        xc_api_key=raw.get("xeno_canto", {}).get("api_key", ""),
        display=dict(raw.get("display", {})),
        default_threshold_set=raw.get("defaults", {}).get("threshold_set", ""),
        default_dataset=raw.get("defaults", {}).get("dataset", ""),
        species_first=tuple(raw.get("defaults", {}).get("species_first", ())),
        terms=dict(raw.get("terms", {})),
        xc_links=bool(raw.get("xeno_canto", {}).get("links", True)),
        panel_file=(resolve("extensions", "panel", "")
                    if raw.get("extensions", {}).get("panel") else None),
    )


def _profiles(configured: Path) -> Path:
    # A pip install has no profiles folder of its own until the user makes one:
    # until then, the profiles that ship with BEX.
    return configured if configured.exists() else bundled("profiles")


INIT_TEMPLATE = """\
# BEX configuration, written by `bex init`. Relative paths resolve against this
# file's folder, so the whole project can move as one.

[paths]
# BEX's working data: score matrices, detections, run manifests, alignments.
store_dir = "store"
# Display spectrograms and other regenerable caches.
cache_dir = "cache"
# Named threshold sets saved from the Thresholds page.
thresholds_dir = "thresholds"
# Plausibility profiles. Until this folder exists, the ones that ship with BEX
# (uk-starter, us-ca-sierra) are used; copy one here to make your own.
profiles_dir = "profiles"
# Only for `bex ingest-sne` (the Sierra Nevada test set): its audio and labels.
# Your own recordings are scanned from wherever they are: `bex scan DIR`.
audio_dir = "data/audio"
labels_dir = "data"

[site]
# Defaults for scans that do not give their own --site/--lat/--lon.
name = ""
# lat = 51.75
# lon = -1.25

[defaults]
profile = "uk-starter"

[display]
# The app's spectrogram, for sounds outside birdsong's range. Any of: fmin, fmax,
# n_fft, hop_length, n_mels. Changing one rebuilds the spectrograms on next use.
# fmin = 150
# fmax = 14000

[xeno_canto]
# Free key from xeno-canto.org, only for cached example audio. Never commit it.
api_key = ""
"""


def assert_audio_available(cfg: Config) -> None:
    """Fail loudly, with the fix in the message, when the cold tier is missing."""
    if not cfg.audio_dir.resolve().exists():
        raise FileNotFoundError(
            f"Audio directory not found: {cfg.audio_dir} (from {cfg.source}). "
            "If it is a symlink to an external drive, plug the drive in; "
            "otherwise fix [paths].audio_dir in your bex.toml."
        )
