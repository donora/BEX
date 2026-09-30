from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def sne_labels_dir() -> Path:
    """The shared BirdSets data dir (annotations.csv, species.csv). Skips cleanly
    on a standalone clone of bex that doesn't sit inside BirdSets."""
    d = REPO.parent / "data"
    if not (d / "species.csv").exists():
        pytest.skip("SNE labels not present (../data/species.csv) — BirdSets-layout only test")
    return d


@pytest.fixture(scope="session")
def sne_audio_dir() -> Path:
    """The SNE audio symlink. Skips when the USB drive isn't mounted, so the
    suite still passes on the train."""
    d = REPO.parent / "data" / "audio"
    if not d.resolve().exists():
        pytest.skip("SNE audio not mounted (data/audio dangling) — plug in the USB drive")
    return d
