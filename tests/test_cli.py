import numpy as np
import soundfile as sf

from bex.cli import main


def test_scan_and_melcache_end_to_end(tmp_path, capsys):
    """The Stage 1 deliverable as one CLI session: arbitrary WAVs -> dataset ->
    explorer-grade spectrograms, all through the config file."""
    audio = tmp_path / "field"
    audio.mkdir()
    rng = np.random.default_rng(1)
    sf.write(audio / "20240501_063000.wav", rng.normal(0, 0.1, 44100 * 2), 44100)
    sf.write(audio / "a.flac", rng.normal(0, 0.1, 22050), 22050)

    cfg = tmp_path / "bex.toml"
    cfg.write_text('[paths]\nstore_dir = "store"\ncache_dir = "cache"\n')

    assert main(["--config", str(cfg), "scan", str(audio), "--name", "glen",
                 "--site", "glen-affric", "--lat", "57.25", "--lon", "-4.92"]) == 0
    assert main(["--config", str(cfg), "melcache", "glen"]) == 0
    assert main(["--config", str(cfg), "datasets"]) == 0

    out = capsys.readouterr().out
    assert "glen: 2 recordings" in out
    assert "built 2 new" in out
    assert (tmp_path / "cache" / "mel" / "glen" / "a.npz").exists()
    assert (tmp_path / "store" / "datasets" / "glen" / "recordings.parquet").exists()


def test_init_writes_a_config_that_loads_and_never_overwrites(tmp_path):
    from bex.cli import main
    from bex.config import load_config

    assert main(["init", str(tmp_path)]) == 0
    cfg = load_config(tmp_path / "bex.toml")
    assert cfg.store_dir == tmp_path / "store"
    # no profiles folder of its own yet: the ones BEX ships with
    assert (cfg.profiles_dir / "uk-starter" / "profile.toml").exists()
    before = (tmp_path / "bex.toml").read_text()
    assert main(["init", str(tmp_path)]) == 1
    assert (tmp_path / "bex.toml").read_text() == before
