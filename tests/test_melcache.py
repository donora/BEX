import numpy as np
import polars as pl
import pytest
import soundfile as sf

from bex.audio import MelParams
from bex.melcache import (
    build_cache,
    build_one,
    display_range,
    load_mel,
    mel_path,
    slice_mel,
)

SR = 22050


@pytest.fixture()
def tone_wav(tmp_path):
    """4 s: 2 s of near-silence, then 2 s of a 2 kHz tone — a spectrogram whose
    correctness is visible in the numbers."""
    t = np.arange(4 * SR) / SR
    y = 0.5 * np.sin(2 * np.pi * 2000 * t) * (t >= 2.0) + 1e-4 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    sf.write(path, y, SR)
    return path


def test_build_and_load(tone_wav, tmp_path):
    out = mel_path(tmp_path / "cache", "demo", "tone")
    shape = build_one(tone_wav, out, MelParams(hop_length=512))
    mel, meta = load_mel(tmp_path / "cache", "demo", "tone")

    assert mel.shape == shape == (128, 1 + 4 * SR // 512)
    assert meta["sr"] == SR and meta["hop_length"] == 512
    assert meta["duration_s"] == pytest.approx(4.0)
    assert np.isfinite(mel).all()

    # Fixed reference, not per-clip max: the loud half must be *much* hotter than
    # the quiet half, in absolute dB.
    quiet = slice_mel(mel, meta, 0.5, 1.5).max()
    loud = slice_mel(mel, meta, 2.5, 3.5).max()
    assert loud - quiet > 40

    # The tone sits where 2 kHz sits: the hottest mel row in the loud half is in
    # the band the params put 2 kHz in (rough check, not a filterbank audit).
    loud_slice = slice_mel(mel, meta, 2.5, 3.5)
    hot_row = int(loud_slice.max(axis=1).argmax())
    assert 30 < hot_row < 90


def test_slice_bounds(tone_wav, tmp_path):
    build_one(tone_wav, mel_path(tmp_path, "d", "tone"))
    mel, meta = load_mel(tmp_path, "d", "tone")
    assert slice_mel(mel, meta, -5.0, 1.0).shape[1] == slice_mel(mel, meta, 0.0, 1.0).shape[1]
    assert slice_mel(mel, meta, 3.9, 99.0).shape[1] > 0
    assert slice_mel(mel, meta, 0.0, 99.0).shape[1] == mel.shape[1]


def test_display_range(tone_wav, tmp_path):
    build_one(tone_wav, mel_path(tmp_path, "d", "tone"))
    mel, meta = load_mel(tmp_path, "d", "tone")
    vmin, vmax = display_range(slice_mel(mel, meta, 0.0, 4.0))
    assert vmin < vmax
    # The tone's energy sits above the display ceiling's neighbourhood of the
    # noise floor — i.e. the interesting content is inside the rendered range.
    assert slice_mel(mel, meta, 2.5, 3.5).max() > vmin

    import numpy as np

    assert display_range(np.full((4, 4), np.nan, dtype=np.float32)) == (0.0, 1.0)


def test_build_cache_resumable(tone_wav, tmp_path):
    recordings = pl.DataFrame(
        {"recording_id": ["tone", "tone2"], "path": ["tone.wav", "tone2.wav"]}
    )
    root = tone_wav.parent
    sf.write(root / "tone2.wav", np.zeros(SR), SR)

    cache = tmp_path / "cache"
    assert build_cache(cache, "d", recordings, root, limit=1) == 1
    assert build_cache(cache, "d", recordings, root) == 1  # only the missing one
    assert build_cache(cache, "d", recordings, root) == 0  # idempotent
