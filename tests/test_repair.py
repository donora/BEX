import numpy as np
import pytest
import soundfile as sf

from bex.ingest import repairs_dir, resolve_audio
from bex.repair import needs_repair, repair_recording


@pytest.fixture()
def flac(tmp_path):
    rng = np.random.default_rng(7)
    y = (rng.normal(0, 0.05, 22050 * 2) * 32767).astype(np.int16)
    p = tmp_path / "rec.flac"
    sf.write(p, y, 22050, subtype="PCM_16")
    return p, y


def test_needs_repair_false_on_healthy(flac):
    assert not needs_repair(flac[0])


def test_repair_roundtrip_lossless(flac, tmp_path):
    src, y = flac
    dst = tmp_path / "out" / "rec.flac"
    dur = repair_recording(src, dst)
    assert dur == pytest.approx(2.0)
    back, sr = sf.read(dst, dtype="int16")
    assert sr == 22050
    np.testing.assert_array_equal(back, y)  # int16 in, int16 out — bit-identical
    assert not dst.with_suffix(".part.flac").exists()


def test_resolve_audio_prefers_repair(tmp_path, flac):
    store, root = tmp_path / "store", flac[0].parent
    plain = resolve_audio(store, "d", root, "rec", "rec.flac")
    assert plain == root / "rec.flac"

    rep = repairs_dir(store, "d") / "rec.flac"
    rep.parent.mkdir(parents=True)
    rep.write_bytes(b"x")
    assert resolve_audio(store, "d", root, "rec", "rec.flac") == rep
