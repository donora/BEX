"""Display-spectrogram cache. Adapted from `birdsong/features.py`.

The sibling's two hard-won choices are kept:

- **One array per recording, not per window** — a display span is a *slice*, so
  the app's zoom level and the models' window grids stay free parameters.
- **dB at a fixed reference (ref=1.0), not per-clip max** — a per-clip reference
  would be "dB relative to the loudest event this hour"; normalisation is a
  display-time decision, made per rendered view.

Two departures, both because this cache serves arbitrary field recordings on a
tight SSD rather than one fixed dataset: each recording's `.npz` is
**self-contained** (mel + its own sample rate + params — a 22.05 kHz WAV and a
48 kHz AudioMoth file coexist in one dataset), and the default hop is **512**
(≈11–16 ms frames — display-grade, half the sibling's size; ~1.9 GB for all of
SNE instead of 3.8 on a drive with 10 GB free). Model adapters never read this
cache; they read audio. This is purely the app's view.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import polars as pl

from .audio import MelParams

DISPLAY_MEL = MelParams(hop_length=512)


def mel_path(cache_dir: str | Path, dataset: str, recording_id: str) -> Path:
    return Path(cache_dir) / "mel" / dataset / f"{recording_id}.npz"


def build_one(
    audio_path: str | Path,
    out_path: str | Path,
    p: MelParams = DISPLAY_MEL,
) -> tuple[int, int]:
    """Compute one recording's whole log-mel and write it. Returns the mel shape."""
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=p.n_fft,
        hop_length=p.hop_length,
        n_mels=p.n_mels,
        fmin=p.fmin,
        fmax=min(p.fmax, sr // 2),
        power=2.0,
    )
    mel_db = librosa.power_to_db(power, ref=1.0, top_db=None).astype(np.float16)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        mel_db=mel_db,
        sr=np.int32(sr),
        duration_s=np.float64(len(y) / sr),
        **{k: np.float64(v) for k, v in asdict(p).items()},
    )
    return mel_db.shape


def build_cache(
    cache_dir: str | Path,
    dataset: str,
    recordings: pl.DataFrame,
    audio_dir: str | Path,
    p: MelParams = DISPLAY_MEL,
    limit: int | None = None,
    progress: Callable[[str], None] = lambda msg: None,
    resolve: Callable[[str, str], Path] | None = None,
) -> int:
    """Build missing cache entries for a dataset. Resumable: existing files are
    skipped, so an interrupted build (or a tight disk) just picks up later.
    `resolve(recording_id, rel_path)` overrides where audio is read from — the
    CLI passes ingest.resolve_audio so repaired copies are preferred."""
    built = 0
    for recording_id, rel in recordings.select("recording_id", "path").iter_rows():
        out = mel_path(cache_dir, dataset, recording_id)
        if out.exists():
            continue
        if limit is not None and built >= limit:
            break
        src = resolve(recording_id, rel) if resolve else Path(audio_dir) / rel
        shape = build_one(src, out, p)
        built += 1
        progress(f"{recording_id}: mel {shape[0]}x{shape[1]} -> {out.name}")
    return built


def load_mel(cache_dir: str | Path, dataset: str, recording_id: str) -> tuple[np.ndarray, dict]:
    """-> (mel_db float32 (n_mels, frames), meta). Meta carries sr/hop/params, so
    time <-> frame conversion never guesses."""
    with np.load(mel_path(cache_dir, dataset, recording_id)) as z:
        mel = z["mel_db"].astype(np.float32)
        meta = {k: z[k].item() for k in z.files if k != "mel_db"}
    return mel, meta


def slice_mel(mel: np.ndarray, meta: dict, start_s: float, end_s: float) -> np.ndarray:
    """The display span [start_s, end_s) as a view of the cached array."""
    frames_per_s = meta["sr"] / meta["hop_length"]
    lo = max(0, int(round(start_s * frames_per_s)))
    hi = min(mel.shape[1], int(round(end_s * frames_per_s)))
    return mel[:, lo:hi]


def display_range(mel_slice: np.ndarray, floor_pct: float = 35.0, ceil_pct: float = 99.8) -> tuple[float, float]:
    """(vmin, vmax) for rendering *this view*: clip the noise floor, keep the calls.

    This is the normalisation the cache deliberately defers (fixed-reference dB on
    disk, contrast decided per rendered span). Percentiles of the visible slice
    mean the same view always fills the colormap, whether it's a quiet night hour
    or a dawn chorus — the absolute numbers stay in the array for anyone who asks.
    """
    finite = mel_slice[np.isfinite(mel_slice)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [floor_pct, ceil_pct])
    return float(vmin), float(max(vmax, vmin + 1.0))
