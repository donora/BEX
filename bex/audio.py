"""Audio access + display spectrograms. Adapted from `birdsong/audio.py`.

Changes from the sibling: no assumed native sample rate (the unlabelled pipeline
takes arbitrary field recordings), and file metadata comes through one function so
the ingest scan and the clip loader agree about what a file is. Same core habit:
never load a whole recording — seek and read only the seconds needed, straight off
cold storage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# librosa is imported lazily inside the functions that need it: model-runner
# envs import this module (via ingest) for file_info alone, and shouldn't pay
# for — or pin — librosa/numba to do so.


@dataclass(frozen=True)
class FileInfo:
    duration_s: float
    sample_rate: int
    channels: int


def file_info(path: str | Path) -> FileInfo:
    """Header-only read: fast, no audio decoded."""
    info = sf.info(str(path))
    return FileInfo(
        duration_s=info.frames / info.samplerate,
        sample_rate=info.samplerate,
        channels=info.channels,
    )


def load_clip(
    path: str | Path,
    start_s: float,
    end_s: float,
    pad_s: float = 1.0,
    sr: int | None = None,
) -> tuple[np.ndarray, int, float, float]:
    """Read just [start_s, end_s] (+ context padding) from a long recording.

    sr=None keeps the file's native rate; adapters that need a fixed model input
    rate pass it explicitly and librosa resamples on the fly.

    Returns (samples, sr, clip_start_s, clip_end_s) — the absolute times in the
    recording that the returned audio covers.
    """
    import librosa

    path = str(path)
    total = sf.info(path).duration
    lo = max(0.0, start_s - pad_s)
    hi = min(total, end_s + pad_s)
    y, out_sr = librosa.load(path, sr=sr, offset=lo, duration=hi - lo, mono=True)
    return y, int(out_sr), lo, hi


@dataclass(frozen=True)
class MelParams:
    """Display-spectrogram knobs (the app's view, not any model's input).

    Defaults follow the sibling project at 32 kHz; fmax is clamped to Nyquist at
    load time so low-sample-rate field recordings don't error.
    """

    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    fmin: int = 150
    fmax: int = 14_000


def mel_spectrogram(y: np.ndarray, sr: int, p: MelParams = MelParams()) -> np.ndarray:
    """Waveform -> log-mel in dB relative to the clip's peak. Shape (n_mels, frames)."""
    import librosa

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
    return librosa.power_to_db(power, ref=np.max)
