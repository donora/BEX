"""Repair recordings libsndfile cannot decode, without touching the originals.

The SNE drive carries four files (every `*_170000.flac` — the 17:00 evening
sessions, presumably one recorder configuration) whose FLAC stream CoreAudio
decodes perfectly but libsndfile rejects with "flac decoder lost sync". Every
consumer in this stack (soundfile, librosa, BirdNET's producer) sits on
libsndfile, so those files are unreadable as-is — silently so, since headers
(`sf.info`) parse fine.

The fix: decode through `audioread` (CoreAudio on macOS, gstreamer/ffmpeg
elsewhere) and re-encode to a clean FLAC under `store/repairs/<dataset>/`,
keyed by recording_id. `ingest.resolve_audio` prefers a repair when one
exists; the cold-tier original is never modified. The transcode is
lossless-to-lossless (int16 PCM in, int16 PCM out) and verified frame-for-frame
on write.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def needs_repair(path: str | Path, probe_s: float = 5.0, full: bool = False) -> bool:
    """Can libsndfile actually *decode* this file? Headers lie (`sf.info`
    succeeds on the broken files), so decoding is the only honest test.

    Default: probe reads at start, middle and end — catches the lost-at-frame-0
    class cheaply. `full=True` decodes the entire stream, because SNE_008 taught
    us the hard way that a file can pass all three probes and still lose sync
    somewhere in the middle.
    """
    try:
        with sf.SoundFile(str(path)) as f:
            if full:
                while len(f.read(f.samplerate * 60)) > 0:
                    pass
                return False
            n = int(f.samplerate * probe_s)
            f.read(n)
            if f.frames > 3 * n:
                f.seek(f.frames // 2)
                f.read(n)
                f.seek(max(0, f.frames - n))
                f.read(n)
        return False
    except sf.LibsndfileError:
        return True


def repair_recording(src: str | Path, dst: str | Path) -> float:
    """Transcode src -> dst (FLAC, int16) via audioread. Returns duration_s.

    Raises if the decoded stream is empty or the written file doesn't read back
    with the same frame count — a repair that can't prove itself is a failure.
    """
    import audioread  # optional dependency; only repairs need it

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.flac")

    with audioread.audio_open(str(src)) as fin:
        sr, ch = fin.samplerate, fin.channels
        frames = 0
        with sf.SoundFile(tmp, "w", samplerate=sr, channels=ch,
                          format="FLAC", subtype="PCM_16") as fout:
            for buf in fin:
                block = np.frombuffer(buf, dtype="<i2").reshape(-1, ch)
                fout.write(block)
                frames += len(block)

    if frames == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{src}: audioread produced no audio")
    check = sf.info(str(tmp))
    if check.frames != frames:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{src}: wrote {frames} frames but file reads back {check.frames}")
    tmp.rename(dst)
    return frames / sr


def build_mel(cache_dir: str | Path, store_dir: str | Path, dataset: str,
              recording_id: str, rel_path: str, audio_root: str | Path) -> str:
    """Build one recording's display spectrogram, repairing it first if
    libsndfile cannot decode it. Returns 'cached', 'built' or 'repaired'.

    Repairing here, at the first step that reads every file end to end, means
    the model runners — which read through `ingest.resolve_audio` too — find a
    clean copy waiting, instead of failing an hour into a run.
    """
    from . import ingest
    from .melcache import build_one, mel_path

    out = mel_path(cache_dir, dataset, recording_id)
    if out.exists():
        return "cached"
    src = ingest.resolve_audio(store_dir, dataset, audio_root, recording_id, rel_path)
    try:
        build_one(src, out)
        return "built"
    except sf.LibsndfileError:
        dst = ingest.repairs_dir(store_dir, dataset) / f"{recording_id}.flac"
        repair_recording(Path(audio_root) / rel_path, dst)
        build_one(dst, out)
        return "repaired"
