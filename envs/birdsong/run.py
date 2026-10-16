#!/usr/bin/env python
"""Birdsong runner: a from-scratch CNN trained on SNE (BirdSets/birdsong) -> the bex store.

The model is a checkpoint from `python -m birdsong.train` (runs/<name>/best.pt):
56 SNE species, 5 s windows at 32 kHz, per-class sigmoid. This runner rebuilds
the exact input the training loop saw — log-mel at ref=1.0 dB, 5 s windows on a
5 s hop, per-window `peak` normalisation — and writes every class score for
every window, with no threshold.

**Leakage guard.** The checkpoint was trained on some SNE recordings and
selected on others; scoring those would flatter it. The runner rebuilds the
training run's recording split and refuses any recording in its train or val
split unless `--allow-seen` is passed. Score it on the held-out `test` split
(or on audio it never saw) for a fair comparison with BirdNET and Perch.

    .venv/bin/python run.py --dataset sne-test --checkpoint ../../../runs/sed-fine/best.pt

No geo model: occ_score is NaN and suppressed is False, as for Perch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from bex import ingest, store
from bex.adapters import scores_to_matrix, vocab_species_keys
from bex.config import load_config
from bex.profiles import load_profile
from bex.schemas import RunManifest
from bex.taxonomy import canonical_from_sne, map_ebird_codes

#: BirdSets/ — where the `birdsong` package lives, relative to this file.
DEFAULT_SRC = Path(__file__).resolve().parents[3]
#: The checkpoint the app's Run button uses: one path in this (gitignored) file,
#: or $BEX_BIRDSONG_CHECKPOINT. The app starts runners without --checkpoint.
CHECKPOINT_FILE = Path(__file__).resolve().parent / "checkpoint.txt"


def default_checkpoint() -> str | None:
    import os
    if os.environ.get("BEX_BIRDSONG_CHECKPOINT"):
        return os.environ["BEX_BIRDSONG_CHECKPOINT"]
    if CHECKPOINT_FILE.exists():
        path = Path(CHECKPOINT_FILE.read_text().strip()).expanduser()
        return str(path if path.is_absolute() else CHECKPOINT_FILE.parent / path)
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", default=default_checkpoint(),
                    help="a birdsong best.pt (default: checkpoint.txt next to this file)")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="directory containing the birdsong package")
    ap.add_argument("--recordings", help="comma-separated recording_ids (default: all)")
    ap.add_argument("--limit", type=int, help="process at most N unprocessed recordings")
    ap.add_argument("--profile", help="plausibility profile (default: from config)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--allow-seen", action="store_true",
                    help="score recordings the checkpoint trained or was selected on")
    ap.add_argument("--run-id", default="", help="resume an existing run")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    if not args.checkpoint:
        ap.error(f"no checkpoint: pass --checkpoint, or put its path in {CHECKPOINT_FILE}")

    sys.path.insert(0, args.src)
    import librosa
    import torch
    from birdsong import data as D
    from birdsong.audio import MelParams
    from birdsong.dataset import normalise
    from birdsong.model import BirdCNN, CNNConfig
    from birdsong.splits import make_split

    cfg = load_config(args.config)
    recordings, annotations = ingest.read_dataset(cfg.store_dir, args.dataset)
    audio_root = ingest.dataset_audio_root(cfg.store_dir, args.dataset)
    if args.recordings:
        wanted = set(args.recordings.split(","))
        recordings = recordings.filter(pl.col("recording_id").is_in(wanted))

    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics_path = ckpt_path.parent / "metrics.json"
    train_args = json.loads(metrics_path.read_text()).get("args", {}) if metrics_path.exists() else {}
    run_name = train_args.get("name", ckpt_path.parent.name)

    # Leakage guard: the split the training run used (deterministic in its seed).
    split = make_split(seed=int(train_args.get("seed", 0)))
    seen = {Path(f).stem: s for f, s in zip(split["filename"], split["split"]) if s != "test"}
    leaked = sorted(r for r in recordings["recording_id"] if r in seen)
    if leaked and not args.allow_seen:
        print(f"refusing: {len(leaked)} recording(s) were in {run_name}'s train/val split "
              f"(e.g. {leaked[0]} -> {seen[leaked[0]]}). Score the test split, or pass "
              "--allow-seen to do it anyway.", file=sys.stderr)
        return 2

    codes = D.species_codes()
    canonical = canonical_from_sne(cfg.labels_dir)
    report = map_ebird_codes(codes, canonical)
    if not report.lossless:
        raise ValueError(f"species mapping not lossless — {report.summary()}")
    common = dict(canonical.select("species_key", "common_name").iter_rows())
    labels = [f"{report.mapping[c]}_{common.get(report.mapping[c], c)}" for c in codes]
    keys = vocab_species_keys(labels)

    model_cfg = dict(ckpt["cfg"])
    model_cfg["channels"] = tuple(model_cfg["channels"])
    model = BirdCNN(CNNConfig(**model_cfg))
    model.load_state_dict(ckpt["model"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

    p = MelParams()
    win_s = 5.0
    fps = p.sr / p.hop_length
    n_frames = int(round(win_s * fps))
    norm = train_args.get("normalisation", "peak")

    profile_name = args.profile or cfg.default_profile
    profile = load_profile(cfg.profiles_dir / profile_name)
    always_keys = profile.plausible_keys(max_tier=2)
    if annotations is not None:
        always_keys |= set(annotations["species_key"].unique())

    if args.run_id:
        manifest = RunManifest.load(cfg.store_dir, args.run_id)
        assert manifest.model_name == "birdsong", manifest.model_name
    else:
        manifest = RunManifest(
            model_name="birdsong",
            model_version=run_name,
            window_s=win_s,
            hop_s=win_s,
            input_sr=p.sr,
            score_transform="sigmoid over the clip logits — the training loop's own readout",
            geofilter={"model": None,
                       "note": "No occurrence model; occ_score is NaN and suppressed is False."},
            profile=profile_name,
            dataset=args.dataset,
            packages={
                "torch": torch.__version__,
                "librosa": importlib.metadata.version("librosa"),
                "python": sys.version.split()[0],
            },
        )
        manifest.save(cfg.store_dir)
    run_id = manifest.run_id
    run_dir = Path(cfg.store_dir) / "runs" / run_id
    (run_dir / "labels.txt").write_text("\n".join(labels) + "\n")
    (run_dir / "checkpoint.json").write_text(json.dumps({
        "path": str(ckpt_path), "sha256": _sha256(ckpt_path), "epoch": ckpt.get("epoch"),
        "cfg": model_cfg, "train_args": train_args, "mel_params": vars(p) if hasattr(p, "__dict__") else str(p),
        "seen_recordings_scored": leaked,
    }, indent=2, default=str))
    print(f"run {run_id} ({run_name}, epoch {ckpt.get('epoch')}, {device}) -> {cfg.store_dir}", flush=True)

    done = set(store.list_scores(cfg.store_dir, run_id))
    todo = recordings.filter(~pl.col("recording_id").is_in(done))
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"{len(done)} recording(s) already scored, {len(todo)} to process", flush=True)

    for row in todo.iter_rows(named=True):
        t0 = time.time()
        path = ingest.resolve_audio(cfg.store_dir, args.dataset, audio_root,
                                    row["recording_id"], row["path"])
        y, _ = librosa.load(str(path), sr=p.sr, mono=True)
        power = librosa.feature.melspectrogram(
            y=y, sr=p.sr, n_fft=p.n_fft, hop_length=p.hop_length,
            n_mels=p.n_mels, fmin=p.fmin, fmax=p.fmax, power=2.0)
        # ref=1.0, then float16 — exactly what the training cache held.
        db = librosa.power_to_db(power, ref=1.0).astype(np.float16).astype(np.float32)

        duration = len(y) / p.sr
        n_win = int(np.floor((duration - win_s) / win_s)) + 1
        starts = np.arange(n_win) * win_s
        batch = []
        for s in starts:
            f0 = int(round(s * fps))
            spec = db[:, f0:f0 + n_frames]
            if spec.shape[1] < n_frames:
                spec = np.pad(spec, ((0, 0), (0, n_frames - spec.shape[1])), mode="edge")
            batch.append(normalise(spec, norm))
        x_all = np.stack(batch)[:, None]  # (windows, 1, mels, frames)

        probs = []
        with torch.no_grad():
            for i in range(0, len(x_all), args.batch):
                xb = torch.from_numpy(x_all[i:i + args.batch]).to(device)
                probs.append(torch.sigmoid(model(xb)).cpu().numpy())
        probs = np.concatenate(probs)

        sm = store.ScoreMatrix(
            recording_id=row["recording_id"],
            run_id=run_id,
            # float32: sigmoid scores crowd near 0 and 1, where float16 is coarse.
            scores=scores_to_matrix(probs, dtype=np.float32),
            start_s=starts.astype(np.float32),
            window_s=win_s,
            species_keys=keys,
        )
        store.write_scores(cfg.store_dir, sm)
        print(f"  {row['recording_id']}: {n_win} windows in {time.time() - t0:.0f}s", flush=True)

    frames = []
    known = set(recordings["recording_id"])
    for recording_id in store.list_scores(cfg.store_dir, run_id):
        if recording_id not in known:
            continue
        sm = store.read_scores(cfg.store_dir, run_id, recording_id)
        frames.append(store.derive_detections(sm, always_keys=always_keys))
    detections = pl.concat(frames)
    out = store.write_detections(cfg.store_dir, run_id, detections)
    print(f"{len(frames)} recordings, {len(detections)} detection rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
