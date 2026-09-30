#!/usr/bin/env python
"""Perch v2 runner: raw all-class scores -> the bex store.

The second arm (PLAN.md §5), and the thing that proves the adapter contract
holds for a differently-shaped model. Three ways Perch differs from BirdNET,
all of them recorded in the run manifest rather than papered over:

- **5 s windows at 32 kHz**, against BirdNET's 3 s at 48 kHz. Kept native
  (resolved D5); cross-model metrics use the shared 1 s grid.
- **Logits, not probabilities — read out with softmax, and that choice is
  measured rather than assumed.** A per-class sigmoid over Perch's logits
  saturates: median 0.27, top classes all above 0.999, and 13,317 of 14,795
  classes clearing 0.01 in a typical window, which makes the score meaningless
  and the detections index unusable. Softmax over the same logits gives a
  sparse, readable distribution — about 6.5 classes above 0.01 per window — and
  is **monotone in the logits**, so every within-window quantity this project
  measures (companion counts, shadowing gaps, ranking) is unchanged in order.
  It also gives impostor mass its natural reading: the share of the model's
  probability that lands on impossible species.

  ⚠️ Two caveats, both recorded in the manifest. Softmax normalises across
  classes, which assumes one dominant sound per window — imperfect for a dawn
  chorus. And θ does **not** mean the same thing for Perch as for BirdNET, so
  arms must be compared at matched operating points (equal detection counts),
  never at equal θ.
- **No geographic filter.** Perch ships no occurrence model, so `occ_score` is
  NaN and `suppressed` is False for every row — the model hides nothing of its
  own accord. Its muddiness is judged by the plausibility profile, which is
  applied identically to every arm and is what makes the comparison fair.

    .venv/bin/python run.py --dataset sne --limit 2
    .venv/bin/python run.py --dataset sne --run-id perch-XXXXXXXX   # resume
    .venv/bin/python run.py --dataset sne --readout sigmoid         # for truth metrics

**Which readout to ask for.** Softmax for the muddiness statistics — they are all
within-window quantities, which is exactly where a softmax is sound, and it keeps
the detections index small. Sigmoid for anything scored against ground truth: a
per-species PR curve ranks *windows* against each other, and a softmax divides by
a per-window sum, so the same bird scores lower in a busy dawn-chorus window than
alone. The two runs coexist; neither replaces the other.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

import birdnet
from bex import ingest, store
from bex.adapters import scores_to_matrix, vocab_species_keys
from bex.config import load_config
from bex.profiles import load_profile
from bex.schemas import RunManifest


def default_workers() -> int:
    """Half the logical cores, leaving the machine usable while this runs.

    Not a performance tweak — a politeness one. See --workers.
    """
    return max(1, (os.cpu_count() or 4) // 2)


def readout_of(manifest: RunManifest) -> str:
    """Which readout produced this run's scores, read back off its manifest.

    The leading word of `score_transform` is load-bearing, not decoration: it is
    the only machine-readable record of the choice, and `RunManifest` has no field
    for it. Anything else raises rather than guessing — a wrong guess here appends
    the wrong kind of score to an existing run.
    """
    t = manifest.score_transform.lower()
    if t.startswith("softmax"):
        return "softmax"
    if t.startswith("per-class sigmoid"):
        return "sigmoid"
    raise SystemExit(
        f"cannot tell which readout produced run {manifest.run_id}: its "
        f"score_transform starts {manifest.score_transform[:40]!r}. Refusing to "
        "resume a run whose scores might be of a different kind."
    )


def check_label_space(keys: list[str]) -> None:
    """Fail loudly if Perch's labels are not canonical-joinable.

    PLAN §3c's cardinal sin is a silent taxonomy mismatch: if Perch emitted
    eBird codes rather than scientific names, every key would still *look* fine
    and would simply never join to BirdNET's, fabricating total disagreement
    between the two models. A scientific name has two words; a code has one.
    """
    single_word = [k for k in keys if " " not in k]
    if len(single_word) > 0.5 * len(keys):
        raise SystemExit(
            f"Perch labels do not look like scientific names — {len(single_word)} "
            f"of {len(keys)} are single tokens, e.g. {single_word[:5]}. These will "
            "not join to BirdNET's keys, so the comparison would be meaningless. "
            "A code -> scientific mapping is needed in bex/taxonomy.py first."
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--recordings", help="comma-separated recording_ids")
    ap.add_argument("--limit", type=int, help="process at most N unprocessed recordings")
    ap.add_argument("--profile", help="plausibility profile (default: from config)")
    ap.add_argument("--embeddings", action="store_true",
                    help="also cache per-window embeddings (Stage 7's probe needs them)")
    ap.add_argument("--readout", choices=("softmax", "sigmoid"), default=None,
                    help="how logits become score_raw. softmax: sparse and readable, "
                         "but normalised WITHIN a window, so it is not comparable "
                         "across windows and depresses per-species AP. sigmoid: "
                         "per-class and monotone in the logit, so across-window "
                         "rankings (and therefore PR curves and AP) are faithful — "
                         "at the cost of a saturated, index-hostile distribution.")
    ap.add_argument("--min-score", type=float, default=None,
                    help="detections-index floor (default: per readout, see below)")
    ap.add_argument("--run-id", default="", help="resume an existing run")
    ap.add_argument("--workers", type=int, default=None,
                    help="""workers the model may spawn. The package's default is one per
                         logical core, and each worker runs TensorFlow with its own
                         thread pool — on a 12-core machine that is 13 processes
                         fighting over 12 cores (observed load average ~80, and the
                         machine unusable). Half the cores leaves the box workable
                         and is not meaningfully slower, because the bottleneck is
                         already memory bandwidth rather than cores.""")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    workers = args.workers if args.workers is not None else default_workers()

    cfg = load_config(args.config)
    recordings, annotations = ingest.read_dataset(cfg.store_dir, args.dataset)
    audio_root = ingest.dataset_audio_root(cfg.store_dir, args.dataset)
    if args.recordings:
        wanted = set(args.recordings.split(","))
        recordings = recordings.filter(pl.col("recording_id").is_in(wanted))

    profile_name = args.profile or cfg.default_profile
    profile = load_profile(cfg.profiles_dir / profile_name)
    always_keys = profile.plausible_keys(max_tier=2)
    if annotations is not None:
        always_keys |= set(annotations["species_key"].unique())

    print("loading Perch v2 (TensorFlow)...", flush=True)
    model = birdnet.load_perch_v2()
    # Perch's labels are already the key — bare scientific names, plus ~80
    # AudioSet event classes ('Bass_drum') whose underscores are not a
    # common-name separator.
    keys = vocab_species_keys(list(model.species_list), split_common=False)
    check_label_space(keys)
    window_s = float(model.get_segment_size_s())
    print(f"  {model.n_species:,} classes · {window_s:.0f}s windows · "
          f"{model.get_sample_rate()} Hz", flush=True)

    # The index floor has to follow the readout. Under softmax about 6.5 classes
    # per window clear 0.01; under sigmoid 13,317 of 14,795 do, which is how two
    # recordings once produced 19.2M index rows. A sigmoid run therefore indexes
    # by RANK (top_k plus the always-keys), not by an absolute score, and 1.1 is
    # the honest way to say "this floor selects nothing".
    def index_floor(readout: str) -> float:
        return 0.01 if readout == "softmax" else 1.1

    # Perch's sigmoid is saturated (median ~0.96 for an abundant species), and
    # float16 steps by 0.00098 up there — stored at that precision it left 1.1%
    # of top-ranked windows distinguishable and made AP a measurement of the
    # store. Softmax spans orders of magnitude and is fine in float16.
    def score_dtype(readout: str):
        return np.float16 if readout == "softmax" else np.float32

    SCORE_TRANSFORMS = {
        "softmax": (
            "softmax over raw logits — sigmoid saturates on this head "
            "(13,317 of 14,795 classes clear 0.01 per window); softmax is "
            "monotone in the logits so within-window rankings are identical. "
            "θ is NOT comparable with BirdNET's: match on detection counts. "
            "⚠️ Normalised within a window, so scores are NOT comparable ACROSS "
            "windows — per-species AP on this run is a lower bound (PLAN §9, "
            "Stage 5.5). Use a sigmoid run for truth metrics."
        ),
        "sigmoid": (
            "per-class sigmoid over raw logits (sensitivity 1.0) — monotone in "
            "each class's logit, so ranking windows by a species' score ranks "
            "them by that species' logit, which is what a PR curve needs. "
            "Saturated by design (median ~0.27 across all classes), so θ here is "
            "not comparable with BirdNET's or with the softmax run's, and the "
            "detections index is rank-based rather than score-based."
        ),
    }

    if args.run_id:
        manifest = RunManifest.load(cfg.store_dir, args.run_id)
        assert manifest.model_name == "perch", manifest.model_name
        window_s = manifest.window_s
        # A resume must not change the readout. Half a run scored with softmax and
        # half with sigmoid would be undetectable downstream — every file looks
        # fine on its own, the manifest would name one transform for scores
        # produced by two, and every metric computed from it would be quietly
        # meaningless. So the manifest decides, and a contradicting flag is fatal.
        stored = readout_of(manifest)
        if args.readout is not None and args.readout != stored:
            raise SystemExit(
                f"run {args.run_id} was scored with --readout {stored}; refusing to "
                f"append {args.readout} scores to it. Start a new run instead "
                f"(drop --run-id) if you want the other readout."
            )
        readout = stored
        if args.readout is None:
            print(f"resuming with the run's own readout: {readout}", flush=True)
    else:
        readout = args.readout or "softmax"
        manifest = RunManifest(
            model_name="perch",
            model_version=str(model.get_version()),
            window_s=window_s,
            hop_s=window_s,
            input_sr=model.get_sample_rate(),
            score_transform=SCORE_TRANSFORMS[readout],
            geofilter={
                "model": None,
                "note": "Perch ships no occurrence model; occ_score is NaN and "
                        "suppressed is False. Implausibility comes from the "
                        "plausibility profile, applied identically to every arm.",
            },
            profile=profile_name,
            dataset=args.dataset,
            packages={
                "birdnet": importlib.metadata.version("birdnet"),
                "tensorflow": importlib.metadata.version("tensorflow"),
                "python": sys.version.split()[0],
            },
        )
        manifest.save(cfg.store_dir)
    run_id = manifest.run_id
    (Path(cfg.store_dir) / "runs" / run_id / "labels.txt").write_text(
        "\n".join(model.species_list) + "\n")
    print(f"run {run_id} -> {cfg.store_dir}", flush=True)

    done = set(store.list_scores(cfg.store_dir, run_id))
    todo = recordings.filter(~pl.col("recording_id").is_in(done))
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"{len(done)} recording(s) already scored, {len(todo)} to process", flush=True)

    for row in todo.iter_rows(named=True):
        t0 = time.time()
        src = str(ingest.resolve_audio(cfg.store_dir, args.dataset, audio_root,
                                       row["recording_id"], row["path"]))
        res = model.predict(
            src,
            top_k=None,
            n_workers=workers,
            default_confidence_threshold=None,
            # Exactly one of these; the package rejects both. See --readout.
            apply_softmax=(readout == "softmax"),
            apply_sigmoid=(readout == "sigmoid"),
            # Any fixed positive sensitivity is monotone in the logit, so AP does
            # not depend on this value; pinned so the manifest means something.
            **({"sigmoid_sensitivity": 1.0} if readout == "sigmoid" else {}),
            show_stats=None,
        )
        probs = np.asarray(res.species_probs)[0]
        assert res.segment_duration_s == window_s, res.segment_duration_s
        store.write_scores(cfg.store_dir, store.ScoreMatrix(
            recording_id=row["recording_id"],
            run_id=run_id,
            scores=scores_to_matrix(probs, dtype=score_dtype(readout)),
            start_s=(np.arange(probs.shape[0]) * window_s).astype(np.float32),
            window_s=window_s,
            species_keys=keys,
        ), dtype=score_dtype(readout))
        if args.embeddings:
            emb = np.asarray(model.encode(src, show_stats=None).embeddings)[0]
            out = Path(cfg.store_dir) / "embeddings" / run_id
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"{row['recording_id']}.npz",
                                embeddings=emb.astype(np.float16))

        n_win = probs.shape[0]
        expected = math.floor(row["duration_s"] / window_s)
        if abs(n_win - expected) > 1:
            print(f"  WARNING {row['recording_id']}: {n_win} windows for "
                  f"{row['duration_s']:.0f}s (expected ~{expected})", flush=True)
        print(f"  {row['recording_id']}: {n_win} windows in {time.time() - t0:.0f}s",
              flush=True)

    # Detections index, rebuilt whole-run so resumed and clean runs converge.
    frames = []
    known = set(recordings["recording_id"])
    for recording_id in store.list_scores(cfg.store_dir, run_id):
        if recording_id not in known:
            continue
        sm = store.read_scores(cfg.store_dir, run_id, recording_id)
        frames.append(store.derive_detections(
            sm,
            min_score=(args.min_score if args.min_score is not None
                       else index_floor(readout)),
            always_keys=always_keys))

    if frames:
        detections = pl.concat(frames)
        out = store.write_detections(cfg.store_dir, run_id, detections)
        print(f"{len(frames)} recordings, {len(detections):,} detection rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
