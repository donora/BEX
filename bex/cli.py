"""The `bex` command — the pipeline without reading the source (PLAN.md §8 rule 3).

    bex ingest-sne                     # labelled test rig -> dataset 'sne'
    bex scan DIR --name my-site ...    # unlabelled folder -> a dataset
    bex melcache DATASET [--limit N]   # build display spectrograms
    bex datasets                       # what the store holds
    bex survey DATASET --setup NAME    # a saved setup's species lists, as CSV
    bex labels DATASET                 # label sets: progress, versions
    bex init [DIR]                     # start a project folder: writes bex.toml
    bex app                            # open the app in your browser

Every subcommand reads `bex.toml` (or the file in $BEX_CONFIG) and prints
what it did and where it put it.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from . import ingest, melcache
from .config import load_config


# The BEX accent (a muted petrol teal; V1.md HM1) passed as flags, because
# Streamlit reads .streamlit/config.toml from the working directory, not the app's.
THEME = ["--theme.light.primaryColor=#3e6a6b", "--theme.dark.primaryColor=#4f868c"]


def _init(folder: Path) -> int:
    from .config import INIT_TEMPLATE

    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "bex.toml"
    if target.exists():
        print(f"{target} already exists — left as it is.")
        return 1
    target.write_text(INIT_TEMPLATE)
    print(f"wrote {target}\n"
          "next: bex scan /path/to/recordings --name my-site --lat .. --lon ..\n"
          "      then run a model (see the Models page), and `bex app` to explore.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bex", description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to bex.toml (default: auto-discover)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest-sne", help="ingest the labelled SNE test rig as dataset 'sne'")

    p_scan = sub.add_parser("scan", help="scan a folder of unlabelled recordings")
    p_scan.add_argument("dir", help="directory of audio files")
    p_scan.add_argument("--name", required=True, help="dataset name in the store")
    p_scan.add_argument("--site", default="", help="site name for the whole folder")
    p_scan.add_argument("--lat", type=float, default=math.nan)
    p_scan.add_argument("--lon", type=float, default=math.nan)

    p_mel = sub.add_parser("melcache", help="build display spectrograms for a dataset")
    p_mel.add_argument("dataset")
    p_mel.add_argument("--limit", type=int, default=None, help="build at most N new entries")

    p_rep = sub.add_parser("repair", help="transcode recordings libsndfile cannot decode")
    p_rep.add_argument("dataset")
    p_rep.add_argument("--full", action="store_true",
                       help="decode every file end-to-end (catches mid-stream corruption; slower)")

    sub.add_parser("datasets", help="list datasets in the store")

    p_sv = sub.add_parser("survey", help="apply a saved survey setup to every recording")
    p_sv.add_argument("dataset")
    p_sv.add_argument("--setup", help="setup name (default: list the saved setups)")
    p_sv.add_argument("--truth", default="",
                      help="label set ref whose fully labelled recordings keep their "
                           "labels: imported, <set>@v<N> or <set> (working copy)")
    p_sv.add_argument("--out", help="CSV path (default: print)")

    p_lb = sub.add_parser("labels", help="label sets for a dataset: progress and versions")
    p_lb.add_argument("dataset")

    sub.add_parser("app", help="launch the Streamlit app")

    p_init = sub.add_parser("init", help="start a project folder: write a bex.toml there")
    p_init.add_argument("dir", nargs="?", default=".", help="folder (default: here)")

    args = parser.parse_args(argv)
    if args.command == "init":
        return _init(Path(args.dir))
    cfg = load_config(args.config)

    if args.command == "ingest-sne":
        recordings, ann = ingest.ingest_sne(cfg.labels_dir, cfg.audio_dir)
        d = ingest.write_dataset(cfg.store_dir, "sne", recordings, ann, audio_root=cfg.audio_dir)
        print(f"sne: {len(recordings)} recordings, {len(ann)} annotations -> {d}")

    elif args.command == "scan":
        recordings = ingest.scan_folder(args.dir, site=args.site, lat=args.lat, lon=args.lon)
        d = ingest.write_dataset(cfg.store_dir, args.name, recordings, audio_root=args.dir)
        hours = recordings["duration_s"].sum() / 3600
        print(f"{args.name}: {len(recordings)} recordings, {hours:.1f} h -> {d}")

    elif args.command == "melcache":
        recordings, _ = ingest.read_dataset(cfg.store_dir, args.dataset)
        audio_root = ingest.dataset_audio_root(cfg.store_dir, args.dataset)
        from . import repair
        n = 0
        for rid, rel in recordings.select("recording_id", "path").iter_rows():
            if args.limit is not None and n >= args.limit:
                break
            # Repairs a file libsndfile cannot decode, then builds from the copy.
            what = repair.build_mel(cfg.cache_dir, cfg.store_dir, args.dataset, rid, rel,
                                    audio_root)
            if what != "cached":
                n += 1
                print(f"{rid}: {what}")
        print(f"{args.dataset}: built {n} new mel cache entr{'y' if n == 1 else 'ies'}")

    elif args.command == "repair":
        from . import repair

        recordings, _ = ingest.read_dataset(cfg.store_dir, args.dataset)
        audio_root = ingest.dataset_audio_root(cfg.store_dir, args.dataset)
        fixed = skipped = 0
        for rid, rel in recordings.select("recording_id", "path").iter_rows():
            dst = ingest.repairs_dir(cfg.store_dir, args.dataset) / f"{rid}.flac"
            if dst.exists():
                skipped += 1
                continue
            src = Path(audio_root) / rel
            if repair.needs_repair(src, full=args.full):
                dur = repair.repair_recording(src, dst)
                fixed += 1
                print(f"  repaired {rid} ({dur:.0f}s) -> {dst}")
        print(f"{args.dataset}: {fixed} repaired, {skipped} already repaired, "
              f"{len(recordings) - fixed - skipped} healthy")

    elif args.command == "app":
        import os
        import subprocess

        # Installed: the app ships inside the package (bex/webapp). In a clone
        # it is app.py at the repo root.
        here = Path(__file__).resolve().parent
        app_path = here / "webapp" / "app.py"
        if not app_path.exists():
            app_path = here.parent / "app.py"
        # The app finds its data through the same bex.toml this command used.
        env = {**os.environ, "BEX_CONFIG": str(cfg.source)}
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app_path), *THEME], env=env)

    elif args.command == "survey":
        return _survey(cfg, args)

    elif args.command == "labels":
        from . import labels
        recordings, _ = ingest.read_dataset(cfg.store_dir, args.dataset)
        if labels.has_imported(cfg.store_dir, args.dataset):
            print(f"{labels.IMPORTED}: the dataset's own annotations (read-only)")
        for name in labels.list_sets(cfg.store_dir, args.dataset):
            ls = labels.load_set(cfg.store_dir, args.dataset, name)
            p = labels.progress(ls, recordings)
            vs = labels.versions(cfg.store_dir, args.dataset, name)
            print(f"{name}: {p['minutes']:.0f} min closed in {p['recordings']} recordings, "
                  f"{p['boxes']} boxes of {p['n_species']} species; sample "
                  f"{p['sample_closed']}/{p['sample_size']}; versions "
                  f"{', '.join(f'v{v}' for v in vs) or 'none'}")

    elif args.command == "datasets":
        for name in ingest.list_datasets(cfg.store_dir):
            recordings, ann = ingest.read_dataset(cfg.store_dir, name)
            tag = f", {len(ann)} annotations" if ann is not None else ""
            print(f"{name}: {len(recordings)} recordings{tag}")

    return 0


def _survey(cfg, args) -> int:
    """V1.2 F4 headlessly: the CSV most users want is the product."""
    from . import labels, setups, stats, store, survey
    from . import thresholds as th
    from .profiles import load_profile
    from .views import list_runs
    names = setups.list_setups(cfg.store_dir)
    if not args.setup:
        for n in names:
            print(f"{n}: {setups.load(cfg.store_dir, n).claim()}")
        return 0
    su = setups.load(cfg.store_dir, args.setup)
    recordings, _ = ingest.read_dataset(cfg.store_dir, args.dataset)
    ids = recordings["recording_id"].to_list()
    runs = [m for m in list_runs(cfg.store_dir) if m.dataset == args.dataset
            and th.identity(m) == su.model
            and len(store.list_scores(cfg.store_dir, m.run_id)) >= len(ids)]
    if not runs:
        print(f"no complete run of {su.model_label} on {args.dataset}", file=sys.stderr)
        return 1
    ref = args.truth
    if ref and "@" not in ref and ref != labels.IMPORTED:
        ref = labels.working_ref(cfg.store_dir, args.dataset, ref)
    truth = labels.load_truth(cfg.store_dir, args.dataset, ref) if ref else None
    det = stats.mark_implausible(store.read_detections(cfg.store_dir, runs[0].run_id),
                                 load_profile(cfg.profiles_dir / su.profile), mode=su.judge)
    out = setups.apply(su, survey.species_only(det), ids, truth)
    print(f"# {su.claim()}", file=sys.stderr)
    if args.out:
        out.write_csv(args.out)
        print(f"{len(out)} rows over {out['recording_id'].n_unique()} recordings -> "
              f"{args.out}", file=sys.stderr)
    else:
        print(out.write_csv(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
