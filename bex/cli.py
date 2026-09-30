"""The `bex` command — the pipeline without reading the source (PLAN.md §8 rule 3).

    bex ingest-sne                     # labelled test rig -> dataset 'sne'
    bex scan DIR --name my-site ...    # unlabelled folder -> a dataset
    bex melcache DATASET [--limit N]   # build display spectrograms
    bex datasets                       # what the store holds
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
        n = melcache.build_cache(
            cfg.cache_dir, args.dataset, recordings, audio_root,
            limit=args.limit, progress=print,
            resolve=lambda rid, rel: ingest.resolve_audio(
                cfg.store_dir, args.dataset, audio_root, rid, rel
            ),
        )
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

    elif args.command == "datasets":
        for name in ingest.list_datasets(cfg.store_dir):
            recordings, ann = ingest.read_dataset(cfg.store_dir, name)
            tag = f", {len(ann)} annotations" if ann is not None else ""
            print(f"{name}: {len(recordings)} recordings{tag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
