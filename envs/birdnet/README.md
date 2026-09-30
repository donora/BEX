# BirdNET runner

BirdNET v2.4 (litert backend — no TensorFlow) as a bex adapter: full 6,522-class
scores per 3 s window with **no confidence threshold and no species filter**, plus
the geo meta-model's complete occurrence vector per (lat, lon, week). Occurrence and
BirdNET's would-have-suppressed flag are recorded per detection row, never applied
destructively — the silenced layer is the thing this project studies.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ../.. --no-deps    # the bex library, sans app deps
```

Model weights (~90 MB) download to the user cache on first run.

## Run

```bash
.venv/bin/python run.py --dataset sne --profile us-ca-sierra              # everything
.venv/bin/python run.py --dataset sne --limit 2                           # smoke test
.venv/bin/python run.py --dataset sne --run-id birdnet-XXXXXXXX           # resume
```

Reads the dataset + audio root from the store, writes score matrices
(`store/scores/<run_id>/`), the derived detections index
(`store/detections/<run_id>.parquet`) and the executed-config manifest
(`store/runs/<run_id>/manifest.json`). Resume skips scored recordings and rebuilds
the detections index for the whole run, so interrupted and clean runs converge.

~5 s per hour of audio on an M-series CPU. The `suppressed` flag reproduces
BirdNET-Analyzer's default species-list behaviour (occurrence < 0.03), recorded in
the manifest as `geofilter.sf_thresh`.
