# Birdsong runner

The from-scratch SNE CNN (`BirdSets/birdsong`, trained with `python -m birdsong.train`)
as a bex adapter: all 56 class scores per 5 s window (5 s hop, 32 kHz), per-class
sigmoid, no threshold. It rebuilds the training input exactly — log-mel at ref=1.0 dB,
per-window `peak` normalisation — so its scores reproduce the training run's test
metrics.

**Leakage guard:** the runner rebuilds the checkpoint's train/val/test split and
refuses any recording from train or val (pass `--allow-seen` to override). Run it on
the test split — `SNE_004, 012, 016, 021, 030, 032` for seed 0 — or on audio the model
never heard.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ../.. --no-deps    # the bex library, sans app deps
echo "../../../runs/sed-fine/best.pt" > checkpoint.txt   # what the app's Run button uses
```

`checkpoint.txt` (gitignored) holds one path, relative to this folder or absolute;
`$BEX_BIRDSONG_CHECKPOINT` overrides it.

## Run

```bash
.venv/bin/python run.py --dataset sne-test-6                                  # checkpoint.txt
.venv/bin/python run.py --dataset sne-test-6 --checkpoint ../../../runs/aug-both/best.pt
```

Writes `store/runs/<run_id>/checkpoint.json` too: the checkpoint's path, sha256, epoch
and training arguments, so a run can be traced back to the exact weights.
