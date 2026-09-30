# Perch runner

Perch v2 as a bex adapter — the second arm, and the proof that the adapter
contract survives a differently-shaped model: **14,795 classes** (birds, plus
frogs, mammals and ~80 AudioSet event classes), **5 s windows at 32 kHz**,
against BirdNET's 6,522 classes and 3 s at 48 kHz. Windows stay native
(PLAN.md D5); cross-model metrics use the shared grid.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ../.. --no-deps    # the bex library, sans app deps
```

TensorFlow is required (Perch is TF-only, unlike BirdNET which runs on litert),
which is exactly why this is a separate environment. Model weights (~379 MB)
download on first run.

## Run

```bash
.venv/bin/python run.py --dataset sne --profile us-ca-sierra
.venv/bin/python run.py --dataset sne --limit 2                    # smoke test
.venv/bin/python run.py --dataset sne --run-id perch-XXXXXXXX      # resume
.venv/bin/python run.py --dataset sne --embeddings                 # + 1536-d vectors
```

~80 s per hour of audio on an M-series CPU (BirdNET is ~5 s — Perch is a much
larger model). Resume by `--run-id`; the detections index rebuilds whole-run.

## Embeddings for an existing run

`run.py --embeddings` only caches vectors for recordings it is *about* to score, and
its resume path skips anything already scored — so a run that was scored without the
flag cannot be back-filled by re-running it. `embed.py` does that job: same model,
same audio, `encode` instead of `predict`, vectors written under the existing run_id.

```bash
.venv/bin/python embed.py --dataset sne --run-id perch-495c1c32
```

Idempotent and resumable. Each recording's window count is **asserted against the
stored score matrix** before the file is written: `embeddings[i]` has to describe the
same audio as `scores[i]`, and a mismatch would train every downstream probe on
quietly shifted labels — a failure that looks like nothing at all. 1,536-d float16,
~2 MB per hour of audio.

## Two things that differ from BirdNET, and why

**Softmax, not sigmoid.** Perch's head emits unnormalised logits. A per-class
sigmoid over them saturates — median 0.27, top classes above 0.999, and 13,317
of 14,795 classes clearing 0.01 in a typical window, which produced 19.2 million
index rows for two recordings. Softmax gives ~6.5 classes above 0.01 per window
(45.8k rows for the same audio) and is **monotone in the logits**, so every
within-window quantity this project measures keeps its ordering. It also gives
impostor mass a natural reading: the share of the model's probability landing on
impossible species.

⚠️ Softmax normalises across classes, which assumes one dominant sound per
window — imperfect in a dawn chorus. And **θ does not mean the same thing here as
for BirdNET**, so arms must be compared at matched detection counts, never at
matched θ. `scripts/compare.py` does that.

**No geofilter.** Perch ships no occurrence model, so `occ_score` is NaN and
`suppressed` is False on every row — the model hides nothing of its own accord.
Its muddiness is judged by the plausibility profile, which is applied identically
to every arm.

## Label space

Perch's labels are bare scientific names, so they join directly to BirdNET's
keys with no taxonomy work. The ~80 AudioSet classes (`Bass_drum`,
`Acoustic_guitar`, `Chicken_and_rooster`) are kept as keys of their own and land
in tier 3 under any profile. Note the underscores there are **not** a common-name
separator as they are in BirdNET's labels — hence `split_common=False`; getting
this wrong collapses `Bass_drum` and `Bass_guitar` onto one key and misaligns
every score column after it.
