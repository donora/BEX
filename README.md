# BEX — Bioacoustics Explorer

**Benchmark and compare bioacoustic identification models on your own recordings.**

Every bioacoustic project makes processing decisions: which model to process the data, which confidence
threshold, whether to trust the model's location filter, etc. BEX runs your recordings
through several models (currently BirdNET and Perch - can also add your own models), scores each against the same
ground truth, and helps you design and quantify a pipeline with a known precision and recall
for every species. It also makes **the geographic filter visible**: what each model
predicted before its location prior silenced it, and how often that filter hides a
bird that was really there.

BEX is a local app (Streamlit) plus a command line. It processes local audio.

## What's in the app

- **Explorer** — one recording at a time: a navigator over the whole recording
  (where each model was right, wrong, or silent), the spectrogram with the
  annotators' boxes, a score lane per model, an audio player on the same time axis,
  and a window inspector.
- **Compare models** — what each model would give you at your threshold rule: how
  often it is right when it reports a bird, how many songs it finds, false
  detections per hour, precision–recall, and species by species.
- **Species scorecard** — one bird at a time: what each model found and missed,
  on which recordings, and what it confuses the bird with.
- **Geofilter forensics** — what a location filter hides, what that costs each
  model in precision and recall, and which birds it suppresses most.
- **Thresholds** — every model's threshold for every species, fitted by a rule
  (e.g. "95% precision"), overridable by hand, and saved as named sets you can
  apply to recordings that have no annotations.

The sidebar holds the experiment's settings (dataset, models, plausibility
profile, threshold rule) and keeps them as you move between pages.

- **NB** - This app can ingest the Sierra Nevada soundscape (open-source, labelled, 33x1hr recordings with expert birdsong labels) as a demonstration set. See below ('Scoring needs annotations') for instructions.

## Install

Python 3.12. Developed and tested on macOS (Apple silicon).

```bash
python3.12 -m venv .venv
.venv/bin/pip install bex-bioacoustics
```

The package is `bex-bioacoustics`; the command and the Python import are `bex`.

## Quick start with your own recordings

**1. Make a project folder.** BEX keeps its working data (scores, spectrograms,
threshold sets) here, and finds it again through the `bex.toml` it writes.

```bash
mkdir my-project && cd my-project
bex init
```

**2. Scan a folder of recordings** (WAV, FLAC, MP3 or OGG; subfolders too). Files
stay where they are. The location is what BirdNET's filter needs; recording times
are read from AudioMoth-style filenames.

```bash
bex scan /path/to/recordings --name my-site --site "Glen Affric" --lat 57.25 --lon -4.92
bex melcache my-site        # the spectrograms the Explorer draws (resumable)
bex repair my-site          # only if some files will not decode
```

**3. Run the models.** Each model runs in its own environment, because BirdNET and
Perch cannot share dependencies with each other or with the app. The runners live
in this repository, so clone it once:

```bash
git clone https://github.com/donora/bex bex-src
cd bex-src/envs/birdnet
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install bex-bioacoustics --no-deps
BEX_CONFIG=/path/to/my-project/bex.toml .venv/bin/python run.py --dataset my-site
```

Perch is the same in `envs/perch` (it needs TensorFlow, so expect a larger install
and ~80 s per hour of audio against BirdNET's ~5 s). Model weights download on
first run. See [envs/birdnet](envs/birdnet/) and [envs/perch](envs/perch/).

**4. Open the app.**

```bash
bex app
```

### Scoring needs annotations

Everything above works on unlabelled audio: you can explore every model's
detections, and apply a saved threshold set. **However, comparing models against ground
truth needs annotations**, and in this version the only importer is for the
annotated Sierra Nevada soundscape set that BEX was built on (`bex ingest-sne`;
point `audio_dir` and `labels_dir` in `bex.toml` at your copy). A general importer
for your own annotations, and a labelling mode in the Explorer, are next on the
roadmap.

## Results so far

On 33 h of strongly annotated Sierra Nevada soundscapes (56 species):

| model | cmAP | micro AP |
|---|---|---|
| BirdNET 2.4 | **0.485** | **0.817** |
| Perch v2 (sigmoid, float32) | 0.375 | 0.646 |
| Perch v2 (softmax) | 0.363 | 0.646 |

- **BirdNET's location filter mostly hides real birds here.** Of the detections
  it suppresses, most were annotated in that very window; one abundant resident
  (Fox Sparrow, just under the filter's occurrence cutoff) accounts for most of it.
- **There is no single good threshold.** Reaching 95% precision takes a threshold
  anywhere from 0.18 to 0.98 for BirdNET depending on the species, which is why
  BEX thresholds species by species.
- **Perch is far "muddier"** — many more species scoring confidently in the same
  window, much of it between close relatives — while finding more of the
  annotated species at a matched number of detections.

Every number is reproducible headlessly (`scripts/evaluate.py`,
`scripts/compare.py`, `scripts/forensics.py`) and re-derived from the raw score
matrices by `scripts/crosscheck.py`. [PLAN.md](PLAN.md) has the full account,
including two representation traps (softmax readout, float16 storage) that were
distorting the comparison and how they were resolved.

## Roadmap

- **Label your own data in the Explorer** — minute-by-minute labelling with the
  models' suggestions as a guide, signed-off chunks, and versioned label sets
  (PLAN.md, Stage 6.5).
- An importer for your own annotations.
- More models: region-tuned classifiers, BirdNET custom classifiers, new Perch
  releases.

[PLAN.md](PLAN.md) is the design document (architecture, metric definitions,
staged roadmap); [V1.md](V1.md) records how version 1 of the app was designed.

## Licences

BEX is Apache-2.0 (see [LICENSE](LICENSE)). It ships **no model weights and no
audio**: you download the models yourself when you set up the runners, and their
own licences apply to your use of them.

- **BirdNET**'s model weights (including its location model) are
  **CC BY-NC-SA 4.0: non-commercial use only.** Research, conservation and
  personal use are fine; using BirdNET for paid work needs a commercial licence
  from the BirdNET team.
- **Perch** is Apache-2.0.
- Xeno-Canto recordings BEX links to are licensed per recording by their recordists.

## Development

```bash
git clone https://github.com/donora/bex && cd bex
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest            # runs without audio or models; app tests skip without a store
bex app                     # or: .venv/bin/streamlit run app.py
```

```
app.py        the app's entry point: page setup and navigation
ui/           the app — sidebar.py (settings), common.py (shared), pages/ (one per page)
bex/          the library — every number the app shows is computed here
profiles/     plausibility profiles (bring your own — PLAN.md §3c)
envs/         one isolated runner environment per model (PLAN.md §3b)
scripts/      the headless versions of the app's analyses
tests/        tests, runnable without audio, models, or the store
```
