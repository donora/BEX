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

**[Take the tour →](TOUR.md)** A page-by-page walk through the app, with
screenshots.

## What's in the app

Every dataset follows one flow from the **Home** page: add it, run the models,
**1 · Label** a sample, **2 · Explore**, **3 · Analyse**, choose a **4 · Survey
protocol**, and apply it to the rest. Or skip labelling and just explore what
the models report.

- **Home** — every dataset with a progress strip through the flow; add a folder
  of recordings, build its spectrograms and start the model runs from here.
- **Label** — label your own recordings: BEX draws a random, stratified sample
  of 60 s chunks and tells you roughly how much labelling your question needs;
  you draw boxes on the spectrogram and close each chunk when every bird in it
  is labelled. The models only show a coarse, species-free activity guide, so
  the labels do not flatter them. Label sets are versioned, exported in SNE's
  CSV columns, and can be imported from CSV, Raven or Audacity.
- **Explorer** — one recording at a time: a navigator over the whole recording
  (where each model was right, wrong, or silent), the spectrogram with the
  annotators' boxes, a score lane per model, an audio player on the same time axis,
  and a window inspector.
- **Compare models** — what each model would give you at your threshold rule: how
  often it is right when it reports a bird, how many songs it finds, false
  detections per hour, precision–recall, and species by species — each with a
  95% interval from resampling recordings, and a paired test of whether one
  model is really better than another on your recordings.
- **Species scorecard** — one bird at a time: what each model found and missed,
  on which recordings, and what it confuses the bird with.
- **Geofilter forensics** — what a location filter hides, what that costs each
  model in precision and recall, and which birds it suppresses most.
- **Thresholds** — every model's threshold for every species, fitted by a rule
  (e.g. "95% precision"), overridable by hand, and saved as named sets you can
  apply to recordings that have no annotations.

- **Survey protocol** — the rule that turns detections into species lists,
  tuned and tested on held-out recordings, saved as a **setup** and applied to
  every recording (`bex survey` does the same from the command line).

The sidebar holds the experiment's settings (dataset, ground truth, models,
plausibility profile, threshold rule) and keeps them as you move between pages.
Its status panel says what every number rests on: how much is labelled, in how
many recordings and sites.

- **NB** - This app can ingest the Sierra Nevada soundscape (open-source, labelled, 33x1hr recordings with expert birdsong labels) as a demonstration set. See below ('Scoring needs annotations') for instructions.

## Simple mode

BEX's **Simple mode** is one guided page, from a folder of recordings to
checked detections: `bex app` opens on it for a new install (a store that
already has datasets opens in Research mode, the full BEX; switch any time
from the top bar). Six steps, each a button in the bar across the top:

1. **Recordings**: a folder of audio (files stay where they are).
2. **Model**: use an existing run, or run a new one. Perch by default
   (Apache-2.0; run with its softmax readout, whose scores spread out).
   BirdNET works too, with a licence note: its weights are non-commercial.
3. **Species**: each window's top three species. Those on the region's
   species list are ticked; give each a common name. Species the model was
   confident about but that aren't on the list are shown apart, as
   **unexpected**: tick one to review it yourself. Low-scoring off-list
   species are folded away as probably noise.
4. **Check**: an overview of every species (dots for the Yes and No still
   needed), then check detections one at a time (Yes, No, Not sure, with
   undo) until each species has 5 Yes and 3 No. Beside each clip you can
   play your own earlier Yeses and follow links to Xeno-canto and iNaturalist
   sounds. A label set made in Research mode can count too. (With EcoLabel,
   below, volunteers each hear 3 of the Yeses as examples and are tested on
   the rest, and species its shared library already covers need fewer.)
5. **Recall**: your answers calibrate the model's scores, so "aim for 80%
   recall" becomes a score floor, with an estimate of the false detections per
   100 that costs. (Tested against SNE's expert labels: aiming for 80% gave
   79%, with 18 false per 100 against an estimate of 16.)
6. **Export**: download the kept detections as a CSV, for a spreadsheet, R
   or Python. Sending them to EcoLabel volunteers is shown but greyed out
   until EcoLabel opens (below).

**EcoLabel: coming soon.** EcoLabel is a companion app, not yet public, in
which volunteers check detections by ear on their phones, with their answers
weighted by how reliable each listener has proved. From Simple mode's Export
step you will choose a survey protocol (with its cost in volunteer answers),
describe the project, and send a bundle of short clips around each detection
(never whole recordings or coordinates). For questions or early access, email
[mara@melusine.uk](mailto:mara@melusine.uk).

**Your own model.** In the Model step, choose *Your own model* for a how-to
and an example file: run your model however you like, save one table (CSV or
Parquet) with `recording`, `start_s`, `end_s`, `species` (scientific name) and
`score` (0 to 1) per window per species, and import it. BEX checks every row
and says what to fix. From the command line:
`bex import-detections my-site detections.csv --model mynet`. (For a model BEX
should run itself each time, write a runner: `envs/README.md`.)

For EcoLabel early-access users, sending from the command line, once steps 1 to 5 and the project description are done in the app (and `BEX_ECOLABEL=1` turns the Export tab on):

```bash
bex export-ecolabel my-site                 # -> my-site.ecolabel.zip
bex upload my-site.ecolabel.zip --key …     # resumable; or set ECOLABEL_KEY
```

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

Everything works on unlabelled audio: you can explore every model's detections
and apply a saved threshold set or survey setup. **Comparing models against
ground truth needs labels**: make them on the **1 · Label** page, import them
there (SNE-style CSV, Raven selection tables, Audacity label tracks), or use the
annotated Sierra Nevada soundscape set BEX was built on (`bex ingest-sne`; point
`audio_dir` and `labels_dir` in `bex.toml` at your copy).

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

- Bouts and per-species rules on the Survey protocol page (V1.1.md, "Later").
- More models: region-tuned classifiers, BirdNET custom classifiers, new Perch
  releases.

[PLAN.md](PLAN.md) is the design document (architecture, metric definitions,
staged roadmap); [V1.md](V1.md), [V1.1.md](V1.1.md) and [V1.2.md](V1.2.md) record how each version of the app was designed.

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
