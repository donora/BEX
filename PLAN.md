# BEX — comparing bioacoustic ID methods, with the geofilter made visible

An app for UK researchers (starting with us) to **compare and contrast methods of
analysing bioacoustic data for bird call and species identification** — BirdNET, Perch,
models we train ourselves — on the same audio, side by side, window by window.

The distinctive contribution, and the reason this isn't just another BirdNET wrapper:
**we open up the stage that every off-the-shelf tool hides — what the classifier
believed *before* the geographic filter silenced it.** When BirdNET scores an
Australasian species highly on a Highland recording and the location filter quietly
suppresses it in favour of the European congener, we want to see that happen, count how
often it happens, measure how confident the model was in the impossible, and compare
that "geographic muddiness" across models — including models tuned to a region, which
should exhibit much less of it.

Birds first. The architecture below deliberately avoids anything bird-specific in the
data contracts, so other taxa (bats, small mammals, orthoptera) are a later extension,
not a rewrite.

---

## 1. Relationship to the sibling projects — what we take, what we don't

```
BirdSets/
├── data/            <- shared: annotations.csv, species.csv, audio symlink to USB
├── birdsong/        <- sibling: from-scratch supervised/SSL pipeline on SNE
├── birdgate/        <- sibling: AudioMoth on-device gate (4-arm comparison)
└── bex/          <- this project (its own git repo)
```

**Taken from `birdsong/` (copied, not imported — see §10 D1):**
- `audio.py` — seek-and-read clip loading off the USB, log-mel with sane 32 kHz params.
- `features.py` — the cache discipline: one float16 array per recording on the SSD,
  raw dB at fixed reference, normalisation deferred to load time.
- `windows.py` — box→window overlap logic (the `min(0.5 s, box duration)` label rule),
  adapted here to map ground-truth boxes onto *each model's* window grid.
- `splits.py` — recording-level splits, if/when we train anything.
- The **explorer-app pattern** from `app.py`: tabs that walk from raw data to result,
  a recording/clip navigator with audio + spectrogram + overlays, every number in the
  app recomputed from the same code path the pipeline uses.

**Taken from `birdgate/` (ideas, not code):**
- The **arm discipline**: every model runs on the same audio, is scored by the same
  metric code, on the same eval recordings — so a difference is attributable.
- The **provenance lesson** (BirdGate SUMMARY §12): manifests must record the config
  that *actually ran*, not a fresh default. Here that means: every inference run writes
  model name + exact version/checkpoint hash + window/hop + geofilter parameters
  (lat/lon/week/threshold) into the run manifest, and every stats table in the app
  displays which manifest it came from.
- The storage tiering + `data/audio` symlink + fail-loudly-if-unmounted assert.

**Explicitly *not* taken:** everything AudioMoth-specific in `birdgate/` — the gate
state machine, adaptive thresholding, energy model, firmware. Out of scope here.

**One ground rule inverted, on purpose.** The sibling `PLAN.md` forbids pretrained
weights because its goal is understanding the machinery. This project's goal is the
opposite: **evaluating the pretrained tools researchers actually use.** BirdNET and
Perch are the subjects of study, not cheating. (Our own from-scratch models can join
the comparison later as additional arms — see Stage 7.)

---

## 2. The data — two pipelines, one contract

### 2a. Labelled pipeline (exists today): SNE as the test rig

The 33 h of Sierra Nevada soundscapes on the USB drive, with 20,147 strong
time × frequency boxes. Yes, it's California, not the UK — and that is a *feature* for
one specific job: **it is the only data we have where we can score the geofilter
itself.** Run BirdNET with the Sierra Nevada lat/lon and ask: did the filter ever
suppress a species that is *actually annotated in the audio*? Did it let through
species that never appear? Ground truth turns "the filter changed the answer" into
"the filter changed the answer *correctly or incorrectly*". No unlabelled UK recording
can give us that.

So SNE plays the same role it played for BirdGate: the rig on which the machinery is
built and validated, before the target data exists in labelled form.

### 2b. Unlabelled pipeline (to build): the actual UK use case

A researcher points the app at a folder (or drive) of field recordings — arbitrary
format (WAV/FLAC/MP3), arbitrary sample rate, arbitrary length, no labels. The
pipeline:

1. **Scan → manifest.** Walk the folder, extract per-file metadata (duration, sample
   rate, channels, timestamp parsed from filename patterns where possible), write
   `recordings.parquet`. Site metadata (lat/lon, name) supplied once per import, since
   the geofilter needs it and filenames rarely carry it.
2. **Normalise on the fly, cache what's hot.** Models want different input rates
   (BirdNET 48 kHz, Perch 32 kHz); resampling happens inside each model adapter.
   What we cache on the SSD is the *display* log-mel (birdsong-style, one array per
   recording) and the *inference outputs* — never resampled copies of the raw audio.
3. **No labels → no label columns**, but the detection schema (§3) is identical, so
   every view in the app works on both; label-dependent panels (scoring vs truth)
   simply don't render.

Labelled data is then just unlabelled data + an annotations table that joins on
recording + time, which is exactly how SNE already ships.

---

## 3. Core architecture

### 3a. The model adapter contract

Every method — BirdNET, Perch, a future fine-tune, a from-scratch sibling model — is
wrapped as an **adapter** with one job: audio in, a standard detections table out.

```
class ModelAdapter:
    name: str                 # "birdnet-2.4", "perch-2.0", ...
    window_s: float           # its native window (3.0, 5.0, ...)
    hop_s: float
    input_sr: int
    def run(recording) -> pl.DataFrame   # the detection schema below
    def taxonomy() -> pl.DataFrame       # its label set -> canonical species keys
```

**Detection schema** (long format, parquet, one file per recording × model run):

| column | meaning |
|---|---|
| `recording_id` | joins to `recordings.parquet` |
| `start_s`, `end_s` | the model's native window |
| `species_key` | canonical key (scientific name; see 3c) |
| `score_raw` | the model's confidence **before any geographic filtering** |
| `occ_score` | the geofilter's occurrence score for this species/site/week (NaN if the model has no geofilter) |
| `suppressed` | bool: would the model's own default pipeline have hidden this row? |
| `rank_in_window` | 1 = top species this window (by `score_raw`) |
| `run_id` | joins to the run manifest (model version, geofilter params, thresholds) |

Two consequences of this schema, both load-bearing:

- **The geofilter becomes data, not behaviour.** We never let a model apply its own
  filter destructively. We extract raw scores *and* the occurrence scores separately,
  and apply the mask ourselves in analysis code — so every suppression is a row we can
  count, not an absence we can't.
- **Every model gets the same treatment.** Perch has no built-in geofilter; we can
  still apply the *same* UK plausibility mask to its raw scores, which makes
  "how much does each model believe in impossible birds" a symmetric question.

**Storage policy for scores — resolved (D3): keep everything, in two layers.**

- **Full score store** (the ground truth of a run): per recording × model, the
  complete float16 score matrix, shape (windows × classes), compressed on disk
  (`.npz`, or zarr if we want chunked reads), plus the class-index → `species_key`
  mapping in the run manifest. ~40–60 MB per hour per model; all 33 SNE hours ×
  3 models ≈ 5–6 GB on the SSD. Every metric we invent later is computable
  retroactively — nothing is ever thrown away.
- **Detections index** (what the app queries): the long-form parquet table above,
  *derived* from the full store (top-20 ∪ `score_raw ≥ 0.01` ∪ the plausibility
  profile ∪ ground-truth species). Because it's derived, it can change shape freely —
  regenerating it is minutes, not a re-run of inference.

Perch embeddings (~1,536-d float16 per window) are cached separately (~4 MB/h)
because Stage 7's region-tuned probe trains on them.

### 3b. Separate runner environments (this one bites if ignored)

BirdNET wants TensorFlow; Perch 2.0 wants its own TF/JAX stack; our siblings use
PyTorch; the app wants none of the above. One venv holding all of them is a known
dependency-hell trap, and version pins that satisfy TF often break something else.

Design: **each adapter is a small CLI run in its own venv** (`envs/birdnet/`,
`envs/perch/`), invoked as a subprocess by the orchestrator, writing parquet to the
store. The app venv (`envs/app/` or `bex/.venv`) needs only
streamlit/polars-or-pandas/librosa/soundfile/altair — it never imports a model. This
also means inference can run on a big machine later while the app runs anywhere.

### 3c. Taxonomy: the unglamorous module that makes everything joinable

Three label vocabularies collide immediately: SNE annotations use **eBird codes**
(2018-era), BirdNET 2.4 uses `Scientific_name_Common name` strings from its own label
file, Perch uses **eBird codes** from a different taxonomy year. eBird taxonomy
revisions rename, split, and lump species annually — a naive string join will silently
drop species and fabricate disagreements between models.

`taxonomy.py`: one canonical table keyed on scientific name, built from the eBird/
Clements taxonomy checklist, with per-model mapping tables and an explicit
**unmapped-labels report** (fail loudly, BirdGate-style). It also carries genus/family,
which the "shadowed congener" metric (§4) needs, and the **UK plausibility tiers**:

- **Tier 0 — British List** (BOU, categories A–C): recorded wild in Britain. ~630 spp.
- **Tier 1 — regular**: breeding/wintering/passage species (a much shorter list);
  optionally seasonal via eBird occurrence.
- **Tier 2 — plausible vagrant**: on the British List but rare (BBRC-class).
- **Tier 3 — implausible**: everything else. The Australasian endemic scoring 0.6 on a
  Highland dawn chorus lives here.

**Plausibility ships as swappable profiles — resolved (D4), and a first-class feature
because other people will use this app.** A profile is a directory:

```
profiles/<name>/
├── profile.toml    # display name, region, source, citation, date
└── tiers.csv       # species_key, tier   (scientific name keyed — nothing else)
```

The app gets a profile picker plus an upload/validate flow (unknown species names
reported, not silently dropped); every statistic is reported *per profile*, and the
run manifest records which profile produced which numbers — swapping a Shetland list
for a Kent list is a dropdown, not a code change. We start with the BOU British List
as Tier 0 plus a hand-curated Tier-1 regular list (yours, when ready, lands as
another profile), and a California profile for SNE validation.

### 3d. Time-grid harmonisation

Models disagree about windows (3 s vs 5 s, different hops), and ground truth is
continuous boxes. Two representations, used for different jobs:

- **Display lanes — native grids, made visible (resolved, D5)**: each model's lane in
  the explorer draws detections on that model's *own* window grid, with the window
  boundaries rendered — seeing that BirdNET chops the same call at 3 s intervals while
  Perch sees it whole in a 5 s window is part of what the app teaches, not a nuisance
  to smooth over. A common 1 s frame grid exists *only* as the substrate for
  cross-model metrics (a window's score spans its frames; overlaps take the max), and
  the app labels any number computed on it as such.
- **Event matching**: for metrics against truth, a detection window matches a
  ground-truth box iff temporal IoU ≥ threshold *or* overlap ≥ `min(0.5 s, box
  duration)` (the birdsong rule, reused so numbers are comparable across siblings).
  Species-level scoring follows the BirdCLEF/soundscape convention: per-window
  multi-label, cmAP + per-species AP, computed per model on that model's own grid, and
  on the shared 1 s grid for cross-model comparison.

---

## 4. The geofilter forensics — definitions, so the stats mean something

All computed from the detection schema; θ_det is the operating detection threshold
(default 0.10, always shown and always sweepable in the app).

- **Suppression event**: window *w*, species *s*, model *m* with
  `score_raw ≥ θ_det` and *s* excluded by the geofilter (or tier ≥ 2 for models
  without one). The atomic unit — each is a browsable row linked to its audio.
- **Confidence-in-the-implausible**: the distribution of `score_raw` over Tier-3
  species, per model. A well-calibrated, well-regionalised model should have almost no
  mass above θ_det; the shape of this curve *is* the model's geographic muddiness.
- **Muddy-window rate**: fraction of detection-positive windows containing ≥ 1
  Tier-3 species above θ_det. The headline per-model number.
- **Impostor mass**: per window, Σ `score_raw` over implausible species ÷ Σ over all
  species (within the stored top-k). Continuous version of the above; robust to θ.
- **Shadowed detection** (the blackbird case): a plausible species *s_p* detected at
  `score_raw = p`, with an implausible species *s_i* in the same window at score
  ≥ p − Δ, flagged **congeneric/confamilial** when taxonomy says so. These are the
  cases where the geofilter is doing *identification work the classifier failed to do*
  — the model didn't distinguish the birds; the map did. Report the rate, the mean
  score gap, and the top offending pairs.
- **Per-species muddiness** `M_m(s)`: over the windows where model *m* detects
  species *s* (`score_raw ≥ θ_det`), the mean **companion count** (how many *other*
  species score within Δ of *s*, or above θ_conf) and the mean impostor mass of those
  windows. Computed in two scopes: **all-species** (general confusability — a
  Skylark-distinctive song vs one that always drags a crowd of confident-ish
  alternatives) and **implausible-only** (the geographic subset). This is the
  species-level answer to "which birds does this model actually *know*, and which does
  it merely rank first".
- **Muddiness variability**: the spread of `M_m(s)` across species — median + IQR per
  model. Low spread = uniformly clean (or uniformly muddy); high spread = muddiness
  concentrated in particular confusable groups. The cross-model view separates
  *hard songs* (muddy for every model) from *model defects* (muddy for one). On
  labelled data, plot `M_m(s)` against per-species AP: if muddiness predicts error,
  it's a usable field proxy for reliability where no labels exist — which is exactly
  the situation UK researchers are in.
- **Filter-error rates (labelled data only)**: false suppression (filter removed a
  species that is annotated in that window — ground truth says the "implausible" bird
  was there) and false admission (filter passed a species never present in 33 h).
  This is what SNE + the California filter is for.
- **The league table**: per model × dataset × profile — muddy-window rate, median
  impostor mass, shadowed-detection rate, per-species muddiness spread (median + IQR),
  plus ordinary detection quality (cmAP vs truth where available) so nobody optimises
  muddiness by detecting nothing.

The claim the app should ultimately let a researcher make, with numbers: *"on UK
audio, model X's raw output assigns __% of its confident detections to species that
are almost certainly absent, and its geographic prior silently rescues __% of them —
whereas region-tuned model Y needs rescuing __% of the time."*

---

## 5. The models (initial arms)

| arm | what | window | classes | geofilter |
|---|---|---|---|---|
| `birdnet` | BirdNET-Analyzer, model v2.4 | 3 s @ 48 kHz | ~6,500 | built-in meta-model: (lat, lon, week) → per-species occurrence score; default pipeline drops species below a threshold. **We call the meta-model directly** to get the occurrence *scores*, and run the classifier unfiltered — both halves land in the schema. Python route: the `birdnet` / `birdnet-analyzer` package (exact pin decided in Stage 2). |
| `perch` | Perch 2.0 embeddings + its species logits | 5 s @ 32 kHz | ~10k eBird classes | none built in — our UK mask applies symmetrically. Also the embedding source for the region-tuned arm. |
| `birdnet-uk` | BirdNET restricted to the UK list | 3 s | Tier-0/1 subset | the trivial "region-tuned" baseline: same logits, mask applied *before* ranking. Costs nothing, establishes whether muddiness is mostly a ranking artefact. |
| *(Stage 7)* `probe-uk` | linear probe on Perch embeddings, trained on UK species (Xeno-Canto focal recordings) | 5 s | UK list | a genuinely region-trained arm, not just region-masked. |
| *(Stage 7, optional)* `sne-cnn` | the sibling's `sed-fine` checkpoint | 5 s | 56 SNE spp | joins only for SNE comparisons; a "perfectly region-tuned" extreme. |

---

## 6. The app

Streamlit, in the house style: tabs that walk from data to conclusion, everything
recomputed from pipeline code, nothing hand-pasted.

1. **Mission** — what this measures and why, the definitions of §4 in prose.
2. **Recordings** — both pipelines' manifests: what's imported, processed, by which
   models, cache/run status. The ingest UI for unlabelled folders lives here.
3. **Explorer** (the heart) — pick a recording; spectrogram + audio player; stacked
   **detection lanes**, one per model, *on its native window grid with the boundaries
   drawn* (D5), coloured by species, with ground-truth boxes overlaid when labelled.
   Suppressed detections render as hollow/hatched marks — the silenced layer made
   visible. Click a window → per-model table: top-k species, `score_raw`, occurrence
   score, tier, suppression flag; shadowed pairs highlighted.
   **Choosing a focus species rebuilds the navigator around that bird.**
   Unfiltered, an hour of dawn chorus is a near-solid wall: on SNE_030 every one
   of 180 bins carries a detection and 162 carry annotation, which says the
   recording is busy and nothing about where any one bird is. Focused, the count
   bars are replaced by **one lane per source** — the annotators' boxes, then each
   arm, showing *presence* rather than counts (a 3 s and a 5 s grid give different
   window counts for the same song, so counts would invite a meaningless
   comparison; they are on hover). Each arm's lane distinguishes **detected** from
   **suppressed-only**, because a species the geofilter silenced must not read as
   one the model never heard — that difference is the project.
   The picker offers **every species annotated in the recording ∪ every species
   any arm detects** (`views.focus_options`), not — as it first did — the primary
   model's loudest-per-bin species above the current θ. That was three kinds of
   too narrow: one model, one threshold, and only species that topped some bin, so
   raising θ or selecting the weakest arm silently removed the bird from the menu.
   On SNE_030 at θ=0.9 the old list would have been nearly empty; the new one
   keeps all 8 annotated species, 3 of them detected by no arm at all — and an arm
   that finds nothing keeps its lane and shows it empty.
4. **Geofilter forensics** — §4's statistics: suppression-event browser (each row
   plays its audio), confidence-in-the-implausible distributions, muddy-window and
   shadowed-detection rates, per-species muddiness (sortable: most/least distinctive
   species per model), the offending-pairs table, θ and Δ as live sliders. The
   plausibility-profile picker and upload flow live here.
5. **Model comparison** — the league table; per-model cmAP/per-species AP vs truth
   (labelled data, built in Stage 5.5); cross-model agreement matrices (where do
   BirdNET and Perch disagree, and does truth side with either); muddiness-variability
   comparison; calibration curves.
   The truth half, as built: a **per-species PR curve** with each arm's operating
   points marked under three rules (precision floor, max F1, max F2) — because
   max-F1 is the textbook answer and usually the wrong one for survey work, where a
   missed bird and a false alarm are not equally costly; a **per-species AP scatter**,
   one point per species, whose distance from the diagonal is the capability
   difference a single league number averages away; and the **per-species threshold
   table**, exportable as CSV, carrying the θ each rule asks for *and the precision it
   achieved* — a θ without that is unfalsifiable.

6. **Species scorecard** — the aggregate turned inside out: pick one bird and ask
   what each arm had the chance to report and what it did with it. Built on the
   Stage 5.5 alignment, so every figure traces to rows.
   **Two denominators, always side by side.** A 30 s box is ten windows on
   BirdNET's 3 s grid and six on Perch's 5 s one, so "windows it should have
   reported" is a property of the grid as much as of the bird and is *not*
   comparable across arms. It is still the right unit for choosing θ, because θ
   is applied per window. The comparable unit is the **vocalisation**: of the
   boxes a human marked, how many did the arm detect at all — one window inside
   a box is a detection, and a box is missed only when the arm was silent across
   every window of it. A third category, **not asked**, holds boxes that own no
   window on this grid (shorter than the 0.5 s floor and straddling a boundary);
   counting those as misses would blame the model for the grid.
   It opens on an **overview of all 56 annotated species at once** — a dumbbell
   per species, one dot per arm at the share of that bird's vocalisations the arm
   detected, dot size the number of vocalisations, the joining line's length the
   amount the choice of model matters for that bird. Size is load-bearing rather
   than decorative: eight species sit at 100% on one to three vocalisations, and
   without the size encoding they would read as the models' greatest successes
   from the most prominent rows of the chart. Ordering by widest disagreement
   surfaces the real ones — Pine Siskin 25% on BirdNET against 100% on Perch,
   Northern Flicker 75% against 21% the other way.
   The overview defaults to **max F1** because every species is reachable under
   it; the 95% precision floor is the rule a survey wants but only 27 of 56
   species reach it on BirdNET and 20 on Perch, so it is a deliberate switch
   rather than half an empty chart on arrival, and the species no threshold can
   satisfy are named rather than drawn at zero.
   Each arm is set to *its own* θ for a rule the reader picks, because θ is not
   comparable between arms and matching on the rule is what makes the columns
   mean the same thing. **Every view says which θ produced it**, added after a
   real confusion: Mountain Chickadee in SNE_030 reads 25% window recall on the
   scorecard and looks well detected in the Explorer, because the scorecard was
   at BirdNET's max-F1 θ of 0.4868 and the Explorer at the sidebar's 0.10. Both
   were correct; neither said so. A **recall-against-θ chart** now marks both
   operating points on the same curve, per arm on its own x axis. It also makes
   the incomparability concrete: at a shared θ=0.10, BirdNET sits at 96% recall
   and 36% precision (7,022 false alarms) while Perch sits at 36% recall and 97%
   precision (38) — one number putting two arms at opposite ends of their curves. Then: per-recording recall (where does it fail — and it
   does, sharply), what the arm named *instead* in the windows it missed, and a
   window-by-window strip against the annotators' boxes.

Everywhere a species name appears — explorer tables, the forensics browser, league
tables — it carries a **reference-audio affordance** (§7): at minimum a Xeno-Canto
link, optionally an inline cached example player. When the app claims BirdNET heard a
Pied Currawong in Perthshire, the researcher should be one click from hearing what a
Pied Currawong sounds like.

---

## 7. Species reference audio — the Xeno-Canto strategy

The requirement: hear an example of any species the app mentions. The constraint:
the models' label spaces cover ~6,500–10,000 species, so bundling audio is out —
several GB minimum, and every clip is someone's CC-licensed recording we'd be
redistributing. Two mechanisms, layered:

**Layer 1 — link-out (v1, covers every species, zero storage, zero keys).**
Xeno-Canto species pages have a stable URL derivable from the scientific name we
already carry as `species_key`:

```
https://xeno-canto.org/species/{Genus}-{species}     e.g. .../species/Turdus-merula
```

One function in `taxonomy.py`, a link wherever a species renders. Needs no API, no
account, works for essentially the entire world list. This ships in the first app
version.

**Layer 2 — on-demand cached examples (enhancement, only for species we actually
meet).** For inline playback without leaving the app: query the Xeno-Canto API for
one representative recording per species (quality `A`, a sane length filter),
download to `cache/xc/`, and store recordist, licence, and XC id alongside — shown
next to the player, as XC's terms require. Fetch lazily (first time a species is
displayed) or pre-warm just the profile's Tier 0/1 (~600 UK species ≈ well under
1 GB). Notes that matter for the open-source goal:

- XC's current API requires a **free per-user key** — it goes in the user's config
  (§8), never in the repo.
- The cache is gitignored and rebuildable; we never commit or redistribute clips.
- Most XC recordings are CC **BY-NC-SA** — fine for a non-commercial research tool,
  and a hosted public instance must keep the attribution visible. Both are easy if
  designed in now, painful retrofitted.
- Macaulay Library (eBird) has broader coverage but doesn't permit this kind of
  embedding/hotlinking; Xeno-Canto is the right backbone, with the eBird species page
  as an optional second link.

Verdict: **Layer 1 is trivially feasible and unconditionally worth it; Layer 2 is
feasible and cheap precisely because it's lazy** — we cache the hundreds of species
that occur in results, not the ten thousand that exist.

---

## 8. Built to share — open-source design rules (resolves D7)

The intent is a public repo other researchers can run, and possibly a hosted
instance later. What that implies, adopted as rules from Stage 0:

1. **Thin app, thick library.** Every metric, join, and transform lives in the
   `bex` package as pure functions over the store; Streamlit only calls and
   renders. This is the one choice that keeps a future migration (proper multi-user
   web hosting) cheap: the data contracts and stats layer are the product, the view
   is replaceable. Corollary: anything the app displays is reproducible headlessly.
2. **No absolute paths, one config file.** `bex.toml` (gitignored; a commented
   `bex.example.toml` in-repo) holds data roots, default site/profile, and the XC
   API key. The `data/audio` symlink discipline survives, but under the configured
   root. A stranger's setup is: copy the example, fill in three paths.
3. **Packaging for humans.** `pyproject.toml` + `uv` lockfiles for the app env *and*
   each runner env (reproducible installs are where open-source audio-ML projects
   usually die); CLI entry points — `bex ingest`, `bex run birdnet`,
   `bex derive`, `bex app` — so the pipeline is usable without reading source.
4. **Ship no data we can't license.** SNE audio, model weights, XC clips: fetched by
   the user via scripts, never committed. Instead the repo carries a **tiny demo
   bundle** — a few minutes of permissively-licensed audio plus a prebuilt store —
   so `git clone` → running app with visible detections in five minutes, no USB
   drive, no model downloads. (Check the BOU list's redistribution terms; if
   restrictive, ship the fetch/convert script rather than the CSV.)
5. **Licences, stated plainly.** Our code: Apache-2.0 (permissive, patent grant,
   comfortable alongside the TF ecosystem). A README section on *upstream* licences,
   because they constrain users: BirdNET's model weights are CC BY-NC-SA, Perch is
   Apache-2.0, XC audio is per-recording CC. The app must surface, not launder, these.
6. **Hosting path.** Streamlit is right for local-run and small hosted demos. A
   hosted instance never runs inference — the runner envs are offline batch jobs by
   design (§3b) — it serves a prebuilt store, which is exactly what Streamlit
   handles well. If BEX later needs uploads and multi-user sessions, rule 1 means
   only the view layer is rebuilt.
7. **Tests + CI on the contracts.** Schema validators, taxonomy joins, and every §4
   stat get unit tests that run without audio, models, or the USB drive (fixtures =
   tiny hand-built parquet). GitHub Actions from the first push. The run-manifest
   provenance (§1) doubles as reproducibility for strangers: any published table is
   re-derivable from a manifest.

---

## 9. Stages

### Stage 0 — Scaffold and contracts ✅ (2026-09-07, 37 tests)
`git init`; `pyproject.toml` + uv lockfile + `bex.toml` config layer (§8 rules
2–3 from day one — retrofitting them is what usually never happens); copy + adapt the
birdsong modules (§1); `envs/` layout; `data/` symlink convention (SNE reuses
`../data`; new UK data gets its own tiered layout); `taxonomy.py` with the canonical
table, per-model mappings, and the profile format (§3c) with BOU + California
profiles; the score-store/detections/recordings/run-manifest schemas as code (typed
constructors + validators, so files can't drift), with contract tests runnable
without audio. **Deliverable:** schemas importable and tested; taxonomy join
demonstrably lossless on SNE's 56 species (unmapped report empty); profile
upload/validate round-trips.

### Stage 1 — Ingest pipelines ✅ (2026-09-07, 48 tests)
Labelled: SNE manifest built from the existing `annotations.csv` unchanged.
Unlabelled: folder scan → `recordings.parquet`, site metadata entry, display-mel cache
(reusing the birdsong cache format so the explorer renders either dataset).
**Deliverable:** both pipelines produce identical-shape manifests; explorer-grade
spectrograms render for a folder of arbitrary WAVs.

*As built:* `bex ingest-sne` / `scan` / `melcache` / `datasets` CLI; datasets
persist under `store/datasets/<name>/` with their audio root recorded; timestamps
parsed from SNE, `YYYYMMDD_HHMMSS`, and classic AudioMoth hex-epoch names; the mel
cache is self-contained per recording (mixed sample rates coexist) at hop 512
(~40 MB/h — display-grade, half the sibling's size on a tight SSD), with
`display_range()` supplying per-view contrast. **Finding:** the schema validator
caught 2 zero-duration point annotations among SNE's 20,147 boxes (zero bandwidth
too — annotation-tool artefacts); they can never label a window under the overlap
rule and are dropped explicitly at ingest, count pinned by test.

### Stage 2 — BirdNET adapter, unfiltered + meta-model ✅ (2026-09-07, 53 tests)
The pivotal stage. In `envs/birdnet/`: run the classifier with **no species filter**
over recordings, capture per-window scores under the §3a storage policy; call the
occurrence meta-model separately for (site, week) and store per-species occurrence
scores; write run manifests (BirdGate lesson — the *executed* config). Validate on a
handful of SNE recordings: sanity-check that confidently annotated common species
score highly. **Deliverable:** `birdnet` arm producing schema-conformant parquet for
all 33 SNE recordings; a smoke-test notebook/table of top detections vs truth.

*As built:* the official `birdnet` 1.1.1 package, litert backend — no TensorFlow
(and it exposes Perch v2, which may simplify Stage 5). `envs/birdnet/run.py` runs
the acoustic model with `top_k=None, default_confidence_threshold=None` (all 6,522
scores per 3 s window, ~5 s per hour of audio) and geo v2.4 with `min_confidence=0`
per (lat, lon, week); occurrence + a `suppressed` flag reproducing BirdNET's own
default filter (`sf_thresh` 0.03) land per row via `adapters.apply_occurrence` —
recorded, never applied. Weeks use BirdNET's 48-week calendar
(`adapters.birdnet_week`); unknown timestamp → year-round aggregation, recorded in
the manifest. Runs resume by `--run-id`: scored recordings skip, the detections
index rebuilds whole-run.

*Finding — four undecodable FLACs, and a silent-fallback trap:* every
`*_170000.flac` (the four 17:00 evening sessions — evidently one recorder
configuration) fails libsndfile with "flac decoder lost sync" at frame 0, while
`sf.info` parses happily and CoreAudio decodes the full 3600 s perfectly. Every
consumer in this stack sits on libsndfile (soundfile, librosa, BirdNET's
producer), so those files were unreadable — and the sibling project never noticed
because librosa silently fell back to audioread/CoreAudio when building its cache.
Fix: `bex repair <dataset>` transcodes via audioread into
`store/repairs/<dataset>/` (int16-to-int16, frame-count verified, originals
untouched), and `ingest.resolve_audio` prefers repairs everywhere audio is read.

*Full run (`birdnet-c102ff26`, all 33 recordings):* 1,044,922 detection rows;
208 MB of score matrices; ~5 s/hour of audio. Annotated species take a mean
**6.7/10** of each recording's top-10 by max score (range 0–10 — the low end is
the sparse night/evening hours). Of 90,792 rows at the standard 0.1 operating
point, **8,423 (9.3%) would have been silently hidden by BirdNET's own species
filter** — and **84% of those are congeners of annotated species**: Black-capped
Chickadee at 0.842 beside Chestnut-backed + Mountain Chickadee, Eastern Towhee
beside Spotted Towhee, Scarlet beside Western Tanager, Wood Thrush beside Hermit
Thrush, an American Woodcock at 0.557 for pure vagrancy. Congener shadowing
dominates the suppressed layer, exactly as §4 theorised.

*Preview of the Stage 4 false-suppression question, already visible:* **Fox
Sparrow — 870 ground-truth boxes, 1,367 windows ≥ 0.5 — has occurrence 0.0317 at
its best week**, straddling the 0.03 threshold across May's weeks, so the default
filter hides an abundantly present species in some weeks and not others. Three
annotated species sit below 0.03 at every observed week (Williamson's Sapsucker,
Evening Grosbeak, Hammond's Flycatcher): outright false suppressions. And
Flammulated Owl scores ≥ 0.5 in 57 evening windows with occ 0.0004 and no
annotation — annotation gap or model error, flagged for Stage 4, not concluded.

### Stage 3 — The app, v1 (visual first — resolved D6) ✅ (2026-09-07, 61 tests)
Tabs 1–3 on the `birdnet` arm: recordings, and the explorer with per-model lanes on
native grids (window boundaries drawn), suppression rendered as the hollow/hatched
layer, click-through audio, Xeno-Canto link-outs (§7 layer 1). SNE + any unlabelled
test folder. **Deliverable:** watching BirdNET's raw vs suppressed output over a real
spectrogram — the core experience, before any aggregate statistic exists.

*Revised after first use:* two things made the explorer hard to read. (a) Only truth
boxes were drawn on the spectrogram, so there was nothing to compare them *to*. Now
a **focus species** selector overlays one bird's model detections (a bar in a
reserved strip along the top, plus a faint tint) against its truth boxes (solid,
time × frequency) — the contrast is the lesson: on Hermit Thrush BirdNET claims
every window at 1.00 where truth localises two intervals, and the bar-vs-box shape
says the model localises in *time only*. Labels are one-per-species (widest block).
(b) Simultaneous species overdrew each other in the truth lane, silently hiding
species; `views.pack_rows` now stacks them. Added a **navigator**: a clickable
whole-recording activity strip (detections, suppressions, annotated coverage per
bin) plus a "jump to" list of busiest / most-confident / most-suppressed moments,
so choosing where to look is no longer guesswork.

*Revised again after a user question ("the lane shows detections but the table's
scores are tiny"):* three separate causes, all in the UI rather than the data.
(a) The inspector read a 3 s window chosen by a **seconds** slider defaulting to
the span's first second, while the lane showed whichever windows cleared θ — at
θ=0.9, 54% of windows in a dawn-chorus hour are empty, so the two were usually
describing different moments. The inspector now selects from the model's actual
window grid (`views.window_grid`), opens on the first window holding a detection
(`views.first_detection`), labels each option with its top species, and is
**marked on the spectrogram and every lane**, so what the table describes is never
in doubt. (b) The lane drew only the top species per window, hiding real
co-detections — 11% of non-empty windows at θ=0.9 and **52% at θ=0.5** carry two
or more species at once; `lane_blocks(per_window=3)` plus `pack_rows` now stacks
them. (c) The table lists the window's top species *regardless of θ*, so its tail
is what the model weighed and rejected — now said explicitly in the caption,
along with the fact that **score and occurrence are independent** (audio vs a
geographic prior that never heard the audio); their disagreement is the subject
of study, not an error.

Also fixed a rule violation introduced by the palette: ranking colours within the
viewport meant a species was repainted when θ or the viewport moved (American
Robin orange at θ=0.9, blue at θ=0.5). Colour is now assigned once per recording
at a fixed threshold, so it follows the bird, never its rank.

### Stage 4 — Geofilter forensics on SNE, cross-checked against the app ✅ (2026-09-08, 85 tests)
Implement §4's metrics in `stats.py` (pure functions over the store — the app and any
script call the same code), including per-species muddiness. Run with the California
profile: suppression events, false suppression/admission vs ground truth, shadowed
congeners. Then the **cross-check D6 asked for**: every headline number the app
displays is independently re-derived by a headless script from the score store, and
the two are compared before we trust either — the BirdGate discipline, applied from
the first result. The forensics tab (tab 4) lands here. **Deliverable:** "what
BirdNET's geofilter actually does, scored against truth" as a table + writeup in this
file, plus the passing cross-check.

*As built:* `bex/stats.py` (§4 as pure functions), `scripts/forensics.py`
(the tables, headless), `scripts/crosscheck.py`, and the app's forensics tab.
Implausibility has two independent judges — the model's own **geofilter** verdict
replayed, and our **profile**, which applies identically to every model and so
keeps Stage 5's comparison fair.

**The cross-check passes exactly.** All 33 recordings: 1,044,922 index rows
re-derived from the raw score matrices row-for-row, and the muddy-window rate and
impostor mass recomputed over all 6,522 classes agree to 3.0e-08 (float32 noise).
That exactness is by construction — impostor mass sums only over species at or
above θ, and the index keeps everything scoring ≥0.01, so for θ≥0.01 the quantity
is computable from the index with no dependence on the storage policy. Defining it
over the whole class vector instead would have put thousands of ~0.001 scores in
the denominator and made the metric an artefact of the model's class count.

**Result 1 — the filter's biggest effect on this data is a false suppression, not
a catch.** At θ=0.1, 22.3% of the 35,979 detection-positive windows carry a
confident impossibility. But **Fox Sparrow alone accounts for 6,504 of the 8,423
suppression events** — a species with 870 ground-truth boxes, detected up to
0.999, whose occurrence (0.0237) sits just below the 0.03 cutoff. It also
dominates the shadowing table: 8 of the 12 worst offending pairs are "some
plausible species, shadowed by Fox Sparrow". The headline is therefore not "the
map rescues the classifier" but **the map overrides a correct classifier, at
scale, for one abundant resident**.

**Result 2 — false suppression 4.70%, false admission 0.39%.** Of 35,314
detections agreeing with an annotation, 1,661 were hidden by the filter (Fox
Sparrow, Evening Grosbeak, Hammond's Flycatcher, Williamson's Sapsucker). Of
82,369 admitted detections, only 323 were of species never annotated. **The filter
is roughly twelve times more likely to hide a real bird than to catch a fictional
one** on this dataset. Three annotated species (Williamson's Sapsucker 0.0113,
Evening Grosbeak 0.0218, Hammond's Flycatcher 0.0274) never reach the cutoff at
any week: the prior has decided against them whatever the audio says.

**Result 3 — congeneric shadowing is rare here (0.47%, 62 of 13,244).** The
Stage 2 note that "84% of suppressed rows are congeners of annotated species" is a
*different* statistic — it asks whether a suppressed species has a relative
somewhere in the dataset, not whether the two competed in the same window. Same-
window congeneric confusion is the sharper claim and it is uncommon on SNE. Worth
re-testing on UK audio, where the European/Australasian blackbird case that
motivated the project actually lives.

**Result 4 — per-species muddiness separates the birds the model knows.** Median
1.67 companions per detection, IQR 0.93–2.49 across 99 species with ≥5 detected
windows. Golden-crowned Kinglet is the standout: **0.44 companions over 11,243
windows** — abundant and unambiguous. At the other end, Green-tailed Towhee (4.49
over 1,146 windows) and Mountain Quail (3.41 over 1,325) are common *and*
confusable, which is exactly the population where a geographic prior does real
work. ⚠️ **A methodological trap, found and fixed:** counting any species within Δ
as a companion confounds the metric with the focal score — at Δ=0.15 a detection
at 0.10 admits everything down to −0.05, so the weakest detections scored as the
muddiest species (median 13.1, and the "muddiest" list was all rare species). A
companion must itself clear θ. Median dropped 13.1 → 1.67 and the ranking became
about species rather than about their scores.

### Stage 5 — Perch adapter + cross-model comparison ✅ (2026-09-08, 89 tests)
`envs/perch/`: Perch 2.0 logits + cached embeddings; the same UK/California masks
applied symmetrically; `birdnet-uk` masked arm (nearly free); tab 5.
**Deliverable:** the league table over three arms on SNE; the adapter contract proven
by a second, differently-shaped model.

*As built:* `envs/perch/` (Perch v2 — 14,795 classes, 5 s windows, 32 kHz,
TensorFlow, ~80 s per hour of audio), `stats.matched_league`,
`scripts/compare.py`, and the app's Compare tab. The contract held with no
schema change; Perch's labels are bare scientific names and join to BirdNET's
keys directly.

**Two traps the real data caught.** (a) A per-class **sigmoid over Perch's
logits saturates** — median 0.27, top classes above 0.999, 13,317 of 14,795
classes clearing 0.01 per window, producing 19.2M index rows for two
recordings. Softmax gives ~6.5 classes above 0.01 per window and is monotone in
the logits, so within-window rankings are unchanged; the sigmoid run was
discarded. (b) Perch's ~80 AudioSet classes (`Bass_drum`, `Bass_guitar`) broke
the BirdNET-style "split on underscore" key rule, collapsing two labels onto one
key — which would have misaligned every score column after it. The split is now
an explicit per-model argument, and the collision guard written in Stage 2
caught it.

**Comparing at matched θ is meaningless** (a sigmoid's 0.1 and a softmax's 0.1
are different claims), so arms are matched on **detection count**. That choice
also makes the comparison *transform-invariant*: matching on count selects the
top-N ranked (window, species) pairs, and both readouts are monotone in the
logits, so the selected set does not depend on which was applied.

**Result — Perch is far muddier at every operating point, and its confusion is
much more congeneric.** Judged by the California profile:

| detection budget | arm | θ | muddy windows | shadowed | congeneric | species reported | truth recall |
|---|---|---|---|---|---|---|---|
| 20,000 | perch-v2 | 0.0903 | **30.4%** | 1,694 | **542** | 439 | 96% |
| 20,000 | birdnet-2.4 | 0.8555 | 0.2% | 1 | 0 | 48 | 62% |
| 60,000 | perch-v2 | 0.0251 | **75.5%** | 23,559 | **3,323** | 1,021 | 98% |
| 60,000 | birdnet-2.4 | 0.1818 | 2.6% | 373 | 13 | 185 | 88% |
| 150,000 | perch-v2 | 0.0103 | 91.7% | 162,289 | 13,505 | 1,791 | 98% |
| 150,000 | birdnet-2.4 | 0.0435 | 12.8% | 12,618 | 456 | 629 | 98% |

The trade is legible: at a 20,000-detection budget Perch recovers **96% of the
56 annotated species against BirdNET's 62%**, but spreads across 439 species
where BirdNET uses 48, and 30% of its detection-positive windows carry a
confident impossibility against BirdNET's 0.2%. **Congeneric shadowing — the
project's founding hypothesis — is where the arms differ most: 542 against 0 at
matched budget, and 3,323 against 13 at the next.** BirdNET's Californian data
showed almost none (Stage 4: 0.47%); Perch shows it abundantly. A UK researcher
using Perch raw would meet far more geographic muddiness, and much of it would
be the confusable-congener kind a location prior is supposed to fix.

⚠️ **A circularity found and guarded.** In `profile` mode the filter-error
metrics are **structurally zero**: `us-ca-sierra` was generated from SNE's
species list, so its plausible set *is* the annotated set and the judge cannot
disagree with truth. `stats.filter_errors` now returns
`degenerate_false_suppression` / `degenerate_false_admission`, and the script,
the app and the comparison all say so rather than reporting a circular 0%. Only
`geofilter` mode scores a real prior against truth (Stage 4's 4.70%/0.39%).

### Stage 5.5 — The truth-metric layer ✅ (2026-09-11, 180 tests)
`bex/truth.py` (alignment) + `bex/metrics.py` (the sweep), `scripts/evaluate.py`,
and the Compare tab's PR section. The half of §3d Stage 5 left unbuilt: until now the
only contact between a model's output and the annotations was set-level ("did this arm
ever report this species?") and an any-overlap join in `stats.filter_errors`. Neither
can say whether the model was right *here*.
**Deliverable:** per-species PR curves and cmAP for every arm, and a per-species
threshold table a survey can actually use.

*As built:* the aligned frame is deliberately the dumbest shape that can be checked
by hand — one dense row per (recording, window, species) with a 0/1 truth and a score,
cached under `store/truth/`. Dense, because the zeros are the negatives and a
precision denominator made of rows filtered out before anyone looked is how evaluation
code lies. Built from the **full score matrices**, not the detections index: the index
drops exactly the low-scoring rows that make up the high-recall end of the curve, so a
curve built from it would be truncated and its AP inflated. D3's "store everything"
paid for itself here — this metric was computed retroactively with no re-run.
`windows.label_windows` finally has a caller, generalised to `label_starts` so truth is
labelled on **the window starts the model actually produced** rather than a grid
recomputed from the duration.

**Three decisions that determine what the numbers mean**, all in `truth.py`'s docstring:
the evaluation label space is the *annotated* species (scoring over Perch's 14,795
classes would count every unlabelled species as a false positive and punish the larger
vocabulary for existing); every arm is scored on the *same* species set, with a species
missing from a model's vocabulary kept and scored zero rather than dropped; and this
assumes the annotations are exhaustive for the species they cover — true for SNE,
**re-check it before trusting these numbers on UK data**.

**Result — on native grids, judged against truth:**

| arm | cmAP | micro AP | θ @ p≥0.95 | recall there | θ @ max F1 |
|---|---|---|---|---|---|
| birdnet-2.4 | **0.485** | **0.817** | 0.8872 | 0.471 | 0.5010 |
| perch-v2 | 0.363 | 0.646 | 0.3696 | 0.266 | 0.0376 |

⚠️ **That table is not a fair comparison, and the reason is a trap this project set
for itself.** Stage 5 chose softmax for Perch because it is "monotone in the logits" —
true *within* a window, which is all the muddiness statistics need, and false *across*
windows, which is all a PR curve does. A per-species curve ranks windows against each
other, and softmax divides by a per-window sum over 14,795 classes, so the same bird
scores lower in a busy dawn-chorus window than alone.

A proxy experiment suggested this was worth a lot: renormalising **BirdNET's**
per-class sigmoid scores within each window cost 0.056 cmAP and 0.168 micro AP, landing
BirdNET's micro AP within 0.003 of Perch's. **That proxy was wrong, and the direct
measurement overturned it** — see "what the sigmoid re-run actually showed" below.
Dividing already-saturated probabilities by their sum is a far more violent and more
biased transform than a softmax over logits, and it does not model Perch.
`truth.across_window_comparable` still flags any softmax run, because on another head
the normaliser may vary much more — but it now reports the measured size rather than an
extrapolation. Note the Stage 5 reason for discarding sigmoid (19.2M index rows) applies
to the *detections index*, not to the score matrices, so both readouts coexist.

**A global θ does not exist.** BirdNET reaches 95% precision on 27 of 56 species, needing
θ from 0.18 to 0.98 (median 0.81); Perch on 20 of 56, θ from 0.016 to 0.87. Two species
wanting θ=0.02 and θ=0.87 for the same precision is not a tuning detail — it is the
argument for the per-species threshold table, which the app exports as CSV.

**What the sigmoid re-run actually showed (2026-09-11).** Perch was re-run with a
per-class sigmoid at float32 and scored against the same truth. The readout *does*
reorder windows substantially — across-window Spearman between the two readouts is
**0.69** (0.43-0.78 over 28 well-labelled species), so the mechanism is real — but the
reordering is close to unbiased with respect to truth, and the aggregate barely moves:

| arm | cmAP | micro AP | rank resolution |
|---|---|---|---|
| birdnet-2.4 | **0.4848** | **0.8172** | 0.924 |
| perch-v2 (sigmoid, float32) | 0.3746 | 0.6462 | 0.958 |
| perch-v2 (softmax, float16) | 0.3629 | 0.6463 | 0.941 |

**+0.0117 cmAP and +0.0001 micro AP.** Individual species swing either way (-0.113 for
*Pipilo maculatus*, +0.146 for *Oreortyx pictus*), and cancel. So **BirdNET's lead over
Perch at window level on SNE is real, not an artefact of our readout choice** — the
opposite of what this section claimed before the measurement existed.

Two practical consequences. The **softmax run stays the working arm**: its AP is now
validated, its θ values are human-readable (median 0.22 for the 95%-precision rule)
where the sigmoid arm's are all ~0.9999, and its index is a third the size. And the
sigmoid float32 run is worth keeping as the control that licenses using it.

**The per-species view, and what it immediately turned up.** `bex/scorecard.py`
plus the app's Species scorecard tab answer the question the league table cannot:
for *Regulus satrapa* (2,696 annotated vocalisations), at each arm's own
95%-precision threshold, BirdNET finds 2,327 of them and Perch 2,591 — **Perch is
better at the unit a surveyor cares about**, while looking worse on window recall
because its 5 s grid gives it 3,856 windows to BirdNET's 5,402. And the
per-recording breakdown is blunt about where BirdNET's average comes from: 82%,
78%, 79% recall on the SNE_029-033 recordings and **0% on both SNE_001 and
SNE_020**, where Perch manages 71% and 60%. A single cmAP cannot say that, and it
is the kind of failure a researcher can go and listen to.

**Per-species AP is where the arms actually differ**, and the scatter says so at a
glance: most species sit below the diagonal (BirdNET ahead) but the best-supported
species cluster on it, and Perch wins outright on some — *Regulus satrapa* 0.936 vs
0.908, **despite the softmax handicap**, which makes it a floor on Perch's true margin.

**The fix, and the second trap underneath it (2026-09-11).** The sigmoid re-run was
made — `envs/perch/run.py` now takes `--readout {softmax,sigmoid}` — and it did *not*
produce a better number. It produced a worse one: cmAP 0.295 against softmax's 0.363.
The cause was not the readout but the **store**.

AP is a ranking statistic, so it is only as good as the store's ability to tell two
windows apart, and D3 stores float16. That is an excellent choice for a softmax —
float16 is a *floating* format, so its relative precision holds all the way down
(spacing 6e-08 near 1e-4, and softmax values span orders of magnitude) — and a bad one
for a **saturated** score, whose informative range is crushed into the top of [0, 1]
where float16 steps by 0.00098. Measured at the top of each species' ranking, the part
AP integrates over:

| arm | distinct values among top-ranked windows (median over species) |
|---|---|
| perch sigmoid, float16 | **0.011** |
| perch softmax, float16 | 0.941 |
| birdnet sigmoid, float16 | 0.924 |

**BirdNET is a sigmoid arm too and is unaffected** — its head is not saturated. So the
hazard is saturation meeting float16, not the readout's name, and `truth.rank_resolution`
therefore *measures* it rather than inferring it from the manifest. Both the script and
the app refuse to present a damaged arm's cmAP as a result.

Storage precision is now a per-run choice (`store.write_scores(..., dtype=)`,
`adapters.scores_to_matrix(..., dtype=)`), and a saturated readout gets float32 —
verified on one recording before committing the run: distinct-in-top-200 goes from
0.250/0.005/0.015 to 1.000/0.525/0.975 for three abundant species, with nothing
saturating to exactly 1.0. Cost: 38 MB per recording against 18 MB, ~1.25 GB for the
arm.

**The general lesson, which outlives this bug:** two of the three headline numbers this
stage produced were artefacts of representation choices made for unrelated reasons —
softmax chosen for index size, float16 chosen for disk — and both looked like clean
measurements of a model. Neither was caught by a test; both were caught by asking "what
would have to be true for this number to mean what it says", and then measuring it.
`across_window_comparable` and `rank_resolution` exist so the next one is caught by the
code instead.

**Also fixed:** arm labels were `model-version`, which stopped being unique the moment
the same model was run twice with different readouts. Two Perch runs both called
themselves `perch-v2`, and every comparison keyed on that label — `compare.py`,
`evaluate.py`, the app's Compare tab — would have silently kept one and dropped the
other, with no error and a table that simply looked like it had fewer arms.
`views.arm_labels` disambiguates by what actually differs (the readout), falling back
to the run id.

### Stage 6 — UK audio for real
First UK recordings through the unlabelled pipeline (our own, plus any collaborator
audio); UK tier lists exercised in anger; muddiness league table on UK data;
θ-sensitivity reported. **Deliverable:** the §4 claim, with the blanks filled in.

### Stage 6.5 — Label your own data in the Explorer (proposed 2026-09-30)
The step that turns the unlabelled pipeline (§2b) into a benchmark: a researcher
loads unlabelled audio, runs BirdNET and Perch over it as a *guide*, and labels a
sample in the Explorer — minute by minute, listening, reading the spectrogram and the
models' suggestions — so every later page can score models on *their* site.

- **Labelling mode in the Explorer.** Step through a recording in 60 s chunks. Draw a
  box on the spectrogram (time × frequency) and give it a species from a searchable
  list (common names, the taxonomy's keys). Edit, move and delete boxes; the audio
  player already runs in recording time under the figure.
- **Suggestions, with provenance.** A model detection can be accepted as a starting
  box, then adjusted. Every label records how it was made (drawn / accepted from
  BirdNET / Perch) and who made it, when.
- **Guard against anchoring.** Labels seeded by a model flatter that model when it is
  later scored on them. A *blind* option hides the model lanes while labelling, and
  the benchmark reports how many labels came from suggestions.
- **Say what was checked.** Each chunk is marked *reviewed — every species heard is
  labelled* or not. Only reviewed chunks count as ground truth, so an unlabelled
  stretch never reads as "no bird here" (the root of finding F1 in V1.md).
- **Label sets, never overwritten.** Labels live in named *label sets*, never in the
  dataset itself. Annotations that came with a dataset (SNE's) are a read-only
  reference set. Adding a box to an already-labelled recording does not change that
  set: it happens in a new set derived from it ("sne + my additions"), which starts as
  a copy and records what it was derived from. Which set counts as ground truth is an
  explicit choice in the sidebar — like the threshold set — so nothing becomes
  "the official labels" by accident, and every page says which set it is scored on.
- **The 60 s chunk is the unit of sign-off.** Each chunk is *open* (editable) or
  *closed* (signed off: every bird heard is labelled, by whom, when). Boxes can only be
  changed in an open chunk; editing a closed one takes an explicit *Reopen* (with an
  optional reason) and a new sign-off. Only closed chunks count as ground truth. A
  recording's status in a set follows from its chunks: *unlabelled* (none closed),
  *in progress*, *labelled* (all closed).
- **History for free: an append-only log.** A label set is stored as a log of events
  — box added, edited, deleted; chunk closed, reopened — each with who, when and
  (for suggestions) which model. The current labels are the log replayed, so the full
  history, "who changed this box", and rolling back to any earlier point come from one
  mechanism, and a save can never destroy what was there. Two open tabs cannot
  overwrite each other: a write is refused if the log has moved since that tab loaded
  it, and the tab reloads.
- **Versions you can cite.** *Publish* freezes a set's current state as a numbered
  version (v1, v2 …). Scores, threshold sets fitted on labels, and cached truth
  alignments are keyed on the set and version, so a benchmark always says exactly
  which labels it used, and publishing v2 never silently changes a v1 result.
- **Out as CSV, in the existing schema** (start, end, low/high Hz, species) plus
  provenance columns, so a label set can leave BEX and come back.

*The one hard part:* drawing boxes. Streamlit has no box-drawing widget, and the
community canvas component is unmaintained. A custom bidirectional component
(`st.components.v2`, inline HTML/JS, no build step) over the spectrogram image is the
plan. Time-only labels (click start, click end — the Explorer already turns clicks into
times) are the fallback if it stalls.

### Stage 7 — Region-tuned arms and beyond
`probe-uk` (linear probe on Perch embeddings, trained on Xeno-Canto UK focal audio —
external *training data*, cheap by design because embeddings are cached); optionally
`sne-cnn` for the SNE-only comparison; then the extension axes: other taxa,
more models (BirdNET custom classifiers, future Perch releases).

*Groundwork done (2026-09-11):* Perch's 1,536-d per-window embeddings are cached for
all 33 SNE recordings under `store/embeddings/perch-495c1c32/` via `envs/perch/embed.py`
— `encode` only, no re-scoring, and each file's window count is **asserted against the
stored score matrix** before it is written, because an embedding that does not index
the same window as its label is a silent, plausible-looking corruption of everything
trained on it. A k-NN over these against labelled exemplars is the cheapest possible
third arm — no training at all — and it lands on Stage 5.5's scoreboard immediately.

*Prerequisite, discharged:* the Perch sigmoid re-run landed (`perch-7de840d8`,
float32) and showed the softmax handicap is worth ~0.01 cmAP, not the ~0.06 the proxy
predicted. New arms can therefore be compared against the **softmax** run, whose θ
values are readable and whose index is smaller, with the sigmoid run kept as the control
that justifies it.

---

## 10. Decisions — resolved 2026-09-07

- **D1 · Copy** the birdsong modules into `bex/`; the repo stands alone, the
  modules will diverge. Accepted cost: fixes don't flow between siblings.
- **D2 · Polars** for the store/stats layer, pandas at the Streamlit edge.
- **D3 · Store everything**: full float16 score matrices per recording × model
  (~5–6 GB for SNE × 3 arms), with the sparse detections parquet as a derived,
  regenerable index. Futureproofing chosen over disk. See §3a.
- **D4 · Plausibility as swappable profiles**, BOU British List first; picker +
  upload in the app; per-profile reporting. Designed for other people's lists from
  the start. See §3c.
- **D5 · Native grids**, drawn visibly in the explorer — the models' differing
  windowing is part of what the app shows, not an implementation detail. The shared
  1 s grid exists only for cross-model metrics and is labelled as such. See §3d.
- **D6 · App before forensics**: Stage 3 is the visual explorer, Stage 4 the SNE
  California-filter forensics with an explicit app-vs-headless-stats cross-check.
- **D7 · Open source is a goal**, hence the design rules in §8: thin app / thick
  library, config file, uv-locked envs, no unlicensable data in the repo, Apache-2.0,
  demo bundle, contract tests + CI. Streamlit stays for v1; rule 1 keeps a proper
  hosted rebuild cheap if it's ever wanted.

---

## 11. Stack

- **App/analysis env:** streamlit, polars (+ pandas at the edges), altair/matplotlib,
  librosa + soundfile, pyarrow. No ML frameworks. Managed with `uv`; lockfile in-repo.
- **Config:** `bex.toml` (data roots, default site + profile, Xeno-Canto API key);
  `bex.example.toml` committed, real one gitignored.
- **`envs/birdnet`:** the BirdNET-Analyzer package (TensorFlow); exact package + pin
  chosen in Stage 2 (the ecosystem has `birdnet`, `birdnetlib`, and the analyzer repo
  proper — we want whichever exposes raw class scores and the meta-model cleanly).
- **`envs/perch`:** Perch 2.0 via its published inference route (Kaggle/TF Hub model +
  `perch-hoplite` tooling); pinned in Stage 5.
- **Python:** 3.12 for the app env (matches the shared sibling venv); runner envs pin
  whatever their model requires, which is precisely why they're separate.
- Raw audio stays on the USB / external storage (cold); mel cache + parquet store on
  SSD (hot); manifests in-repo (meta). `.gitignore`: `data/audio`, `cache/`, `envs/`,
  `store/`.
