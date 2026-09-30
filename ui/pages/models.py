"""BEX · Models."""
from ui.common import *  # noqa: F403 — the shared app toolkit
from ui import sidebar


def render(ctx) -> None:
    dataset = ctx.dataset
    recordings = ctx.recordings
    ds_runs = ctx.ds_runs
    run_label = ctx.run_label
    n_scored = ctx.n_scored
    picked = ctx.picked
    included = ctx.included
    profile = ctx.profile

    heading(
        "The models being compared",
        "One row per run of a model over this dataset. **Window / hop** is the "
        "audio each score describes — a 3 s and a 5 s model are not scoring the "
        "same thing, and every view keeps each model on its own grid. **Readout** "
        "is how the model's raw outputs became scores; it is why θ is not "
        "comparable between models. **Geofilter** is the location filter the model "
        "ships with, if any — models without one are judged by the plausibility "
        "profile only.", level="###")
    if not ds_runs:
        st.info("No model has been run on this dataset yet — see envs/birdnet/README.md.")
    else:
        st.dataframe(pl.DataFrame([{
            "model": run_label[m.run_id],
            "included": m.run_id in picked,
            "name": m.model_name,
            "version": m.model_version,
            "window_s": m.window_s,
            "hop_s": m.hop_s,
            "classes": get_vocabulary_size(m.run_id),
            "readout": views.readout_token(m.score_transform) or "—",
            "geofilter": m.geofilter.get("model") or "none",
            "recordings": f"{n_scored[m.run_id]}/{len(recordings)}",
            "profile": m.profile or "—",
            "created": m.created_utc[:10],
            "run_id": m.run_id,
        } for m in ds_runs]), width="stretch", hide_index=True, column_config={
            "window_s": st.column_config.NumberColumn("window (s)", format="%.1f"),
            "hop_s": st.column_config.NumberColumn("hop (s)", format="%.1f"),
            "classes": st.column_config.NumberColumn("classes", format="%d"),
            "profile": st.column_config.TextColumn("run's default profile"),
        })
        st.markdown("### How to add a model")
        st.markdown(
            """
Each model runs from the command line in **its own Python environment**, under
`envs/<model>/` — BirdNET, Perch and any future model need libraries that
cannot share one install. A run reads a dataset from the store and writes its
scores back; the app picks it up on the next refresh.

**BirdNET** (v2.4 — ~5 s per hour of audio on an M-series CPU)

```bash
cd envs/birdnet
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ../.. --no-deps
.venv/bin/python run.py --dataset my-site --limit 2      # quick check first
.venv/bin/python run.py --dataset my-site                 # then everything
```

**Perch** (v2 — ~80 s per hour of audio; needs TensorFlow)

```bash
cd envs/perch
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ../.. --no-deps
.venv/bin/python run.py --dataset my-site --readout sigmoid
```

- A run that stops part-way can be resumed: re-run with
  `--run-id <the run's id>` and it skips what is already scored.
- The models page above shows each run's coverage; a run is only compared with
  others once it has scored every recording.
- Model weights download on the first run (~90 MB BirdNET, ~380 MB Perch).
- Full details: `envs/birdnet/README.md`, `envs/perch/README.md`.

**Your own model** joins the same way: a runner in `envs/<your-model>/` that
writes each recording's full score matrix and a run manifest through
`bex.store`, using BirdNET's and Perch's runners as templates. The app needs
nothing else to compare it.
            """
        )

        st.markdown("##### Run manifests")
        st.caption("The configuration each run *actually executed* — every number in "
                   "the app traces back to one of these.")
        for m in ds_runs:
            with st.expander(f"{run_label[m.run_id]} — {m.run_id}"):
                st.code(m.to_json(), language="json")


render(sidebar.render())
