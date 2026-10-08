"""BEX · Recordings."""
from ui.common import *  # noqa: F403 — the shared app toolkit
from ui import sidebar


def render(ctx) -> None:
    datasets = ctx.datasets
    dataset = ctx.dataset
    recordings = ctx.recordings

    st.header("Datasets in the store")
    st.dataframe(
        pl.DataFrame([views.dataset_summary(cfg.store_dir, d, cfg.cache_dir) for d in datasets]),
        width="stretch", hide_index=True,
    )
    st.subheader(dataset)
    st.dataframe(recordings, width="stretch", hide_index=True, height=300)

    st.markdown("### How to add recordings")
    st.markdown(
        """
Recordings are added from the command line, from the BEX folder, and appear
here on the next refresh.

**1. Scan a folder of audio into a new dataset**

```bash
.venv/bin/bex scan /path/to/recordings --name my-site \\
    --site "Wood Farm" --lat 52.41 --lon -1.53
```

- Every `.wav`, `.flac`, `.mp3` and `.ogg` file under the folder is included,
  subfolders too. Files stay where they are; BEX records where to find them.
- `--lat` / `--lon` locate the whole folder. BirdNET's location filter needs
  them: without them no filter verdict is recorded, and the geofilter judge and
  *Geofilter forensics* have nothing to show for that dataset.
- **Recording times come from filenames**: `20240501_063000`-style names
  (modern AudioMoth, most recorders) and classic AudioMoth hex names
  (`5AFC0A30`) are recognised. A file whose name holds no time still loads,
  but BirdNET's filter then falls back to year-round occurrence.

**2. Build the spectrograms the Explorer draws**

```bash
.venv/bin/bex melcache my-site
```

**3. If a file will not decode**, `bex repair my-site` transcodes the ones the
audio library cannot read (`--full` for all of them).

**Then run a model over the new dataset** — see the **Models** page.

**Or add it from the app**: *Home → Add a dataset* scans a folder, builds the
spectrograms and starts the model runs.

**Labels.** A scanned folder is unlabelled: BEX can show what each model
detects, but cannot score them until there is ground truth. Make it on the
**1 · Label** page (a random sample of minutes, labelled by ear), or import
existing annotations there — SNE-style CSV, Raven selection tables or Audacity
label tracks.
        """
    )


render(sidebar.render())
