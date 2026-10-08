"""BEX · Home — every dataset and how far through the flow it is (V1.2 F0, F1)."""
from ui.common import *  # noqa: F403 — the shared app toolkit
from bex import flow, runners
from bex.profiles import ProfileError, profile_from_csv, save_profile, suggest_profile
from ui import sidebar

def masthead() -> None:
    """The logo — B and EX in their two colours — over the name it abbreviates,
    with the same letters picked out, and one line on what BEX is for."""
    c = brand()
    st.markdown(
        f"<div style='background:linear-gradient(120deg, {c['wash']}, transparent 70%);"
        "border-radius:14px; padding:1.6rem 1.8rem 1.4rem; margin:0.4rem 0 1.2rem'>"
        "<div style='font-size:4.2rem; font-weight:800; letter-spacing:0.04em; "
        "line-height:1'>"
        f"<span style='color:{c['b']}'>B</span><span style='color:{c['ex']}'>EX</span>"
        "</div>"
        "<div style='font-size:1.45rem; font-weight:600; margin-top:0.35rem; "
        "opacity:0.92'>"
        f"<span style='color:{c['b']}'>B</span>ioacoustics "
        f"<span style='color:{c['ex']}'>Ex</span>plorer</div>"
        "<div style='margin-top:0.8rem; max-width:46rem; opacity:0.85'>"
        "Run your recordings through several bird-ID models side by side, measure "
        "each against the same ground truth, and choose how to process the rest of "
        "your data on the evidence rather than on defaults.</div>"
        "</div>", unsafe_allow_html=True)


STATE_STYLE = {
    "done": ("✓", "#3e8e5e"), "partial": ("◐", "#c9892b"), "todo": ("○", "#8a8880"),
    "waiting": ("·", "#b9b8b0"),
}


def strip_html(steps: dict, nxt: str | None) -> str:
    """The flow as a row of chips, one per step, the next one outlined."""
    c = brand()
    chips = []
    for key, label in flow.STEPS.items():
        state, detail = steps[key]
        mark, colour = STATE_STYLE[state]
        ring = f"box-shadow:0 0 0 2px {c['b']};" if key == nxt else ""
        chips.append(
            f"<span title='{detail}' style='display:inline-block; white-space:nowrap; "
            f"padding:0.15rem 0.55rem; margin:0.15rem 0.2rem 0.15rem 0; border-radius:999px;"
            f"background:{c['tint']}; {ring} font-size:0.82rem'>"
            f"<span style='color:{colour}; font-weight:700'>{mark}</span> {label}</span>")
    return "<div>" + "".join(chips) + "</div>"


def go(dataset: str, page: str, view: str | None = None) -> None:
    """Make a dataset current and open a page, optionally in a given view."""
    st.session_state["dataset"] = dataset
    if view:
        sidebar.set_view(dataset, view)
    st.switch_page(page)


def labels_fact(d: str, recs: pl.DataFrame) -> str:
    """'labels: none yet' / 'every recording (imported)' / '42 min in 18 recordings (set)'."""
    if labels.has_imported(cfg.store_dir, d) and not labels.list_sets(cfg.store_dir, d):
        return "labels: every recording (imported)"
    sets = labels.list_sets(cfg.store_dir, d)
    if not sets:
        return "labels: none yet"
    t = labels.truth_from_set(labels.load_set(cfg.store_dir, d, sets[-1]), recs, "")
    more = f" (+{len(sets) - 1} more set{'s' * (len(sets) > 2)})" if len(sets) > 1 else ""
    return (f"labels: {t.hours * 60:,.0f} min in {len(t.recordings)} recording"
            f"{'s' * (len(t.recordings) != 1)} ({sets[-1]}){more}")


def dataset_card(d: str) -> None:
    recs, _ = get_dataset(d)
    hours = float(recs["duration_s"].sum()) / 3600
    sites = recs["site"].replace("", None).drop_nulls().n_unique()
    p = flow.progress(cfg.store_dir, cfg.cache_dir, d)
    steps, nxt = p["steps"], p["next"]
    ready = steps["models"][0] == "done"
    with st.container(border=True):
        top, buttons = st.columns([2.4, 1.3], vertical_alignment="top")
        with top:
            current = " · <i>current</i>" if st.session_state.get("dataset") == d else ""
            st.markdown(f"#### {d}")
            st.markdown(f"<div style='font-size:0.88rem; opacity:0.8'>{sites} "
                        f"site{'s' * (sites != 1)} · {len(recs):,} recordings · "
                        f"{hours:,.1f} h · {labels_fact(d, recs)}{current}</div>",
                        unsafe_allow_html=True)
            st.markdown(strip_html(steps, nxt), unsafe_allow_html=True)
            if nxt:
                st.caption(f"Next: **{flow.STEPS[nxt]}** — {steps[nxt][1]}.")
        with buttons:
            if not ready:
                st.caption("Set this dataset up below: spectrograms first, then the "
                           "models. Everything else waits on those.")
            else:
                if nxt and nxt not in ("labelled", "explored") and st.button(
                        f"Continue: {flow.STEPS[nxt]}", key=f"h_next_{d}",
                        type="primary", width="stretch"):
                    go(d, flow.PAGE_OF[nxt])
                # Two plain actions, never a commitment: either can be done first,
                # and the sidebar switches between the views at any time.
                if st.button("Label" if p["has_labels"] else "Start labelling",
                             key=f"h_label_{d}", width="stretch",
                             type="primary" if nxt == "labelled" else "secondary",
                             help="Label a random sample of minutes, so the models can "
                                  "be scored on this site."):
                    go(d, "ui/pages/label.py")
                if st.button("Explore all recordings", key=f"h_exp_{d}", width="stretch",
                             type="primary" if nxt == "explored" else "secondary",
                             help="Every recording, what each model reports, unscored. "
                                  "Switch to the labelled data at any time."):
                    go(d, "ui/pages/explorer.py", sidebar.VIEW_ALL)
        # Setting up happens here, in the card, not on another page: until the
        # spectrograms and model runs exist there is nowhere useful to go.
        if steps["models"][0] != "done":
            st.divider()
            setup_steps(d)
        else:
            with st.expander("Profile, spectrograms and models"):
                setup_steps(d)


def add_dataset() -> None:
    """F1: a folder of audio -> a dataset, its spectrograms, and model runs."""
    st.markdown(
        "Point BEX at a folder of recordings (`.wav`, `.flac`, `.mp3`, `.ogg`, "
        "subfolders included). Files stay where they are. Recording times are read "
        "from file names (AudioMoth and most recorders); the location lets BirdNET's "
        "own filter run, and places your recordings in the sample's strata.")
    with st.form("add_dataset"):
        folder = st.text_input("Folder of recordings", placeholder="/path/to/recordings")
        a, b = st.columns(2)
        name = a.text_input("Dataset name", placeholder="wood-farm-2024")
        site = b.text_input("Site name", value=cfg.site_name)
        c, d = st.columns(2)
        lat = c.number_input("Latitude", -90.0, 90.0, float(cfg.site_lat or 0.0),
                             format="%.4f")
        lon = d.number_input("Longitude", -180.0, 180.0, float(cfg.site_lon or 0.0),
                             format="%.4f")
        profiles = list_profiles(cfg.profiles_dir)
        prof = st.selectbox(
            "Plausibility profile", [SUGGEST, *profiles],
            help="The list of species plausible at this site: the judge every page "
                 "applies, and what the model runs keep low scores for. *Suggest from "
                 "the location* picks the profile whose area contains the coordinates. "
                 "It can be changed later on the dataset's card. Not in the list? "
                 "Upload your own under *Upload a plausibility profile* below.")
        ok = st.form_submit_button("Add the dataset", type="primary")
    if ok:
        name = name.strip()
        if not labels.valid_name(name):
            st.error("Give the dataset a name: letters, digits, - _ and . only.")
            return
        if name in ingest.list_datasets(cfg.store_dir):
            st.error(f"There is already a dataset called {name}.")
            return
        try:
            with st.spinner("reading every file's header…"):
                recs = ingest.scan_folder(folder.strip(), site=site.strip(),
                                          lat=float(lat), lon=float(lon))
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            st.error(str(e))
            return
        if prof == SUGGEST:
            prof = suggest_profile(cfg.profiles_dir, float(lat), float(lon))
            if prof is None:
                st.error(f"No profile covers {lat:.2f}, {lon:.2f}. Choose one from the "
                         "list (or add your own to the profiles folder — PLAN.md §3c).")
                return
        ingest.write_dataset(cfg.store_dir, name, recs, audio_root=folder.strip())
        flow.set_dataset_meta(cfg.store_dir, name, profile=prof)
        get_dataset.clear()
        st.session_state["dataset"] = name
        st.session_state["home_flash"] = (f"Added **{name}**: {len(recs)} recordings, "
                                          f"{recs['duration_s'].sum() / 3600:.1f} h, "
                                          f"profile *{prof}*. "
                                          "Now build its spectrograms and run the models.")
        st.rerun()


SUGGEST = "Suggest from the location"


def profile_setting(d: str, recs: pl.DataFrame) -> None:
    """The dataset's plausibility profile: its default everywhere, and what model
    runs are told. Changing it affects runs started afterwards only."""
    profiles = list_profiles(cfg.profiles_dir)
    own = flow.dataset_meta(cfg.store_dir, d).get("profile")
    lat, lon = recs["lat"].drop_nans().first(), recs["lon"].drop_nans().first()
    hint = (suggest_profile(cfg.profiles_dir, float(lat), float(lon))
            if lat is not None and lon is not None else None)
    a, b = st.columns([1.2, 2], vertical_alignment="bottom")
    pick = a.selectbox("Plausibility profile", profiles,
                       index=profiles.index(own) if own in profiles else
                       (profiles.index(hint) if hint in profiles else 0),
                       key=f"h_prof_{d}")
    if pick != own:
        if b.button("Save as this dataset's profile", key=f"h_prof_save_{d}"):
            flow.set_dataset_meta(cfg.store_dir, d, profile=pick)
            st.session_state.pop(f"profile_{d}", None)
            st.rerun()
        b.caption(("Not set yet" if not own else f"Currently *{own}*")
                  + (f" · suggested for this location: *{hint}*" if hint else ""))
    else:
        b.caption("The default on every page for this dataset, and what model runs "
                  "started from here are told.")


def upload_profile() -> None:
    """A species list for a site, uploaded as CSV -> a profile anyone can pick."""
    st.markdown(
        "A **plausibility profile** is the list of birds plausible where you "
        "recorded. Every page uses it to judge a detection: a confident score for a "
        "species that is not on the list is a model mistaking something it heard "
        "for a bird that could not be there.\n\n"
        "Upload a **CSV with one row per plausible species**:\n\n"
        "```\nspecies_key,tier,common_name\nErithacus rubecula,1,European Robin\n"
        "Turdus merula,1,Eurasian Blackbird\nUpupa epops,2,Eurasian Hoopoe\n```\n"
        "- **species_key** — the scientific name. A `common_name` (or "
        "`scientific_name`) column works instead; names are matched to the species "
        "the models know, and any that match nothing are listed for you to fix.\n"
        "- **tier** *(optional, 1 if left out)* — 0 on the national list, 1 regular "
        "(breeding, wintering, passage), 2 plausible vagrant, 3 implausible.\n"
        "- Species **not on the list count as implausible**, so list only what is "
        "plausible. Any other columns are kept.")
    uk = cfg.profiles_dir / "uk-starter" / "tiers.csv"
    if uk.exists():
        st.download_button("The UK starter list, as a template (CSV)", uk.read_bytes(),
                           "uk-starter_tiers.csv", "text/csv", key="h_prof_template")
    with st.form("upload_profile"):
        up = st.file_uploader("Species list (CSV)", type=["csv"])
        a, b = st.columns(2)
        name = a.text_input("Short name", placeholder="wood-farm")
        title = b.text_input("Display name", placeholder="Wood Farm, Warwickshire")
        region = st.text_input("Region", placeholder="e.g. West Midlands, England")
        st.caption("Optionally, the area it covers, so BEX can suggest it for "
                   "recordings made there (decimal degrees).")
        c1, c2, c3, c4 = st.columns(4)
        box = [c1.number_input("South (min lat)", -90.0, 90.0, 0.0, format="%.3f"),
               c2.number_input("West (min lon)", -180.0, 180.0, 0.0, format="%.3f"),
               c3.number_input("North (max lat)", -90.0, 90.0, 0.0, format="%.3f"),
               c4.number_input("East (max lon)", -180.0, 180.0, 0.0, format="%.3f")]
        go_up = st.form_submit_button("Add the profile", type="primary")
    if not go_up:
        return
    name = name.strip()
    if up is None or not labels.valid_name(name):
        st.error("Choose a file, and a short name (letters, digits, - _ . only).")
        return
    if name in list_profiles(cfg.profiles_dir):
        st.error(f"There is already a profile called {name}.")
        return
    lookup = {}
    for key, common in get_names().items():
        lookup[key.lower()] = key
        if common:
            lookup[common.lower()] = key
    try:
        prof, unmapped = profile_from_csv(up.getvalue().decode("utf-8", errors="replace"),
                                          name, title.strip(), lookup, region.strip())
    except ProfileError as e:
        st.error(f"Could not read it: {e}")
        return
    if unmapped:
        st.error(f"{len(unmapped)} name(s) match no species the models know: "
                 + ", ".join(unmapped[:25]) + (" …" if len(unmapped) > 25 else "")
                 + ". Use the scientific name for these, or check the spelling, and "
                 "upload again.")
        return
    has_box = box[2] > box[0] and box[3] > box[1]
    save_profile(cfg.profiles_dir, prof, bbox=tuple(box) if has_box else None)
    st.session_state["home_flash"] = (
        f"Added profile **{name}**: {len(prof.tiers)} species. Choose it for a dataset "
        "on its card, or when adding one" + (", or let BEX suggest it from the location."
                                            if has_box else "."))
    st.rerun()


def setup_steps(d: str) -> None:
    """The dataset's profile, its spectrograms, then model runs — inside its card."""
    recs, _ = get_dataset(d)
    n = len(recs)
    profile_setting(d, recs)
    from bex import repair
    from bex.melcache import mel_path
    mels = sum(mel_path(cfg.cache_dir, d, r).exists() for r in recs["recording_id"])

    st.markdown("**1. Spectrograms** — what the Explorer and the Label page draw. "
                "Building them reads every file end to end, and a file the audio "
                "library cannot decode is repaired into a clean copy here, before a "
                "model run trips over it.")
    st.progress(mels / n, text=f"{mels} of {n} built")
    if mels < n and st.button(f"Build the spectrograms ({n - mels} to go)",
                              key=f"h_mel_{d}", type="primary"):
        audio_root = ingest.dataset_audio_root(cfg.store_dir, d)
        bar = st.progress(0.0)
        repaired, failed = [], []
        todo = [(r, p) for r, p in recs.select("recording_id", "path").iter_rows()
                if not mel_path(cfg.cache_dir, d, r).exists()]
        for i, (rid, rel) in enumerate(todo):
            bar.progress(i / len(todo), text=f"{rid} ({i + 1} of {len(todo)}) — a file "
                         "that needs repairing takes a minute or so")
            try:
                if repair.build_mel(cfg.cache_dir, cfg.store_dir, d, rid, rel,
                                    audio_root) == "repaired":
                    repaired.append(rid)
            except Exception as e:  # noqa: BLE001 — report and carry on with the rest
                failed.append(f"{rid}: {e}")
        bar.empty()
        get_mel.clear()
        st.session_state["home_flash"] = (
            f"Spectrograms built for **{d}**."
            + (f" Repaired {len(repaired)} file(s) the audio library could not decode: "
               + ", ".join(repaired) + "." if repaired else "")
            + (f" **{len(failed)} failed** — " + "; ".join(failed) if failed else ""))
        st.rerun()

    st.markdown("**2. Model runs** — each model scores every recording, in its own "
                "environment, in the background.")
    models_panel(d, n, ready=mels == n)


@st.fragment(run_every=5)
def models_panel(d: str, n: int, ready: bool) -> None:
    """One progress bar per model, refreshed every few seconds while it runs."""
    have = runners.available()
    any_running = False
    for model in runners.known():
        runs = [m for m in views.list_runs(cfg.store_dir)
                if m.dataset == d and m.model_name.lower().startswith(model)]
        scored = max((len(store.list_scores(cfg.store_dir, m.run_id)) for m in runs),
                     default=0)
        j = runners.job(cfg.store_dir, d, model)
        running = bool(j and j["running"])
        any_running |= running
        label, bar = st.columns([1, 3], vertical_alignment="center")
        label.markdown(f"**{model}**")
        bar.progress(scored / n, text=f"{scored} of {n} recordings scored"
                     + (" · running" if running else " ✓" if scored >= n else ""))
        if running:
            with st.expander("Log", expanded=False):
                st.code(runners.tail(j["log"]) or "starting…", language=None)
        elif j and j.get("exit_code") not in (None, 0):
            st.caption(f"⚠️ The last {model} run stopped with an error "
                       f"(exit code {j['exit_code']}).")
        elif model not in have:
            st.caption(f"The {model} environment is not installed — see "
                       f"`envs/{model}/README.md`.")
        elif scored < n:
            if j and not running and scored < n and runners.tail(j["log"]):
                with st.expander("Last run stopped early — log"):
                    st.code(runners.tail(j["log"], 15), language=None)
            if st.button(f"Run {model}" + (" (resume)" if scored else ""),
                         key=f"h_run_{d}_{model}", disabled=not ready,
                         help=None if ready else "Build the spectrograms first: that is "
                         "where unreadable files are repaired."):
                try:
                    extra = ["--run-id", runs[0].run_id] if runs and scored else []
                    runners.launch(cfg.store_dir, d, model, cfg.source, extra)
                except FileNotFoundError as e:
                    st.error(str(e))
                st.rerun()
    st.page_link("ui/pages/models.py", label="Add a model", icon="➕",
                 help="How to install another model's runner — your own included.")
    flag = f"_h_was_running_{d}"
    if any_running:
        st.session_state[flag] = True
        st.caption("Runs carry on if you leave this page. Don't edit BEX's code while "
                   "one runs: the runner imports it.")
    elif st.session_state.pop(flag, False):
        # A run just finished: redraw the whole page once, so the card's steps
        # and buttons catch up.
        complete_runs.clear()
        st.rerun(scope="app")


def render() -> None:
    masthead()
    st.markdown(
        "Every dataset goes through the same flow: **label** a random sample of it, "
        "**explore** what each model hears, **analyse** how well each does against "
        "your labels, choose a **survey protocol**, and **apply** it to the rest. Or "
        "skip the labels and just explore what the models report — you can start "
        "labelling at any point.")
    flash = st.session_state.pop("home_flash", None)
    if flash:
        st.success(flash)
    datasets = ingest.list_datasets(cfg.store_dir)

    st.markdown("### Your datasets")
    if not datasets:
        st.info("No datasets yet. Add a folder of recordings below to begin — or, for "
                "the demonstration set, `bex ingest-sne` (see the README).")
    for d in datasets:
        dataset_card(d)

    with st.expander("➕ Add a dataset", expanded=not datasets):
        add_dataset()
    with st.expander("📄 Upload a plausibility profile"):
        upload_profile()

    st.page_link("ui/pages/about.py", label="About BEX — what it measures and why",
                 icon="ℹ️")


render()
