"""Simple mode: the guided path from a folder of recordings to checked
detections, and on to EcoLabel volunteers (bex/ecolabel.py does the work;
this is the walk-through).

    1 Recordings → 2 Model → 3 Species → 4 Check → 5 Recall → 6 Export
"""
import hashlib
import math
import time
from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st

from bex import ecolabel as E
from bex import flow, ingest, labels, prefs, repair, runners
from bex.profiles import suggest_profile
from ui.common import cfg, get_dataset, ordered_datasets

STEPS = ["Recordings", "Model", "Species", "Check", "Recall", "Export"]
NEW = "➕ New project from a folder…"
ACCENT, ACCENT_DEEP = "#3e6a6b", "#2f5657"
DONE = "#3f9a52"            # the ticks: readable on light and dark pages
WARN = "#c98a1a"
CHECK_VIEWS = ["Overview", "Checking", "Your answers"]


# ----------------------------------------------------------------- cached ---
@st.cache_data(show_spinner=False)
def pool_for(dataset: str, run_id: str) -> pl.DataFrame:
    return E.candidate_pool(cfg.store_dir, dataset, run_id)


@st.cache_data(show_spinner=False)
def names_and_tiers(dataset: str):
    tiers, pdir = E._tiers(cfg, dataset)
    return tiers, E.common_names(cfg.store_dir, pdir)


@st.cache_data(show_spinner=False, max_entries=64)
def clip_media(path: str, t0: float, t1: float, hi0: float, hi1: float) -> tuple[bytes, bytes]:
    return (E.clip_bytes(Path(path), t0, t1), E.spectrogram_png(Path(path), t0, t1, hi0, hi1))


def settings(d: str) -> dict:
    return E.load_settings(cfg.store_dir, d)


def save(d: str, **changes) -> None:
    s = settings(d)
    s.update(changes)
    E.save_settings(cfg.store_dir, d, s)


def species_for(d: str, s: dict) -> pl.DataFrame:
    tiers, names = names_and_tiers(d)
    return E.species_table(pool_for(d, s["run_id"]), tiers, names, s)


def server_url() -> str:
    return prefs.get(cfg.store_dir, "ecolabel_server", "http://localhost:8000")


def library(d: str) -> dict:
    """EcoLabel's shared library, per species (asked at most every 10 minutes;
    {} when the server can't be reached, and then everything is checked here)."""
    k = f"eco_lib_{server_url()}"
    got = st.session_state.get(k)
    if got is None or time.time() - got[0] > 600:
        cov = E.library_coverage(server_url())
        got = (time.time(), cov)
        st.session_state[k] = got
        if cov and settings(d).get("library_coverage") != cov:
            save(d, library_coverage=cov)
    return got[1]


def about_ok(s: dict) -> bool:
    a = s["about"]
    return bool(a["project"]["title"] and a["project"]["summary"] and a["group"]["name"])


def go(step: str) -> None:
    st.session_state["eco_step"] = step


# ----------------------------------------------------------------- status ---
def status(d: str | None) -> dict[str, bool]:
    """Which steps are done, for the stepper and for where to open."""
    done = dict.fromkeys(STEPS, False)
    if not d:
        return done
    done["Recordings"] = True
    s = settings(d)
    runs = E.runs_for(cfg.store_dir, d)
    done["Model"] = bool(s["run_id"]) and any(r["run_id"] == s["run_id"] for r in runs)
    if not done["Model"]:
        return done
    done["Species"] = s.get("included") is not None
    done["Check"] = E.load_checks(cfg.store_dir, d).height >= E.MIN_CALIBRATION
    done["Recall"] = s.get("floor") is not None
    done["Export"] = bool(s.get("uploaded") or s.get("exported"))
    return done


def stepper(d: str | None, current: str) -> None:
    """The steps, left to right: each is a button, the current one filled,
    finished ones ticked in green, arrows for the direction of travel."""
    done = status(d)
    css = [
        ".st-key-eco_stepper [data-testid='stHorizontalBlock']{align-items:center}",
        ".st-key-eco_stepper button{min-height:3.3rem;font-size:1.02rem;font-weight:600;border-radius:12px;"
        "white-space:nowrap}",
        ".eco-arrow{text-align:center;font-size:1.4rem;opacity:0.45;line-height:3.3rem}",
    ]
    for i, name in enumerate(STEPS):
        sel = f".st-key-eco_st_{i} button"
        if name == current:
            css.append(f"{sel}{{background:{ACCENT};border-color:{ACCENT};color:#fff}}"
                       f"{sel} *{{color:#fff !important}}{sel}:hover{{background:{ACCENT_DEEP}}}")
        elif done[name]:
            css.append(f"{sel}{{background:{DONE}1f;border-color:{DONE}88}}{sel} *{{color:{DONE} !important}}")
    st.markdown(f"<style>{''.join(css)}</style>", unsafe_allow_html=True)
    widths = []
    for i in range(len(STEPS)):
        widths += [1, 0.18] if i < len(STEPS) - 1 else [1]
    with st.container(key="eco_stepper"):
        cols = st.columns(widths, gap="small")
        for i, name in enumerate(STEPS):
            with cols[2 * i].container(key=f"eco_st_{i}"):
                tick = " ✓" if done[name] else ""
                st.button(f"{i + 1} · {name}{tick}", key=f"eco_go_{name}", width="stretch",
                          disabled=d is None and name != "Recordings", on_click=go, args=(name,))
            if i < len(STEPS) - 1:
                cols[2 * i + 1].markdown("<div class='eco-arrow'>→</div>", unsafe_allow_html=True)


def nav(back: str | None, forward: str | None, ok: bool = True, why: str = "") -> None:
    st.divider()
    a, _, b = st.columns([1, 3, 1])
    if back:
        a.button(f"‹ {back}", key=f"eco_back_{back}", on_click=go, args=(back,))
    if forward:
        b.button(f"{forward} ›", key=f"eco_fwd_{forward}", type="primary", disabled=not ok, help=why or None,
                 on_click=go, args=(forward,), width="stretch")


# ================================================================== steps ===
def step_recordings(d: str | None) -> None:
    st.markdown("#### Your recordings")
    if d:
        recs, _ = get_dataset(d)
        meta = flow.dataset_meta(cfg.store_dir, d)
        st.success(f"**{d}**: {len(recs)} recordings, {recs['duration_s'].sum() / 3600:.1f} hours"
                   f"{' · species list *' + meta['profile'] + '*' if meta.get('profile') else ''}.")
        st.caption("Files stay where they are. Only short clips around detections ever leave this computer "
                   "(if you send them to EcoLabel), never whole recordings or your exact location.")
        nav(None, "Model")
        return
    st.markdown("Point BEX at a folder of recordings (`.wav`, `.flac`, `.mp3`, `.ogg`, subfolders included). "
                "Recording times are read from file names (AudioMoth and most recorders).")
    with st.form("eco_new"):
        folder = st.text_input("Folder of recordings", placeholder="/path/to/recordings")
        a, b = st.columns(2)
        name = a.text_input("Project name (no spaces)", placeholder="glen-affric-dawn-2026")
        site = b.text_input("Site name (yours, not shared)", value=cfg.site_name)
        c, e = st.columns(2)
        lat = c.number_input("Latitude", -90.0, 90.0, float(cfg.site_lat or 0.0), format="%.4f",
                             help="Used here to pick the region's species list. Never sent anywhere.")
        lon = e.number_input("Longitude", -180.0, 180.0, float(cfg.site_lon or 0.0), format="%.4f")
        ok = st.form_submit_button("Add the recordings", type="primary")
    if ok:
        name = name.strip()
        if not labels.valid_name(name):
            st.error("Give it a name: letters, digits, - _ and . only.")
            return
        if name in ingest.list_datasets(cfg.store_dir):
            st.error(f"There's already a project called {name}.")
            return
        try:
            with st.spinner("Reading every file's header…"):
                recs = ingest.scan_folder(folder.strip(), site=site.strip(), lat=float(lat), lon=float(lon))
        except (FileNotFoundError, RuntimeError, ValueError) as err:
            st.error(str(err))
            return
        prof = suggest_profile(cfg.profiles_dir, float(lat), float(lon))
        ingest.write_dataset(cfg.store_dir, name, recs, audio_root=folder.strip())
        flow.set_dataset_meta(cfg.store_dir, name, profile=prof or "")
        get_dataset.clear()
        st.session_state.update(eco_dataset=name, dataset=name, eco_step="Model")
        if not prof:
            st.session_state["eco_flash"] = ("No species list covers this location yet, so every species the "
                                             "model reports will be listed: untick the impossible ones in step 3. "
                                             "(Research mode → Home can add a regional list.)")
        st.rerun()


@st.fragment(run_every=4)
def run_progress(d: str, model: str, n: int) -> None:
    j = runners.job(cfg.store_dir, d, model)
    runs = [r for r in E.runs_for(cfg.store_dir, d) if r.get("model_name") == model]
    from bex import store
    scored = max((len(store.list_scores(cfg.store_dir, r["run_id"])) for r in runs), default=0)
    st.progress(min(1.0, scored / max(1, n)), text=f"{model}: {scored} of {n} recordings"
                + (" · running" if j and j["running"] else ""))
    if j and j["running"]:
        with st.expander("Log"):
            st.code(runners.tail(j["log"]) or "starting…", language=None)
    elif st.session_state.pop(f"eco_running_{d}", False):
        st.rerun(scope="app")


def run_label(r: dict) -> str:
    when = str(r.get("created_utc", ""))[:16].replace("T", " ")
    soft = " · softmax" if "softmax" in str(r.get("score_transform", "")) else ""
    return f"{r.get('model_name', '?').capitalize()} {r.get('model_version', '')}{soft} · {when or r['run_id']}"


def step_model(d: str) -> None:
    recs, _ = get_dataset(d)
    s = settings(d)
    runs = E.runs_for(cfg.store_dir, d)
    # A run started here and now finished: use it.
    pend = st.session_state.get(f"eco_newrun_{d}")
    if pend:
        j = runners.job(cfg.store_dir, d, pend)
        if not (j and j["running"]):
            st.session_state.pop(f"eco_newrun_{d}")
            mine = [r for r in runs if r.get("model_name") == pend]
            if mine:
                newest = max(mine, key=lambda r: r.get("created_utc", ""))
                save(d, run_id=newest["run_id"], floor=None)
                s = settings(d)
                st.toast(f"The new {pend} run is finished, and selected.")

    st.markdown("#### Existing model runs")
    if runs:
        st.caption("Runs already made over these recordings. Choose the one to use.")
        labels_ = {r["run_id"]: run_label(r) for r in runs}
        current = s["run_id"] if s["run_id"] in labels_ else runs[0]["run_id"]
        if current != s["run_id"]:
            save(d, run_id=current, floor=None)
            s = settings(d)
        rkey = f"eco_run_{d}"
        if st.session_state.get(rkey) != current:
            st.session_state[rkey] = current
        st.radio("Run to use", list(labels_), key=rkey, format_func=labels_.get, label_visibility="collapsed",
                 on_change=lambda: save(d, run_id=st.session_state[rkey], floor=None))
        choice = current
        pool = pool_for(d, choice)
        st.caption(f"{pool.height:,} candidate detections: each 3 s window's top {E.POOL_RANK} species.")
        if E.saturated(pool):
            st.warning("This run's scores pile up near 1, so they can't tell good detections from bad (Perch's "
                       "sigmoid readout does this). Run a new Perch model below: this path uses its softmax "
                       "readout, whose scores spread out.")
        warn = E.model_licence(choice.split("-")[0])
        if warn:
            st.info(warn)
    else:
        st.caption("None yet: run a model below.")

    st.markdown("#### Run a new model")
    st.caption("Adds another run to the list above; it doesn't replace them. When it finishes, it's selected."
               if runs else "Scores every recording for every species the model knows.")
    have = runners.available()
    model = st.radio("Model", [m for m in ("perch", "birdnet") if m in runners.known()], horizontal=True,
                     format_func=lambda m: "Perch (recommended)" if m == "perch" else "BirdNET (non-commercial licence)")
    if E.model_licence(model):
        st.caption("⚠️ " + E.model_licence(model))
    j = runners.job(cfg.store_dir, d, model)
    if model not in have:
        st.error(f"The {model} environment isn't installed yet: see `envs/{model}/README.md`.")
    elif j and j["running"]:
        st.session_state[f"eco_running_{d}"] = True
        run_progress(d, model, len(recs))
        st.caption("It carries on if you leave this page. About 80 s per hour of audio for Perch.")
    elif st.button(f"Run {model.capitalize()}", type="primary" if not runs else "secondary"):
        audio_root = ingest.dataset_audio_root(cfg.store_dir, d)
        bar = st.progress(0.0, text="Checking every recording can be read…")
        for i, (rid, rel) in enumerate(recs.select("recording_id", "path").iter_rows()):
            src = ingest.resolve_audio(cfg.store_dir, d, audio_root, rid, rel)
            if src.parent != ingest.repairs_dir(cfg.store_dir, d) and repair.needs_repair(src):
                dst = ingest.repairs_dir(cfg.store_dir, d) / f"{rid}.flac"
                dst.parent.mkdir(parents=True, exist_ok=True)
                repair.repair_recording(src, dst)
            bar.progress((i + 1) / len(recs), text=f"Checking every recording can be read… {i + 1} of {len(recs)}")
        runners.launch(cfg.store_dir, d, model, cfg.source, E.RUN_ARGS.get(model))
        st.session_state[f"eco_running_{d}"] = True
        st.session_state[f"eco_newrun_{d}"] = model
        st.rerun()
    nav("Recordings", "Species", ok=bool(s["run_id"]) and bool(runs),
        why="Run a model first (or wait for it to finish).")


# ---------------------------------------------------------------- species ---
def _shade(values, top: float, log: bool = False) -> list[str]:
    """One hue, light to dark, by magnitude: a tint behind the number, so the
    text keeps its own colour in either theme."""
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append("")
            continue
        x = (math.log1p(v) / math.log1p(top)) if log else (v / top)
        x = max(0.0, min(1.0, x))
        out.append(f"background-color: rgba(62,106,107,{0.06 + 0.52 * x:.2f})")
    return out


def species_editor(frame: pl.DataFrame, key: str, tick_label: str, tick_help: str, top_n: int):
    df = frame.select("include", "name", "species_key", "detections", "best").to_pandas()
    styled = (df.style.apply(lambda c: _shade(c, 1.0), subset=["best"])
              .apply(lambda c: _shade(c, max(top_n, 1), log=True), subset=["detections"]))
    return st.data_editor(
        styled, hide_index=True, width="stretch", key=key, height=min(560, 38 + 35 * max(1, frame.height)),
        column_config={
            "include": st.column_config.CheckboxColumn(tick_label, help=tick_help, width="small"),
            "name": st.column_config.TextColumn("Common name", help="What volunteers see"),
            "species_key": st.column_config.TextColumn("Species", disabled=True),
            "detections": st.column_config.NumberColumn("Detections", disabled=True, format="%d",
                                                        help="Windows where it's among the top 3 (darker: more)"),
            "best": st.column_config.NumberColumn("Best score", format="%.2f", disabled=True,
                                                  help="The model's most confident detection (darker: surer)")})


def step_species(d: str) -> None:
    s = settings(d)
    sp = species_for(d, s)
    tiers, _ = names_and_tiers(d)
    top_n = int(sp["detections"].max() or 1)
    main = sp.filter(pl.col("plausible"))
    unexpected = sp.filter(pl.col("unexpected"))
    low = sp.filter(~pl.col("plausible") & ~pl.col("unexpected"))

    st.markdown("#### On this region's species list" if tiers is not None else "#### Species the model found")
    st.caption(("Ticked species go to volunteers. " if tiers is not None else
                "No regional species list covers this location, so everything is listed: **untick anything that "
                "can't be here.** ")
               + "Each needs a common name (what volunteers see): add any that are missing. Darker cells: more "
                 "detections, or a surer best score.")
    ed_main = species_editor(main, f"eco_sp_{d}", "Check", "Ask volunteers about this species", top_n)

    ed_ux = None
    if unexpected.height:
        st.markdown(f"<style>.st-key-eco_ux_box{{border:1px solid {WARN}99 !important;"
                    f"background:{WARN}12;border-radius:12px}}</style>", unsafe_allow_html=True)
        with st.container(key="eco_ux_box", border=True):
            st.markdown(f"#### ⚠️ Unexpected here: {unexpected.height} species")
            st.markdown(f"The model was **confident** about these (a score of {E.UNEXPECTED_SCORE:.1f} or more), "
                        "but they **aren't on this region's list**. Most are mistakes, often a similar-sounding "
                        "species; one could be a real surprise worth recording. Tick any you want to look at: "
                        "they come to **you** in the EcoLabel portal, never to volunteers.")
            ed_ux = species_editor(unexpected, f"eco_ux_{d}", "Look at it", "Send to you, not to volunteers", top_n)

    ed_low = None
    if low.height:
        with st.expander(f"Not on the list, low scores: {low.height} species (probably noise)"):
            st.caption(f"The model never scored these above {E.UNEXPECTED_SCORE:.1f}, and the region's list says "
                       "they're unlikely. Tick one only if you know it's here: it then goes to volunteers, like "
                       "the list above.")
            ed_low = species_editor(low, f"eco_low_{d}", "Check", "Ask volunteers about this species", top_n)

    frames = [f for f in (ed_main, ed_ux, ed_low) if f is not None]
    crowd = [f for f in (ed_main, ed_low) if f is not None]
    inc_crowd = sum(int(f["include"].sum()) for f in crowd)
    det = sum(int(f[f["include"]]["detections"].sum()) for f in crowd)
    unnamed = sum(int((f[f["include"]]["name"].fillna("").str.strip() == "").sum()) for f in frames)
    n_ux = int(ed_ux["include"].sum()) if ed_ux is not None else 0
    st.caption(f"{inc_crowd} species for volunteers, {det:,} detections"
               + (f"; {n_ux} unexpected for you" if n_ux else "") + "."
               + (f" ⚠️ {unnamed} ticked without a name won't be included." if unnamed else ""))
    if st.button("Save the species list", type="primary"):
        names = {k: v.strip() for f in frames for k, v in zip(f["species_key"], f["name"].fillna("")) if v.strip()}
        included = [k for f in frames for k in f[f["include"]]["species_key"]]
        save(d, included=included, names=names, floor=None)
        st.toast("Saved")
        go("Check")
        st.rerun()
    nav("Model", "Check", ok=s.get("included") is not None, why="Save the species list first.")


# ------------------------------------------------------------------ check ---
def check_state(d: str, s: dict):
    pool = pool_for(d, s["run_id"])
    sp = species_for(d, s).filter(pl.col("include"), pl.col("name").str.len_chars() > 0, ~pl.col("unexpected"))
    checks = E.load_checks(cfg.store_dir, d)
    keys = sp["species_key"].to_list()
    left = (pool.filter(pl.col("species_key").is_in(keys)).join(checks.select(E.KEY), on=E.KEY, how="anti")
            .group_by("species_key").agg(left=pl.len()))
    ready = (E.readiness(checks, keys, library(d)).join(sp.select("species_key", "name", "detections"), on="species_key")
             .join(left, on="species_key", how="left").with_columns(pl.col("left").fill_null(0)))
    ready = ready.with_columns(progress=pl.Series([E.progress(r) for r in ready.iter_rows(named=True)],
                                                  dtype=pl.Float64))
    # A fixed order (most detections first): finished species stay put, so progress shows.
    return pool, checks, ready.sort(["detections", "species_key"], descending=[True, False])


PIP_CSS = (
    "<style>.pip{display:inline-block;width:13px;height:13px;border-radius:50%;margin-right:4px;"
    "vertical-align:middle;border:1.5px solid rgba(128,128,128,0.55);box-sizing:border-box}"
    ".pip.yes{background:#0E8A6A;border-color:#0E8A6A}.pip.no{background:#C9533B;border-color:#C9533B}"
    ".pip.lib{opacity:0.4}.pip-gap{display:inline-block;width:14px}"
    ".pip-legend{font-size:0.85rem;opacity:0.85;margin:0.2rem 0 0.6rem}.pip-legend .pip{margin-right:2px}"
    ".eco-sci{font-size:0.8rem;opacity:0.65;font-style:italic}"
    "[class*='st-key-eco_row_']{border-bottom:1px solid rgba(128,128,128,0.15);padding-bottom:4px}</style>")


def pip_row(here: int, lib: int, need: int, kind: str) -> str:
    """`need` dots: filled for answers here, pale for the library's, hollow for still needed."""
    h = min(here, need)
    lb = min(lib, need - h)
    return (f"<i class='pip {kind}'></i>" * h + f"<i class='pip {kind} lib'></i>" * lb
            + "<i class='pip'></i>" * (need - h - lb))


def pips(r: dict) -> str:
    return ("<span title='Yes'>" + pip_row(r["known_yes"], r["lib_yes"], E.MIN_YES, "yes") + "</span>"
            "<span class='pip-gap'></span><span title='No'>"
            + pip_row(r["known_no"], r["lib_no"], E.MIN_NO, "no") + "</span>")


def next_species(ready: pl.DataFrame, after: str | None = None) -> str | None:
    """The next species still needing answers (most detections first)."""
    todo = ready.filter(~pl.col("ready"), pl.col("left") > 0).sort("detections", descending=True)["species_key"]
    todo = [k for k in todo if k != after]
    return todo[0] if todo else None


def open_species(key: str | None) -> None:
    if key:
        st.session_state["eco_chk_sp"] = key
    st.session_state["eco_chk_view"] = "Checking"


def answer(d: str, c: dict, key: str, verdict: str) -> None:
    E.add_check(cfg.store_dir, d, c["recording_id"], c["start_s"], key, verdict, labeller="researcher")
    _, _, ready = check_state(d, settings(d))
    r = ready.filter(pl.col("species_key") == key).row(0, named=True)
    if r["ready"] and not st.session_state.get(f"eco_was_ready_{key}"):
        nxt = next_species(ready, key)
        name = r["name"]
        if nxt:
            st.session_state["eco_chk_sp"] = nxt
            st.session_state["eco_chk_flash"] = f"✓ {name} is ready for volunteers. On to the next species."
        else:
            st.session_state["eco_chk_flash"] = f"✓ {name} is ready. That was the last species needing answers."
            st.session_state["eco_chk_view"] = "Overview"


def undo_last(d: str) -> None:
    c = E.load_checks(cfg.store_dir, d)
    if c.height:
        last = c.sort("at").row(-1, named=True)
        E.remove_check(cfg.store_dir, d, last["recording_id"], last["start_s"], last["species_key"])
        st.session_state["eco_chk_sp"] = last["species_key"]


def check_overview(d: str, s: dict, checks: pl.DataFrame, ready: pl.DataFrame) -> None:
    n, n_ready = ready.height, int(ready["ready"].sum())
    answered = checks.filter(pl.col("verdict") != "unsure").height
    m1, m2, m3 = st.columns(3)
    m1.metric("Species ready for volunteers", f"{n_ready} of {n}")
    m2.metric("Your answers", f"{checks.height:,}")
    m3.metric("Recall estimate", "ready" if answered >= E.MIN_CALIBRATION else f"{E.MIN_CALIBRATION - answered} to go",
              help=f"Needs {E.MIN_CALIBRATION} clear answers (Yes or No) across species")
    total = float(ready["progress"].sum()) / max(1, n)
    st.progress(total, text=f"Overall: {total:.0%}")
    nxt = next_species(ready)
    b1, b2 = st.columns([1, 3], vertical_alignment="center")
    b1.button("Carry on checking ›" if checks.height else "Start checking ›", type="primary", width="stretch",
              disabled=nxt is None, on_click=open_species, args=(nxt,))
    b2.caption("Starts with the species that still needs answers and has the most detections, and moves on by "
               "itself when one is ready. Or click any species below to check it.")
    with st.expander(f"What “ready” needs: {E.MIN_YES} Yes and {E.MIN_NO} No"):
        st.markdown(f"- **{E.MIN_YES} Yes**: each volunteer hears {E.EXAMPLES} of them as examples (“this is what "
                    f"it sounds like”), and the other {E.MIN_YES - E.EXAMPLES} are hidden among their questions to "
                    "check them. Which ones is shuffled per volunteer, so every Yes does both jobs.\n"
                    f"- **{E.MIN_NO} No**: hidden among their questions too, so saying Yes to everything doesn't pay.\n"
                    "- 📚 **The shared library** counts: species other projects have already confirmed need fewer "
                    "answers from you, or none.\n"
                    "- Your answers also turn the model's scores into the recall estimate (next step), so check "
                    f"at least {E.MIN_CALIBRATION} in all, even if the library covers everything.\n\n"
                    "A species you can't finish (too few detections, or none you're sure of) isn't lost: its "
                    "detections come to you in the portal instead of going to volunteers.")

    def state(r):
        if r["ready"] and r["covered"] and not (r["known_yes"] or r["known_no"]):
            return "📚 Covered by the library"
        if r["ready"]:
            return "✓ Ready"
        if r["left"] == 0:
            return "Ran out: comes to you"
        return "In progress" if r["references"] + r["known_yes"] + r["known_no"] + r["unsure"] else "Not started"

    st.markdown(PIP_CSS, unsafe_allow_html=True)
    st.markdown("<div class='pip-legend'>" + pip_row(1, 0, 1, "yes") + " Yes you confirmed &nbsp; "
                + pip_row(1, 0, 1, "no") + " No &nbsp; " + pip_row(0, 1, 1, "yes") + " from the library &nbsp; "
                + pip_row(0, 0, 1, "yes") + " still needed</div>", unsafe_allow_html=True)
    widths = [2.6, 2.3, 1.7, 0.9, 1.0]
    head = st.columns(widths, vertical_alignment="center")
    for h, c in zip(["Species", f"{E.MIN_YES} Yes · {E.MIN_NO} No", "Status", "Detections", ""], head):
        c.caption(h)
    with st.container(height=min(620, 20 + 58 * ready.height), border=False):
        for r in ready.iter_rows(named=True):
            k = r["species_key"]
            with st.container(key=f"eco_row_{hashlib.md5(k.encode()).hexdigest()[:10]}"):
                c = st.columns(widths, vertical_alignment="center")
                c[0].markdown(f"**{r['name']}**<br><span class='eco-sci'>{k}</span>", unsafe_allow_html=True)
                c[1].markdown(pips(r), unsafe_allow_html=True)
                c[2].markdown(state(r))
                c[3].markdown(f"{r['detections']:,}")
                c[4].button("Check ›", key=f"eco_chk_go_{k}", width="stretch", disabled=r["left"] == 0,
                            type="primary" if k == nxt else "secondary", on_click=open_species, args=(k,),
                            help="No unchecked detections left" if r["left"] == 0 else None)

    with st.expander("Already have expert labels for these recordings?"):
        sets = labels.list_sets(cfg.store_dir, d)
        st.caption("A BEX label set's finished (closed) minutes count as known answers, and its boxes can be "
                   "examples. Make or import one in Research mode → Label.")
        if sets:
            refs = [""] + [labels.working_ref(cfg.store_dir, d, n) for n in sets]
            cur = s.get("label_ref", "")
            pick = st.selectbox("Use a label set", refs, index=refs.index(cur) if cur in refs else 0,
                                format_func=lambda x: "None" if not x else labels.describe_ref(x))
            if pick != cur:
                save(d, label_ref=pick)
                st.rerun()


def check_one(d: str, pool: pl.DataFrame, checks: pl.DataFrame, ready: pl.DataFrame) -> None:
    keys = ready["species_key"].to_list()
    if st.session_state.get("eco_chk_sp") not in keys:
        st.session_state["eco_chk_sp"] = next_species(ready) or keys[0]
    fmt = {r["species_key"]: ("✓ " if r["ready"] else "") + f"{r['name']}  ·  {r['progress']:.0%}"
           for r in ready.iter_rows(named=True)}
    top1, top2, top3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    key = top1.selectbox("Species", keys, format_func=fmt.get, key="eco_chk_sp")
    nxt = next_species(ready, key)
    top2.button("Next species ›", width="stretch", disabled=nxt is None, on_click=open_species, args=(nxt,))
    top3.button("↶ Undo last", width="stretch", disabled=checks.height == 0, on_click=undo_last, args=(d,))
    r = ready.filter(pl.col("species_key") == key).row(0, named=True)
    st.session_state[f"eco_was_ready_{key}"] = r["ready"]
    c = E.next_to_check(pool, checks, key, library(d))
    a, b = st.columns([3, 2], vertical_alignment="top")
    with b:
        st.markdown(f"**{r['name']}** · *{key}*")
        for label, have, lib, need in (("Yes", r["known_yes"], r["lib_yes"], E.MIN_YES),
                                       ("No", r["known_no"], r["lib_no"], E.MIN_NO)):
            st.progress(min(1.0, (have + lib) / need),
                        text=f"{label}: {min(have + lib, need)} of {need}" + (f" ({lib} from the library)" if lib else ""))
        st.caption(f"Each volunteer hears {E.EXAMPLES} of the Yeses as examples and is tested on the rest.")
        if r["ready"]:
            st.success("Ready for volunteers. More answers still sharpen the recall estimate.")
        st.caption("Not sure? Say so: only clear answers count. A species you can't finish comes to you in the "
                   "portal instead of going to volunteers.")
    with a:
        if c is None:
            st.info("You've seen every detection of this species.")
            return
        src = E._readable(cfg.store_dir, d, c["recording_id"], E.audio_path(cfg.store_dir, d, c["recording_id"]))
        t0, t1 = max(0.0, c["start_s"] - 1.5), c["end_s"] + 1.5
        wav, png = clip_media(str(src), t0, t1, c["start_s"], c["end_s"])
        st.image(png, width="stretch",
                 caption=f"{c['recording_id']} · {c['start_s']:.0f}–{c['end_s']:.0f} s · model score {c['score']:.2f}"
                         " · the bright part is the model's window")
        st.audio(wav, format="audio/wav", autoplay=True)
        st.markdown(f"**Is this a {r['name']}?**")
        x1, x2, x3 = st.columns(3)
        x1.button("Yes", type="primary", width="stretch", on_click=answer, args=(d, c, key, "yes"))
        x2.button("No", width="stretch", on_click=answer, args=(d, c, key, "no"))
        x3.button("Not sure", width="stretch", on_click=answer, args=(d, c, key, "unsure"),
                  help="Faint, buried, or you can't tell: only clear answers count")


def check_answers(d: str, checks: pl.DataFrame, ready: pl.DataFrame) -> None:
    if checks.height == 0:
        st.info("No answers yet.")
        return
    names = dict(zip(ready["species_key"], ready["name"]))
    c = checks.sort("at", descending=True)
    verdict = {"yes": "Yes", "no": "No", "unsure": "Not sure"}
    table = pl.DataFrame({
        "When": [str(x)[:16].replace("T", " ") for x in c["at"]],
        "Species": [names.get(k, k) for k in c["species_key"]],
        "Answer": [verdict.get(v, v) for v in c["verdict"]],
        "Recording": c["recording_id"], "At (s)": c["start_s"]})
    st.caption("Newest first. Select answers to take them back: the detections return to the queue.")
    ev = st.dataframe(table.to_pandas(), hide_index=True, width="stretch", on_select="rerun",
                      selection_mode="multi-row", key=f"eco_ans_{d}",
                      column_config={"At (s)": st.column_config.NumberColumn(format="%.0f")})
    rows = ev.selection.rows if ev and ev.selection else []
    if st.button(f"Take back {len(rows)} answer{'s' if len(rows) != 1 else ''}", disabled=not rows):
        for i in rows:
            x = c.row(i, named=True)
            E.remove_check(cfg.store_dir, d, x["recording_id"], x["start_s"], x["species_key"])
        st.session_state.pop(f"eco_ans_{d}", None)
        st.rerun()


def step_check(d: str) -> None:
    s = settings(d)
    pool, checks, ready = check_state(d, s)
    if ready.height == 0:
        st.info("No species to check: choose some in step 3.")
        nav("Species", "Recall", ok=False)
        return
    if (msg := st.session_state.pop("eco_chk_flash", None)):
        st.toast(msg)
    if st.session_state.get("eco_chk_view") not in CHECK_VIEWS:
        st.session_state["eco_chk_view"] = "Overview"
    st.segmented_control("View", CHECK_VIEWS, key="eco_chk_view", label_visibility="collapsed")
    view = st.session_state.get("eco_chk_view") or "Overview"
    if view == "Overview":
        check_overview(d, s, checks, ready)
    elif view == "Checking":
        check_one(d, pool, checks, ready)
    else:
        check_answers(d, checks, ready)
    nav("Species", "Recall", ok=checks.height >= E.MIN_CALIBRATION,
        why=f"Check at least {E.MIN_CALIBRATION} detections, across species, to estimate recall.")


# ----------------------------------------------------------------- recall ---
def step_recall(d: str) -> None:
    st.markdown("#### How much to keep")
    s = settings(d)
    pool = pool_for(d, s["run_id"])
    sp = species_for(d, s)
    checks = E.load_checks(cfg.store_dir, d)
    inc = pool.join(sp.filter(pl.col("include"), pl.col("name").str.len_chars() > 0).select("species_key"),
                    on="species_key")
    p = E.calibrate(inc, checks)
    if p is None:
        st.info(f"Check at least {E.MIN_CALIBRATION} detections (Yes and No) in step 4 first: your answers are what "
                "turn the model's scores into an estimate of recall.")
        nav("Check", "Export", ok=False)
        return
    tab = E.recall_table(inc, p)
    st.markdown("Aim high for **recall** (finding the calls that are there); the false detections that come with "
                "it get weeded out by checking (yours, or EcoLabel volunteers').")
    target = st.slider("Aim for recall of", 50, 99, int(round(100 * float(s.get("target_recall") or 0.8))), 1,
                       format="%d%%", help="Of the calls the model can find at all.") / 100
    row = E.floor_for_recall(tab, target)
    m1, m2, m3 = st.columns(3)
    m1.metric("Detections kept", f"{row['kept']:,}")
    m2.metric("False ones per 100", f"{row['false_per_100']:.0f}", help="Estimated from your answers")
    m3.metric("Real calls found", f"{row['recall']:.0%}", help=f"About {row['expected_real']:,.0f} real calls")
    chart = (alt.Chart(tab.to_pandas()).mark_line(strokeWidth=2, color=ACCENT)
             .encode(x=alt.X("recall:Q", title="Recall (share of real calls kept)", axis=alt.Axis(format="%")),
                     y=alt.Y("false_per_100:Q", title="False detections per 100 kept"),
                     tooltip=[alt.Tooltip("recall:Q", format=".0%"), alt.Tooltip("false_per_100:Q", format=".0f"),
                              alt.Tooltip("kept:Q", format=",", title="detections")]))
    dot = alt.Chart(pl.DataFrame([row]).to_pandas()).mark_point(size=120, filled=True, color=ACCENT).encode(
        x="recall:Q", y="false_per_100:Q")
    st.altair_chart(chart + dot, width="stretch")
    st.caption(f"Model score floor: {row['floor']:.3f}. Estimated from {checks.filter(pl.col('verdict') != 'unsure').height} "
               "of your answers; checking more makes it surer. Recall here is of the calls the model detects at all: "
               "calls it never noticed can't be kept.")
    if st.button("Use this", type="primary"):
        save(d, floor=row["floor"], target_recall=target)
        go("Export")
        st.rerun()
    nav("Check", "Export", ok=s.get("floor") is not None, why="Press “Use this” to set your recall target.")


# ----------------------------------------------------------------- export ---
PROTOCOL_CHOICES = {
    "Which species are there": {"mode": "presence", "block_s": 300, "confirmations": 1, "give_up_after": 5},
    "Activity every 5 minutes": {"mode": "activity", "block_s": 300, "confirmations": 1, "give_up_after": 3},
    "Activity every minute": {"mode": "activity", "block_s": 60, "confirmations": 1, "give_up_after": 3},
    "Every call": {"mode": "every_call", "block_s": 300, "confirmations": 1, "give_up_after": 0},
}


def about_form(d: str, s: dict) -> None:
    a = s["about"]
    with st.form("eco_about"):
        t = st.text_input("Project title", a["project"]["title"], placeholder="Dawn chorus, Glen Affric")
        summ = st.text_area("What it’s about", a["project"]["summary"], height=80)
        why = st.text_area("Why it matters", a["project"]["why"], height=80)
        g1, g2 = st.columns(2)
        gname = g1.text_input("Your group or organisation", a["group"]["name"])
        gkind = g2.text_input("What kind of group", a["group"]["kind"], placeholder="University research group")
        gbio = st.text_area("About you", a["group"]["bio"], height=80)
        gweb = st.text_input("Website (optional)", a["group"]["website"])
        s1, s2 = st.columns(2)
        place = s1.text_input("Place", a["site"]["name"] or cfg.site_name, placeholder="Scottish Highlands")
        hab = s2.text_input("Habitat", a["site"]["habitat"], placeholder="pine forest")
        when = st.text_input("When", a["site"]["when"], placeholder="Three mornings in May 2026, 5 to 6 am")
        desc = st.text_area("About the place", a["site"]["description"], height=80)
        photo = st.file_uploader("A photo (optional): the place, the team, the kit in the field",
                                 type=["jpg", "jpeg", "png"])
        p1, p2 = st.columns(2)
        cap = p1.text_input("Photo caption", a["photo"]["caption"])
        cred = p2.text_input("Photo credit", a["photo"]["credit"])
        ok = st.form_submit_button("Save", type="primary")
    if ok:
        about = {"project": {"title": t, "summary": summ, "why": why},
                 "group": {"name": gname, "kind": gkind, "bio": gbio, "website": gweb},
                 "site": {"name": place, "habitat": hab, "description": desc, "when": when},
                 "photo": {"caption": cap, "credit": cred}}
        if photo is not None:
            from PIL import Image, ImageOps
            img = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
            img.thumbnail((1600, 1600))
            clean = Image.new("RGB", img.size)
            clean.paste(img)                      # no EXIF, so no GPS, leaves this machine
            clean.save(E.photo_path(cfg.store_dir, d), format="JPEG", quality=85)
        save(d, about=about)
        st.toast("Saved")
        st.rerun()
    if E.photo_path(cfg.store_dir, d).exists():
        st.image(str(E.photo_path(cfg.store_dir, d)), width=320, caption="Your photo (location data removed)")


def share_form(d: str, s: dict) -> None:
    """Consent to add this project's confirmed answers to EcoLabel's shared
    library, under the terms the server publishes (their version is recorded)."""
    terms = E.library_terms(server_url())
    cur = s.get("library") or {}
    if terms is None:
        st.caption("Connect to an EcoLabel server (below) to see the library's terms. Until then nothing is shared.")
        return
    st.markdown(terms.get("summary", ""))
    with st.expander(f"Read the terms ({terms['version']})"):
        st.markdown(terms.get("text", ""))
    agreed = cur.get("share") and cur.get("terms") == terms["version"]
    a, b = st.columns([2, 3], vertical_alignment="bottom")
    on = a.checkbox("Share my confirmed clips, under these terms", value=bool(agreed), key=f"eco_share_{d}")
    credit = b.text_input("Credit them to", cur.get("credit") or s["about"]["group"]["name"], key=f"eco_credit_{d}",
                          disabled=not on)
    new = ({"share": True, "terms": terms["version"], "credit": credit.strip(),
            "agreed": cur.get("agreed") if agreed else time.strftime("%Y-%m-%dT%H:%M:%S")} if on else {"share": False})
    if {k: new.get(k) for k in ("share", "terms", "credit")} != {k: cur.get(k) for k in ("share", "terms", "credit")}:
        save(d, library=new)
    if cur.get("share") and not agreed:
        st.warning("The terms have changed since you agreed: tick again to agree to the new version.")


def export_ecolabel(d: str, s: dict, kept: pl.DataFrame, checks: pl.DataFrame) -> None:
    st.caption("Volunteers in the EcoLabel app check the detections for you, and their skill-weighted answers "
               "settle each one. Only short clips around detections leave this computer.")
    st.markdown("##### 1 · Survey protocol")
    kept_crowd = kept.filter(~pl.col("unexpected")).join(checks.select(E.KEY), on=E.KEY, how="anti")
    p = E.calibrate(kept_crowd, checks)
    rows = []
    for name, proto in PROTOCOL_CHOICES.items():
        est = E.estimate_protocol(kept_crowd, p, proto)
        rows.append({"Protocol": name, "What it tells you": E.PROTOCOLS[proto["mode"]],
                     "Detections checked": est["checks"],
                     "Volunteer answers": f"{est['answers'][0]:,}–{est['answers'][1]:,}"})
    st.dataframe(pl.DataFrame(rows).to_pandas(), hide_index=True, width="stretch")
    names = list(PROTOCOL_CHOICES)
    cur = next((n for n, v in PROTOCOL_CHOICES.items() if v == s["protocol"]), names[0])
    pick = st.radio("Protocol", names, index=names.index(cur), horizontal=True,
                    help="You can change it later in the EcoLabel portal.")
    if PROTOCOL_CHOICES[pick] != s["protocol"]:
        save(d, protocol=PROTOCOL_CHOICES[pick])
    st.caption("“Which species are there” and “Activity” stop asking about a species once it's confirmed, and "
               "if its best detections all come back No, ask you to confirm it's absent.")

    st.markdown("##### 2 · About the project" + (" ✓" if about_ok(s) else ""))
    st.caption("Shown to volunteers beside every clip. Keep places general (a region, not coordinates): it "
               "protects sensitive species.")
    with st.expander("Edit" if about_ok(s) else "Fill this in", expanded=not about_ok(s)):
        about_form(d, s)

    st.markdown("##### 3 · Share with the species library? (optional)")
    share_form(d, s)

    st.markdown("##### 4 · Build and send")
    out = Path(cfg.store_dir) / "datasets" / d / "ecolabel" / "out"
    z = out / f"{d}.ecolabel.zip"
    if st.button("Build the bundle", type="primary" if not z.exists() else "secondary", disabled=not about_ok(s),
                 help=None if about_ok(s) else "Fill in the title, summary and your group first"):
        bar = st.progress(0.0, text="Starting…")
        try:
            r = E.build_bundle(cfg, d, out, progress=lambda f, m: bar.progress(min(1.0, f), text=m))
            bar.progress(1.0, text="Zipping…")
            z = E.zip_bundle(Path(r["folder"]))
            st.session_state["eco_built"] = r
        except ValueError as err:
            st.error(str(err))
    if z.exists():
        r = st.session_state.get("eco_built")
        st.success(f"Bundle ready: {z.stat().st_size / 1e6:,.0f} MB"
                   + (f" · {r['candidates']:,} detections, {r['known_answers']} known answers, "
                      f"{r['references']} examples, {r['ready_species']} of {r['species']} species ready for "
                      "volunteers" if r else ""))
        st.caption(f"`{z}`")
    server = st.text_input("EcoLabel server", server_url())
    key = st.text_input("Your access key", type="password", help="From EcoLabel. Kept for this session only.",
                        key="eco_key")
    if st.button("Send to EcoLabel", type="primary", disabled=not (z.exists() and key)):
        prefs.put(cfg.store_dir, "ecolabel_server", server)
        bar = st.progress(0.0, text="Starting…")
        try:
            res = E.upload(z, server, key, progress=lambda f, m: bar.progress(min(1.0, f), text=m))
        except (RuntimeError, OSError) as err:
            st.error(f"{err}\n\nPress Send again to carry on where it stopped.")
        else:
            save(d, uploaded={"project": res.get("project"), "at": time.strftime("%Y-%m-%d %H:%M"),
                              "portal": res.get("portal")})
            st.balloons()
            st.success(f"Sent. It's a **draft** in the EcoLabel portal: review it there and press **Launch** when "
                       f"you're ready. [Open it in the portal]({res.get('portal')})")
    if s.get("uploaded"):
        u = s["uploaded"]
        st.caption(f"Last sent {u['at']} as **{u['project']}**. [Open in the portal]({u.get('portal')})")
    st.caption(f"From the command line: `bex export-ecolabel {d}` then `bex upload {d}.ecolabel.zip --key …`")


def export_file(d: str, kept: pl.DataFrame, checks: pl.DataFrame, names: dict[str, str]) -> None:
    st.caption("Every detection kept at your recall target, with your own answers where you gave one: for a "
               "spreadsheet, R or Python.")
    verdict = checks.select(*E.KEY, pl.col("verdict").alias("your_check"))
    out = (kept.join(verdict, on=E.KEY, how="left")
           .with_columns(common_name=pl.col("species_key").replace_strict(names, default=""))
           .select("recording_id", "start_s", "end_s", "species_key", "common_name", "score", "unexpected",
                   "your_check")
           .sort("recording_id", "start_s", "species_key"))
    st.dataframe(out.head(200).to_pandas(), hide_index=True, width="stretch", height=260)
    if st.download_button(f"Download {out.height:,} detections (CSV)", out.write_csv().encode(),
                          file_name=f"{d}-detections.csv", mime="text/csv", type="primary"):
        save(d, exported=time.strftime("%Y-%m-%d %H:%M"))


def step_export(d: str) -> None:
    s = settings(d)
    pool = pool_for(d, s["run_id"])
    sp = species_for(d, s)
    checks = E.load_checks(cfg.store_dir, d)
    kept = E.kept_candidates(pool, sp, checks, s.get("floor"))
    names = dict(zip(sp["species_key"], sp["name"]))
    eco, file = st.tabs(["🌿 Send to EcoLabel volunteers", "⬇️ Detections file"])
    with eco:
        export_ecolabel(d, s, kept, checks)
    with file:
        export_file(d, kept, checks, names)
    nav("Recall", None)


# ==================================================================== page ===
def render() -> None:
    names = ordered_datasets()
    cur = st.session_state.get("eco_dataset") or st.session_state.get("dataset")
    options = names + [NEW]
    pick = st.selectbox("Project", options, index=options.index(cur) if cur in names else len(options) - 1,
                        key="eco_pick")
    d = None if pick == NEW else pick
    if d != st.session_state.get("eco_dataset"):
        st.session_state["eco_dataset"] = d
        st.session_state["eco_chk_view"] = "Overview"
        if d:
            st.session_state["dataset"] = d
            done = status(d)
            st.session_state["eco_step"] = next((n for n in STEPS if not done[n]), "Export")
        else:
            st.session_state["eco_step"] = "Recordings"
    step = st.session_state.get("eco_step", "Recordings")
    if step not in STEPS or not d:
        step = "Recordings" if not d else "Export"
    if d and step not in ("Recordings", "Model") and not status(d)["Model"]:
        step = "Model"
    stepper(d, step)
    if (msg := st.session_state.pop("eco_flash", None)):
        st.info(msg)
    {"Recordings": lambda: step_recordings(d), "Model": lambda: step_model(d), "Species": lambda: step_species(d),
     "Check": lambda: step_check(d), "Recall": lambda: step_recall(d), "Export": lambda: step_export(d)}[step]()


render()
