"""BEX · 1 · Label — label a sample of your recordings (V1.2 L1–L8).

The first step of the flow for a dataset that has no ground truth. You label a
random sample of 60 s chunks, a chunk at a time, listening and reading the
spectrogram; each chunk is closed when every bird heard in it is labelled, and
only closed chunks count as truth. The models' only part here is a coarse
activity guide on the navigator — no species, nothing on the spectrogram —
because labels that start from a model's suggestions flatter that model.
"""
from ui.common import *  # noqa: F403 — the shared app toolkit
from ui import sidebar

NEW_SET = "➕ New label set…"
VIEWS = ["Label", "Plan and sample", "Progress", "Publish and share", "How to"]


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


# --------------------------------------------------------------------------- #
# Loading and saving
# --------------------------------------------------------------------------- #

def load(dataset: str, name: str) -> labels.LabelSet:
    return labels.load_set(cfg.store_dir, dataset, name)


def save(ls: labels.LabelSet) -> bool:
    """Write the working copy; on a clash with another tab, say so and reload."""
    try:
        labels.save_set(cfg.store_dir, ls)
    except labels.StaleWrite:
        st.session_state["lab_flash"] = ("error", "This label set was changed in another "
                                         "tab or by someone else, so your last change "
                                         "was not saved. It has been reloaded: please "
                                         "redo it.")
        return False
    get_truth.clear()
    sidebar.sample_state.clear()
    return True


def species_options(ls: labels.LabelSet, profile) -> tuple[list[str], callable]:
    """Every species any model knows, with the ones this set already uses first,
    then the plausibility profile's, then the rest — a searchable list."""
    names = get_names()
    used = ls.boxes["species_key"].value_counts(sort=True)["species_key"].to_list()
    plausible = sorted(profile.plausible_keys(2) if profile else [],
                       key=lambda k: names.get(k, k))
    rest = sorted(names, key=lambda k: names.get(k, k))
    seen, order = set(), []
    for k in [labels.UNKNOWN_BIRD, *used, *plausible, *rest]:
        if k not in seen:
            seen.add(k)
            order.append(k)

    def fmt(k: str) -> str:
        if k == labels.UNKNOWN_BIRD:
            return "Unknown bird (heard, not identified)"
        common = names.get(k)
        return f"{common} — {k}" if common and common != k else k
    return order, fmt


# --------------------------------------------------------------------------- #
# A new set
# --------------------------------------------------------------------------- #

def new_set_form(dataset: str, labeller: str) -> None:
    st.markdown("#### Start a label set")
    st.markdown(
        "A label set holds your labels for this dataset, kept apart from the "
        "recordings and from any annotations that came with them. You can have "
        "several (one per labeller, or one per survey season), publish frozen "
        "versions to cite, and choose in the sidebar which one the scoring pages use.")
    existing = labels.list_sets(cfg.store_dir, dataset)
    sources = {"": "Empty — label from scratch"}
    if labels.has_imported(cfg.store_dir, dataset):
        sources[labels.IMPORTED] = "A copy of the imported annotations (to add to them)"
    for n in existing:
        sources[n] = f"A copy of {n}"
    with st.form("new_set"):
        name = st.text_input("Name", placeholder="e.g. wood-farm-2024",
                             help="Letters, digits, - _ and . only.")
        src = st.radio("Start from", list(sources), format_func=sources.get)
        ok = st.form_submit_button("Create", type="primary")
    if ok:
        if not labeller:
            st.error("Enter your name at the top first — every label records who made it.")
            return
        try:
            labels.create_set(cfg.store_dir, dataset, name.strip(), labeller,
                              derived_from=src or None)
        except (ValueError, FileExistsError) as e:
            st.error(str(e))
            return
        # The picker is already drawn this run, so the choice is parked and
        # applied before it is drawn on the next.
        st.session_state["lab_select_pending"] = (dataset, name.strip(), "Plan and sample")
        st.rerun()


# --------------------------------------------------------------------------- #
# The activity guide (L4)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False, max_entries=8)
def get_activity(run_ids: tuple[str, ...], share: float) -> tuple[pl.DataFrame, dict]:
    """Every model's permissive bird-activity windows, merged; no species."""
    parts, bars = [], {}
    for rid in run_ids:
        w, bar = views.activity_windows(get_detections(rid), share)
        parts.append(w)
        bars[rid] = bar
    return (pl.concat(parts) if parts else pl.DataFrame(
        schema={"recording_id": pl.Utf8, "start_s": pl.Float64, "end_s": pl.Float64})), bars


def navigator(ls, rec_id: str, duration: float, chunk: tuple[float, float],
              activity: pl.DataFrame | None) -> float | None:
    """The whole recording: the activity guide as grey bars, the 60 s chunks
    coloured by state, the current chunk outlined. A click returns a time."""
    n_bins = 180
    bins = (views.activity_bins(activity, rec_id, duration, n_bins)
            if activity is not None else
            views.activity_bins(pl.DataFrame(schema={"recording_id": pl.Utf8,
                                                     "start_s": pl.Float64,
                                                     "end_s": pl.Float64}),
                                rec_id, duration, n_bins))
    starts = labels.chunk_starts(duration)
    closed = set(ls.chunks.filter((pl.col("recording_id") == rec_id)
                                  & (pl.col("state") == labels.CLOSED))["start_s"].to_list())
    sampled = set(ls.sample.filter(pl.col("recording_id") == rec_id)["start_s"].to_list())
    ch = pd.DataFrame([{"start": float(s), "end": labels.chunk_end(float(s), duration),
                        "state": ("closed" if s in closed else
                                  "in the sample" if s in sampled else "not sampled"),
                        "label": f"{mmss(s)}–{mmss(labels.chunk_end(float(s), duration))}"}
                       for s in starts])
    x = alt.X("start:Q", title=None, scale=alt.Scale(domain=[0, duration], nice=False),
              axis=alt.Axis(labelExpr="floor(datum.value/60) + ':' + "
                                      "(datum.value%60 < 10 ? '0' : '') + datum.value%60",
                            tickCount=12))
    pick = alt.selection_point(fields=["start"], name="chunk", on="click", empty=False)
    state_colour = alt.Scale(domain=["closed", "in the sample", "not sampled"],
                             range=["#3e8e5e", "#2a78d6", "#d8d7d2"])
    strip = alt.Chart(ch).mark_rect(height=14, stroke="#ffffff", strokeWidth=1).encode(
        x=x, x2="end:Q", color=alt.Color("state:N", scale=state_colour,
                                         legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("label:N", title="chunk"), alt.Tooltip("state:N")],
    ).add_params(pick).properties(height=18)
    cur = alt.Chart(pd.DataFrame({"start": [chunk[0]], "end": [chunk[1]]})).mark_rect(
        fill=None, stroke="#1d1b17", strokeWidth=2.2).encode(x="start:Q", x2="end:Q")
    act = alt.Chart(bins.rename({"bin_start_s": "start", "bin_end_s": "end"}).to_pandas()
                    ).mark_rect(color="#8a8880").encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[0, duration], nice=False), axis=None),
        x2="end:Q",
        y=alt.Y("activity:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=None),
        tooltip=[alt.Tooltip("activity:Q", title="share with possible birdsong",
                             format=".0%")],
    ).properties(height=46, title=alt.Title(
        "possible birdsong (permissive guide, no species)" if activity is not None
        else "no activity guide (no model has scored this recording)",
        fontSize=10, anchor="start"))
    cur_top = alt.Chart(pd.DataFrame({"start": [chunk[0]], "end": [chunk[1]]})).mark_rect(
        fill="#2a78d6", fillOpacity=0.10, stroke="#2a78d6").encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[0, duration], nice=False)), x2="end:Q")
    chart = alt.vconcat(alt.layer(act, cur_top), alt.layer(strip, cur), spacing=4
                        ).configure_view(strokeWidth=0)
    ev = st.altair_chart(chart, width="stretch", on_select="rerun", key=f"lab_nav_{rec_id}")
    try:
        sel = (ev or {}).get("selection", {}).get("chunk") or []
    except AttributeError:
        sel = []
    if sel:
        t = float(sel[0]["start"])
        if st.session_state.get("_lab_nav_seen") != (rec_id, t):
            st.session_state["_lab_nav_seen"] = (rec_id, t)
            return t
    return None


# --------------------------------------------------------------------------- #
# The labelling view (L5)
# --------------------------------------------------------------------------- #

#: Seconds of the neighbouring minutes shown either side of the chunk, shaded:
#: enough to see a call that runs over the boundary, and to box it.
MARGIN = 5.0


def chunk_figure(mel, meta, t0, t1, chunk: tuple[float, float], boxes: pl.DataFrame,
                 pending: dict | None, corner: dict | None, fmt,
                 expert: pl.DataFrame | None = None):
    """The chunk's spectrogram, with a shaded margin of its neighbours, its boxes,
    the first corner of a box being drawn and the box waiting for a species.
    Returns the PNG and the axes box (figure fractions) for turning a click back
    into time and frequency."""
    fig, ax = plt.subplots(figsize=(14, 4.4))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.12)
    n_rows = mel.shape[0]
    s = slice_mel(mel, meta, t0, t1)
    vmin, vmax = display_range(s)
    ax.imshow(s, origin="lower", aspect="auto", cmap="magma",
              extent=[t0, t1, 0, n_rows], vmin=vmin, vmax=vmax)
    khz = [k for k in (0.5, 1, 2, 4, 8, 12, 16) if k * 1000 < min(meta["fmax"], meta["sr"] / 2)]
    ax.set_yticks([views.hz_to_mel_row(k * 1000, meta) for k in khz],
                  [f"{k:g}" for k in khz])
    ax.set_ylabel("kHz")
    ax.set_ylim(0, n_rows)
    ax.set_xlim(t0, t1)
    ticks = np.arange(np.ceil(t0 / 5) * 5, t1 + 1e-6, 5)
    ax.set_xticks(ticks, [mmss(t) for t in ticks])
    # The neighbouring minutes: visible and clickable, but plainly not this chunk.
    for a, b in ((t0, chunk[0]), (chunk[1], t1)):
        if b > a:
            ax.axvspan(a, b, facecolor="#f2f1ec", alpha=0.35, zorder=2)
    for x in chunk:
        if t0 < x < t1:
            ax.axvline(x, color="#f2f1ec", linewidth=1.2, linestyle=(0, (3, 2)), zorder=3)
    palette = views.CATEGORICAL
    keys = list(dict.fromkeys(boxes["species_key"].to_list()))
    for b in boxes.iter_rows(named=True):
        colour = ("#ffffff" if b["species_key"] == labels.UNKNOWN_BIRD
                  else palette[keys.index(b["species_key"]) % len(palette)])
        lo = views.hz_to_mel_row(b["low_hz"], meta) if np.isfinite(b["low_hz"]) else 0
        hi = views.hz_to_mel_row(b["high_hz"], meta) if np.isfinite(b["high_hz"]) else n_rows
        x0, x1 = max(b["start_s"], t0), min(b["end_s"], t1)
        ax.add_patch(plt.Rectangle((x0, lo), x1 - x0, hi - lo, fill=False,
                                   edgecolor=colour, linewidth=1.8, zorder=4))
        name = fmt(b["species_key"]).split(" — ")[0]
        ax.text(x0 + 0.1, min(hi + 1.5, n_rows - 6), name, fontsize=7.5, color=colour,
                va="bottom", ha="left", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#12100f", alpha=0.75,
                          edgecolor="none"))
    # Practice: once the minute is closed, the experts' boxes, dashed green — each
    # species named once per run of calls, not on every box.
    last_named: dict[str, float] = {}
    for b in (expert.sort("start_s").iter_rows(named=True) if expert is not None else []):
        lo = views.hz_to_mel_row(b["low_hz"], meta) if np.isfinite(b["low_hz"]) else 0
        hi = views.hz_to_mel_row(b["high_hz"], meta) if np.isfinite(b["high_hz"]) else n_rows
        x0, x1 = max(b["start_s"], t0), min(b["end_s"], t1)
        if x1 <= x0:
            continue
        ax.add_patch(plt.Rectangle((x0, lo), x1 - x0, hi - lo, fill=False,
                                   edgecolor="#3ecf8e", linewidth=1.6,
                                   linestyle=(0, (4, 2)), zorder=5))
        if b["start_s"] - last_named.get(b["species_key"], -1e9) > 0.25 * (t1 - t0):
            last_named[b["species_key"]] = b["start_s"]
            ax.text(x0 + 0.1, max(lo - 1.5, 2),
                    "expert: " + fmt(b["species_key"]).split(" — ")[0],
                    fontsize=7, color="#3ecf8e", va="top", ha="left", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#12100f", alpha=0.75,
                              edgecolor="none"))
    if pending:
        lo, hi = (views.hz_to_mel_row(pending["low_hz"], meta),
                  views.hz_to_mel_row(pending["high_hz"], meta))
        x0, x1 = max(pending["start_s"], t0), min(pending["end_s"], t1)
        if x1 > x0:
            ax.add_patch(plt.Rectangle((x0, lo), x1 - x0, hi - lo, fill=True,
                                       facecolor="#ffffff22", edgecolor="#ffffff",
                                       linewidth=1.6, linestyle=(0, (4, 2)), zorder=6))
            # The ✕ that discards it, at the top-right corner (clicked: see
            # discard_hit).
            xt, yt = discard_point(pending, t0, t1, n_rows, meta)
            ax.plot(xt, yt, marker="o", markersize=15, markerfacecolor="#d0021b",
                    markeredgecolor="#ffffff", markeredgewidth=1.2, zorder=8,
                    clip_on=False)
            ax.plot(xt, yt, marker="x", markersize=7, markeredgecolor="#ffffff",
                    markeredgewidth=2.0, zorder=9, clip_on=False)
    if corner and t0 <= corner["t"] <= t1:
        row = views.hz_to_mel_row(corner["hz"], meta)
        ax.axvline(corner["t"], color="#ffffff", linewidth=0.8, linestyle=(0, (2, 3)),
                   zorder=6)
        ax.axhline(row, color="#ffffff", linewidth=0.8, linestyle=(0, (2, 3)), zorder=6)
        ax.plot(corner["t"], row, marker="+", markersize=30, markeredgewidth=2.6,
                color="#ffffff", zorder=7)
        ax.plot(corner["t"], row, marker="o", markersize=7, markerfacecolor="none",
                markeredgecolor="#ffffff", markeredgewidth=1.4, zorder=7)
    pos = ax.get_position()
    x_ticks = [float(t) for t in ax.get_xticks()]
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf), pos, x_ticks


def discard_point(pending: dict, t0: float, t1: float, n_rows: int,
                  meta) -> tuple[float, float]:
    """Where the pending box's ✕ sits, in (time, mel row): its top-right corner,
    pulled inside the view when the box runs off an edge."""
    span = t1 - t0
    xt = min(pending["end_s"], t1 - 0.012 * span)
    yt = min(views.hz_to_mel_row(pending["high_hz"], meta), n_rows * 0.96)
    return xt, yt


def discard_hit(click: dict, pending: dict, pos, t0: float, t1: float, n_rows: int,
                meta, radius_px: float = 14.0) -> bool:
    """Did this click land on the pending box's ✕?"""
    w, h = click.get("width"), click.get("height")
    if not w or not h:
        return False
    xt, yt = discard_point(pending, t0, t1, n_rows, meta)
    px = (pos.x0 + (xt - t0) / (t1 - t0) * (pos.x1 - pos.x0)) * w
    py = (1 - (pos.y0 + yt / n_rows * (pos.y1 - pos.y0))) * h
    return (click["x"] - px) ** 2 + (click["y"] - py) ** 2 <= radius_px ** 2


def click_to_point(click: dict, pos, t0: float, t1: float, n_rows: int,
                   meta) -> tuple[float, float] | None:
    """Pixel click -> (time in the recording, frequency in Hz); None outside the
    spectrogram itself (on the axis labels, say)."""
    w, h = click.get("width"), click.get("height")
    if not w or not h:
        return None
    fx = (click["x"] / w - pos.x0) / (pos.x1 - pos.x0)
    fy = ((1 - click["y"] / h) - pos.y0) / (pos.y1 - pos.y0)
    if not (0 <= fx <= 1 and 0 <= fy <= 1):
        return None
    return (round(t0 + fx * (t1 - t0), 2),
            round(views.mel_row_to_hz(fy * (n_rows - 1), meta), -1))


def mmss1(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


@st.cache_resource(show_spinner=False, max_entries=6)
def get_matrix(run_id: str, recording_id: str):
    """A recording's full score matrix for one run (tens of MB for Perch: held
    as a resource, not copied per call)."""
    return store.read_scores(cfg.store_dir, run_id, recording_id)


def use_species(key: str) -> None:
    st.session_state["lab_species"] = key


def suggestions(ctx, rec_id: str, pending: dict, box_key: tuple, opened: bool,
                fmt) -> None:
    """The models' suggestions for a box the labeller has already drawn — only
    on request, and the box remembers that they were seen (`assisted`), so the
    benchmark can say how many of its labels came with help."""
    runs = {ctx.run_label[m.run_id]: m.run_id for m in ctx.included
            if rec_id in set(store.list_scores(cfg.store_dir, m.run_id))}
    if not runs:
        return
    if not opened:
        if st.button("💡 Suggest birds", key="lab_sugg_open",
                     help="What the models heard inside your box, best first, with "
                          "reference recordings to compare. The box is then marked "
                          "as labelled with suggestions."):
            st.session_state["lab_sugg_for"] = box_key
            st.rerun()
        return
    mats = {label: get_matrix(rid, rec_id) for label, rid in runs.items()}
    sug = views.box_suggestions(mats, pending["start_s"], pending["end_s"], top=8)
    st.caption("**What the models heard inside your box**, best first (models "
               "agreeing count for more). These are guesses and often wrong: listen "
               "to the reference recordings and compare with your bird before you "
               "choose. This box will be marked *labelled with suggestions*.")
    profile = ctx.profile
    for i, r in enumerate(sug.iter_rows(named=True), start=1):
        key = r["species_key"]
        ranks = " · ".join(f"{label} #{r[f'{label} rank']}" for label in runs
                           if r.get(f"{label} rank") is not None)
        odd = profile is not None and profile.tier_of(key) >= 3
        a, b, c = st.columns([3.2, 1.1, 0.7], vertical_alignment="center")
        a.markdown(f"**{i}. {fmt(key).split(' — ')[0]}** <span style='opacity:0.65'>"
                   f"*{key}*</span><br><span style='font-size:0.8rem; opacity:0.7'>"
                   f"{ranks}{' · ⚠ not expected here' if odd else ''}</span>",
                   unsafe_allow_html=True)
        b.markdown(f"[🔊 Xeno-Canto]({xc_species_url(key)})")
        c.button("Use", key=f"lab_use_{i}", on_click=use_species, args=(key,),
                 width="stretch")


def hear(rec_id: str, start_s: float, end_s: float) -> None:
    st.session_state["lab_focus"] = (rec_id, start_s, end_s)


def practice_feedback(ls, reference: pl.DataFrame, rec_id: str, start: float, end: float,
                      fmt) -> None:
    """After a practice minute is closed: your boxes against the experts'."""
    fb = labels.compare_to_reference(ls.boxes, reference, rec_id, start, end)
    name = lambda k: fmt(k).split(" — ")[0]     # noqa: E731
    if not fb["n_calls"] and not fb["extra"]:
        st.success("The experts heard no birds in this minute either. ✓")
        return
    n_sp, n_found = len(fb["expert_species"]), len(fb["found_species"])
    all_sp = n_found == n_sp and n_sp > 0
    all_calls = fb["n_found"] == fb["n_calls"] and fb["n_calls"] > 0
    head = (f"The experts labelled **{n_sp} species** in **{fb['n_calls']} calls** here. "
            f"You found **{n_found} of {n_sp}** species{' ✓' if all_sp else ''}, and "
            f"boxed **{fb['n_found']} of {fb['n_calls']}** calls{' ✓' if all_calls else ''}.")
    (st.success if all_sp and all_calls else st.info)(head)
    if all_sp and not all_calls:
        st.caption("Every call counts, not just every species: models are scored window "
                   "by window, so a call without a box reads as *no bird here* and makes "
                   "a model that heard it look wrong. The unboxed calls are below.")
    st.toggle("Show the experts' boxes (dashed green)", value=True, key="lab_show_expert")
    marks = {"found": "✓", "named differently": "≈", "missed": "✗"}
    for i, c in enumerate(fb["calls"]):
        a, b = st.columns([4, 1], vertical_alignment="center")
        what = name(c["species_key"])
        tail = ("you found it" if c["outcome"] == "found" else
                f"you named it {name(c['yours'])}" if c["outcome"] == "named differently"
                else "missed")
        a.markdown(f"{marks[c['outcome']]} **{what}** · {mmss1(c['start_s'])}–"
                   f"{mmss1(c['end_s'])} · {tail}")
        b.button("▶ hear it", key=f"lab_hear_{i}", on_click=hear,
                 args=(rec_id, c["start_s"], c["end_s"]), width="stretch",
                 help="Marks it on the player's timeline: press ▶ box to listen.")
    if fb["extra"]:
        st.caption("Your boxes the experts did not label here (a bird they did not "
                   "mark, or one named differently): "
                   + ", ".join(f"{name(y['species_key'])} at {mmss1(y['start_s'])}"
                               for y in fb["extra"]))
    nxt = labels.next_chunk(ls)
    if nxt and st.button("Next practice minute ▶", type="primary", key="lab_prac_next"):
        go_to(nxt["recording_id"], nxt["start_s"])
        st.rerun()


def go_to(rec_id: str, start_s: float) -> None:
    """Open a chunk. A first corner already placed is kept, so a box can be
    finished in the next minute; a box waiting for its species is not."""
    st.session_state.update(lab_rec=rec_id, lab_start=float(start_s), lab_pending=None)


def label_view(ctx, ls: labels.LabelSet, labeller: str, activity, share: float) -> None:
    dataset, recordings = ctx.dataset, ctx.recordings
    practice = labels.is_practice(cfg.store_dir, dataset, ls.name)
    reference = get_dataset(dataset)[1] if practice else None
    rec_ids = recordings["recording_id"].to_list()
    duration_of = dict(zip(recordings["recording_id"], recordings["duration_s"]))
    path_of = dict(zip(recordings["recording_id"], recordings["path"]))

    # ---- which chunk ------------------------------------------------------- #
    if st.session_state.get("lab_rec") not in duration_of:
        nxt = labels.next_chunk(ls)
        if nxt:
            go_to(nxt["recording_id"], nxt["start_s"])
        else:
            go_to(rec_ids[0], 0.0)
    rec_id, start = st.session_state["lab_rec"], float(st.session_state["lab_start"])
    duration = duration_of[rec_id]
    end = labels.chunk_end(start, duration)
    in_sample = ls.sample.filter((pl.col("recording_id") == rec_id)
                                 & ((pl.col("start_s") - start).abs() < 1e-6))
    order = int(in_sample["order"][0]) if len(in_sample) else None
    state = ls.chunk_state(rec_id, start)

    nxt = labels.next_chunk(ls)
    c0, c1, c2, c3 = st.columns([0.9, 1.3, 1.3, 2.1], vertical_alignment="bottom")
    # The neighbouring minutes of this recording: where a call that runs over
    # the boundary is finished (its first corner stays put).
    p1, p2 = c0.columns(2)
    if p1.button("◀", key="lab_prev_min", disabled=start <= 0, width="stretch",
                 help="The previous minute of this recording"):
        go_to(rec_id, labels.chunk_of(start - 1))
        st.rerun()
    if p2.button("▶", key="lab_next_min", disabled=end >= duration - 1e-6,
                 width="stretch", help="The next minute of this recording — to "
                 "finish a box that runs over the boundary, say"):
        go_to(rec_id, end)
        st.rerun()
    if c1.button("Next chunk in the sample ▶", type="primary", disabled=nxt is None,
                 width="stretch",
                 help="The first chunk of the random sample not yet closed."):
        go_to(nxt["recording_id"], nxt["start_s"])
        st.rerun()
    with c2.popover("Choose a chunk by hand", width="stretch"):
        st.caption("Chunks you choose yourself count as truth, but they are marked "
                   "*chosen by hand* and the headline numbers come from the random "
                   "sample — choosing where to label is how a benchmark gets biased.")
        r = st.selectbox("Recording", rec_ids, index=rec_ids.index(rec_id), key="lab_pick_rec")
        cs = labels.chunk_starts(duration_of[r])
        s = st.selectbox("Chunk", list(cs), format_func=lambda v: f"{mmss(v)}–"
                         f"{mmss(labels.chunk_end(v, duration_of[r]))}", key="lab_pick_start")
        if st.button("Go", key="lab_pick_go"):
            go_to(r, s)
            st.rerun()
    done = ls.sample.join(ls.chunks.filter(pl.col("state") == labels.CLOSED),
                          on=["recording_id", "start_s"], how="semi").height
    c3.markdown(
        f"<div style='text-align:right'><b>{rec_id}</b> · {mmss(start)}–{mmss(end)} · "
        + (f"sample chunk <b>{order + 1}</b> of {len(ls.sample)} "
           f"({in_sample['stratum'][0]})" if order is not None else "<b>chosen by hand</b>")
        + f" · <b>{state}</b><br><span style='opacity:0.7'>{done} of {len(ls.sample)} "
          "sampled chunks closed</span></div>", unsafe_allow_html=True)

    # ---- the navigator, with the activity guide only ------------------------ #
    clicked = navigator(ls, rec_id, duration, (start, end), activity)
    if clicked is not None:
        go_to(rec_id, labels.chunk_of(clicked + 0.01))
        st.rerun()
    st.caption(
        f"**Grey bars**: where BirdNET or Perch hear *something* bird-like — the "
        f"{share:.0%} of each model's windows it scores highest, with **no species "
        "named**. It is a deliberately loose guide to where the recording is busy or "
        "quiet, and it misses birds: it does not choose what you label and it does "
        "not appear on the spectrogram. Click a chunk in the strip to open it.")

    # ---- the spectrogram ---------------------------------------------------- #
    try:
        mel, meta = get_mel(dataset, rec_id)
    except FileNotFoundError:
        st.warning(f"No spectrogram cached for {rec_id} yet.")
        if st.button("Build it now", key="lab_mel"):
            build_mel(dataset, recordings.filter(pl.col("recording_id") == rec_id))
            st.rerun()
        return

    order_keys, fmt = species_options(ls, ctx.profile)
    t0, t1 = max(0.0, start - MARGIN), min(duration, end + MARGIN)
    shown = ls.boxes.filter((pl.col("recording_id") == rec_id) & (pl.col("end_s") > t0)
                            & (pl.col("start_s") < t1)).sort("start_s")
    pending = st.session_state.get("lab_pending")
    corner = st.session_state.get("lab_corner")
    if corner and corner["rec"] != rec_id:
        corner = None
    editable = state == labels.OPEN

    # Two clicks make a box: the first drops a corner marker, the second the
    # opposite corner. Nothing happens silently mid-drag, and between the two
    # you can listen, step to the next minute, or cancel.
    if editable and not pending:
        if corner is None:
            st.markdown("**Click one corner of a call**, then the opposite corner. The "
                        "shaded strips either side are the neighbouring minutes: a box "
                        "may run into them.")
        else:
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            c1.markdown(f"First corner at **{mmss1(corner['t'])}**, "
                        f"**{corner['hz'] / 1000:.1f} kHz** — now click the opposite "
                        "corner. For a call that runs on, step to the next minute "
                        "(▶ above) and click it there.")
            if c2.button("Cancel", key="lab_corner_cancel", width="stretch"):
                st.session_state["lab_corner"] = None
                st.rerun()
    show_expert = (practice and not editable and reference is not None
                   and st.session_state.get("lab_show_expert", True))
    expert = (reference.filter((pl.col("recording_id") == rec_id) & (pl.col("end_s") > t0)
                               & (pl.col("start_s") < t1)) if show_expert else None)
    img, pos, x_ticks = chunk_figure(mel, meta, t0, t1, (start, end), shown, pending,
                                     corner, fmt, expert)
    click = streamlit_image_coordinates(img, width="stretch",
                                        key=f"lab_img_{rec_id}_{start:g}",
                                        cursor="crosshair" if editable else "default")
    # The component hands back its last click on every rerun, so each click is
    # used at most once — and every click is marked as seen, even one ignored
    # because a box is waiting for its species or the minute is closed.
    # Otherwise a stray or double click made then would come back as a new
    # first corner after *Discard* (found 2026-10-08).
    stamp = (click or {}).get("unix_time")
    seen = st.session_state.setdefault("_lab_clicks_seen", set())
    fresh = bool(click) and stamp is not None and stamp not in seen
    if fresh:
        seen.add(stamp)
    if fresh and editable and pending and discard_hit(click, pending, pos, t0, t1,
                                                      mel.shape[0], meta):
        st.session_state["lab_pending"] = None      # the box's ✕: same as Discard
        st.rerun()
    if fresh and editable and not pending:
        pt = click_to_point(click, pos, t0, t1, mel.shape[0], meta)
        if pt is not None:
            if corner is None:
                st.session_state["lab_corner"] = {"rec": rec_id, "t": pt[0], "hz": pt[1]}
            else:
                a, b = sorted((corner["t"], pt[0]))
                lo, hi = sorted((corner["hz"], pt[1]))
                if b - a >= 0.05:
                    st.session_state["lab_pending"] = {"start_s": a, "end_s": b,
                                                       "low_hz": lo, "high_hz": hi}
                    st.session_state["lab_corner"] = None
            st.rerun()

    try:
        audio_root = ingest.dataset_audio_root(cfg.store_dir, dataset)
        src = ingest.resolve_audio(cfg.store_dir, dataset, audio_root, rec_id, path_of[rec_id])
        y, sr, clip_lo, _ = load_clip(src, t0, t1, pad_s=0.0)
        # The box to hear on its own: the one just drawn, else the one selected
        # in the table below — marked on the timeline, with ▶ box to play it.
        focus = None
        heard = st.session_state.get("lab_focus")
        if pending:
            focus = (pending["start_s"], pending["end_s"])
        elif heard and heard[0] == rec_id and heard[1] < t1 and heard[2] > t0:
            focus = (heard[1], heard[2])
        else:
            sel = st.session_state.get(f"lab_boxes_{rec_id}_{start:g}")
            rows = (sel or {}).get("selection", {}).get("rows", []) if sel else []
            listed = ls.chunk_boxes(rec_id, start, end)
            if rows and rows[0] < len(listed):
                r = listed.row(rows[0], named=True)
                focus = (r["start_s"], r["end_s"])
        if focus:
            focus = (max(focus[0], t0), min(focus[1], t1))
        st.iframe(player_html(encode_mp3(y, sr), t0, t1, clip_lo, focus,
                              pos.x0, pos.x1, x_ticks, window_label="box", clock=True),
                  height=72)
        if not pending and not focus and len(ls.chunk_boxes(rec_id, start, end)):
            st.caption("Select a box in the table below to mark it on the timeline and "
                       "play it on its own.")
    except FileNotFoundError as e:
        st.info(f"Audio unavailable ({e}).")

    # ---- name the box ------------------------------------------------------------ #
    left, right = st.columns([1.35, 1])
    with left:
        if not editable and practice and reference is not None:
            practice_feedback(ls, reference, rec_id, start, end, fmt)
        elif not editable:
            closer = ls.chunks.filter((pl.col("recording_id") == rec_id)
                                      & ((pl.col("start_s") - start).abs() < 1e-6))
            st.info(f"This chunk is **closed** — signed off by "
                    f"{closer['closed_by'][0] or 'someone'}. Reopen it to change its boxes.")
        elif pending:
            crosses = len(labels.chunks_touched(pending["start_s"], pending["end_s"])) > 1
            st.markdown(f"**New box** · {mmss1(pending['start_s'])}–"
                        f"{mmss1(pending['end_s'])} · {pending['low_hz'] / 1000:.1f}–"
                        f"{pending['high_hz'] / 1000:.1f} kHz"
                        + (" · *crosses into the next minute*" if crosses else ""))
            box_key = (rec_id, pending["start_s"], pending["end_s"],
                       pending["low_hz"], pending["high_hz"])
            assisted = st.session_state.get("lab_sugg_for") == box_key
            last = st.session_state.get("lab_last_species")
            if st.session_state.get("lab_species") not in order_keys:
                st.session_state["lab_species"] = last if last in order_keys else None
            sp = st.selectbox("Species", order_keys, format_func=fmt,
                              placeholder="Type to search…", key="lab_species")
            a, b = st.columns(2)
            if a.button("Add box", type="primary", disabled=sp is None, width="stretch"):
                try:
                    labels.add_box(ls, rec_id, pending["start_s"], pending["end_s"],
                                   pending["low_hz"], pending["high_hz"], sp, labeller,
                                   assisted=assisted)
                except (ValueError, labels.ClosedChunk) as e:
                    st.error(str(e))
                else:
                    if save(ls):
                        st.session_state.update(lab_pending=None, lab_last_species=sp)
                    st.rerun()
            if b.button("Discard", width="stretch"):
                st.session_state["lab_pending"] = None
                st.rerun()
            suggestions(ctx, rec_id, pending, box_key, assisted, fmt)
        else:
            st.caption("Name each box with its species. A bird you hear but cannot "
                       "identify is *Unknown bird*.")

    # ---- this chunk's boxes ---------------------------------------------------- #
    with right:
        mine = ls.chunk_boxes(rec_id, start, end)
        st.markdown(f"**Boxes in this chunk** ({len(mine)})")
        if len(mine) and ((mine["start_s"] < start) | (mine["end_s"] > end)).any():
            st.caption("Boxes marked ↔ cross into a neighbouring minute. They count as "
                       "truth in every closed minute they touch, and can only be changed "
                       "while all of those minutes are open.")
        if mine.is_empty():
            st.caption("None yet. A chunk closed with no boxes means *listened, no birds*.")
        else:
            table = mine.select(
                (pl.col("species_key").map_elements(lambda k: fmt(k).split(" — ")[0],
                                                    return_dtype=pl.Utf8)
                 + pl.when((pl.col("start_s") < start) | (pl.col("end_s") > end))
                 .then(pl.lit(" ↔")).otherwise(pl.lit(""))).alias("species"),
                pl.col("start_s").map_elements(mmss1, return_dtype=pl.Utf8).alias("from"),
                pl.col("end_s").map_elements(mmss1, return_dtype=pl.Utf8).alias("to"),
                (pl.col("low_hz") / 1000).round(1).alias("low kHz"),
                (pl.col("high_hz") / 1000).round(1).alias("high kHz"),
                "labeller")
            ev = st.dataframe(table, hide_index=True, width="stretch",
                              on_select="rerun", selection_mode="single-row",
                              key=f"lab_boxes_{rec_id}_{start:g}")
            rows = (ev or {}).get("selection", {}).get("rows", []) if editable else []
            if rows:
                b = mine.row(rows[0], named=True)
                with st.container(border=True):
                    new_sp = st.selectbox("Change species to", order_keys, format_func=fmt,
                                          index=order_keys.index(b["species_key"])
                                          if b["species_key"] in order_keys else None,
                                          key=f"lab_edit_sp_{b['box_id']}")
                    e1, e2 = st.columns(2)
                    try:
                        if e1.button("Save", key=f"lab_edit_{b['box_id']}", width="stretch"):
                            labels.update_box(ls, b["box_id"], labeller, species_key=new_sp)
                            save(ls)
                            st.rerun()
                        if e2.button("Delete box", key=f"lab_del_{b['box_id']}",
                                     width="stretch"):
                            labels.delete_box(ls, b["box_id"])
                            save(ls)
                            st.rerun()
                    except labels.ClosedChunk as e:
                        st.error(str(e))

    # ---- close / reopen ------------------------------------------------------------ #
    st.divider()
    if editable:
        heard_all = st.checkbox(
            "I listened to the whole minute, including stretches the guide left grey, "
            "and every bird I heard has a box.", key=f"lab_ack_{rec_id}_{start:g}")
        if st.button("Close this chunk ✓", type="primary", disabled=not heard_all,
                     help="Signs the chunk off as ground truth. You can reopen it later."):
            if not labeller:
                st.error("Enter your name at the top first.")
                return
            labels.close_chunk(ls, rec_id, start, end, labeller)
            if save(ls):
                nxt = labels.next_chunk(ls)
                # In practice, stay to read the feedback; otherwise move on.
                if order is not None and nxt and not practice:
                    go_to(nxt["recording_id"], nxt["start_s"])
            st.rerun()
    else:
        with st.popover("Reopen this chunk"):
            reason = st.text_input("Why? (optional)", key=f"lab_reason_{rec_id}_{start:g}")
            if st.button("Reopen", key=f"lab_reopen_{rec_id}_{start:g}"):
                labels.reopen_chunk(ls, rec_id, start, reason)
                save(ls)
                st.rerun()


def build_mel(dataset: str, recs: pl.DataFrame) -> None:
    from bex import repair
    audio_root = ingest.dataset_audio_root(cfg.store_dir, dataset)
    with st.spinner("building the spectrogram (repairing the file first if it will "
                    "not decode)…"):
        for rid, rel in recs.select("recording_id", "path").iter_rows():
            repair.build_mel(cfg.cache_dir, cfg.store_dir, dataset, rid, rel, audio_root)
    get_mel.clear()


# --------------------------------------------------------------------------- #
# Plan and sample (L2, L3)
# --------------------------------------------------------------------------- #

#: Minutes to suggest for a first sample, and for each top-up after it.
START_MINUTES = 5


def add_to_sample(ls: labels.LabelSet, recordings: pl.DataFrame, labeller: str,
                  minutes: int = 0, whole: int = 0) -> None:
    """Draw more of the random sample. The seed is the sample's current size,
    so every draw is reproducible without anyone having to choose one."""
    seed = len(ls.sample)
    ls.sample = labels.draw_sample(recordings, minutes, seed=seed,
                                   whole_recordings=whole, existing=ls.sample)
    ls.meta.setdefault("draws", []).append({"chunks": minutes, "whole": whole,
                                            "seed": seed, "by": labeller})
    (labels.set_dir(cfg.store_dir, ls.dataset, ls.name) / "meta.json").write_text(
        json.dumps(ls.meta, indent=2) + "\n")
    save(ls)


def plan_view(ctx, ls: labels.LabelSet, labeller: str) -> None:
    recordings = ctx.recordings
    st.markdown("#### What to label")
    st.markdown(
        "Say how many minutes you want to label. BEX picks them **at random**, "
        "balanced across sites, times of day and recordings, and hands them to you "
        "one at a time on the *Label* view. Picking at random matters: labelling only "
        "where the models fired would leave out the birds they miss.")
    first = ls.sample.is_empty()
    a, b = st.columns([1, 2], vertical_alignment="bottom")
    n = a.number_input("Minutes to label" if first else "Minutes to add", 1, 600,
                       START_MINUTES, step=1, key="lab_minutes",
                       help=f"{START_MINUTES} minutes is a good start: enough to get a "
                            "feel for the recordings. Add more whenever you like — the "
                            "sample stays balanced however far you get.")
    if b.button(f"Pick {int(n)} minute{'s' * (n != 1)}" if first
                else f"Add {int(n)} more minute{'s' * (n != 1)}", type="primary",
                disabled=ls.read_only):
        add_to_sample(ls, recordings, labeller, minutes=int(n))
        # The view switch is already drawn this run: park the change for the next.
        st.session_state["lab_select_pending"] = (ls.dataset, ls.name, "Label")
        st.rerun()

    if not ls.sample.is_empty():
        closed = ls.chunks.filter(pl.col("state") == labels.CLOSED).select(
            "recording_id", "start_s", closed=pl.lit(True))
        view = (ls.sample.join(closed, on=["recording_id", "start_s"], how="left")
                .with_columns(closed=pl.col("closed").fill_null(False)))
        done = int(view["closed"].sum())
        st.progress(done / len(view), text=f"{done} of {len(view)} minutes labelled")
        with st.expander("The minutes picked"):
            st.dataframe(view.select(
                (pl.col("order") + 1).alias("#"), "recording_id",
                pl.col("start_s").map_elements(mmss, return_dtype=pl.Utf8).alias("from"),
                pl.col("stratum").alias("site · time of day"), "closed"),
                hide_index=True, width="stretch", height=260)

    projection(ctx, ls)


def projection(ctx, ls: labels.LabelSet) -> None:
    """L2 step 3: measure interval widths on what is labelled so far, against a
    model, and project how much more labelling the target needs."""
    st.markdown("#### How much more?")
    if not ctx.included:
        st.caption("Run a model over this dataset first: the projection measures how "
                   "much a model's scores vary between your labelled recordings.")
        return
    ref = labels.working_ref(cfg.store_dir, ls.dataset, ls.name)
    t = get_truth(ls.dataset, ref)
    if t is None or len(t.recordings) < 2:
        st.caption("Label minutes in at least two recordings and BEX will "
                   "measure how much your recordings vary, and project from that.")
        return
    labels_of = {m.run_id: ctx.run_label[m.run_id] for m in ctx.included}
    a, b = st.columns(2)
    rid = a.selectbox("Measure with", list(labels_of), format_func=labels_of.get,
                      key="lab_proj_run")
    target = b.select_slider("Target: 95% interval of ±", [0.03, 0.05, 0.075, 0.10, 0.15],
                             value=0.10, key="lab_proj_target")
    if not has_aligned(rid, ls.dataset, ref):
        st.caption(f"{len(t.recordings)} recordings have closed chunks. Measuring "
                   "aligns the model against them (seconds for a small sample).")
        if st.button("Measure now", key="lab_proj_go"):
            build_alignment(rid, ls.dataset, "native", ref)
            st.rerun()
        return
    bm = get_benchmark(rid, ls.dataset, ctx.judge, ctx.profile_name,
                       resolved_key(ctx.resolved[rid]), ctx.p_floor, ctx.min_support,
                       ctx.min_pos, ref)
    per_rec = bm["per_recording"]
    rows = []
    for stat in ("precision", "window_recall", "found"):
        lo, hi = bm["intervals"][stat]
        half = (hi - lo) / 2
        proj = labels.project(half, len(per_rec), target)
        v = bm["summary"][stat]
        rows.append({
            "measure": uncertainty.STAT_NAMES[stat],
            "now": f"{v:.0%}" if np.isfinite(v) else "—",
            "± now": f"±{half:.2f}" if np.isfinite(half) else "too few recordings",
            "recordings needed": (f"{proj[0]}–{proj[1]}" if proj else "—"),
        })
    st.dataframe(pl.DataFrame(rows), hide_index=True, width="stretch")
    mins = t.hours * 60
    per = mins / max(1, len(t.recordings))
    worst = max((labels.project((bm["intervals"][s][1] - bm["intervals"][s][0]) / 2,
                                len(per_rec), target) or (0, 0))[1]
                for s in ("precision", "window_recall"))
    if worst > len(per_rec):
        rate = labels.labelling_rate(ls)
        more = (worst - len(per_rec)) * per
        st.markdown(
            f"So far: **{mins:.0f} min** across **{len(per_rec)}** recordings. To reach "
            f"±{target:.2f} on precision and recall, label roughly **{worst - len(per_rec)} "
            f"more recordings** at your current {per:.1f} min each — about **{more:.0f} "
            "more minutes** of audio"
            + (f", or **{more / rate:.1f} hours** of labelling at your rate so far."
               if rate else ".")
            + " A range, not a promise: it assumes the next recordings vary as much as "
              "these did.")
    elif len(per_rec) >= uncertainty.MIN_UNITS:
        st.success(f"Precision and recall are already within ±{target:.2f}.")
    else:
        st.caption(f"Intervals need at least {uncertainty.MIN_UNITS} labelled recordings.")
    st.caption(f"Measured with {labels_of[rid]} at the sidebar's thresholds, on "
               f"*{labels.describe_ref(ref)}*.")


# --------------------------------------------------------------------------- #
# Progress (L7)
# --------------------------------------------------------------------------- #

def progress_view(ctx, ls: labels.LabelSet) -> None:
    recordings = ctx.recordings
    p = labels.progress(ls, recordings)
    m = st.columns(5)
    m[0].metric("Minutes closed", f"{p['minutes']:.0f}")
    m[1].metric("Recordings", p["recordings"], help=f"{p['complete_recordings']} end to end")
    m[2].metric("Sites", p["sites"])
    m[3].metric("Species", p["n_species"], help=f"{p['unknown']} unknown-bird labels")
    m[4].metric("Sample closed", f"{p['sample_closed']}/{p['sample_size']}",
                help=f"plus {p['by_hand']} chunk(s) chosen by hand")
    rate = labels.labelling_rate(ls)
    if rate:
        st.caption(f"Your labelling rate so far: about **{rate:.0f} minutes of audio per "
                   "hour** of work.")
    if p["all_boxes"]:
        st.caption(f"**{p['assisted']} of {p['all_boxes']} boxes** were labelled after "
                   "viewing the models' suggestions. Those labels lean towards what the "
                   "models already hear, which flatters them a little when they are "
                   "scored; the share is recorded so it can be checked.")

    left, right = st.columns(2)
    with left:
        heading("Species found against minutes labelled",
                "Each step is a chunk closed, in the order they were closed. When the "
                "curve flattens, more labelling has stopped turning up new species — a "
                "better signal to stop than a fixed number of hours. It never quite "
                "flattens for rare birds.", level="#####")
        acc = labels.accumulation(ls)
        if len(acc) > 1:
            st.altair_chart(style_chart(alt.Chart(acc.to_pandas()).mark_line(
                interpolate="step-after", color=brand()["b"]).encode(
                x=alt.X("minutes:Q", title="minutes closed"),
                y=alt.Y("species:Q", title="species"))).properties(height=220),
                width="stretch")
        else:
            st.caption("Close a few chunks to see the curve.")
    with right:
        heading("Coverage: closed minutes by site and time of day",
                "Gaps here are gaps in what the benchmark can say: a model's dawn "
                "chorus performance tells you little about its night.", level="#####")
        closed = ls.chunks.filter(pl.col("state") == labels.CLOSED)
        if closed.is_empty():
            st.caption("Nothing closed yet.")
        else:
            cov = (labels.with_strata(labels.all_chunks(recordings))
                   .join(closed.select("recording_id", "start_s"),
                         on=["recording_id", "start_s"], how="semi")
                   .with_columns(site=pl.col("stratum").str.split(" · ").list.get(0),
                                 band=pl.col("stratum").str.split(" · ").list.get(1),
                                 mins=(pl.col("end_s") - pl.col("start_s")) / 60)
                   .group_by("site", "band").agg(minutes=pl.col("mins").sum()))
            st.altair_chart(style_chart(alt.Chart(cov.to_pandas()).mark_rect().encode(
                x=alt.X("band:N", title=None, sort=["dawn", "day", "dusk", "night",
                                                    "time unknown"]),
                y=alt.Y("site:N", title=None),
                color=alt.Color("minutes:Q", scale=alt.Scale(scheme="greens"),
                                legend=alt.Legend(title="min", format=".0f")),
                tooltip=["site", "band", alt.Tooltip("minutes:Q", format=".0f")],
            )).properties(height=max(80, 36 * cov["site"].n_unique())), width="stretch")

    st.markdown("##### What has been found")
    nm = lambda k: views.display_name(k, get_names())
    if p["species"].is_empty():
        st.caption("No birds labelled in closed chunks yet.")
    else:
        st.dataframe(p["species"].with_columns(
            species=pl.col("species_key").map_elements(nm, return_dtype=pl.Utf8),
            per_hour=(pl.col("boxes") / max(p["minutes"] / 60, 1e-9)).round(1),
            reach_50=pl.when(pl.col("boxes") >= 50).then(pl.lit("enough for a threshold"))
            .otherwise(((50 - pl.col("boxes")) / pl.col("boxes") * p["minutes"] / 60)
                       .round(1).cast(pl.Utf8) + " more hours (approx.)"))
            .select("species", "boxes", "chunks", "per_hour", "reach_50"),
            hide_index=True, width="stretch", column_config={
                "boxes": st.column_config.NumberColumn("labelled calls"),
                "chunks": st.column_config.NumberColumn("chunks it is in"),
                "per_hour": st.column_config.NumberColumn("calls per hour"),
                "reach_50": st.column_config.TextColumn(
                    "to reach ~50 calls", help="At the rate it has turned up so far.")})


# --------------------------------------------------------------------------- #
# Publish and share (L6, L8)
# --------------------------------------------------------------------------- #

def share_view(ctx, ls: labels.LabelSet, labeller: str) -> None:
    dataset, recordings = ctx.dataset, ctx.recordings
    vs = labels.versions(cfg.store_dir, dataset, ls.name)
    st.markdown("#### Publish a version")
    st.markdown(
        "The working copy changes as you label. **Publishing** freezes it as a numbered "
        "version (v1, v2 …) that never changes again: choose it as the ground truth in "
        "the sidebar, and every score, threshold set and report built on it says which "
        "version it used. Publishing v2 later does not change anything built on v1.")
    pub = ls.meta.get("published", {})
    if vs:
        st.caption("Published: " + ", ".join(f"v{v} ({pub.get(str(v), '')[:10]})" for v in vs))
    if st.button(f"Publish as v{(vs or [0])[-1] + 1}", type="primary"):
        try:
            n = labels.publish(cfg.store_dir, ls)
        except labels.StaleWrite:
            st.error("The set changed in another tab; reload and publish again.")
        else:
            st.session_state["lab_flash"] = ("success", f"Published {ls.name} v{n}.")
            st.rerun()

    st.markdown("#### Export")
    st.markdown(
        "In **SNE's columns** (Filename, Start Time (s), End Time (s), Low Freq (Hz), "
        "High Freq (Hz), Species), plus who labelled each box and whether its chunk is "
        "closed — a box in an open chunk is a bird, but its neighbours may not be "
        "labelled yet. The chunks file says exactly which minutes were listened to.")
    names = get_names()
    a, b = st.columns(2)
    a.download_button("Boxes (CSV)", labels.export_csv(ls, recordings, names),
                      f"{dataset}_{ls.name}_annotations.csv", "text/csv")
    b.download_button("Chunks (CSV)", labels.export_chunks_csv(ls),
                      f"{dataset}_{ls.name}_chunks.csv", "text/csv")

    st.markdown("#### Import annotations as a new set")
    st.markdown(
        "A CSV in SNE's columns (as exported above), a **Raven** selection table, or an "
        "**Audacity** label track (one recording per file: its name must match the "
        "recording's). Species may be scientific or common names; anything that cannot "
        "be matched is listed and the import is refused, never silently dropped.")
    with st.form("import"):
        up = st.file_uploader("Annotations file", type=["csv", "txt", "tsv"])
        name = st.text_input("New set name")
        exhaustive = st.checkbox(
            "Every bird in these recordings is labelled (exhaustive)",
            help="Yes: every chunk of the recordings in the file is closed — the file "
                 "is ground truth for them. No: the boxes count as birds that were "
                 "there, but nothing says what was *not* there, so they cannot score "
                 "false alarms until you close chunks yourself.")
        go = st.form_submit_button("Import")
    if go and up is not None:
        lookup = {}
        for k, common in names.items():
            lookup[k], lookup[k.lower()] = k, k
            if common:
                lookup[common.lower()] = k
        lookup[labels.UNKNOWN_BIRD] = labels.UNKNOWN_BIRD
        try:
            boxes, unmapped = labels.read_annotation_file(
                up.getvalue().decode("utf-8", errors="replace"), up.name, recordings, lookup)
            if unmapped:
                st.error(f"{len(unmapped)} name(s) match no species: "
                         + ", ".join(unmapped[:20]) + ". Fix them in the file and import "
                         "again.")
                return
            labels.import_boxes(cfg.store_dir, dataset, name.strip(), boxes, recordings,
                                exhaustive, labeller or "imported")
        except (ValueError, FileExistsError, StopIteration) as e:
            st.error(f"Could not import: {e}")
            return
        st.session_state["lab_select_pending"] = (dataset, name.strip(), "Progress")
        st.session_state["lab_flash"] = ("success", f"Imported {len(boxes)} boxes as "
                                         f"{name.strip()}.")
        st.rerun()


def howto_view() -> None:
    st.markdown("""
#### Labelling, in five minutes

**What closing a chunk promises.** A closed chunk says *every bird I heard in this
minute has a box*. That is what lets BEX score a model's false alarms: a detection
in a closed chunk with no matching box is wrong. A chunk that is not closed says
nothing, so it is left out of every score. A closed chunk with no boxes means
*listened, no birds*, which is truth too.

**Why BEX picks the minutes.** If you label where the models fired, you mostly
label the birds they already find, and every model looks better than it is. The
random sample is spread across sites, times of day and recordings, and in an
order where you can stop at any point and still have a balanced sample.

**Why the models stay quiet.** Labels that start from a model's suggestion
flatter that model when it is scored on them, and steer you towards the birds the
models find and away from the ones they miss. The grey activity guide on the
navigator only says where the recording is busy. It names no species and it
misses birds.

**How much to label.** Start small on *Plan and sample* — 5 minutes — and add
more as you go. Once a few recordings have labels, *How much more?* measures how
much your recordings vary and projects how much more you need.

**Reading the spectrogram.**
- Birdsong is mostly between 1 and 8 kHz: tonal strokes, whistles, trills,
  repeated phrases.
- **Wind** and handling noise are broad smears at the bottom. **Rain** is
  scattered vertical dots across all frequencies. **Insects** are steady,
  high-pitched bands that do not change for many seconds.
- Listen first, then look. Use the player under the spectrogram: it is on the
  same time axis.
- One box per call or per continuous song bout. Long bouts can be one long box.
- If you can hear a bird but cannot tell what it is, label it **Unknown bird**
  rather than guessing. Unknown birds are counted, and left out of species scores.

**Practise first.** On a dataset that came with expert labels (the demo set),
*New to labelling? Practise on minutes the experts have done* gives you a few
minutes to label; as you close each, BEX shows which of the experts' calls you
found, named differently or missed, with each to listen to. Practice labels
never count as ground truth.

**A worked pilot.**
1. *Plan and sample* → say how many minutes (5 is a good start) → **Pick**.
2. *Label* → **Next chunk in the sample**. Listen to the whole minute.
3. Click one corner of each call, then the opposite corner; name the species,
   **Add box**. A call that runs past the minute can be boxed into the shaded
   strip, or finished in the next minute (▶ steps there) with its second click.
4. Tick *I listened to the whole minute*, then **Close this chunk**. BEX moves on
   to the next one.
5. After about 20 chunks, *Plan and sample* → **Measure now** to see how much
   more you need.
6. When you are done, *Publish and share* → **Publish** a version, and choose it
   as the ground truth in the sidebar.
""")


# --------------------------------------------------------------------------- #

def practice_offer(ctx, dataset: str, labeller: str) -> None:
    """On a dataset with expert labels: try labelling and see how you did."""
    with st.expander("🎓 New to labelling? Practise on minutes the experts have done"):
        st.markdown(
            f"*{dataset}* came with expert labels. Label a few of its minutes "
            "yourself and, as you close each one, BEX compares your boxes with the "
            "experts' — which birds you found, which you named differently, and "
            "which you missed, with each to listen to. It is the quickest way to "
            "learn what birdsong looks like on a spectrogram. Practice labels are "
            "kept apart, and never count as ground truth.")
        if st.button("Start practising (5 minutes)", type="primary", key="lab_prac_go",
                     disabled=not labeller, help=None if labeller else
                     "Enter your name first (top right)."):
            ls = labels.create_practice(cfg.store_dir, dataset, labeller, ctx.recordings)
            st.session_state["lab_select_pending"] = (dataset, ls.name, "Label")
            st.session_state.pop("lab_rec", None)
            st.rerun()


def render(ctx) -> None:
    dataset = ctx.dataset
    st.markdown(f"## 1 · Label — {dataset}")
    flash = st.session_state.pop("lab_flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])

    c1, c2 = st.columns([2, 1])
    labeller = c2.text_input("Your name", key="labeller", persist_state="session",
                             help="Recorded on every box and every chunk you close.")
    sets = labels.list_sets(cfg.store_dir, dataset)
    key = f"label_set_{dataset}"
    pending = st.session_state.pop("lab_select_pending", None)
    if pending and pending[0] == dataset:
        st.session_state[key] = pending[1]
        st.session_state["label_view"] = pending[2]
    if st.session_state.get(key) not in [*sets, NEW_SET]:
        real = [n for n in sets if not labels.is_practice(cfg.store_dir, dataset, n)]
        st.session_state[key] = (real or sets or [NEW_SET])[0]
    choice = c1.selectbox(
        "Label set you are working on", [*sets, NEW_SET], key=key,
        format_func=lambda n: f"🎓 {n} (practice)"
        if n != NEW_SET and labels.is_practice(cfg.store_dir, dataset, n) else n)
    if labels.has_imported(cfg.store_dir, dataset) and not (
            choice != NEW_SET and labels.is_practice(cfg.store_dir, dataset, choice)):
        practice_offer(ctx, dataset, labeller)
    if choice == NEW_SET:
        if labels.has_imported(cfg.store_dir, dataset):
            st.info(f"**{dataset}** came with annotations, so it can be scored already. "
                    "Start a set here to add to them (a copy) or to label it yourself.")
        new_set_form(dataset, labeller)
        return

    ls = load(dataset, choice)
    in_use = ctx.truth_ref and labels.parse_ref(ctx.truth_ref)[0] == ls.name
    if labels.is_practice(cfg.store_dir, dataset, ls.name):
        st.info("🎓 **Practice.** These minutes were labelled by experts. Label each one "
                "as if it were your own, close it, and BEX shows what the experts "
                "heard — what you found, named differently or missed — with each call "
                "to listen to. Practice labels never count as ground truth.")
    elif not in_use:
        st.caption(f"The other pages are not scoring against *{ls.name}* yet: choose "
                   f"**{sidebar.VIEW_LABELLED}** in the sidebar, then *{ls.name}* under "
                   "**Scored against**.")
    if not labeller:
        st.warning("Enter your name (top right) to label: every box and sign-off "
                   "records who made it.")

    if st.session_state.get("label_view") not in VIEWS:
        st.session_state["label_view"] = "Plan and sample" if ls.sample.is_empty() else "Label"
    view = st.segmented_control("View", VIEWS, key="label_view",
                                label_visibility="collapsed") or "Label"

    if view == "Label":
        share = st.session_state.get("lab_share", 0.30)
        runs = tuple(r for r in complete_runs(dataset)
                     if r in {m.run_id for m in ctx.included})
        activity = get_activity(runs, share)[0] if runs else None
        if ls.sample.is_empty():
            st.info("No sample drawn yet — go to **Plan and sample** first, or choose a "
                    "chunk by hand.")
        label_view(ctx, ls, labeller, activity, share)
        with st.expander("Activity guide settings"):
            st.slider("Share of each model's windows to highlight", 0.05, 0.6, 0.30, 0.05,
                      key="lab_share",
                      help="The guide highlights the windows each model scores most "
                           "bird-like, this share of them across the whole dataset. "
                           "Higher misses less and highlights more noise.")
            if runs:
                bars = get_activity(runs, share)[1]
                st.caption("Bars in use: " + ", ".join(
                    f"{ctx.run_label[r]} ≥ {b:.3g}" for r, b in bars.items()))
    elif view == "Plan and sample":
        plan_view(ctx, ls, labeller)
    elif view == "Progress":
        progress_view(ctx, ls)
    elif view == "Publish and share":
        share_view(ctx, ls, labeller)
    else:
        howto_view()


render(sidebar.render())
