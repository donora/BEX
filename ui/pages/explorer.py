"""BEX · Explorer."""
from ui.common import *  # noqa: F403 — the shared app toolkit
from ui import sidebar


def render(ctx) -> None:
    from bex import flow
    if not flow.dataset_meta(cfg.store_dir, ctx.dataset).get("explored"):
        flow.set_dataset_meta(cfg.store_dir, ctx.dataset, explored=True)
    runs = ctx.runs
    dataset = ctx.dataset
    recordings = ctx.recordings
    annotations = ctx.annotations
    run_label = ctx.run_label
    included = ctx.included
    profile_name = ctx.profile_name
    profile = ctx.profile
    judge = ctx.judge
    fixed = ctx.fixed
    spec = ctx.spec
    resolved = ctx.resolved
    fitted_here = ctx.fitted_here

    # ---- which recording ---------------------------------------------------- #
    c1, c2 = st.columns([3, 1])
    rec_id = c1.selectbox("Recording", recordings["recording_id"].to_list(),
                          key="explorer_rec", persist_state="session")
    rec_row = recordings.filter(pl.col("recording_id") == rec_id).to_dicts()[0]
    duration = rec_row["duration_s"]
    span_s = c2.selectbox("Span (s)", [15, 30, 60, 120, 300], index=2,
                          key="explorer_span", persist_state="session")
    # Right and wrong need the whole recording labelled: a detection in a chunk
    # nobody has closed is not wrong, just not yet judged.
    t_obj = ctx.truth_obj
    if t_obj is not None and rec_id not in t_obj.complete:
        done = t_obj.covered_s.get(rec_id, 0.0)
        st.caption(f"ℹ️ This recording is {'partly labelled' if done else 'not labelled'} "
                   f"in *{t_obj.label}* ({done / 60:.0f} of {duration / 60:.0f} min "
                   "closed), so it is shown unscored: what each model reported, "
                   "with no right or wrong.")
        annotations = None

    # Every included run that has scored THIS recording — a run that has not
    # reached it would draw an empty lane and read as a silent model.
    models = {run_label[m.run_id]: m for m in included
              if rec_id in set(store.list_scores(cfg.store_dir, m.run_id))}
    if not models:
        st.info("No included model has scored this recording — see the Models page.")
        return

    det_by_model = {
        label: get_judged(m.run_id, judge, profile_name, resolved_key(resolved[m.run_id]))
        for label, m in models.items()
    }
    names_by_model = {label: get_names() for label in models}

    def name_of(key: str) -> str:
        return views.display_name(key, get_names())

    # A new recording starts at its beginning, with nothing inspected yet.
    if st.session_state.get("_explorer_rec") != rec_id:
        st.session_state.update(_explorer_rec=rec_id, viewport_start=0.0,
                                inspect_t=None, nav_seen={})
    max_start = max(0.0, duration - span_s)
    st.session_state.update(_span=float(span_s), _max_start=max_start,
                            _duration=float(duration))
    t0 = min(float(st.session_state.get("viewport_start", 0.0)), max_start)
    t1 = t0 + span_s

    # ---- focus species ------------------------------------------------------ #
    # Truth ∪ every model's detections — see views.focus_options for why.
    opts = views.focus_options(det_by_model, rec_id, None, annotations)
    opt_meta = {r["species_key"]: r for r in opts.iter_rows(named=True)}

    def focus_label(key: str) -> str:
        if key.startswith("—"):
            return key
        r = opt_meta.get(key, {})
        if r.get("in_truth") and not r.get("n_models"):
            tag = "annotated · no model detects it"
        elif r.get("in_truth"):
            tag = f"annotated · {r['n_models']} of {len(models)} models detect it"
        else:
            tag = "detected, not annotated"
        return f"{name_of(key)} — {tag}"

    focus_opts = ["— all species —"] + opts["species_key"].to_list()
    # Keyed so another page (the Species scorecard's "Listen in the Explorer")
    # can open this recording already focused on a bird; a remembered choice that
    # this recording does not offer is dropped rather than left pointing nowhere.
    if st.session_state.get("explorer_focus") not in focus_opts:
        st.session_state.pop("explorer_focus", None)
    focus = st.selectbox(
        "Focus species", focus_opts, key="explorer_focus",
        format_func=focus_label,
        help="Narrows the navigator and the lanes to one bird. Every species "
             "annotated in this recording is offered, plus every species any model "
             "detects — a bird no model finds is the one most worth looking at.",
    )
    focus_key = None if focus.startswith("—") else focus

    # ---- navigator (V1 E1) -------------------------------------------------- #
    labelled = annotations is not None
    up_outcomes = list(views.OUTCOMES if labelled else views.UNLABELLED_OUTCOMES)
    heading(
        "Where the birdsong is"
        + (f" — **{name_of(focus_key)}** only" if focus_key else ""),
        "The whole recording, one chart per model, counted in **species** per time "
        "slot (not windows — a 3 s model produces more windows than a 5 s one for "
        "the same song, so window counts would show a difference that is not "
        "there).\n\n"
        "**Above the line: what the model reported**, split by outcome:\n"
        "- **correct** — the species is annotated in that window;\n"
        "- **wrong** — it is not;\n"
        "- **filter caught a mistake** — not annotated, and the judge would hide it;\n"
        "- **filter hid a real bird** — annotated, and the judge would hide it "
        "anyway. This is the failure this app exists to surface.\n\n"
        "**Below the line: annotated birds the model missed** in that slot.\n\n"
        "The top chart is the ground truth: annotated species per slot. "
        "'Annotated in the window' uses the same overlap rule as the scorecard, so "
        "the two agree. The judge is the sidebar's; each model's threshold is its "
        "own per-species θ (Thresholds page).\n\n"
        "**Click anywhere** to move the view there.",
        level="#####")
    key_items = up_outcomes + (["missed"] if labelled else [])
    st.markdown(
        "<div style='font-size:0.85rem;line-height:1.9'>"
        + (f"<span style='white-space:nowrap'><span style='color:{views.TRUTH_COLOUR};"
           "font-size:1.1rem'>■</span> ground truth</span> &nbsp; " if labelled else "")
        + " &nbsp; ".join(
            f"<span style='white-space:nowrap'><span style='color:"
            f"{views.OUTCOME_COLOURS[o]};font-size:1.1rem'>■</span> {o}</span>"
            for o in key_items)
        + ("<br><b>↑ above the line</b>: what the model reported &nbsp;·&nbsp; "
           "<b>↓ below</b>: annotated birds it missed" if labelled else
           "<br><b>↑</b> what the model reported — no annotations, so no right or "
           "wrong")
        + "</div>", unsafe_allow_html=True)

    N_BINS = 180
    nav_by_model = {
        label: views.navigator_bins(det, annotations, rec_id, duration, N_BINS, focus_key)
        for label, det in det_by_model.items()
    }
    first_nav = next(iter(nav_by_model.values()))
    bin_w = float(first_nav["bin_end_s"][0] - first_nav["bin_start_s"][0])
    up_max = max(1, max(int(nb.select(pl.sum_horizontal(
        [f"n_{o}" for o in up_outcomes])).to_series().max() or 0)
        for nb in nav_by_model.values()))
    down_max = (max(1, max(int(nb["n_missed"].max() or 0)
                           for nb in nav_by_model.values())) if labelled else 0)

    x_axis = alt.X("bin_start_s:Q", title=None,
                   scale=alt.Scale(domain=[0, duration], nice=False))
    # One click parameter per chart — Altair will not share one across a vconcat
    # — and whichever fired says where to go.
    pickers: list[str] = []

    def picker() -> alt.Parameter:
        pickers.append(f"pick{len(pickers)}")
        return alt.selection_point(fields=["bin_start_s"], name=pickers[-1],
                                   on="click", empty=False)
    inspect_now = st.session_state.get("inspect_t")
    viewport = pd.DataFrame({"start": [t0], "end": [t1],
                             "label": [f"{t0:.0f}–{t1:.0f} s"]})

    def overlays(height: int, with_label: bool = False) -> list:
        """Viewport box (and its times), and the inspected instant."""
        out = [alt.Chart(viewport).mark_rect(
            fill="#2a78d6", fillOpacity=0.08, stroke="#2a78d6", strokeWidth=1.5,
        ).encode(x="start:Q", x2="end:Q").properties(height=height)]
        if with_label:
            out.append(alt.Chart(viewport).mark_text(
                align="left", baseline="top", dx=4, dy=2, fontSize=11,
                fontWeight="bold", color="#1f5fae",
            ).encode(x="start:Q", y=alt.value(0), text="label:N"))
        if inspect_now is not None:
            out.append(alt.Chart(pd.DataFrame({"t": [inspect_now]})).mark_rule(
                color="#1d1b17", strokeWidth=1, strokeDash=[3, 2]).encode(x="t:Q"))
        return out

    def hit_layer(nb: pl.DataFrame, height: int, outs: list[str]) -> alt.Chart:
        """Invisible full-height cells: the click target, and the bin's tooltip."""
        hit = nb.select("bin_start_s", "bin_end_s").to_pandas()
        tips = [alt.Tooltip("time:N", title="time")]
        hit["time"] = [f"{a:.0f}–{b:.0f} s" for a, b in
                       zip(hit["bin_start_s"], hit["bin_end_s"])]
        for i, o in enumerate(outs):
            hit[f"t{i}"] = [names_of(k, name_of) for k in nb[f"names_{o}"]]
            tips.append(alt.Tooltip(f"t{i}:N", title=o))
        return alt.Chart(hit).mark_rect(opacity=0).encode(
            x=x_axis, x2="bin_end_s:Q", tooltip=tips,
        ).add_params(picker()).properties(height=height)

    parts = []
    if labelled:
        truth_df = (first_nav.select("bin_start_s", "bin_end_s", "n_truth")
                    .filter(pl.col("n_truth") > 0).with_columns(y0=pl.lit(0))
                    .to_pandas())
        truth_max = max(1, int(first_nav["n_truth"].max() or 0))
        truth_bars = alt.Chart(truth_df).mark_rect(color=views.TRUTH_COLOUR).encode(
            x=x_axis, x2="bin_end_s:Q", y2="y0:Q",
            y=alt.Y("n_truth:Q", title=None, axis=alt.Axis(tickCount=2, format="d"),
                    scale=alt.Scale(domain=[0, truth_max])),
        ).properties(height=45, title=alt.Title("ground truth — annotated species",
                                                fontSize=10, anchor="start"))
        parts.append(alt.layer(truth_bars, *overlays(45, with_label=True),
                               hit_layer(first_nav, 45, ["truth"])))

    for i, (label, nb) in enumerate(nav_by_model.items()):
        # Stacked by hand into explicit [y0, y1] segments: outcomes upward from
        # the line in their fixed order, misses downward from it.
        long, base = [], pl.lit(0)
        for o in up_outcomes:
            long.append(nb.select("bin_start_s", "bin_end_s", y0=base,
                                  y1=base + pl.col(f"n_{o}"), outcome=pl.lit(o),
                                  n=pl.col(f"n_{o}")))
            base = base + pl.col(f"n_{o}")
        if labelled:
            long.append(nb.select("bin_start_s", "bin_end_s", y0=pl.lit(0),
                                  y1=-pl.col("n_missed"), outcome=pl.lit("missed"),
                                  n=pl.col("n_missed")))
        long_df = pl.concat(long, how="vertical_relaxed").filter(
            pl.col("n") > 0).to_pandas()
        bars = alt.Chart(long_df).mark_rect().encode(
            x=x_axis, x2="bin_end_s:Q", y2="y0:Q",
            y=alt.Y("y1:Q", title=None,
                    scale=alt.Scale(domain=[-down_max, up_max]),
                    axis=alt.Axis(tickCount=3, format="d", labelExpr="abs(datum.value)")),
            color=alt.Color("outcome:N", legend=None, scale=alt.Scale(
                domain=list(views.OUTCOME_COLOURS),
                range=list(views.OUTCOME_COLOURS.values()))),
        ).properties(height=70, title=alt.Title(label, fontSize=10, anchor="start"))
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            color="#8a8880", strokeWidth=0.8).encode(y="y:Q")
        tip_outs = up_outcomes + (["missed"] if labelled else [])
        parts.append(alt.layer(bars, zero,
                               *overlays(70, with_label=not labelled and i == 0),
                               hit_layer(nb, 70, tip_outs)))

    chart = alt.vconcat(*parts, spacing=6).resolve_scale(y="independent").configure_view(
        strokeWidth=0)
    event = st.altair_chart(chart, width="stretch", on_select="rerun",
                            key="nav")

    try:
        chosen = (event or {}).get("selection", {})
    except AttributeError:
        chosen = {}
    seen = st.session_state.get("nav_seen")
    clicked, st.session_state["nav_seen"] = views.new_click(
        chosen, pickers, seen if isinstance(seen, dict) else {})
    if clicked is not None:
        st.session_state["viewport_start"] = min(
            max(0.0, clicked - span_s / 2 + bin_w / 2), max_start)
        st.session_state["inspect_t"] = clicked + bin_w / 2
        st.rerun()

    # ---- where am I, and small steps (V1 E3) --------------------------------- #
    hops = [m.hop_s for m in models.values()]
    step = min(hops)
    inspect_t = st.session_state.get("inspect_t")
    if inspect_t is None or not t0 <= inspect_t < t1:
        starts = [views.first_detection(det, rec_id, t0, t1, None)
                  for det in det_by_model.values()]
        starts = [s for s in starts if s is not None and s >= t0]
        inspect_t = (min(starts) + 0.01) if starts else t0 + 0.01
        st.session_state["inspect_t"] = inspect_t

    b1, b2, mid, b3, b4 = st.columns([1, 1, 4, 1, 1], vertical_alignment="center")
    b1.button("⏮ span", on_click=shift_view, kwargs={"d_view": -float(span_s)},
              disabled=t0 <= 0, width="stretch", help="Previous span")
    b2.button("◀ window", on_click=shift_view, kwargs={"d_inspect": -step},
              width="stretch", help=f"Inspect {step:g} s earlier")
    mid.markdown(
        f"<div style='text-align:center'>Viewing <b>{t0:.0f}–{t1:.0f} s</b> of "
        f"{duration:,.0f} s &nbsp;·&nbsp; inspecting <b>{inspect_t:.1f} s</b></div>",
        unsafe_allow_html=True)
    b3.button("window ▶", on_click=shift_view, kwargs={"d_inspect": step},
              width="stretch", help=f"Inspect {step:g} s later")
    b4.button("span ⏭", on_click=shift_view, kwargs={"d_view": float(span_s)},
              disabled=t0 >= max_start, width="stretch", help="Next span")

    # ---- data for this span ------------------------------------------------ #
    lanes = {
        label: views.lane_blocks(det, rec_id, t0, t1, None, per_window=3)
        for label, det in det_by_model.items()
    }
    truth_lane = (views.truth_blocks(annotations, rec_id, t0, t1)
                  if annotations is not None else pl.DataFrame())
    # Colour by what is actually drawn, over the whole recording (so a bird keeps
    # its colour as the view pans): annotated species first, then the
    # most-detected. Anything past the eight colours is grey, and named where
    # it is drawn (see grey_label).
    palette = views.species_palette(
        *[det.filter((pl.col("recording_id") == rec_id) & pl.col("above"))
          for det in det_by_model.values()],
        priority=(annotations.filter(pl.col("recording_id") == rec_id)
                  if annotations is not None else None))
    if focus_key and palette.get(focus_key, views.OTHER_COLOR) == views.OTHER_COLOR:
        palette[focus_key] = views.CATEGORICAL[0]

    try:
        mel, meta = get_mel(dataset, rec_id)
        have_mel = True
    except FileNotFoundError:
        have_mel = False
        st.warning(f"No display spectrogram cached — run `bex melcache {dataset}`.")

    # ---- the figure: spectrogram, truth lane, one score lane per model ----- #
    has_truth = annotations is not None
    ratios = [7] + ([1.1] if has_truth else []) + [2.4] * len(models)
    fig, axes = plt.subplots(
        len(ratios), 1, sharex=True,
        figsize=(14, 4.6 + (0.55 if has_truth else 0) + 1.15 * len(models)),
        gridspec_kw={"height_ratios": ratios, "hspace": 0.12},
    )
    # Fixed margins in inches, not fractions: the default bottom margin grows
    # with the figure and left a band of white between the time axis and the
    # audio player lined up beneath it.
    fig_h = fig.get_figheight()
    fig.subplots_adjust(top=1 - 0.42 / fig_h, bottom=0.5 / fig_h)
    axes = np.atleast_1d(axes)
    ax_spec, lane_axes = axes[0], list(axes[1:])
    n_rows = mel.shape[0] if have_mel else 128

    if have_mel:
        s = slice_mel(mel, meta, t0, t1)
        vmin, vmax = display_range(s)
        ax_spec.imshow(s, origin="lower", aspect="auto", cmap="magma",
                       extent=[t0, t1, 0, n_rows], vmin=vmin, vmax=vmax)
        ax_spec.set_ylabel("mel band")
    else:
        ax_spec.text(0.5, 0.5, "no spectrogram cache", transform=ax_spec.transAxes,
                     ha="center", color="#9a9992")
    ax_spec.set_ylim(0, n_rows)

    def label(ax, x, y, text, color, va="bottom", size=7.5):
        ax.text(x, y, text, fontsize=size, color=color, va=va, ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#12100f",
                          edgecolor="none", alpha=0.78))

    # Ground truth: solid boxes, localised in time AND frequency.
    shown_truth = truth_lane
    if focus_key and not shown_truth.is_empty():
        shown_truth = shown_truth.filter(pl.col("species_key") == focus_key)
    tl = views.label_anchors(shown_truth) if not shown_truth.is_empty() else pl.DataFrame()
    truth_anchor = ({(r["species_key"], r["start_s"]) for r in tl.to_dicts()}
                    if not tl.is_empty() else set())
    for b in (shown_truth.to_dicts() if not shown_truth.is_empty() else []):
        color = palette.get(b["species_key"], views.OTHER_COLOR)
        lo = views.hz_to_mel_row(b["low_hz"], meta) if have_mel and np.isfinite(b["low_hz"]) else 0
        hi = views.hz_to_mel_row(b["high_hz"], meta) if have_mel and np.isfinite(b["high_hz"]) else n_rows
        x0, x1 = max(b["start_s"], t0), min(b["end_s"], t1)
        ax_spec.add_patch(plt.Rectangle((x0, lo), x1 - x0, hi - lo, fill=False,
                                        edgecolor=color, linewidth=1.6, linestyle="-"))
        if (b["species_key"], b["start_s"]) in truth_anchor:
            label(ax_spec, x0 + 0.15, min(hi + 1.5, n_rows - 8),
                  name_of(b["species_key"]), color)

    # A focus species' detections, shaded through the spectrogram, with one row
    # per model in a strip along the top (in lane order) so two models' claims
    # about the same second sit one above the other. Without a focus nothing is
    # drawn here: every species at once buries the audio, and the score lanes
    # below carry the same information.
    BAR_H = n_rows * 0.030
    strip_top = n_rows * 0.995
    for i, (mlabel, m) in enumerate(models.items() if focus_key else []):
        row_top = strip_top - i * (BAR_H * 1.55)
        ax_spec.text(t0 + 0.1, row_top - BAR_H / 2, mlabel, fontsize=6.5,
                     color="#f2f1ec", va="center", ha="left", zorder=6,
                     bbox=dict(boxstyle="square,pad=0.15", facecolor="#12100f",
                               edgecolor="none", alpha=0.7))
        spans_v = views.species_spans(det_by_model[mlabel], rec_id, t0, t1,
                                      None, focus_key, suppressed=None)
        imp = det_by_model[mlabel].filter(
            (pl.col("species_key") == focus_key) & pl.col("implausible"))
        focus_implausible = len(imp) > 0
        for b in spans_v.to_dicts():
            color = palette.get(focus_key, views.OTHER_COLOR)
            x0, x1 = max(b["start_s"], t0), min(b["end_s"], t1)
            ax_spec.add_patch(plt.Rectangle(
                (x0, 0), x1 - x0, n_rows, facecolor=to_rgba(color, 0.08),
                edgecolor=to_rgba(color, 0.7), linewidth=1.0,
                linestyle=(0, (1, 2)) if focus_implausible else (0, (5, 3)),
                zorder=3))
            ax_spec.add_patch(plt.Rectangle(
                (x0, row_top - BAR_H), x1 - x0, BAR_H,
                facecolor="none" if focus_implausible else to_rgba(color, 0.92),
                edgecolor=color, hatch="///" if focus_implausible else None,
                linewidth=1.1, zorder=4))

    # The inspected instant: one line through the spectrogram, its time written on.
    ax_spec.axvline(inspect_t, color="#ffffff", linewidth=1.3, zorder=5)
    label(ax_spec, inspect_t + 0.12, n_rows * 0.04, f"{inspect_t:.1f} s", "#ffffff",
          size=7.5)

    WRONG_MARK, HIDDEN_MARK = "#d0021b", "#f2b705"

    def grey_label(ax, x, y, key, va="bottom"):
        """Name a grey mark where it is drawn — past the eight colours a species
        has no colour of its own, so without this it would be anonymous."""
        ax.text(x, y, name_of(key), fontsize=6, color="#3a3a35", va=va, ha="left",
                zorder=7, clip_on=True,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="#ffffffc8",
                          edgecolor="none"))

    def run_starts(blocks: pl.DataFrame) -> set[tuple[str, float]]:
        """(species, start) of the first window of each unbroken run of a species,
        so a bird singing across ten windows is named once, not ten times."""
        firsts, last_end = set(), {}
        for b in blocks.sort("start_s").iter_rows(named=True):
            k = b["species_key"]
            if b["start_s"] > last_end.get(k, -1e9) + 1e-6:
                firsts.add((k, b["start_s"]))
            last_end[k] = max(b["end_s"], last_end.get(k, -1e9))
        return firsts

    title = (f"{rec_id} · {t0:.0f}–{t1:.0f} s · θ: {spec.label} · judge: {judge}"
             + (f" · focus: {name_of(focus_key)}" if focus_key else ""))
    ax_spec.set_title(title, loc="left", fontsize=11)

    # ---- lanes: truth boxes, then one score lane per model (V1 E5) ----------- #
    lane_i = 0
    if has_truth:
        ax = lane_axes[lane_i]
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel("truth", rotation=0, ha="right", va="center", fontsize=8.5)
        if not truth_lane.is_empty():
            packed = views.pack_rows(truth_lane)
            h = 0.76 / (int(packed["row"].max()) + 1)
            named = set()
            for b in packed.to_dicts():
                color = palette.get(b["species_key"], views.OTHER_COLOR)
                if focus_key and b["species_key"] != focus_key:
                    color = "#dedcd3"
                x0, x1 = max(b["start_s"], t0), min(b["end_s"], t1)
                gap = 0.004 * span_s
                ax.add_patch(plt.Rectangle((x0 + gap, 0.12 + b["row"] * h),
                                           max(0, x1 - x0 - 2 * gap), h * 0.88,
                                           facecolor=color, linewidth=0))
                # Once per species per view: truth boxes are long and many.
                if color == views.OTHER_COLOR and b["species_key"] not in named:
                    named.add(b["species_key"])
                    grey_label(ax, x0 + gap, 0.12 + b["row"] * h + h * 0.44,
                               b["species_key"], va="center")
        ax.axvline(inspect_t, color="#1d1b17", linewidth=0.9, linestyle=(0, (3, 2)))
        lane_i += 1

    for mlabel, m in models.items():
        ax = lane_axes[lane_i]
        lane_i += 1
        ax.set_ylim(0, 1.12)   # headroom for the ✕ / ⚠ marks above the top lines
        ax.set_ylabel(f"{mlabel}\n{m.window_s:g} s windows", rotation=0, ha="right",
                      va="center", fontsize=8.5)
        # Everything below the baseline is "under threshold" — never drawn here.
        ax.axhspan(0, th.LANE_BASELINE, facecolor="#00000012", edgecolor="none")
        ax.axhline(th.LANE_BASELINE, color="#9a9992", linewidth=0.6)
        ax.set_yticks([th.LANE_BASELINE, 1.0], ["threshold", "strongest"], fontsize=7)
        ax.tick_params(axis="y", length=0, labelcolor="#6b6a63")
        for x in np.arange(np.floor(t0 / m.hop_s) * m.hop_s, t1 + 1e-9, m.hop_s):
            ax.axvline(x, color="#00000014", linewidth=0.5)
        win = views.window_at(det_by_model[mlabel], rec_id, inspect_t)
        if win:
            ax.axvspan(win[0], win[1], facecolor="#2a78d614", edgecolor="#2a78d6",
                       linewidth=1.0, zorder=0)
            ax.text(win[0] + 0.05, 1.02, f"{win[0]:g}–{win[1]:g} s", fontsize=6.5,
                    color="#1f5fae", va="top", ha="left")
        marks = views.lane_lines(det_by_model[mlabel], rec_id, t0, t1, annotations)
        firsts = run_starts(marks) if not marks.is_empty() else set()
        gap = 0.006 * span_s
        for b in marks.to_dicts():
            color = palette.get(b["species_key"], views.OTHER_COLOR)
            if focus_key and b["species_key"] != focus_key:
                color = "#dedcd3"
            x0, x1 = max(b["start_s"], t0) + gap, min(b["end_s"], t1) - gap
            ax.hlines(b["strength"], x0, max(x0, x1), colors=color, linewidth=2.4,
                      linestyles=(0, (2, 1.2)) if b["implausible"] else "solid",
                      zorder=3 if not focus_key or b["species_key"] == focus_key else 2)
            # Colour says which species; a symbol at the line's top left says
            # what kind of error it is. Symbols, not colours, because the species
            # palette already uses every hue an outcome colour could take.
            mark = None
            if b["correct"] is False and not b["implausible"]:
                mark = "wrong"
            elif b["correct"] is True and b["implausible"]:
                mark = "hidden"
            if mark and (not focus_key or b["species_key"] == focus_key):
                mx, my = x0 + 0.006 * span_s, b["strength"] + 0.075
                if mark == "wrong":
                    ax.plot(mx, my, marker="X", markersize=8, color=WRONG_MARK,
                            markeredgecolor="#ffffff", markeredgewidth=0.7, zorder=6)
                else:
                    ax.plot(mx, my, marker="^", markersize=8.5, color=HIDDEN_MARK,
                            markeredgecolor="#1d1b17", markeredgewidth=0.7, zorder=6)
                    ax.text(mx, my - 0.012, "!", fontsize=5.5, fontweight="bold",
                            color="#1d1b17", ha="center", va="center", zorder=7)
            if color == views.OTHER_COLOR and (b["species_key"], b["start_s"]) in firsts:
                grey_label(ax, x0 + (0.016 * span_s if mark else 0),
                           b["strength"] + 0.03, b["species_key"])

    lane_axes[-1].set_xlabel("time (s)")
    for ax in axes:
        ax.set_xlim(t0, t1)

    # ---- render, and make it click-to-inspect ------------------------------- #
    axes_box = ax_spec.get_position()
    x_ticks = [float(t) for t in ax_spec.get_xticks()]
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    click = streamlit_image_coordinates(Image.open(buf), width="stretch",
                                        key=f"spec_click_{rec_id}", cursor="crosshair")
    stamp = (click or {}).get("unix_time")
    if click and click.get("width") and stamp != st.session_state.get("spec_click_seen"):
        st.session_state["spec_click_seen"] = stamp
        target = views.click_to_time(click["x"] / click["width"], axes_box.x0,
                                     axes_box.x1, t0, t1)
        if target is not None:
            st.session_state["inspect_t"] = target
            st.rerun()

    # ---- audio: one timeline in recording time, lined up under the figure --- #
    wins = [w for w in (views.window_at(det, rec_id, inspect_t)
                        for det in det_by_model.values()) if w]
    inspected = (min(w[0] for w in wins), max(w[1] for w in wins)) if wins else None
    try:
        audio_root = ingest.dataset_audio_root(cfg.store_dir, dataset)
        src = ingest.resolve_audio(cfg.store_dir, dataset, audio_root, rec_id, rec_row["path"])
        y, sr, clip_lo, _ = load_clip(src, t0, t1, pad_s=0.0)
        # Our own generated HTML, never user input — st.iframe runs it as-is.
        st.iframe(player_html(encode_mp3(y, sr), t0, t1, clip_lo, inspected,
                              axes_box.x0, axes_box.x1, x_ticks), height=72)
    except FileNotFoundError as e:
        st.info(f"Audio unavailable ({e}). The spectrogram cache still renders.")

    in_view = {focus_key} if focus_key else set(
        [k for v, h in lanes.values()
         for k in v["species_key"].to_list() + h["species_key"].to_list()]
        + (truth_lane["species_key"].to_list() if not truth_lane.is_empty() else [])
    )
    if annotations is not None:
        st.markdown(
            "<div style='font-size:0.85rem'>"
            "<span style='color:#d0021b;font-weight:700'>✕</span> wrong — not "
            "annotated at that time &nbsp;·&nbsp; "
            "<span style='color:#b8860b'>⚠</span> a real bird the filter would hide "
            "&nbsp;·&nbsp; <b>- - -</b> dashed: implausible under the judge"
            "</div>", unsafe_allow_html=True)
    k1, k2 = st.columns([0.92, 0.08], vertical_alignment="top")
    if in_view:
        order = list(views.CATEGORICAL) + [views.OTHER_COLOR]
        chips = [
            f'<span style="white-space:nowrap"><span style="color:'
            f'{palette.get(k, views.OTHER_COLOR)}">■</span> '
            f'<a href="{xc_species_url(k)}" target="_blank" title="{k} on Xeno-Canto">'
            f'{name_of(k)}</a></span>'
            for k in sorted(in_view, key=lambda k: (
                order.index(palette.get(k, views.OTHER_COLOR)), name_of(k)))
        ]
        k1.markdown(" &nbsp; ".join(chips), unsafe_allow_html=True)
    with k2.popover("ⓘ", width="stretch"):
        st.markdown(
            "**Reading the figure.** Click anywhere on it to inspect that instant; "
            "the ◀ ▶ buttons step by one window or one span.\n\n"
            "**Listening.** The player under the figure is on the same time axis, in "
            "recording time. The blue band is the inspected window — **▶ window** "
            "plays just that — and clicking the bar plays from that point. The red "
            "line is the playhead.\n\n"
            "- **Solid boxes on the spectrogram** are ground truth, annotated in "
            "time *and* frequency. The models localise in time only.\n"
            "- **Score lanes**: one per model. Each line is a species detected "
            "above *its own* threshold, across the window that detected it. "
            "Height is how far it cleared that threshold, compared with this "
            "model's other detections — the top of the lane is its strongest. "
            "It is a picture, not a measurement: exact scores are in the window "
            "inspector, and **the same height in two lanes does not mean the two "
            "models were equally sure**, because their scores are not comparable.\n"
            "- **Dashed lines** are implausible under the sidebar's judge.\n"
            "- **✕** (red cross) marks a wrong detection: the species is not "
            "annotated at that time, even if it is annotated elsewhere in the "
            "recording. **⚠** marks a real, annotated bird the filter would hide. "
            "No mark: correct, or a mistake the filter would have caught.\n"
            "- The shaded band in each lane is each model's **own window** around "
            "the inspected instant — the 3 s and 5 s grids do not line up.\n"
            "- **Colour** is fixed per species across every model and the whole "
            "recording. Annotated species get colours first, then the most-"
            "detected; past eight colours a species is **grey**, and grey marks "
            "carry their name. Names link to reference recordings on Xeno-Canto."
            "\n\n"
            + ("Thresholds here were fitted on these same recordings, which "
               "flatters every model — see the Thresholds page." if fitted_here
               else ""))

    # ---- window inspector, across every model ------------------------------ #
    heading(
        f"Window inspector — {inspect_t:.1f} s",
        "Each model is asked for **its own** window containing this instant, and the "
        "window is shown alongside: a 3 s and a 5 s grid do not line up, so "
        "pretending there is one shared window would compare different audio. "
        "Listed **regardless of threshold** — the tail is what each model weighed "
        "and rejected; **θ** is the threshold that species has for that model, and "
        "**above** says whether it cleared it. **Score** is what the audio sounds "
        "like; **occurrence** is an independent geographic prior that never heard "
        "it.", level="####")
    table = views.multi_window_table(det_by_model, rec_id, inspect_t,
                                     profile, names_by_model)
    if table.is_empty():
        st.caption("No stored scores in this window for any model.")
    else:
        st.dataframe(
            table, width="stretch", hide_index=True, height=min(420, 60 + 32 * len(table)),
            column_config={
                "score_raw": st.column_config.ProgressColumn(
                    "score", min_value=0.0, max_value=1.0, format="%.3f"),
                "theta": st.column_config.NumberColumn("θ", format="%.4f"),
                "above": st.column_config.CheckboxColumn("above θ"),
                "occ_score": st.column_config.NumberColumn("occurrence", format="%.4f"),
                "suppressed": st.column_config.CheckboxColumn("geofiltered"),
                "xeno_canto": st.column_config.LinkColumn("reference audio",
                                                          display_text="🔊 Xeno-Canto"),
            },
        )
        if annotations is not None:
            if inspected:
                lo, hi = inspected
                w_truth = views.truth_blocks(annotations, rec_id, lo, hi)
                names = sorted({name_of(k) for k in w_truth["species_key"].to_list()})
                st.caption(f"Annotated in {lo:g}–{hi:g} s: " + (", ".join(names) or "—"))


render(sidebar.render())
