"""BEX · Geofilter forensics."""
from ui.common import *  # noqa: F403 — the shared app toolkit
from ui import sidebar

# What the filter hides, in the navigator's colours for the same outcomes.
# Lost keeps the navigator's "filter hid a real bird" purple; removed is a bold
# green here rather than the navigator's pale grey, which vanished against the
# let-through track. Not the Explorer's ground-truth green (#1baf7a).
MISTAKE_COLOUR = "#2e8b3e"
LOST_COLOUR = views.OUTCOME_COLOURS["filter hid a real bird"]
TRACK_COLOUR = "#dcdad3"


def render(ctx) -> None:
    dataset = ctx.dataset
    recordings = ctx.recordings
    annotations = ctx.annotations
    run_label = ctx.run_label
    included = ctx.included
    model_style = ctx.model_style
    profile_name = ctx.profile_name
    judge = ctx.judge
    delta = ctx.delta
    spec = ctx.spec
    resolved = ctx.resolved
    picked = ctx.picked

    def fname(key: str) -> str:
        return views.display_name(key, get_names())

    heading(
        "Geofilter forensics",
        "A location filter suppresses birds it does not expect at the site. This "
        "page asks **what that actually does to each model's output**: how much it "
        "hides, whether it was right to, and which birds each model detects that "
        "are then suppressed. (How confusable each bird is — its crowd — is on the "
        "Species scorecard.)\n\n"
        "**The judge** (sidebar) decides what is suppressed. *profile* is "
        "the plausibility list, applied to every model the same way — the fair "
        "comparison. *geofilter* is each model's own location filter, replayed — "
        "only models that ship one have it. Either way this page calls what the "
        "judge rules out **hidden**, and the rest **let through**.\n\n"
        "Every model is read at its own per-species thresholds from the sidebar "
        "rule, as on every other page, over every species it can name — most of "
        "what a filter removes is birds nobody annotated, because they are not here.",
        level="##")
    rule_tag = rule_banner(
        spec, f"Judge: <b>{judge}</b>. Change it in the sidebar; thresholds species by "
              "species on the Thresholds page.")

    # Say which filter, every time: under the profile judge nothing is a geofilter.
    filt = {"geofilter": "the geofilter", "profile": "the plausibility profile"}.get(
        judge, "the filter")
    models = sorted(included, key=lambda m: model_style[m.run_id]["rank"])
    if not models:
        st.info("No models included — pick some in the sidebar.")
        return
    colour = {m.run_id: model_style[m.run_id]["colour"] for m in models}
    label = {m.run_id: run_label[m.run_id] for m in models}
    order = [label[m.run_id] for m in models]
    model_scale = alt.Scale(domain=order, range=[colour[m.run_id] for m in models])
    no_filter = {m.run_id for m in models
                 if judge == "geofilter" and not m.geofilter.get("model")}

    with st.spinner("judging every model's detections…"):
        G = {m.run_id: get_geo(m.run_id, dataset, judge, profile_name,
                               resolved_key(resolved[m.run_id]))
             for m in models}
    judged_models = [m for m in models if m.run_id not in no_filter]

    # ---- what the filter does, per model ------------------------------------ #
    heading(
        f"What the filter does to each model — at {rule_tag}",
        "One card per model, about every detection it reports above its "
        "thresholds — before the filter acts.\n\n"
        "- **Hides N%** — the share of those detections the filter removes. The "
        "top bar is all of them; the coloured slice at its left is what it hides.\n"
        "- **The bar underneath** zooms into that slice and splits it by the "
        "annotations: **mistakes removed** — the bird was not annotated in that "
        "window, so hiding it helped — and **correct detections lost** — the bird "
        "*was* annotated there, so hiding it cost you a real detection. The two add "
        "up to everything it hides. Annotations are never complete, so a few "
        "\"mistakes removed\" may be birds nobody marked: that number is a little "
        "flattering.\n"
        "- **Precision without → with the filter** — of the detections you see, "
        "the share that are right: everything, or only what the filter lets "
        "through.\n"
        "- **Recall without → with** — of the annotated songs, the share the model "
        "finds, counted per song as on the Compare page. Read the two together: a "
        "filter that hides nearly everything can still lift precision, and only "
        "recall shows what it cost.\n"
        "- **A suppressed bird in N% of windows** — of the windows where the model "
        "detects anything, how many also hold a bird the filter suppresses.\n"
        "- **Always hides N of the annotated species** — one cell per annotated "
        "species; filled are those the filter suppresses every time the model scores "
        "them, so the filter hides them whatever the audio says. *Which species* "
        "lists them.")
    if any(G[m.run_id]["circular"] for m in judged_models) and judge == "profile":
        st.warning(
            f"**Circular under this judge.** No annotated species is ruled out by the "
            f"*{profile_name}* profile — it was derived from these annotations — so "
            "*correct detections lost* is zero by construction, not by merit. Switch "
            "the judge to **geofilter** to test a real location prior.")
    cols = st.columns(len(models))
    for col, m in zip(cols, models):
        g, rid = G[m.run_id], m.run_id
        with col.container(border=True):
            st.markdown(f"<span style='color:{colour[rid]}; font-size:1.1rem'>●</span> "
                        f"**{label[rid]}**", unsafe_allow_html=True)
            if rid in no_filter:
                st.markdown("*Ships no geofilter, so nothing is hidden under this "
                            "judge.* Switch the judge to **profile** to compare it.")
                continue
            o, total = g["outcomes"], g["reported"]
            if not total:
                st.markdown("*Reports nothing at these thresholds.*")
                continue
            hidden = sum(o.get(k, 0) for k in (*geofilter.HIDDEN, "hidden"))
            st.markdown(f"Hides <b style='font-size:1.35rem'>{hidden / total:.1%}</b> "
                        f"of its detections<br><span style='font-size:0.8rem; "
                        f"opacity:0.7'>{hidden:,} of the {total:,} it reports above "
                        "its thresholds</span>", unsafe_allow_html=True)
            st.altair_chart(hidden_bar(o, total, colour[rid]), width="stretch")
            if "correct" in o and hidden:
                removed, lost = o["filter caught a mistake"], o["filter hid a real bird"]
                st.markdown(
                    "<div style='font-size:0.85rem; line-height:1.6'>"
                    f"<span style='color:{MISTAKE_COLOUR}'>■</span> <b>{removed:,}</b> "
                    f"mistakes removed ({removed / hidden:.0%})<br>"
                    f"<span style='color:{LOST_COLOUR}'>■</span> <b>{lost:,}</b> correct "
                    f"detections lost ({lost / hidden:.0%})</div>", unsafe_allow_html=True)
            rows_pr = [("Precision", "of what you see, the share that is right",
                        geofilter.precision_with_and_without(o))]
            if g["songs"]:
                rows_pr.append(("Recall", "of the annotated songs, the share found",
                                geofilter.recall_with_and_without(g["songs"])))
            for name, what, (before, after) in rows_pr:
                if before == before and after == after:
                    st.markdown(before_after(name, what, before, after),
                                unsafe_allow_html=True)
            mw = g["muddy"]
            if mw["windows"]:
                st.markdown(
                    f"<div style='font-size:0.85rem; margin-top:0.4rem'>A suppressed bird "
                    f"in <b>{mw['muddy_window_rate']:.0%}</b> of the windows where it reports "
                    "anything</div>", unsafe_allow_html=True)
            if g["annotated_species"]:
                ks, n_ann = g["unreportable"], g["annotated_species"]
                st.markdown(
                    "<div style='font-size:0.85rem; margin-top:0.6rem'>Always hides "
                    f"<b style='font-size:1.1rem'>{len(ks)}</b> of the {n_ann} annotated "
                    "species</div>", unsafe_allow_html=True)
                st.altair_chart(unit_bar(len(ks), n_ann), width="stretch")
                if ks:
                    with st.popover("Which species", width="stretch"):
                        st.markdown(
                            f"{filt[0].upper() + filt[1:]} suppresses these every time this model scores "
                            "them, so the filter hides them whatever the audio says:\n\n"
                            + "\n".join(f"- {fname(k)}" for k in ks))

    if not judged_models:
        return

    # ---- the most-suppressed birds, species by species ---------------------- #
    heading(
        f"The birds {filt} suppresses most — detected above threshold, then hidden "
        f"— at {rule_tag}",
        f"Every species {filt} suppresses, with how many times each model detected it "
        "above its threshold before it was hidden. One column per model, one row per "
        "bird. Bar length is detections, or the recordings they are spread over "
        "(*Order by*) — a bird detected in one noisy recording is a glitch, one "
        "detected across most recordings is something the model persistently hears "
        "in this soundscape (often a local bird with a similar call).\n\n"
        "**⚠** marks a bird that *was* annotated in some of those windows — there "
        "the suppression cost a correct detection.")
    imp_rows = []
    for m in judged_models:
        for r in G[m.run_id]["impossible"].iter_rows(named=True):
            imp_rows.append({**r, "model": label[m.run_id]})
    if not imp_rows:
        st.info(f"No model detects a bird {filt} suppresses at these thresholds.")
    else:
        imp = pd.DataFrame(imp_rows)
        n_recs = max(1, len(recordings))
        o1, o2 = st.columns([2, 1])
        rank_by = o1.radio("Order by", ["most detections", "most recordings"],
                           horizontal=True, key="geo_imp_order")
        top_n = o2.number_input("Birds shown", 5, 60, 15, 5, key="geo_imp_top")
        key_col = "detections" if rank_by == "most detections" else "recordings"
        ranking = imp.groupby("species_key")[key_col].max().sort_values(ascending=False)
        keep = list(ranking.index[:top_n])
        real_any = set(imp[imp["real"] > 0]["species_key"])
        imp = imp[imp["species_key"].isin(keep)].copy()
        imp["row"] = [("⚠ " if k in real_any else "") + fname(k) for k in imp["species_key"]]
        imp["rec_share"] = imp["recordings"] / n_recs
        imp["value"] = imp[key_col]
        tips = [alt.Tooltip("model:N"), alt.Tooltip("row:N", title="bird"),
                alt.Tooltip("detections:Q"), alt.Tooltip("recordings:Q"),
                alt.Tooltip("rec_share:Q", title="share of recordings", format=".0%"),
                alt.Tooltip("peak:Q", title="strongest score", format=".3f"),
                alt.Tooltip("real:Q", title="annotated after all")]
        st.altair_chart(style_chart(swim_lanes(
            imp, [("⚠ " if k in real_any else "") + fname(k) for k in keep],
            [m for m in judged_models], label, colour, key_col, tips)), width="content")
        st.caption("⚠ = annotated in some of those windows (there the suppression "
                   "cost a correct detection).")
        with st.container(border=True):
            explorer_jump(keep, fname, [(m.run_id, resolved_key(resolved[m.run_id]))
                                        for m in judged_models], judge, profile_name)

    # ---- every number ------------------------------------------------------- #
    with st.expander("All the numbers"):
        st.dataframe(pl.DataFrame([{
            "model": label[m.run_id], "reported": G[m.run_id]["reported"],
            **G[m.run_id]["outcomes"],
            "windows with a suppressed bird": G[m.run_id]["muddy"]["muddy_window_rate"],
        } for m in judged_models]), width="stretch", hide_index=True, column_config={
            "windows with a suppressed bird": st.column_config.NumberColumn(format="%.3f")})
        if imp_rows:
            st.markdown("**Suppressed birds**")
            st.dataframe(pd.DataFrame(imp_rows).assign(
                species=lambda d: [fname(k) for k in d["species_key"]])
                [["model", "species", "detections", "recordings", "peak", "real"]],
                width="stretch", hide_index=True, height=240,
                column_config={"real": "annotated after all",
                               "peak": st.column_config.NumberColumn(format="%.3f")})
        matched_league(dataset, picked, profile_name, delta)


def swim_lanes(df: pd.DataFrame, rows: list[str], models, label: dict, colour: dict,
               x_title: str, tips: list, label_limit: int = 220) -> alt.HConcatChart:
    """One column per model, one striped row per item, a bar in each cell.

    Grouped bars (a bar per model inside each row) turn into a puzzle as soon as
    a model has nothing for some rows: the colours stop repeating in step and a
    bar can no longer be told apart from its neighbour's. Here a model always has
    its own column, so a missing bar is an empty cell."""
    n = len(models)
    width = max(140, min(280, 620 // max(n, 1)))
    x_max = float(df["value"].max()) * 1.18 or 1.0
    stripes = pd.DataFrame({"row": rows, "lo": 0.0, "hi": x_max,
                            "band": [i % 2 for i in range(len(rows))]})
    stripes = stripes[stripes["band"] == 0]
    x_scale = alt.Scale(domain=[0, x_max], nice=False)
    panels = []
    for i, m in enumerate(models):
        name = label[m.run_id]
        sub = df[df["model"] == name]
        y = alt.Y("row:N", sort=rows, scale=alt.Scale(domain=rows), title=None,
                  axis=alt.Axis(labelLimit=label_limit, labelOverlap=False) if i == 0
                  else None)
        # A translucent grey, not a light fill: it reads as a faint stripe on a
        # light or a dark theme alike. The stripe is scenery, so no tooltip.
        band = alt.Chart(stripes).mark_rect(color="#8a8980", opacity=0.09).encode(
            y=y, x=alt.X("lo:Q", scale=x_scale), x2="hi:Q", tooltip=alt.value(None))
        bars = alt.Chart(sub).mark_bar(color=colour[m.run_id], cornerRadiusEnd=2,
                                       height=12).encode(
            y=y, x=alt.X("value:Q", scale=x_scale, title=x_title,
                         axis=alt.Axis(tickCount=4, grid=False)),
            tooltip=tips)
        panels.append(alt.layer(band, bars).properties(
            width=width, height=alt.Step(20),
            title=alt.Title(name, color=colour[m.run_id], fontSize=12, anchor="start")))
    return alt.hconcat(*panels, spacing=10)


def before_after(name: str, what: str, before: float, after: float) -> str:
    """One metric without and with the filter, and the change in points."""
    change = (after - before) * 100
    tone = "#2e9d57" if change > 0.05 else "#d6453a" if change < -0.05 else ink("label")
    return (
        f"<div style='margin-top:0.5rem; font-size:0.85rem'><b>{name}</b> "
        f"<span style='opacity:0.6; font-size:0.75rem'>{what}</span>"
        "<div style='display:flex; align-items:baseline; gap:0.45rem; flex-wrap:wrap'>"
        f"<span><span style='opacity:0.7'>without filter</span> "
        f"<b style='font-size:1.15rem'>{before:.1%}</b></span>"
        "<span style='opacity:0.5'>→</span>"
        f"<span><span style='opacity:0.7'>with</span> "
        f"<b style='font-size:1.15rem'>{after:.1%}</b></span>"
        f"<span style='color:{tone}; font-weight:600'>{change:+.1f} pts</span>"
        "</div></div>")


def unit_bar(hidden: int, total: int) -> alt.Chart:
    """One cell per annotated species, the always-hidden ones filled — a count
    you can see rather than a percentage you have to read."""
    df = pd.DataFrame([{"x0": i / total, "x1": (i + 1) / total,
                        "part": "always hidden" if i < hidden else "can be reported"}
                       for i in range(total)])
    # Built like outcome_bar (a fixed-height bar mark): a bare rect this short
    # rendered nothing inside a card.
    return alt.Chart(df).mark_bar(height=16, stroke="#ffffff",
                                  strokeWidth=1 if total <= 80 else 0).encode(
        x=alt.X("x0:Q", scale=alt.Scale(domain=[0, 1]), axis=None), x2="x1:Q",
        color=alt.Color("part:N", legend=None, scale=alt.Scale(
            domain=["always hidden", "can be reported"], range=[LOST_COLOUR, TRACK_COLOUR])),
        tooltip=[alt.Tooltip("part:N")],
    ).properties(height=24)


def hidden_bar(o: dict[str, int], total: int, colour: str) -> alt.Chart:
    """Top: every reported detection, the hidden slice at the left. Bottom: that
    slice blown up to full width and split into mistakes removed and correct
    detections lost, joined to it by two guide lines — the zoom."""
    hidden = sum(o.get(k, 0) for k in (*geofilter.HIDDEN, "hidden"))
    h = hidden / total if total else 0.0
    rows = [{"x0": 0.0, "x1": 1.0, "y0": 2.3, "y1": 3.0, "part": "let through",
             "n": total - hidden, "share": 1 - h}]
    known = "correct" in o
    if known and hidden:
        removed, lost = o["filter caught a mistake"], o["filter hid a real bird"]
        split = removed / hidden
        rows += [
            {"x0": 0.0, "x1": h * split, "y0": 2.3, "y1": 3.0, "part": "mistakes removed",
             "n": removed, "share": removed / total},
            {"x0": h * split, "x1": h, "y0": 2.3, "y1": 3.0,
             "part": "correct detections lost", "n": lost, "share": lost / total},
            {"x0": 0.0, "x1": split, "y0": 0.0, "y1": 0.9, "part": "mistakes removed",
             "n": removed, "share": split},
            {"x0": split, "x1": 1.0, "y0": 0.0, "y1": 0.9,
             "part": "correct detections lost", "n": lost, "share": 1 - split}]
    elif hidden:
        rows.append({"x0": 0.0, "x1": h, "y0": 2.3, "y1": 3.0, "part": "hidden",
                     "n": hidden, "share": h})
    df = pd.DataFrame(rows)
    x = alt.Scale(domain=[0, 1])
    y = alt.Scale(domain=[0, 3])
    parts = alt.Scale(
        domain=["let through", "mistakes removed", "correct detections lost", "hidden"],
        range=[TRACK_COLOUR, MISTAKE_COLOUR, LOST_COLOUR, colour])
    bars = alt.Chart(df).mark_rect(stroke="#ffffff", strokeWidth=1).encode(
        x=alt.X("x0:Q", scale=x, axis=None), x2="x1:Q",
        y=alt.Y("y0:Q", scale=y, axis=None), y2="y1:Q",
        color=alt.Color("part:N", scale=parts, legend=None),
        tooltip=[alt.Tooltip("part:N"), alt.Tooltip("n:Q", title="detections", format=","),
                 alt.Tooltip("share:Q", format=".1%")])
    layers = [bars]
    if known and hidden:
        # The zoom: from the ends of the hidden slice down to the ends of its blow-up.
        guides = pd.DataFrame([{"xa": 0.0, "xb": 0.0, "ya": 2.3, "yb": 0.9},
                               {"xa": h, "xb": 1.0, "ya": 2.3, "yb": 0.9}])
        layers.append(alt.Chart(guides).mark_rule(color="#b5b4ad", strokeDash=[3, 2]).encode(
            x=alt.X("xa:Q", scale=x), x2="xb:Q", y=alt.Y("ya:Q", scale=y), y2="yb:Q"))
        labels = pd.DataFrame([r for r in rows if r["y0"] == 0.0 and r["x1"] - r["x0"] >= 0.14])
        if len(labels):
            labels["mid"] = (labels["x0"] + labels["x1"]) / 2
            labels["y"] = 0.45
            labels["txt"] = [f"{s_:.0%}" for s_ in labels["share"]]
            layers.append(alt.Chart(labels).mark_text(fontSize=11, fontWeight="bold").encode(
                x=alt.X("mid:Q", scale=x), y=alt.Y("y:Q", scale=y), text="txt:N",
                color=alt.condition(alt.FieldEqualPredicate(field="part",
                                                            equal="mistakes removed"),
                                    alt.value("#ffffff"), alt.value("#ffffff"))))
    return alt.layer(*layers).properties(height=62)


def go_to_explorer(recording_id: str, start_s: float, species_key: str) -> None:
    st.session_state.update(
        explorer_rec=recording_id, _explorer_rec=recording_id, explorer_focus=species_key,
        viewport_start=max(0.0, start_s - 5.0), inspect_t=start_s + 0.5, nav_seen={})
    st.switch_page("ui/pages/explorer.py")


def explorer_jump(species: list[str], fname, runs: list[tuple[str, str]], judge: str,
                  profile_name: str) -> None:
    """Pick one of the charted suppressed birds and open its strongest detection,
    over every model, in the Explorer."""
    l1, l2 = st.columns([3, 1], vertical_alignment="bottom")
    sp = l1.selectbox("Listen to one of these suppressed birds", species,
                      format_func=fname, key="geo_imp_sp")
    if l2.button("Hear it in the Explorer", icon="🔎", key="geo_imp_go"):
        best = None
        for run_id, rkey in runs:
            j = get_judged(run_id, judge, profile_name, rkey)
            hit = (j.filter(pl.col("above") & pl.col("implausible")
                            & (pl.col("species_key") == sp))
                    .sort("score_raw", descending=True))
            if len(hit) and (best is None or hit["score_raw"][0] > best["score_raw"]):
                best = hit.row(0, named=True)
        if best:
            go_to_explorer(best["recording_id"], float(best["start_s"]), sp)


def matched_league(dataset: str, picked: list[str], profile_name: str, delta: float) -> None:
    """Every model at the same number of detections — the §4 statistics at a
    matched operating point, not the sidebar's thresholds (moved here in V1 C1)."""
    ready = [r for r in complete_runs(dataset) if r in picked]
    if len(ready) < 2:
        return
    heading(
        "Every model, at matched detection counts",
        "The PLAN §4 statistics with every model thresholded to make **the same "
        "number of detections**, at one θ each, rather than at the sidebar's "
        "per-species thresholds: how often a model reports a suppressed bird rises "
        "with how much it reports at all, so comparing models that report "
        "different amounts compares nothing. The θ each model needed is itself a fact about the model. Judged "
        "by the plausibility profile, so the numbers mean the same thing for every "
        "model. Reproducible headlessly with `scripts/forensics.py`; "
        "`scripts/crosscheck.py` re-derives them from the raw score matrices.",
        level="#####")
    with st.spinner("computing the matched league…"):
        C = get_comparison(tuple(ready), profile_name, delta, dataset)
    show = [c for c in ["arm", "theta", "detections", "detection_windows",
                        "muddy_window_rate", "median_impostor_mass",
                        "false_suppressions", "false_admissions"]
            if c in C["table"].columns]
    st.dataframe(
        C["table"].select(show).rename({"arm": "model"}), width="stretch",
        hide_index=True,
        column_config={
            "theta": st.column_config.NumberColumn("θ used", format="%.4f"),
            "muddy_window_rate": st.column_config.NumberColumn(
                "muddy windows", format="%.3f"),
            "median_impostor_mass": st.column_config.NumberColumn(
                "impostor mass", format="%.4f"),
        },
    )


render(sidebar.render())
