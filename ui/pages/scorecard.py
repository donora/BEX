"""BEX · Species scorecard."""
from ui.common import *  # noqa: F403 — the shared app toolkit
from ui import sidebar


def render(ctx) -> None:
    dataset = ctx.dataset
    recordings = ctx.recordings
    annotations = ctx.annotations
    picked = ctx.picked
    included = ctx.included
    judge = ctx.judge
    model_style = ctx.model_style
    profile_name = ctx.profile_name
    delta = ctx.delta
    source = ctx.source
    fallback = ctx.fallback
    spec = ctx.spec
    overrides = ctx.overrides
    resolved = ctx.resolved
    fitted_here = ctx.fitted_here

    heading(
        "Species scorecard",
        "The Compare page asks which model is better overall. This asks the "
        "question a surveyor asks about one bird: **what did each model have the "
        "chance to report, how much of it did it find, and how often did it claim "
        "the bird where nobody heard it?**\n\n"
        f"**The rule in force — {spec.label}:** "
        f"{spec.describe()[0].upper() + spec.describe()[1:]}"
        + ("\n\n⚠️ The thresholds were fitted on these same recordings, which "
           "flatters every model — most of all on rare species." if fitted_here else ""),
        level="##")
    rule_tag = rule_banner(spec)

    sc_ready = [r for r in complete_runs(dataset) if r in picked]
    sc_grid = "native"
    sc_unaligned = [r for r in sc_ready
                    if not truth.aligned_path(cfg.store_dir, dataset, r, sc_grid).exists()]
    if annotations is None:
        st.info(f"Dataset **{dataset}** is unlabelled — there is no truth to score against.")
    elif not sc_ready:
        st.info("No run has scored every recording in this dataset yet.")
    elif sc_unaligned:
        st.info(
            f"{len(sc_unaligned)} run(s) need an alignment against truth first. "
            "Build it on the Compare models page, or run "
            "`scripts/evaluate.py --dataset " + dataset + "`."
        )
    else:
        sc_manifests = {m.run_id: m for m in views.list_runs(cfg.store_dir)}
        sc_label_of = views.arm_labels([sc_manifests[r] for r in sc_ready])
        sc_arms = {sc_label_of[r]: r for r in sc_ready}
        sc_order = sorted(sc_arms, key=lambda m: model_style[sc_arms[m]]["rank"])
        # The model palette, as on Compare: never an outcome colour.
        sc_colours = {m: model_style[sc_arms[m]]["colour"] for m in sc_order}
        sc_fname = lambda k: views.display_name(k, get_names())

        first_aligned, _ = get_aligned(sc_arms[sc_order[0]], dataset, sc_grid)
        index = scorecard.species_index(first_aligned, annotations)

        # Every arm at its own per-species θ from the sidebar rule — the same
        # thresholds the Explorer draws with, overrides and fallbacks included.
        rule_name = spec.label
        frames = {label: get_aligned(sc_arms[label], dataset, sc_grid)[0]
                  for label in sc_order}
        all_species = index["species_key"].to_list()
        theta_map = {label: {sp: resolved[sc_arms[label]].theta_of(sp)
                             for sp in all_species}
                     for label in sc_order}

        # ---- overview: every species, every arm ---------------------------- #
        heading(
            f"Every annotated species, and how much of it each model finds — at {rule_tag}",
            "One dot per model per species: the share of that bird's annotated "
            "songs the model detected at least once, at its own threshold for the "
            "species. Dot size is how many songs there are, so a lonely dot high on "
            "the chart is a bird the recordings barely test rather than one the "
            "model has mastered. The grey line joins the models, so its length is "
            "how much the choice of model matters for that bird — where they agree "
            "exactly the dots coincide.")
        with st.spinner("scoring every species…"):
            over_rows = []
            for label in sc_order:
                o = scorecard.detection_overview(frames[label], annotations,
                                                 theta_map[label])
                if not o.is_empty():
                    over_rows.append(o.with_columns(pl.lit(label).alias("arm")))

        if not over_rows:
            st.info("No annotated species to score.")
        else:
            over = (pl.concat(over_rows)
                    .with_columns(pl.col("species_key")
                                  .map_elements(sc_fname, return_dtype=pl.Utf8)
                                  .alias("species")))
            unreachable = over.filter(~pl.col("reachable"))
            plotted = over.filter(pl.col("reachable")).to_pandas()

            f1c, f2c = st.columns([1, 2])
            min_boxes = f1c.number_input(
                "Hide species with fewer than N vocalisations", 0, 500, 0, 5,
                key="sc_minboxes",
                help="A rate over three vocalisations is not a measurement. Raise "
                     "this to see only the birds the eval set can actually judge.")
            sort_by = f2c.selectbox(
                "Order species by", ["best model", "widest disagreement",
                                     "most vocalisations"] + sc_order,
                key="sc_sort")

            plotted = plotted[plotted["boxes"] >= min_boxes]
            if not len(plotted):
                st.info("Nothing left after that filter.")
            else:
                agg = plotted.groupby("species")["detection_rate"]
                if sort_by == "best model":
                    rank = agg.max().sort_values(ascending=False)
                elif sort_by == "widest disagreement":
                    rank = (agg.max() - agg.min()).sort_values(ascending=False)
                elif sort_by == "most vocalisations":
                    rank = (plotted.groupby("species")["boxes"].max()
                            .sort_values(ascending=False))
                else:
                    one = plotted[plotted["arm"] == sort_by]
                    rank = (one.set_index("species")["detection_rate"]
                            .reindex(sorted(set(plotted["species"])))
                            .sort_values(ascending=False, na_position="last"))
                y_order = list(rank.index)

                y_enc = alt.Y("species:N", title=None, sort=y_order,
                              axis=alt.Axis(labelLimit=220))
                spread = (plotted.groupby("species")["detection_rate"]
                          .agg(["min", "max"]).reset_index())
                # padding keeps a dot sitting at 100% from being sliced in half by
                # the plot edge — and 100% is exactly where the interesting dots are.
                x_scale = alt.Scale(domain=[0, 1], padding=14)
                link = alt.Chart(spread).mark_rule(
                    stroke="#c9c8c2", strokeWidth=2
                ).encode(x=alt.X("min:Q", title="vocalisations detected",
                                 scale=x_scale,
                                 axis=alt.Axis(format=".0%", tickCount=5)),
                         x2="max:Q", y=y_enc)
                dots = alt.Chart(plotted).mark_circle(
                    opacity=0.9, stroke="#fcfcfb", strokeWidth=1.5
                ).encode(
                    x=alt.X("detection_rate:Q", scale=x_scale),
                    y=y_enc,
                    size=alt.Size("boxes:Q", title="vocalisations",
                                  scale=alt.Scale(type="sqrt", range=[30, 420])),
                    color=alt.Color("arm:N", title=None,
                                    scale=alt.Scale(domain=sc_order,
                                                    range=[sc_colours[a]
                                                           for a in sc_order]),
                                    legend=alt.Legend(orient="top",
                                                      direction="horizontal")),
                    tooltip=[alt.Tooltip("species:N"), alt.Tooltip("arm:N"),
                             alt.Tooltip("detection_rate:Q", title="detected",
                                         format=".1%"),
                             alt.Tooltip("detected:Q", title="vocalisations found"),
                             alt.Tooltip("askable:Q", title="of"),
                             alt.Tooltip("recordings:Q", title="recordings"),
                             alt.Tooltip("theta:Q", title="θ used", format=".4f")],
                )
                st.altair_chart(
                    style_chart(alt.layer(link, dots)
                                .properties(height=alt.Step(19))),
                    width="stretch",
                )

            if not unreachable.is_empty():
                counts_by_arm = (unreachable.group_by("arm").agg(n=pl.len())
                                 .sort("arm"))
                st.caption(
                    "Not plotted, because no threshold reaches "
                    f"**{rule_name}** for them at all: "
                    + ", ".join(f"{r['arm']} {r['n']}" for r in
                                counts_by_arm.iter_rows(named=True))
                    + f" of {over['species_key'].n_unique()} species. That is a "
                      "result about the arm, not a gap in the chart.")
                with st.expander("Which species, and for which arms"):
                    st.dataframe(
                        unreachable.select("arm", "species", "boxes", "recordings")
                        .sort(["arm", "boxes"], descending=[False, True]),
                        width="stretch", hide_index=True,
                        column_config={"boxes": "vocalisations"})

        st.divider()

        # ---- pick one and drill in ----------------------------------------- #
        pick_sp = st.selectbox(
            "Species", index["species_key"].to_list(), key="sc_species", persist_state="session",
            format_func=lambda k: (
                f"{sc_fname(k)} — {index.filter(pl.col('species_key') == k)['boxes'][0]:,} "
                f"annotated vocalisations"),
        )

        details = {
            label: get_species_detail(sc_arms[label], dataset, judge, profile_name,
                                      resolved_key(resolved[sc_arms[label]]), pick_sp, delta)
            for label in sc_order}
        info = index.filter(pl.col("species_key") == pick_sp).row(0, named=True)
        st.markdown(
            f"### {sc_fname(pick_sp)}\n"
            f"**{info['boxes']:,} annotated songs** across **{info['recordings']} "
            f"recordings**, {info['seconds'] / 60:.0f} minutes of labelled song in total.")

        # ---- one card per model (V1 S2) ------------------------------------ #
        heading(
            f"How each model handles it — at {rule_tag}",
            "One card per model, for this bird only, at the thresholds in force.\n\n"
            "- **Threshold** — the θ this model uses for this bird, and where it came "
            "from: the rule, the fallback (when the rule cannot be met or there is "
            "too little data) or a hand override.\n"
            "- **When it reports this bird** — every window in which it named this "
            "bird, by outcome: right, wrong, and those the filter would hide.\n"
            "- **False detections per hour** — wrong reports of this bird you would "
            "see, counted as events, per hour of audio.\n"
            "- **When this bird is singing** — of its annotated songs, how many the "
            "model found at least once. Counted per song, so a 3 s and a 5 s model "
            "compare fairly.")
        st.markdown(
            "<div style='font-size:0.85rem'>" + outcome_key(list(benchmark.REPORT_OUTCOMES))
            + " &nbsp; <span style='color:" + views.OUTCOME_COLOURS["missed"]
            + "'>■</span> missed</div>", unsafe_allow_html=True)
        SOURCE_WORDS = {"rule": "from the rule", "fallback": "fallback",
                        "override": "set by hand", "set": "from the saved set",
                        "none": "no threshold"}
        cols = st.columns(len(sc_order))
        for col, label in zip(cols, sc_order):
            d, r = details[label], resolved[sc_arms[label]]
            s_, w = d["summary"], d["windows"]
            row = r.table.filter(pl.col("species_key") == pick_sp)
            src = row["source"][0] if len(row) else r.default_source
            why = row["reason"][0] if len(row) else ""
            theta_here = r.theta_of(pick_sp)
            with col.container(border=True):
                st.markdown(f"**{label}**")
                st.markdown(
                    f"<div style='font-size:0.85rem'>θ <b>{theta_here:.4g}</b> · "
                    f"{SOURCE_WORDS.get(src, src)}"
                    + (f" <span style='opacity:0.7'>({why})</span>"
                       if src in ("fallback", "none") and why else "")
                    + "</div>", unsafe_allow_html=True)
                st.caption("When it reports this bird")
                rep_total = sum(s_["report"].values())
                if rep_total:
                    st.altair_chart(outcome_bar({k: v / rep_total for k, v in s_["report"].items()},
                                                list(benchmark.REPORT_OUTCOMES)), width="stretch")
                    st.markdown(f"≈ **{s_['false_per_hour']:.1f}** false detections per "
                                "hour of audio")
                else:
                    st.markdown("*Never reports it at this threshold.*")
                st.caption("When this bird is singing")
                sing_total = sum(s_["singing"].values())
                if sing_total:
                    st.altair_chart(outcome_bar(
                        {("correct" if k == "found" else k): v / sing_total
                         for k, v in s_["singing"].items()},
                        ["correct", "filter hid a real bird", "missed"]), width="stretch")
                    st.markdown(f"Finds **{s_['singing']['found']:,}** of "
                                f"**{sing_total:,}** songs ({s_['found']:.0%})")
        if st.button("Tune this species' thresholds on the Thresholds page",
                     icon="🎚️", key="sc_to_thresholds"):
            st.session_state["th_species"] = pick_sp
            st.switch_page("ui/pages/thresholds.py")

        # ---- recording by recording (V1 S3) -------------------------------- #
        heading(
            f"Which recordings each model does well and badly on — at {rule_tag}",
            "One row per recording, and for each model two columns side by side.\n\n"
            "- **Songs found** — a bar: blue for the share of this bird's annotated "
            "songs the model found, grey for the share it missed, labelled found / "
            "annotated (3/4).\n"
            "- **False detections** — times it reported this bird where nobody "
            "annotated it, counted as events.\n\n"
            "**How strong a colour is shows how many** songs or detections there are, "
            "on one scale for both columns. So a recording where a model found 2 of 2 "
            "songs but made 30 false detections shows a faint full bar beside a deep "
            "orange cell — a perfect score on almost no evidence, next to a real "
            "problem.\n\n"
            "A bird a model handles well overall can be invisible on one recorder or "
            "in the dawn hour, and one number for the whole dataset hides that. "
            "Recordings where the bird is not annotated at all still appear if a "
            "model reported it there — that is where false detections happen; their "
            "*songs found* cell is blank.")
        rec_rows = []
        for label in sc_order:
            t = details[label]["recordings"]
            for r_ in t.iter_rows(named=True):
                rec_rows.append({"recording": r_["recording_id"], "model": label,
                                 "songs": r_["songs"], "found": r_["found"],
                                 "found_n": r_["found_n"],
                                 "false": r_["false_events"]})
        if not rec_rows:
            st.info("No recording has this bird annotated or reported.")
        else:
            recs_df = pd.DataFrame(rec_rows)
            # Rows: most songs first, then most false detections — the recordings
            # worth looking at lead either way.
            order_recs = (recs_df.groupby("recording")
                          .agg(songs=("songs", "max"), false=("false", "max"))
                          .sort_values(["songs", "false"], ascending=False).index.tolist())
            y = alt.Y("recording:N", sort=order_recs, title=None,
                      axis=alt.Axis(labelLimit=240, labelOverlap=False))
            # Songs found: each cell is a bar — blue for the share found, grey for
            # the share missed — and its strength is the number of songs, so
            # 2/2 is a faint full bar and 601/800 a deep three-quarter one.
            # False detections: counts, shaded by count. One square-root scale
            # for both, so a faint 2/2 beside a deep 30 reads as what it is.
            count_max = max(1, int(recs_df["songs"].max()), int(recs_df["false"].max()))
            strength = lambda n: 0.22 + 0.78 * (n / count_max) ** 0.5
            panels = []
            for i, label in enumerate(sc_order):
                sub = recs_df[recs_df["model"] == label].copy()
                rows = []
                for r_ in sub.itertuples():
                    share = (r_.found_n / r_.songs) if r_.songs else 0.0
                    a_ = strength(r_.songs) if r_.songs else 0.0
                    rows.append({"recording": r_.recording, "x0": 0.0, "x1": share,
                                 "part": "found", "alpha": a_, "songs": r_.songs,
                                 "found_n": r_.found_n, "false": r_.false})
                    rows.append({"recording": r_.recording, "x0": share,
                                 "x1": 1.0 if r_.songs else 0.0, "part": "missed",
                                 "alpha": a_, "songs": r_.songs,
                                 "found_n": r_.found_n, "false": r_.false})
                bars_df = pd.DataFrame(rows)
                sub["found_txt"] = [f"{int(f)}/{int(n)}" if n else "" for f, n in
                                    zip(sub["found_n"], sub["songs"])]
                sub["false_txt"] = [str(int(v)) if v else "" for v in sub["false"]]
                sub["alpha"] = [strength(n) if n else 0.0 for n in sub["songs"]]
                sub["false_alpha"] = [strength(v) if v else 0.0 for v in sub["false"]]
                # Cell geometry as data, not constants: Vega-Lite rejects layers
                # that mix datum positions with field positions on one axis.
                sub["lo"], sub["hi"], sub["mid"] = 0.0, 1.0, 0.5
                yy = (y if i == 0 else
                      alt.Y("recording:N", sort=order_recs, title=None, axis=None))
                tips = [alt.Tooltip("recording:N"),
                        alt.Tooltip("found_n:Q", title="songs found"),
                        alt.Tooltip("songs:Q", title="songs annotated"),
                        alt.Tooltip("false:Q", title="false detections")]
                unit = alt.Scale(domain=[0, 1])
                x_unit = alt.X("lo:Q", scale=unit, axis=None, title=None)
                track = alt.Chart(sub).mark_rect(color="#f5f4f0").encode(
                    y=yy, x=x_unit, x2="hi:Q")
                found = alt.Chart(bars_df).mark_rect(stroke="#ffffff", strokeWidth=1).encode(
                    y=yy,
                    x=alt.X("x0:Q", scale=unit, axis=None,
                            title=None), x2="x1:Q",
                    color=alt.Color("part:N", legend=None, scale=alt.Scale(
                        domain=["found", "missed"],
                        range=[views.OUTCOME_COLOURS["correct"],
                               views.OUTCOME_COLOURS["missed"]])),
                    opacity=alt.Opacity("alpha:Q", scale=None, legend=None),
                    tooltip=tips)
                found_t = alt.Chart(sub).mark_text(fontSize=10, fontWeight="bold").encode(
                    y=yy, x=alt.X("mid:Q", scale=unit, axis=None), text="found_txt:N",
                    color=alt.condition("datum.alpha > 0.72", alt.value("#ffffff"),
                                        alt.value("#1d1b17")))
                found_panel = alt.layer(track, found, found_t).properties(
                    width=150, height=alt.Step(20),
                    title=alt.Title("songs found", fontSize=11, fontWeight="normal", anchor="middle"))
                # Recording names only once, at the far left.
                y_bare = alt.Y("recording:N", sort=order_recs, title=None, axis=None)
                false_ = alt.Chart(sub).mark_rect(stroke="#ffffff", strokeWidth=1,
                                                   color=views.OUTCOME_COLOURS["wrong"]).encode(
                    y=y_bare, x=x_unit, x2="hi:Q",
                    opacity=alt.Opacity("false_alpha:Q", scale=None, legend=None),
                    tooltip=tips)
                false_t = alt.Chart(sub).mark_text(fontSize=10, fontWeight="bold").encode(
                    y=y_bare, x=alt.X("mid:Q", scale=unit, axis=None), text="false_txt:N",
                    color=alt.condition("datum.false_alpha > 0.72", alt.value("#ffffff"),
                                        alt.value("#1d1b17")))
                false_panel = alt.layer(false_, false_t).properties(
                    width=60, height=alt.Step(20),
                    title=alt.Title("false det.", fontSize=11, fontWeight="normal", anchor="middle"))
                panels.append(alt.hconcat(found_panel, false_panel, spacing=4,
                                          title=alt.Title(label, fontSize=12,
                                                          anchor="middle")))
            st.altair_chart(style_chart(alt.hconcat(*panels, spacing=18)),
                            width="content")
            st.markdown(
                "<div style='font-size:0.85rem'><span style='color:"
                f"{views.OUTCOME_COLOURS['correct']}'>■</span> songs found &nbsp; "
                f"<span style='color:{views.OUTCOME_COLOURS['missed']}'>■</span> songs "
                f"missed &nbsp; <span style='color:{views.OUTCOME_COLOURS['wrong']}'>■"
                "</span> false detections &nbsp;·&nbsp; stronger colour = more songs or "
                "detections</div>", unsafe_allow_html=True)

            # V1 S6: the window-by-window strip is the Explorer's job — go there.
            l1, l2 = st.columns([3, 1], vertical_alignment="bottom")
            listen = l1.selectbox("Recording", order_recs, key=f"sc_listen_{pick_sp}")
            if l2.button("Listen in the Explorer", icon="🔎", key="sc_listen_go"):
                boxes_here = annotations.filter(
                    (pl.col("recording_id") == listen)
                    & (pl.col("species_key") == pick_sp)).sort("start_s")
                start = float(boxes_here["start_s"][0]) if len(boxes_here) else 0.0
                st.session_state.update(
                    explorer_rec=listen, _explorer_rec=listen, explorer_focus=pick_sp,
                    viewport_start=max(0.0, start - 5.0), inspect_t=start + 0.5,
                    nav_seen={})
                st.switch_page("ui/pages/explorer.py")

        # ---- what it gets confused with (V1 S5) ---------------------------- #
        heading(
            f"What this bird gets confused with — at {rule_tag}",
            "Two different failures.\n\n"
            "**Its crowd (muddiness).** When a model *does* report this bird, which "
            f"other species score within Δ = {delta:.2f} of it (the sidebar's "
            "shadowing band) and clear their own thresholds too. A distinctive song "
            "sings alone; a confusable one drags a crowd. **⚠** marks species the "
            "judge calls implausible here — where the map, not the model, is doing "
            "the identifying.\n\n"
            "**What it is mistaken for.** When this bird *was* singing and a model "
            "missed it, which other annotated bird it reported in those windows "
            "instead. A close relative near the top means the model heard something "
            "and named the wrong bird — a different failure from hearing nothing.")
        # Row by row, so the two charts start at the same height: headers, then
        # charts, then the left panel's per-model summaries underneath.
        h_left, h_right = st.columns(2)
        h_left.markdown("**Its crowd** — who scores almost as high when it is reported")
        h_right.markdown("**What it is mistaken for** — reported instead when it was missed")
        c_left, c_right = st.columns(2)

        def bars(df: pd.DataFrame, x_title: str) -> alt.Chart:
            return alt.Chart(df).mark_bar().encode(
                x=alt.X("share:Q", title=x_title, axis=alt.Axis(format=".0%", tickCount=4)),
                y=alt.Y("species:N", title=None, sort="-x", axis=alt.Axis(labelLimit=200)),
                yOffset=alt.YOffset("model:N", sort=sc_order),
                color=alt.Color("model:N", scale=alt.Scale(
                    domain=sc_order, range=[sc_colours[m] for m in sc_order]),
                    legend=alt.Legend(orient="top", title=None, labelLimit=160,
                                      columns=2)),
                tooltip=[alt.Tooltip("model:N"), alt.Tooltip("species:N"),
                         alt.Tooltip("windows:Q"),
                         alt.Tooltip("share:Q", format=".1%")],
            # With models offset inside each species row, the step sizes one
            # model's bar — so a species row is len(models) × this tall.
            ).properties(height=alt.Step(9))

        with c_left:
            crowd_rows = []
            for label in sc_order:
                for r_ in details[label]["crowd"].iter_rows(named=True):
                    crowd_rows.append({
                        "model": label, "windows": r_["windows"], "share": r_["share"],
                        "species": ("⚠ " if r_["implausible"] else "")
                                   + sc_fname(r_["species_key"])})
            if crowd_rows:
                st.altair_chart(style_chart(bars(pd.DataFrame(crowd_rows),
                                                 "share of its windows")), width="stretch")
            else:
                st.caption("Nothing else scores within Δ when it is reported — it sings "
                           "alone.")
            for label in sc_order:
                cs = details[label]["crowd_summary"]
                if cs["windows"]:
                    st.markdown(
                        f"<div style='font-size:0.85rem'><span style='color:"
                        f"{sc_colours[label]}'>●</span> <b>{label}</b>: on average "
                        f"<b>{cs['companions']:.1f}</b> others within Δ, "
                        f"<b>{cs['implausible_companions']:.1f}</b> of them implausible "
                        f"({cs['windows']:,} windows)</div>", unsafe_allow_html=True)
        with c_right:
            conf_rows = []
            for label in sc_order:
                for r_ in details[label]["mistaken_for"].iter_rows(named=True):
                    conf_rows.append({"model": label, "windows": r_["windows"],
                                      "share": r_["share"],
                                      "species": sc_fname(r_["species_key"])})
            if conf_rows:
                st.altair_chart(style_chart(bars(pd.DataFrame(conf_rows),
                                                 "share of its missed windows")),
                                width="stretch")
            else:
                st.caption("When it was missed, no other annotated bird was reported "
                           "instead — the models went quiet rather than naming another.")

        # ---- every number --------------------------------------------------- #
        with st.expander("All the numbers"):
            st.dataframe(pl.DataFrame([{
                "model": label,
                "θ": resolved[sc_arms[label]].theta_of(pick_sp),
                "songs found": details[label]["summary"]["singing"]["found"],
                "songs hidden by filter": details[label]["summary"]["singing"]["filter hid a real bird"],
                "songs missed": details[label]["summary"]["singing"]["missed"],
                "windows reported": details[label]["windows"]["reported"],
                "windows right": details[label]["windows"]["caught"],
                "windows labelled": details[label]["windows"]["labelled"],
                "precision": details[label]["summary"]["precision"],
                "false detections / hour": details[label]["summary"]["false_per_hour"],
            } for label in sc_order]), width="stretch", hide_index=True, column_config={
                "θ": st.column_config.NumberColumn(format="%.4f"),
                "precision": st.column_config.NumberColumn(format="%.3f"),
                "false detections / hour": st.column_config.NumberColumn(format="%.2f"),
            })
            st.caption("Windows depend on each model's grid (a 30 s song is ten 3 s "
                       "windows but six 5 s ones), so compare models on songs; windows "
                       "are what a threshold is applied to.")
            if rec_rows:
                st.dataframe(pd.DataFrame(rec_rows), width="stretch", hide_index=True)


render(sidebar.render())
