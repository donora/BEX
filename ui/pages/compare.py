"""BEX · Compare models — what each model would give you, at the sidebar's settings."""
from ui.common import *  # noqa: F403 — the shared app toolkit
from ui import sidebar


def render(ctx) -> None:
    dataset = ctx.dataset
    annotations = ctx.annotations
    picked = ctx.picked
    run_label = ctx.run_label
    model_style = ctx.model_style
    profile_name = ctx.profile_name
    judge = ctx.judge
    delta = ctx.delta
    spec = ctx.spec
    p_floor = ctx.p_floor
    min_support = ctx.min_support
    min_pos = ctx.min_pos
    min_labelled = ctx.min_labelled
    resolved = ctx.resolved
    fitted_here = ctx.fitted_here

    heading(
        "Compare models",
        "**If you processed your audio with each model, at the sidebar's thresholds "
        "and settings, what would you get?** Every number here is at the thresholds "
        "in force — the same ones the Explorer draws with — and judged against the "
        "annotations, on the annotated species only (a detection of anything else "
        "cannot be judged).\n\n"
        f"**The rule in force — {spec.label}:** "
        f"{spec.describe()[0].upper() + spec.describe()[1:]} Judge: **{judge}**.\n\n"
        "Change the rule, the judge or the models in the sidebar and the whole page "
        "follows."
        + ("\n\n⚠️ The thresholds were fitted on these same recordings, which "
           "flatters every model — most of all on rare species." if fitted_here else ""),
        level="##")
    # The rule decides every number below, so it is stated up front, in words,
    # not left to a caption or an ⓘ.
    rule_tag = rule_banner(spec)

    ready = [r for r in complete_runs(dataset) if r in picked]
    if annotations is None:
        st.info(f"**{dataset}** has no annotations, so there is nothing to measure the "
                "models against. The Explorer still shows what each one detects.")
        return
    if not ready:
        st.info("No included model has scored every recording in this dataset yet — "
                "a partly-scored run would look like a silent model on everything it "
                "has not reached. See the Models page.")
        return
    unaligned = [r for r in ready
                 if not truth.aligned_path(cfg.store_dir, dataset, r, "native").exists()]
    if unaligned:
        st.info(f"{len(unaligned)} run(s) need aligning against the annotations first. "
                "This reads every stored score matrix, takes a minute, and is cached.")
        if st.button(f"Align {len(unaligned)} run(s)", key="pr_build", type="primary"):
            for rid in unaligned:
                build_alignment(rid, dataset, "native")
            st.rerun()
        return

    labels = {run_label[r]: r
              for r in sorted(ready, key=lambda r: model_style[r]["rank"])}
    order = list(labels)
    colour = {m: model_style[r]["colour"] for m, r in labels.items()}
    shape = {m: model_style[r]["shape"] for m, r in labels.items()}
    model_colour = alt.Scale(domain=order, range=[colour[m] for m in order])
    model_shape = alt.Scale(domain=order, range=[shape[m] for m in order])
    with st.spinner("measuring every model at the thresholds in force…"):
        B = {m: get_benchmark(rid, dataset, judge, profile_name,
                              resolved_key(resolved[rid]), p_floor, min_support, min_pos)
             for m, rid in labels.items()}
    nm = lambda k: views.display_name(k, get_names())

    for m, b in B.items():
        if b["resolution"]["damaged"]:
            st.warning(f"**{m}**: its stored scores are too coarse to rank reliably, so "
                       "its curve and AP are not a fair measure of the model (details "
                       "under *All the numbers*). Its outcomes at the thresholds in "
                       "force below are unaffected.")

    # ---- 1 · headline cards: start with what each model would give you ------- #
    heading(
        f"What each model would give you — at {rule_tag}",
        "One card per model, at the thresholds in force.\n\n"
        "- **When it reports a bird** — every window it reports, by outcome. Blue is "
        "right; orange is a bird that was not there. A detection the judge calls "
        "implausible is one the filter would remove before you saw it: purple if it "
        "was right after all (a real bird hidden), grey if hiding it was correct.\n"
        "- **False detections per hour** — wrong detections you would actually see, "
        "counted as events (one bird wrongly reported across three windows is one "
        "false detection), per hour of audio.\n"
        "- **When a bird is singing** — of the annotated songs, how many it found at "
        "least once. Counted per song, not per window, so a 3 s and a 5 s model are "
        "compared fairly.\n"
        "- **Species by species** — one dot per species with enough labelled data "
        f"(≥ {min_labelled} labelled windows, set in the sidebar), at the share of "
        "its songs found. A tight cluster is a consistent model; a spread one is "
        "excellent on some birds and blind to others.")
    st.markdown(
        "<div style='font-size:0.85rem'>" + outcome_key(list(benchmark.REPORT_OUTCOMES))
        + " &nbsp; <span style='color:" + views.OUTCOME_COLOURS["missed"]
        + "'>■</span> missed</div>", unsafe_allow_html=True)
    per_row = 3
    for start in range(0, len(order), per_row):
        cols = st.columns(per_row)
        for col, m in zip(cols, order[start:start + per_row]):
            s, ps = B[m]["summary"], B[m]["per_species"]
            with col.container(border=True):
                st.markdown(f"**{m}**")
                rep_total = sum(s["report"].values()) or 1
                st.caption("When it reports a bird")
                st.altair_chart(outcome_bar({k: v / rep_total for k, v in s["report"].items()},
                                            list(benchmark.REPORT_OUTCOMES)), width="stretch")
                st.markdown(f"≈ **{s['false_per_hour']:.0f}** false detections per hour "
                            "of audio")
                sing_total = sum(s["singing"].values()) or 1
                st.caption("When a bird is singing")
                st.altair_chart(outcome_bar(
                    {("correct" if k == "found" else k): v / sing_total
                     for k, v in s["singing"].items()},
                    ["correct", "filter hid a real bird", "missed"]), width="stretch")
                st.markdown(f"Finds **{s['found']:.0%}** of songs")

                ranked = ps.filter((pl.col("n_positive") >= min_labelled)
                                   & pl.col("found").is_not_null())
                st.caption(f"Species by species ({ranked.height} with enough data)")
                if not ranked.is_empty():
                    strip = ranked.with_columns(
                        species=pl.col("species_key").map_elements(nm, return_dtype=pl.Utf8)
                    ).to_pandas()
                    st.altair_chart(
                        alt.Chart(strip).mark_tick(thickness=2, size=18, opacity=0.8,
                                                   color=colour[m]).encode(
                            x=alt.X("found:Q", title=None,
                                    scale=alt.Scale(domain=[0, 1]),
                                    axis=alt.Axis(format=".0%", tickCount=5)),
                            tooltip=[alt.Tooltip("species:N"),
                                     alt.Tooltip("found:Q", title="songs found",
                                                 format=".0%")],
                        ).properties(height=34), width="stretch")
                    lo, hi = ranked["found"].min(), ranked["found"].max()

                    def who(v):
                        ks = ranked.filter(pl.col("found") == v)["species_key"].to_list()
                        return ", ".join(nm(k) for k in ks[:2]) + (
                            f" and {len(ks) - 2} more" if len(ks) > 2 else "")
                    st.markdown(f"<div style='font-size:0.85rem'>worst <b>{lo:.0%}</b> — "
                                f"{who(lo)}<br>best <b>{hi:.0%}</b> — {who(hi)}</div>",
                                unsafe_allow_html=True)

    # ---- 2 · precision–recall, with where your settings put each model -------- #
    heading(
        f"Precision and recall — where {rule_tag} puts each model",
        "The standard picture of a detector. **Precision** (up): when the model "
        "reports a bird, how often it is right. **Recall** (across): of the windows "
        "where a bird was singing, how many it reported.\n\n"
        "Each **line** is a model swept across every possible threshold, with all "
        "annotated species pooled. Each **marker** is where the sidebar's settings "
        "actually put it. Because those thresholds are set species by species, the "
        "marker need not sit on the pooled line: **above the line means the "
        "per-species thresholds beat any single threshold.**\n\n"
        "Up and to the right is better. The shape of a line says how much precision "
        "a model gives up to find more birds.")
    st.markdown(
        "**Lines**: one threshold for every species, swept through every value. "
        f"**Markers**: your rule, {rule_tag} — a threshold per species. A marker "
        "above its line means per-species thresholds beat any single one.")
    curves = pd.concat([B[m]["micro"].assign(model=m) for m in order])
    points = pd.DataFrame([{"model": m, "precision": B[m]["summary"]["precision"],
                            "recall": B[m]["summary"]["window_recall"], "zero": 0.0}
                           for m in order])
    axis = alt.Axis(format=".0%", tickCount=5)
    lines = alt.Chart(curves).mark_line(strokeWidth=2, opacity=0.9).encode(
        x=alt.X("recall:Q", title="recall", scale=alt.Scale(domain=[0, 1]), axis=axis),
        y=alt.Y("precision:Q", title="precision", scale=alt.Scale(domain=[0, 1]), axis=axis),
        color=alt.Color("model:N", scale=model_colour,
                        legend=alt.Legend(orient="top", title=None)),
        order="theta:Q")
    marks = alt.Chart(points).mark_point(size=220, filled=True, stroke="#ffffff",
                                         strokeWidth=1.5).encode(
        x="recall:Q", y="precision:Q",
        color=alt.Color("model:N", scale=model_colour, legend=None),
        shape=alt.Shape("model:N", scale=model_shape, legend=None),
        tooltip=[alt.Tooltip("model:N"),
                 alt.Tooltip("precision:Q", format=".1%"),
                 alt.Tooltip("recall:Q", format=".1%")])
    guide_x = alt.Chart(points).mark_rule(strokeDash=[4, 3], strokeWidth=1.2,
                                          opacity=0.7).encode(
        x="recall:Q", y="zero:Q", y2="precision:Q",
        color=alt.Color("model:N", scale=model_colour, legend=None))
    guide_y = alt.Chart(points).mark_rule(strokeDash=[4, 3], strokeWidth=1.2,
                                          opacity=0.7).encode(
        y="precision:Q", x="zero:Q", x2="recall:Q",
        color=alt.Color("model:N", scale=model_colour, legend=None))
    st.altair_chart(style_chart((lines + guide_x + guide_y + marks).properties(height=360)),
                    width="stretch")
    st.markdown(
        "<div style='font-size:0.9rem'>At " + spec.label + ": " + " &nbsp;·&nbsp; ".join(
            f"<span style='color:{colour[m]}'>●</span> <b>{m}</b> precision "
            f"<b>{B[m]['summary']['precision']:.0%}</b>, recall "
            f"<b>{B[m]['summary']['window_recall']:.0%}</b>" for m in order)
        + "</div>", unsafe_allow_html=True)

    # ---- 3 · species by species, side by side --------------------------------- #
    heading(
        f"Where the models differ, species by species — at {rule_tag}",
        "One row per species with enough labelled data, one marker per model, "
        "sorted so the species where the choice of model matters most come first.\n\n"
        "- **Songs found** — of that bird's annotated songs, the share each model "
        "found (the misses question).\n"
        "- **Right when it says so** — of the windows where a model reported that "
        "bird, the share that were right (the false-alarm question).\n\n"
        "Read them together: a model can find every song by reporting the bird "
        "everywhere, and the right-hand panel is where that shows.\n\n"
        "**Average precision** instead summarises each model's whole precision–"
        "recall curve for the species — every threshold at once. It is the fairest "
        "measure of the *model*, independent of your settings, but abstract, and "
        "low for rare species even when the model is decent.")
    metric = st.radio("Show", ["songs found & right when it says so",
                               "average precision (all thresholds)"],
                      horizontal=True, key="cmp_metric", persist_state="session",
                      label_visibility="collapsed")
    long = pl.concat([
        B[m]["per_species"].filter(pl.col("n_positive") >= min_labelled)
        .with_columns(model=pl.lit(m)) for m in order])
    if long.is_empty():
        st.info(f"No species has {min_labelled} or more labelled windows.")
    else:
        long = long.with_columns(
            species=pl.col("species_key").map_elements(nm, return_dtype=pl.Utf8))
        key_metric = "found" if metric.startswith("songs") else "ap"
        spread = (long.group_by("species")
                  .agg(gap=pl.col(key_metric).max() - pl.col(key_metric).min())
                  .sort("gap", descending=True, nulls_last=True))
        sort = spread["species"].to_list()
        df = long.to_pandas()
        y = alt.Y("species:N", sort=sort, title=None,
                  # Every species named: at a tight step the axis drops alternate
                  # labels, and the markers then read as belonging to the wrong row.
                  axis=alt.Axis(labelLimit=180, labelFontSize=11, labelOverlap=False))

        def panel(field: str, title: str) -> alt.Chart:
            rng = (long.group_by("species")
                   .agg(lo=pl.col(field).min(), hi=pl.col(field).max()).to_pandas())
            x = alt.X(f"{field}:Q", title=title, scale=alt.Scale(domain=[0, 1]),
                      axis=alt.Axis(format=".0%", tickCount=5, orient="top"))
            link = alt.Chart(rng).mark_rule(color="#c9c7bf", strokeWidth=3).encode(
                y=y, x=alt.X("lo:Q", scale=alt.Scale(domain=[0, 1])), x2="hi:Q")
            dots = alt.Chart(df).mark_point(size=90, filled=True, opacity=0.95).encode(
                y=y, x=x,
                color=alt.Color("model:N", scale=model_colour,
                                legend=alt.Legend(orient="top", title=None)),
                shape=alt.Shape("model:N", scale=model_shape, legend=None),
                tooltip=[alt.Tooltip("species:N"), alt.Tooltip("model:N"),
                         alt.Tooltip(f"{field}:Q", title=title, format=".1%"),
                         alt.Tooltip("n_positive:Q", title="labelled windows")])
            return (link + dots).properties(height=alt.Step(22), width=330)

        chart = (alt.hconcat(panel("found", "songs found"),
                             panel("precision", "right when it says so"))
                 if key_metric == "found" else panel("ap", "average precision"))
        st.altair_chart(style_chart(chart.resolve_scale(y="shared")
                                    if key_metric == "found" else chart), width="content")
        excluded = B[order[0]]["per_species"].filter(
            pl.col("n_positive") < min_labelled).height
        if excluded:
            st.caption(f"{excluded} annotated species have fewer than {min_labelled} "
                       "labelled windows and are left out.")

    # ---- 4 · coverage --------------------------------------------------------- #
    heading(
        f"Which species each model finds at all — at {rule_tag}",
        "Every annotated species, grouped by which models find at least one of its "
        "songs at the thresholds in force. **Only one model** is the interesting "
        "group: a species one model finds and the others miss entirely, confirmed "
        "by the annotations — a real difference in what the models can hear, worth "
        "listening to in the Explorer (pick it as the focus species).")
    species_all = sorted(set().union(*[
        set(B[m]["per_species"].filter(pl.col("vocalisations") > 0)["species_key"])
        for m in order]))
    found_by = {m: set(B[m]["per_species"].filter(pl.col("found_n") > 0)["species_key"])
                for m in order}
    groups = benchmark.coverage(found_by, species_all)
    gdf = pd.DataFrame([{"group": g, "species": len(v)} for g, v in groups.items()])
    group_colour = {"all models": views.OUTCOME_COLOURS["correct"],
                    "some, not all": "#9ec5f4", "no model": views.OUTCOME_COLOURS["missed"]}
    group_colour.update({f"only {m}": colour[m] for m in order})
    st.altair_chart(style_chart(
        alt.Chart(gdf).mark_bar(height=18, cornerRadiusEnd=3).encode(
            y=alt.Y("group:N", sort=list(groups), title=None,
                    axis=alt.Axis(labelLimit=220)),
            x=alt.X("species:Q", title="annotated species", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("group:N", legend=None, scale=alt.Scale(
                domain=list(group_colour), range=list(group_colour.values()))),
            tooltip=["group:N", "species:Q"],
        ).properties(height=alt.Step(26))), width="stretch")
    for g, members in groups.items():
        if members and g != "all models":
            with st.expander(f"{g} — {len(members)} species"):
                st.write(", ".join(nm(k) for k in members))

    # ---- 5 · every number ----------------------------------------------------- #
    with st.expander("All the numbers"):
        st.markdown("##### At the thresholds in force")
        st.dataframe(pl.DataFrame([{
            "model": m,
            "precision": B[m]["summary"]["precision"],
            "recall (windows)": B[m]["summary"]["window_recall"],
            "songs found": B[m]["summary"]["found"],
            "false detections / hour": B[m]["summary"]["false_per_hour"],
            **{f"reported: {k}": v for k, v in B[m]["summary"]["report"].items()},
            **{f"songs: {k}": v for k, v in B[m]["summary"]["singing"].items()},
            "cmAP": B[m]["cmap"], "micro AP": B[m]["micro_ap"],
        } for m in order]), width="stretch", hide_index=True, column_config={
            "precision": st.column_config.NumberColumn(format="%.3f"),
            "recall (windows)": st.column_config.NumberColumn(format="%.3f"),
            "songs found": st.column_config.NumberColumn(format="%.3f"),
            "false detections / hour": st.column_config.NumberColumn(format="%.1f"),
            "cmAP": st.column_config.NumberColumn(format="%.4f"),
            "micro AP": st.column_config.NumberColumn(format="%.4f"),
        })
        st.markdown("##### Every species")
        wide = pl.concat([
            B[m]["per_species"].select(
                "species_key", "n_positive", "vocalisations",
                pl.col("found").alias(f"{m} · found"),
                pl.col("precision").alias(f"{m} · precision"),
                pl.col("ap").alias(f"{m} · AP"))
            for m in order[:1]] + [
            B[m]["per_species"].select(
                "species_key",
                pl.col("found").alias(f"{m} · found"),
                pl.col("precision").alias(f"{m} · precision"),
                pl.col("ap").alias(f"{m} · AP"))
            for m in order[1:]], how="align").with_columns(
            species=pl.col("species_key").map_elements(nm, return_dtype=pl.Utf8))
        wide = wide.select("species", *[c for c in wide.columns if c != "species"])
        st.dataframe(wide, width="stretch", hide_index=True, height=360)
        st.download_button(
            "Download per-species table (CSV)", wide.write_csv().encode(),
            file_name=f"bex-compare-{dataset}-{th.slug(spec.label)}.csv",
            mime="text/csv", key="cmp_export",
            help="Every species, every model, at the thresholds in force.")

        st.markdown("##### Across every threshold (cmAP)")
        grid_label = st.radio(
            "Window grid", ["native", "shared-1s"], horizontal=True, key="pr_grid",
            persist_state="session",
            help="native: each model's own windows (3 s BirdNET, 5 s Perch). "
                 "shared-1s: every model projected onto 1 s frames, so the curves "
                 "share a denominator.")
        grid_name = "native" if grid_label == "native" else "shared-1s"
        todo = [r for r in ready
                if not truth.aligned_path(cfg.store_dir, dataset, r, grid_name).exists()]
        if todo:
            if st.button(f"Align {len(todo)} run(s) on the {grid_name} grid",
                         key="pr_build_grid"):
                for rid in todo:
                    build_alignment(rid, dataset, grid_name)
                st.rerun()
        else:
            cards = {m: get_scorecard(rid, dataset, grid_name, p_floor, min_support,
                                      min_pos) for m, rid in labels.items()}
            league = pl.DataFrame([{**c["row"], "model": m} for m, c in cards.items()])
            st.dataframe(league.select("model", *[c for c in ("cmap", "micro_ap",
                                                              "n_species", "n_excluded")
                                                  if c in league.columns]),
                         width="stretch", hide_index=True, column_config={
                             "cmap": st.column_config.NumberColumn("cmAP", format="%.4f"),
                             "micro_ap": st.column_config.NumberColumn("micro AP",
                                                                       format="%.4f"),
                             "n_species": "species scored", "n_excluded": "excluded"})
            st.caption(
                "cmAP: average precision per species, averaged over species unweighted "
                "(a scarce warbler gets the same vote as a robin in a third of all "
                "windows). Micro AP pools every (window, species) pair, and is "
                "dominated by whichever birds sing most. Species with fewer than "
                f"{min_pos} labelled windows are left out of cmAP.")
        manifests = {m.run_id: m for m in views.list_runs(cfg.store_dir)}
        for m, rid in labels.items():
            ok, why = truth.across_window_comparable(manifests[rid])
            if not ok:
                st.caption(f"**{m}** — {why}")
            res = B[m]["resolution"]
            if res["damaged"]:
                st.caption(
                    f"**{m}** — only {res['median_distinct_fraction']:.1%} of its "
                    "top-ranked windows have scores the store can tell apart: its "
                    "saturated sigmoid scores were stored as float16, whose steps near "
                    "1.0 are coarser than the differences between them. AP measures "
                    "the storage format here, not the model; a float32 re-run fixes it.")


render(sidebar.render())
