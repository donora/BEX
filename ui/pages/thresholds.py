"""BEX · Thresholds."""
from ui.common import *  # noqa: F403 — the shared app toolkit
from ui import sidebar
from ui.sidebar import NO_SET


def render(ctx) -> None:
    dataset = ctx.dataset
    truth_ref = ctx.truth_ref
    recordings = ctx.recordings
    annotations = ctx.annotations
    has_alignment = ctx.has_alignment
    run_label = ctx.run_label
    model_style = ctx.model_style
    included = ctx.included
    ident_of = ctx.ident_of
    rule_kind = ctx.rule_kind
    sets = ctx.sets
    active_set = ctx.active_set
    loaded = ctx.loaded
    source = ctx.source
    p_floor = ctx.p_floor
    min_support = ctx.min_support
    min_pos = ctx.min_pos
    fallback = ctx.fallback
    fixed = ctx.fixed
    spec = ctx.spec
    overrides = ctx.overrides
    unsaved = ctx.unsaved
    resolved = ctx.resolved
    threshold_notes = ctx.threshold_notes
    fitted_here = ctx.fitted_here
    n_overrides = ctx.n_overrides

    heading(
        "Every threshold, and why it is what it is",
        "The sidebar's **rule** gives every model a θ for every species. Each θ "
        "comes from the first of these that has an answer:\n\n"
        "1. **override** — set by hand, below;\n"
        "2. **rule** — the rule fitted on this dataset's annotations (or **set** — "
        "taken from a saved threshold set);\n"
        "3. **fallback** — the model-wide θ for the rule, or a fixed θ;\n"
        "4. **none** — the model has no operating point for the species, and its "
        "detections of it are not drawn.\n\n"
        "Species that are never annotated (most of a model's thousands of classes) "
        "have no curve to fit, so they always take the fallback.\n\n"
        "**Save** your overrides as a named set to keep them: a set holds the rule, "
        "the overrides and every resolved θ, so it can threshold an unlabelled "
        "dataset later or be shared with someone else.",
        level="###")

    st.markdown(
        f"Rule **{spec.label}**"
        + (f" · thresholds from **{source}**" if source and rule_kind != "fixed" else "")
        + f" · fallback **{fallback}** · set **{active_set}**"
        + (" · **unsaved changes**" if unsaved else ""))
    if fitted_here:
        st.caption(
            "⚠️ Fitted and scored on the same recordings, which flatters every "
            "model — most of all on rare species. Treat these thresholds as a "
            "starting point for other data, not a measured property."
            + (f" **{n_overrides} θ set by hand** while looking at the curves they "
               "are scored on are more optimistic still." if n_overrides else ""))
    for note in threshold_notes:
        st.warning(note)
    if loaded:
        stale = loaded.stale(set(ident_of.values()))
        if stale:
            st.warning(
                f"Set *{loaded.name}* has entries for models that are not included "
                "here, which are not applied: "
                + "; ".join(f"**{m}** ({n} entries)" for m, n in stale.items())
                + ". A θ only means something for the model version and readout "
                  "that produced it.")

    if not included:
        st.info("No models included — pick some in the sidebar.")
    else:
        nm = lambda k: views.display_name(k, get_names())

        # ---- overview: species × model ------------------------------------ #
        st.markdown("#### Overview")
        long_rows = []
        for m in included:
            r = resolved[m.run_id]
            card = (get_scorecard(m.run_id, dataset, "native", p_floor, min_support,
                                  min_pos, truth_ref) if has_alignment(m.run_id) else None)
            for row in r.table.iter_rows(named=True):
                pr = (th.point_at(card["curves"][row["species_key"]], row["theta"])
                      if card and row["species_key"] in card["curves"] else {})
                long_rows.append({
                    "species": nm(row["species_key"]), "species_key": row["species_key"],
                    "model": run_label[m.run_id], "theta": row["theta"],
                    "set_by": row["source"], "precision": pr.get("precision"),
                    "recall": pr.get("recall"), "why": row["reason"]})
            long_rows.append({
                "species": "(every other species)", "species_key": "",
                "model": run_label[m.run_id], "theta": r.default_theta,
                "set_by": r.default_source, "precision": None, "recall": None,
                "why": "species with no curve of their own"})
        overview = pl.DataFrame(long_rows, schema={
            "species": pl.Utf8, "species_key": pl.Utf8, "model": pl.Utf8,
            "theta": pl.Float64, "set_by": pl.Utf8, "precision": pl.Float64,
            "recall": pl.Float64, "why": pl.Utf8})

        f1c, f2c = st.columns([2, 3])
        needle = f1c.text_input("Filter species", key="th_filter",
                                placeholder="name or scientific name")
        shown_src = f2c.multiselect("Set by", th.SOURCES, default=list(th.SOURCES),
                                    key="th_src_filter")
        view = overview.filter(pl.col("set_by").is_in(shown_src))
        if needle:
            view = view.filter(
                pl.col("species").str.to_lowercase().str.contains(needle.lower(),
                                                                  literal=True)
                | pl.col("species_key").str.to_lowercase().str.contains(
                    needle.lower(), literal=True))
        # Problem cells first: what fell back, what has no θ, what was hand-set.
        rank = {"none": 0, "fallback": 1, "override": 2, "set": 3, "rule": 4}
        view = view.with_columns(
            _r=pl.col("set_by").replace_strict(rank, default=5)).sort(
            ["_r", "species", "model"]).drop("_r")
        st.dataframe(view, width="stretch", hide_index=True, height=360,
                     column_config={
                         "species_key": "scientific name",
                         "theta": st.column_config.NumberColumn("θ", format="%.4f"),
                         "set_by": "set by",
                         "precision": st.column_config.NumberColumn(format="%.3f"),
                         "recall": st.column_config.NumberColumn(format="%.3f"),
                     })

        # ---- one species: curves, candidates, overrides ------------------- #
        st.markdown("#### One species")
        curve_species = sorted(
            {k for m in included for k in resolved[m.run_id].table["species_key"]},
            key=lambda k: nm(k).lower())
        if not curve_species:
            st.info("No species has a curve here — there are no annotations to fit "
                    "on. Thresholds come from the fixed θ or the saved set.")
        else:
            pick = st.selectbox("Look up a species", curve_species, key="th_species", persist_state="session",
                                format_func=lambda k: f"{nm(k)} ({k})")
            curve_rows, point_rows = [], []
            for m in included:
                label = run_label[m.run_id]
                cp = (get_curve_points(m.run_id, dataset, pick, p_floor, min_support,
                                       truth_ref)
                      if has_alignment(m.run_id) else None)
                if cp is None:
                    continue
                curve_rows.append(cp["frame"].assign(model=label))
                card = get_scorecard(m.run_id, dataset, "native", p_floor,
                                     min_support, min_pos, truth_ref)
                curve = card["curves"][pick]
                for name, t in [*cp["candidates"].items(),
                                ("in force", resolved[m.run_id].theta_of(pick))]:
                    pt = th.point_at(curve, t)
                    if pt["precision"] == pt["precision"]:
                        point_rows.append({"model": label, "point": name, "theta": t,
                                           "precision": pt["precision"],
                                           "recall": pt["recall"]})
            if curve_rows:
                curves_df = pd.concat(curve_rows)
                pts_df = pd.DataFrame(point_rows)
                model_scale = alt.Scale(
                    domain=[run_label[m.run_id] for m in included],
                    range=[model_style[m.run_id]["colour"] for m in included])
                lines = alt.Chart(curves_df).mark_line(strokeWidth=2).encode(
                    x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("model:N", scale=model_scale,
                                    legend=alt.Legend(orient="top", title=None)),
                    order="theta:Q")
                pts = alt.Chart(pts_df).mark_point(size=110, filled=True,
                                                   strokeWidth=1.5).encode(
                    x="recall:Q", y="precision:Q",
                    color=alt.Color("model:N", scale=model_scale, legend=None),
                    shape=alt.Shape("point:N", legend=alt.Legend(orient="top",
                                                                 title=None)),
                    tooltip=["model:N", "point:N",
                             alt.Tooltip("theta:Q", title="θ", format=".4f"),
                             alt.Tooltip("precision:Q", format=".3f"),
                             alt.Tooltip("recall:Q", format=".3f")])
                st.altair_chart(style_chart(alt.layer(lines, pts).properties(height=320)),
                                width="stretch")
            else:
                st.caption("No precision–recall curve for this species here.")

            st.markdown("**Set θ by hand**")
            with st.form(f"overrides_{pick}"):
                choices = {}
                for m in included:
                    label, ident = run_label[m.run_id], ident_of[m.run_id]
                    r = resolved[m.run_id]
                    row = r.table.filter(pl.col("species_key") == pick)
                    now_theta = r.theta_of(pick)
                    now_src = row["source"][0] if len(row) else r.default_source
                    cp = (get_curve_points(m.run_id, dataset, pick, p_floor,
                                           min_support, truth_ref)
                          if has_alignment(m.run_id) else None)
                    opts = ["use the rule"] + [f"{n} (θ = {t:.4f})" for n, t in
                                               (cp["candidates"].items() if cp else [])]
                    opts.append("custom")
                    current = overrides.get((ident, pick))
                    idx = 0
                    if current is not None:
                        idx = next((i for i, o in enumerate(opts)
                                    if o.endswith(f"(θ = {current:.4f})")),
                                   len(opts) - 1)
                    c1, c2, c3 = st.columns([2, 2, 1])
                    c1.markdown(f"**{label}**  \nnow θ = {now_theta:.4f} · *{now_src}*")
                    choice = c2.selectbox("θ", opts, index=idx, key=f"ovsel_{ident}_{pick}",
                                          label_visibility="collapsed")
                    custom = c3.number_input(
                        "custom θ", 0.0, 1.0,
                        value=float(current if current is not None
                                    else (now_theta if now_theta == now_theta else 0.1)),
                        step=0.001, format="%.4f", key=f"ovnum_{ident}_{pick}",
                        label_visibility="collapsed")
                    choices[ident] = (choice, custom, cp)
                if st.form_submit_button("Apply"):
                    new = dict(overrides)
                    for ident, (choice, custom, cp) in choices.items():
                        if choice == "use the rule":
                            new.pop((ident, pick), None)
                        elif choice == "custom":
                            new[(ident, pick)] = float(custom)
                        else:
                            name = choice.rsplit(" (θ =", 1)[0]
                            new[(ident, pick)] = float(cp["candidates"][name])
                    st.session_state["overrides"] = new
                    st.rerun()

        # ---- named sets ---------------------------------------------------- #
        st.markdown("#### Threshold sets")
        st.caption(f"Saved in `{cfg.thresholds_dir}` — one readable JSON file per set.")

        def current_set(name: str, author: str = "", created: str = "") -> th.ThresholdSet:
            return th.build_set(name, spec, [resolved[m.run_id] for m in included],
                                {k: v for k, v in overrides.items()
                                 if k[0] in set(ident_of.values())},
                                author=author, dataset=dataset, created_utc=created)

        s1, s2 = st.columns(2)
        with s1:
            if loaded:
                st.markdown(f"Active: **{loaded.name}**"
                            + (f" · by {loaded.author}" if loaded.author else "")
                            + (" · **unsaved changes**" if unsaved else " · saved"))
                if st.button("Save", disabled=not unsaved, key="set_save"):
                    ts = current_set(loaded.name, loaded.author, loaded.created_utc)
                    th.save_set(cfg.thresholds_dir, ts)
                    st.session_state["set_loaded"] = ts.to_json()
                    st.rerun()
                with st.popover("Rename or delete"):
                    new_name = st.text_input("New name", value=loaded.name,
                                             key="set_rename_name")
                    if st.button("Rename", key="set_rename"):
                        if new_name.strip() and new_name != loaded.name:
                            ts = th.ThresholdSet.from_json(loaded.to_json())
                            ts.name = new_name.strip()
                            th.save_set(cfg.thresholds_dir, ts)
                            th.delete_set(cfg.thresholds_dir, loaded.name)
                            st.session_state["set_loaded"] = ts.to_json()
                            st.session_state["active_set_pending"] = ts.name
                            st.rerun()
                    sure = st.checkbox(f"Yes, delete “{loaded.name}”", key="set_del_ok")
                    if st.button("Delete", disabled=not sure, key="set_delete"):
                        th.delete_set(cfg.thresholds_dir, loaded.name)
                        st.session_state["set_loaded"] = None
                        st.session_state["overrides"] = {}
                        st.session_state["active_set_pending"] = NO_SET
                        st.rerun()
            else:
                st.markdown("No set active — the rule alone"
                            + (f", plus **{n_overrides} unsaved override(s)**."
                               if n_overrides else "."))
        with s2:
            with st.form("set_save_as", clear_on_submit=True):
                new_name = st.text_input("Save as", placeholder="e.g. dawn survey v1")
                author = st.text_input("Author (optional)")
                if st.form_submit_button("Save as new set"):
                    if not new_name.strip():
                        st.error("A set needs a name.")
                    elif new_name.strip() in sets:
                        st.error(f"“{new_name.strip()}” exists — pick another name, "
                                 "or activate it and Save.")
                    else:
                        ts = current_set(new_name.strip(), author.strip())
                        th.save_set(cfg.thresholds_dir, ts)
                        st.session_state["set_loaded"] = ts.to_json()
                        st.session_state["active_set_pending"] = ts.name
                        st.rerun()

        d1, d2 = st.columns(2)
        d1.download_button(
            "Download current thresholds (JSON)",
            current_set(loaded.name if loaded else "unsaved thresholds",
                        loaded.author if loaded else "").to_json().encode(),
            file_name=f"bex-thresholds-{dataset}-{th.slug(active_set)}.json",
            mime="application/json", key="set_download",
            help="The rule, the overrides and every resolved θ — the same file a "
                 "saved set is, to share or keep with a paper.")
        up = d2.file_uploader("Add a set from a file", type=["json"], key="set_upload")
        if up is not None and st.session_state.get("set_upload_seen") != up.file_id:
            st.session_state["set_upload_seen"] = up.file_id
            try:
                ts = th.ThresholdSet.from_json(up.getvalue().decode())
            except (ValueError, KeyError, TypeError) as e:
                d2.error(f"Not a BEX threshold set: {e}")
            else:
                if ts.name in sets:
                    d2.error(f"A set called “{ts.name}” already exists.")
                else:
                    th.save_set(cfg.thresholds_dir, ts)
                    d2.success(f"Added “{ts.name}” — choose it in the sidebar.")
                    st.rerun()


render(sidebar.render())
