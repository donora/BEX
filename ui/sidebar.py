"""The sidebar: the experiment's global settings, set once for every page (V1 S1–S5).

`render()` draws the sidebar, resolves every included model's thresholds, and
returns everything the pages need as one namespace. Pages that have no use for
the settings (Home, About) do not call it, so they have no sidebar.

Every widget here keeps its value for the whole session (`persist_state=
"session"`), so visiting a page without the sidebar does not reset the
experiment.
"""
from __future__ import annotations

from types import SimpleNamespace

from ui.common import *  # noqa: F403 — the shared app toolkit

NO_SET = "none (rule only)"
FIT_HERE, FROM_SET = "fit on this dataset", "the set's saved thresholds"
JUDGES = ["profile", "geofilter", "either"]


def render() -> SimpleNamespace:
    datasets = ingest.list_datasets(cfg.store_dir)
    runs = views.list_runs(cfg.store_dir)
    if not datasets:
        st.error("No datasets in the store yet — run `bex ingest-sne` or `bex scan`.")
        st.stop()

    for _k, _v in {"p_floor": 0.95, "min_support": 10, "min_pos": 1,
                   "min_labelled": 10}.items():
        st.session_state.setdefault(_k, _v)
    st.session_state.setdefault("overrides", {})     # {(model identity, species): θ}
    st.session_state.setdefault("set_loaded", None)  # the active set's JSON, as loaded
    # A save made on the Thresholds page selects the saved set, but the sidebar's
    # widget already exists by then — so it is parked here and applied on the rerun.
    if "active_set_pending" in st.session_state:
        st.session_state["active_set"] = st.session_state.pop("active_set_pending")

    def load_active_set() -> None:
        """Selecting a set loads its rule into the sidebar and its overrides."""
        name = st.session_state.get("active_set", NO_SET)
        if name == NO_SET:
            st.session_state["overrides"] = {}
            st.session_state["set_loaded"] = None
            return
        ts = th.load_set(cfg.thresholds_dir, name)
        spec = ts.spec
        st.session_state.update(rule_kind=spec.kind, p_floor=spec.precision_floor,
                                min_support=spec.min_support, min_pos=spec.min_positives,
                                min_labelled=spec.min_labelled,
                                fallback=spec.fallback)
        for ident, value in spec.fixed:
            st.session_state[f"fixed::{ident}"] = value
        st.session_state["overrides"] = {(r["model"], r["species_key"]): float(r["theta"])
                                         for r in ts.overrides}
        st.session_state["set_loaded"] = ts.to_json()

    with st.sidebar:
        st.markdown("### Experiment settings")
        dataset = st.selectbox("Dataset", datasets, key="dataset",
                               persist_state="session")
        recordings, annotations = get_dataset(dataset)

        def has_alignment(run_id: str) -> bool:
            """Can this run's rule be fitted here: annotations, and a cached alignment."""
            return annotations is not None and truth.aligned_path(
                cfg.store_dir, dataset, run_id, "native").exists()

        ds_runs = [m for m in runs if m.dataset == dataset]
        run_label = views.arm_labels(ds_runs)
        model_style = views.model_styles(ds_runs)   # one colour and shape per model
        n_scored = {m.run_id: len(store.list_scores(cfg.store_dir, m.run_id)) for m in ds_runs}

        picked = st.multiselect(
            "Models", [m.run_id for m in ds_runs], default=[m.run_id for m in ds_runs],
            key=f"models_{dataset}", persist_state="session",
            # Coverage matters: a partly-scored run looks like a silent model on
            # every recording it has not reached yet.
            format_func=lambda r: f"{run_label[r]} · {n_scored[r]}/{len(recordings)} rec",
            help="Which runs every page includes. Details of each are on the Models page.",
        )
        included = [m for m in ds_runs if m.run_id in picked]
        manifest_of = {m.run_id: m for m in ds_runs}
        ident_of = {m.run_id: th.identity(m) for m in included}

        profiles = list_profiles(cfg.profiles_dir)
        run_default = next((m.profile for m in included if m.profile), cfg.default_profile)
        profile_name = st.selectbox(
            "Plausibility profile", profiles, key="profile_name",
            persist_state="session",
            index=profiles.index(run_default) if run_default in profiles else 0,
            help="The list of species plausible here. It is the judge that applies to "
                 "every model identically.")
        profile = load_profile(cfg.profiles_dir / profile_name)

        judge = st.radio(
            "Implausibility judge", JUDGES, horizontal=True, key="judge",
            persist_state="session",
            help="profile: the plausibility list above, applied to every model the same "
                 "way. geofilter: each model's own location filter, replayed — only "
                 "models that ship one have it. either: implausible by any.")
        no_geo = [run_label[m.run_id] for m in included if not m.geofilter.get("model")]
        if judge != "profile" and no_geo:
            st.caption("⚠️ " + ", ".join(no_geo) + " ships no geofilter"
                       + (", so nothing is implausible for it under this judge."
                          if judge == "geofilter" else "; only the profile judges it."))

        delta = st.slider("Δ (shadowing band)", 0.05, 0.50, 0.15, 0.05, key="delta",
                          persist_state="session",
                          help="A plausible detection with an implausible rival scoring "
                               "within Δ of it is shadowed — the map, not the model, "
                               "made the identification.")

        st.markdown("**Thresholds**")
        rule_kind = st.selectbox(
            "Threshold rule", th.RULE_KINDS, key="rule_kind", persist_state="session",
            help="How each model's threshold is chosen, species by species. θ is not "
                 "comparable between models, so every model gets its own. Everything "
                 "this produces, and hand overrides, are on the Thresholds page.")
        sets = th.list_sets(cfg.thresholds_dir)
        if st.session_state.get("active_set") not in [NO_SET, *sets]:
            st.session_state["active_set"] = NO_SET
        active_set = st.selectbox("Threshold set", [NO_SET, *sets], key="active_set",
                                  persist_state="session",
                                  on_change=load_active_set,
                                  help="A saved, named set of thresholds and overrides. "
                                       "Choosing one loads its rule and its overrides.")
        loaded = (th.ThresholdSet.from_json(st.session_state["set_loaded"])
                  if st.session_state["set_loaded"] else None)

        can_fit = any(has_alignment(m.run_id) for m in included)
        source_options = ([FIT_HERE] if can_fit else []) + ([FROM_SET] if loaded else [])
        if rule_kind == "fixed" or not source_options:
            source = FIT_HERE if can_fit else None
        else:
            if st.session_state.get("th_source") not in source_options:
                st.session_state.pop("th_source", None)
            source = st.radio("Thresholds from", source_options, key="th_source",
                              persist_state="session",
                              help="Fit the rule on this dataset's annotations, or reuse "
                                   "the thresholds a saved set was fitted on — the way "
                                   "to threshold recordings with no annotations.")

        with st.expander("Rule settings"):
            p_floor = st.slider("Precision floor", 0.50, 0.99, step=0.01, key="p_floor",
                                persist_state="session",
                                help="For the precision-floor rule: the highest recall "
                                     "available while staying this precise. Also the "
                                     "floor the Compare page reports.")
            min_support = st.slider(
                "Minimum predicted windows", 1, 200, step=1, key="min_support",
                persist_state="session",
                help="A rule may only pick an operating point that flags at least this "
                     "many windows. Deep in a ranking one lucky row can reach precision "
                     "1.0 on a single prediction, and θ would then be chosen by noise.")
            min_labelled = st.slider(
                "Minimum labelled windows to fit a θ", 1, 200, step=1, key="min_labelled",
                persist_state="session",
                help="A species needs at least this many labelled windows before the "
                     "rule is fitted to it; below that it takes the fallback. With very "
                     "few labelled windows max F1 and max F2 pick absurdly low "
                     "thresholds — one labelled window can buy thousands of false "
                     "alarms. Set to 1 to fit every species regardless.")
            min_pos = st.slider(
                "Minimum labelled windows to count in cmAP", 1, 200, step=1, key="min_pos",
                persist_state="session",
                help="Species with fewer labelled windows than this are left out of "
                     "cmAP. The count excluded is always shown.")
            fallback = st.radio(
                "Fallback", th.FALLBACKS, key="fallback",
                persist_state="session", horizontal=True,
                help="What a species gets when the rule has no answer for it — too few "
                     "labelled windows, never annotated, or unreachable. model-wide: "
                     "the rule applied to all of the model's species pooled. fixed: "
                     "the fixed θ below.")
            fixed = []
            if rule_kind == "fixed" or fallback == "fixed":
                st.caption("Fixed θ per model — scores are on different scales, so one "
                           "number cannot serve them all.")
                for rid, ident in ident_of.items():
                    st.session_state.setdefault(f"fixed::{ident}", th.DEFAULT_FIXED)
                    v = st.number_input(f"θ · {run_label[rid]}", 0.0, 1.0, step=0.01,
                                        format="%.4f", key=f"fixed::{ident}",
                                        persist_state="session")
                    fixed.append((ident, float(v)))

        spec = th.RuleSpec(rule_kind, p_floor, min_support, min_pos, fallback,
                           tuple(sorted(fixed)), min_labelled=min_labelled)
        overrides: dict[tuple[str, str], float] = st.session_state["overrides"]

        def fingerprint(s: th.RuleSpec, ov: dict) -> str:
            # Fixed values only count when they are in use, and only for the models
            # on screen, or a set saved with other models would always look edited.
            uses_fixed = s.kind == "fixed" or s.fallback == "fixed"
            fx = {i: s.fixed_for(i) for i in ident_of.values()} if uses_fixed else {}
            body = {**s.to_dict(), "fixed": fx}
            return json.dumps([body, sorted((m, sp, round(t, 6))
                                            for (m, sp), t in ov.items())])

        if loaded:
            unsaved = fingerprint(spec, overrides) != fingerprint(
                loaded.spec, {(r["model"], r["species_key"]): r["theta"]
                              for r in loaded.overrides})
        else:
            unsaved = bool(overrides)

    # ---- resolve every included model's thresholds, once, for every page ---------- #
    resolved: dict[str, th.Resolved] = {}
    threshold_notes: list[str] = []
    for m in included:
        ident = ident_of[m.run_id]
        ov = {sp: t for (mi, sp), t in overrides.items() if mi == ident}
        if source == FROM_SET and loaded is not None and rule_kind != "fixed":
            snap, default = loaded.snapshot_for(ident)
            if snap is None and default is None:
                threshold_notes.append(f"**{run_label[m.run_id]}** is not in set "
                                       f"*{loaded.name}*, so it has no thresholds.")
            resolved[m.run_id] = th.resolve(ident, spec, snapshot=snap or {},
                                            snapshot_default=default, overrides=ov)
        elif source == FIT_HERE and has_alignment(m.run_id):
            card = get_scorecard(m.run_id, dataset, "native", p_floor, min_support, min_pos)
            resolved[m.run_id] = th.resolve(ident, spec, card["curves"], card["micro"],
                                            overrides=ov)
        else:
            if rule_kind != "fixed" and fallback != "fixed":
                threshold_notes.append(
                    f"**{run_label[m.run_id]}** has nothing to fit its rule on (no "
                    "annotations or no alignment) and no saved set — choose a fixed θ or "
                    "a threshold set.")
            resolved[m.run_id] = th.resolve(ident, spec, overrides=ov)

    fitted_here = source == FIT_HERE and rule_kind != "fixed"
    n_overrides = sum(1 for (mi, _) in overrides if mi in set(ident_of.values()))

    with st.sidebar:
        for note in threshold_notes:
            st.caption("⚠️ " + note)
        st.divider()
        # S5: the whole experiment definition in one line, so a screenshot records it.
        st.caption(
            f"**{dataset}** · {len(included)} model(s) · profile *{profile_name}* · "
            f"judge *{judge}* · Δ {delta:.2f} · θ: {spec.label}"
            + (f" ({source})" if source and rule_kind != "fixed" else "")
            + f", fallback {fallback}"
            + f" · set *{active_set}*" + (" — **unsaved changes**" if unsaved else "")
            + (f" · {n_overrides} override(s)" if n_overrides else ""))

    return SimpleNamespace(**{k: v for k, v in locals().items()
                              if not k.startswith("_")})
