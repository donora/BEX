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


VIEW_LABELLED, VIEW_ALL = "Labelled data, scored", "All recordings, unscored"


def view_key(dataset: str) -> str:
    return f"view_{dataset}"


def label_options(dataset: str) -> dict[str, str]:
    """Every set of labels a dataset can be scored on: {choice id: label}. Ids
    are stable across edits; the ref (with its content hash) is resolved from them."""
    opts = {}
    if labels.has_imported(cfg.store_dir, dataset):
        opts[labels.IMPORTED] = "Imported annotations"
    for name in labels.list_sets(cfg.store_dir, dataset):
        if labels.is_practice(cfg.store_dir, dataset, name):
            continue          # practice labels are for learning, never ground truth
        opts[f"{name}@working"] = f"{name} — working copy"
        for v in reversed(labels.versions(cfg.store_dir, dataset, name)):
            opts[f"{name}@v{v}"] = f"{name} — v{v} (published)"
    return opts


def default_labels(dataset: str, opts: dict[str, str]) -> str | None:
    """Imported annotations when there are some; else the most recently
    changed set's working copy."""
    if labels.IMPORTED in opts:
        return labels.IMPORTED
    sets = [n for n in labels.list_sets(cfg.store_dir, dataset)
            if not labels.is_practice(cfg.store_dir, dataset, n)]
    if sets:
        newest = max(sets, key=lambda n: max(
            (p.stat().st_mtime for p in labels.set_dir(cfg.store_dir, dataset, n).iterdir()),
            default=0))
        return f"{newest}@working"
    return None


def resolve_truth(dataset: str, choice: str) -> str:
    """Choice id -> ref. A working copy's ref carries the hash of its files, so
    every cache built on it expires the moment it is edited."""
    if choice.endswith("@working"):
        return labels.working_ref(cfg.store_dir, dataset, choice[:-len("@working")])
    return choice


def current_view(dataset: str) -> str:
    """The dataset's view: labelled data when it has labels and nobody chose
    otherwise; all recordings when it has none (or someone switched)."""
    has = bool(label_options(dataset))
    v = st.session_state.get(view_key(dataset))
    if v not in (VIEW_LABELLED, VIEW_ALL):
        v = VIEW_LABELLED if has else VIEW_ALL
        st.session_state[view_key(dataset)] = v
    return v


def set_view(dataset: str, view: str) -> None:
    """For buttons elsewhere (top bar, Home, page lines): switch, before the
    sidebar's widget is drawn this run."""
    st.session_state[view_key(dataset)] = view


def truth_choice(dataset: str) -> str:
    """The view switch, and under it which labels the scored view uses (V1.2
    F0, L6). Returns the ref, or "" for all recordings, unscored. Switching is
    free in both directions: neither view closes off the other."""
    opts = label_options(dataset)
    view = current_view(dataset)
    view = st.segmented_control(
        "Looking at", [VIEW_LABELLED, VIEW_ALL], key=view_key(dataset),
        persist_state="session", width="stretch",
        help="Labelled data, scored: only what has been labelled, with every "
             "number judged against the labels. All recordings, unscored: every "
             "recording, showing what the models report, with no right or wrong. "
             "Switch back and forth freely.") or view
    if view == VIEW_ALL:
        return ""
    if not opts:
        st.caption("No labels yet, so there is nothing to score against.")
        st.page_link("ui/pages/label.py", label="Start labelling", icon="🏷️")
        return ""
    key, auto = f"truth_{dataset}", f"_truth_auto_{dataset}"
    # Until someone picks, the default follows the data: the first label set
    # made on a dataset is used without a trip to the sidebar.
    if st.session_state.get(key) not in opts or st.session_state.get(auto, False):
        st.session_state[key] = default_labels(dataset, opts)
        st.session_state[auto] = True

    def picked() -> None:
        st.session_state[auto] = False

    choice = st.selectbox("Scored against", list(opts), key=key, format_func=opts.get,
                          persist_state="session", on_change=picked,
                          help="Which labels every page scores against. A working "
                               "copy changes as you label; a published version never "
                               "does, so cite that one.")
    return resolve_truth(dataset, choice)


@st.cache_data(show_spinner=False, max_entries=16)
def sample_state(dataset: str, ref: str) -> tuple[int, int] | None:
    """(sampled chunks closed, sample size) for a set ref; None without a sample."""
    name, version = labels.parse_ref(ref)
    if not ref or name == labels.IMPORTED:
        return None
    ls = labels.load_set(cfg.store_dir, dataset, name, version)
    if ls.sample.is_empty():
        return None
    closed = ls.chunks.filter(pl.col("state") == labels.CLOSED)
    done = ls.sample.join(closed, on=["recording_id", "start_s"], how="semi")
    return len(done), len(ls.sample)


def status_panel(dataset: str, recordings: pl.DataFrame, ref: str,
                 t: "labels.Truth | None") -> None:
    """What every number on every page rests on, said once (V1.2 F3)."""
    hours = float(recordings["duration_s"].sum()) / 3600
    sites = recordings["site"].replace("", None).drop_nulls().n_unique()
    c = brand()
    lines = [f"<b style='font-size:1.05rem'>{dataset}</b>",
             f"{sites} site{'s' * (sites != 1)} · {len(recordings):,} recordings · "
             f"{hours:,.1f} h of audio"]
    if t is None:
        lines.append(f"<span style='color:{c['deep']}'><b>All recordings, unscored.</b>"
                     "</span> Every number is what the models reported, with no right "
                     "or wrong.")
    else:
        lab_sites = recordings.filter(pl.col("recording_id").is_in(t.recordings))[
            "site"].replace("", None).drop_nulls().n_unique()
        lines.append(f"<span style='color:{c['deep']}'><b>Labelled data, scored</b></span>"
                     f" against {t.label}.")
        if t.exhaustive:
            lines.append(f"<b>Labelled:</b> every recording, end to end ({t.hours:,.1f} h)")
        else:
            lines.append(f"<b>Labelled:</b> {t.hours * 60:,.0f} min across "
                         f"{len(t.recordings)} recordings and {lab_sites} "
                         f"site{'s' * (lab_sites != 1)}"
                         + (f"; {len(t.complete)} recording(s) end to end"
                            if t.complete else ""))
        lines.append(f"{len(t.annotations):,} labelled birds of "
                     f"{t.annotations['species_key'].n_unique()} species"
                     + (f" · {t.n_unknown} unknown" if t.n_unknown else ""))
        ss = sample_state(dataset, ref)
        if ss:
            lines.append(f"Labelling sample: {ss[0] / ss[1]:.0%} closed ({ss[0]}/{ss[1]})")
    st.markdown(
        f"<div style='border-left:4px solid {c['b']}; background:{c['tint']};"
        "padding:0.55rem 0.8rem; border-radius:6px; font-size:0.86rem; line-height:1.55;"
        "margin-bottom:0.5rem'>" + "<br>".join(lines) + "</div>",
        unsafe_allow_html=True)


def view_line(dataset: str, t: "labels.Truth | None") -> None:
    """One line at the top of every page with settings: what is in view, and
    the switch to the other view."""
    has = bool(label_options(dataset))
    a, b = st.columns([5, 1.4], vertical_alignment="center")
    if t is None:
        a.caption(f"👁 **All recordings, unscored** — what the models report on every "
                  f"recording of *{dataset}*, with no right or wrong.")
        if has:
            b.button("Switch to labelled data", key="view_line_switch", width="stretch",
                     on_click=set_view, args=(dataset, VIEW_LABELLED))
    else:
        where = ("every recording" if t.exhaustive else
                 f"{t.hours * 60:,.0f} min of labelled audio in {len(t.recordings)} "
                 "recordings")
        a.caption(f"👁 **Labelled data, scored** — on {where}, against *{t.label}*.")
        b.button("Switch to all recordings", key="view_line_switch", width="stretch",
                 on_click=set_view, args=(dataset, VIEW_ALL))


@st.cache_data(show_spinner=False, ttl=60)
def borrowed_fit(ident: str, exclude: str) -> tuple[str, str] | None:
    """A run of the same model identity on another dataset, already aligned
    against that dataset's imported annotations: (run_id, dataset)."""
    for m in views.list_runs(cfg.store_dir):
        if (m.dataset != exclude and th.identity(m) == ident
                and truth.aligned_path(cfg.store_dir, m.dataset, m.run_id, "native",
                                       labels.IMPORTED).exists()):
            return m.run_id, m.dataset
    return None


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
        if st.session_state.get("dataset") not in datasets:
            st.session_state.pop("dataset", None)
        dataset = st.selectbox("Dataset", datasets, key="dataset",
                               persist_state="session",
                               help="Choose a dataset here or on the Home page.")
        recordings, _ = get_dataset(dataset)
        truth_ref = truth_choice(dataset)
        truth_obj = get_truth(dataset, truth_ref)
        annotations = truth_obj.annotations if truth_obj is not None else None
        status_panel(dataset, recordings, truth_ref, truth_obj)
    view_line(dataset, truth_obj)
    with st.sidebar:
        st.markdown("### Experiment settings")

        def has_alignment(run_id: str) -> bool:
            """Can this run's rule be fitted here: labels, and a cached alignment."""
            return annotations is not None and has_aligned(run_id, dataset, truth_ref)

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
        # The dataset's own profile (chosen when it was added), then the one its
        # runs were made with, then the config's — per dataset, so moving between
        # a UK and a Californian dataset never carries the wrong list across.
        from bex import flow
        own = flow.dataset_meta(cfg.store_dir, dataset).get("profile")
        run_default = own or next((m.profile for m in included if m.profile),
                                  cfg.default_profile)
        pkey = f"profile_{dataset}"
        if st.session_state.get(pkey) not in profiles:
            st.session_state[pkey] = run_default if run_default in profiles else profiles[0]
        profile_name = st.selectbox(
            "Plausibility profile", profiles, key=pkey, persist_state="session",
            help="The list of species plausible here. It is the judge that applies to "
                 "every model identically. Each dataset keeps its own; set its default "
                 "on its card on Home.")
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
    borrowed: dict[str, str] = {}   # model label -> the labelled dataset it was fitted on
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
            card = get_scorecard(m.run_id, dataset, "native", p_floor, min_support, min_pos,
                                 truth_ref)
            resolved[m.run_id] = th.resolve(ident, spec, card["curves"], card["micro"],
                                            overrides=ov)
        elif (not truth_ref and rule_kind != "fixed"
              and (lent := borrowed_fit(ident, dataset)) is not None):
            # Nothing to fit on in this view: the same model's rule, fitted on a
            # labelled dataset — the way BEX is meant to reach unlabelled audio.
            lend_run, lend_ds = lent
            card = get_scorecard(lend_run, lend_ds, "native", p_floor, min_support,
                                 min_pos, labels.IMPORTED)
            resolved[m.run_id] = th.resolve(ident, spec, card["curves"], card["micro"],
                                            overrides=ov)
            borrowed[run_label[m.run_id]] = lend_ds
        else:
            if rule_kind != "fixed" and fallback != "fixed":
                threshold_notes.append(
                    f"**{run_label[m.run_id]}** has nothing to fit its rule on (no "
                    "ground truth, or not yet aligned against it) and no saved set — "
                    "choose a fixed θ or a threshold set.")
            resolved[m.run_id] = th.resolve(ident, spec, overrides=ov)

    fitted_here = source == FIT_HERE and rule_kind != "fixed"
    n_overrides = sum(1 for (mi, _) in overrides if mi in set(ident_of.values()))

    if borrowed:
        lenders = sorted(set(borrowed.values()))
        threshold_notes.insert(0, (
            "This view has no labels to fit thresholds on, so each model uses the "
            f"rule fitted on **{', '.join(lenders)}**'s labels (same model and readout): "
            + ", ".join(f"{m} ← {d}" for m, d in borrowed.items())
            + ". Label this dataset to fit its own."))

    with st.sidebar:
        for note in threshold_notes:
            st.caption("⚠️ " + note)
        st.divider()
        # S5: the whole experiment definition in one line, so a screenshot records it.
        st.caption(
            f"**{dataset}** · truth: *{labels.describe_ref(truth_ref) if truth_ref else 'none'}*"
            f" · {len(included)} model(s) · profile *{profile_name}* · "
            f"judge *{judge}* · Δ {delta:.2f} · θ: {spec.label}"
            + (f" ({source})" if source and rule_kind != "fixed" else "")
            + f", fallback {fallback}"
            + f" · set *{active_set}*" + (" — **unsaved changes**" if unsaved else "")
            + (f" · {n_overrides} override(s)" if n_overrides else ""))

    return SimpleNamespace(**{k: v for k, v in locals().items()
                              if not k.startswith("_")})
