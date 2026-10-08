"""BEX · Survey protocol — from detections to a species list, with a tier for an
expert to check (V1.1). The engine is `bex.survey`."""
from ui.common import *  # noqa: F403 — the shared app toolkit
from ui import sidebar

import math

FLOOR_OPTS = [*survey.FLOORS, None]
SPAN_OPTS = list(survey.SPANS)
#: Below the annotated line: the species that are there. Above: listed, not there.
BELOW = ["firm, right", "check, real", "filter hid a real bird", "missed"]
ABOVE = ["firm, wrong", "check, not there", "firm, can't judge", "check, can't judge"]
WORDS = {
    "firm, right": "firm, and there", "check, real": "to check, and there",
    "filter hid a real bird": "there, but hidden by the filter",
    "missed": "there, but not found", "firm, wrong": "firm, but not there",
    "check, not there": "to check, not there",
    "firm, can't judge": "firm, outside the label set",
    "check, can't judge": "to check, outside the label set",
}
AUTO, HAND = "Optimise automatically", "Set by hand"
EACH = "each model at its own best"
#: How the verdict says a model won on each objective.
VERBS = {"recall_reviewed": "finds the most birds after checking",
         "firm_recall": "finds the most birds with no checking",
         "checks_per_recording": "needs the fewest checks per recording",
         "firm_precision": "has the most trustworthy firm calls"}
PCT = {"firm_precision", "recall_reviewed", "firm_recall", "check_yield"}


def cell_text(c: dict, rule: survey.SurveyRule) -> str:
    """A sweep cell as a rule, in words: `rule` (for the check count and "where")
    moved to the cell's floors and firm count."""
    return survey.swept(rule, c["floor"], c["k"], c.get("check_floor")).describe()


def floor_label(p) -> str:
    return "the sidebar's thresholds" if p is None else f"precision ≥ {p:g}"


def span_label(s) -> str:
    return "anywhere in the recording" if s is None else f"within {s:g} s"


def fmt(stat: str, v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.0%}" if stat in PCT else f"{v:.1f}"


def key_html(order: list[str]) -> str:
    return " &nbsp; ".join(
        f"<span style='white-space:nowrap'><span style='color:"
        f"{views.SURVEY_COLOURS[o]}'>■</span> {WORDS[o]}</span>" for o in order)


def render(ctx) -> None:
    dataset = ctx.dataset
    truth_ref = ctx.truth_ref
    recordings = ctx.recordings
    annotations = ctx.annotations
    picked = ctx.picked
    run_label = ctx.run_label
    model_style = ctx.model_style
    profile_name = ctx.profile_name
    judge = ctx.judge
    spec = ctx.spec
    resolved = ctx.resolved
    manifest_of = ctx.manifest_of

    heading(
        "Survey protocol",
        "**Which species were in this recording?** A model reports detections, "
        "window by window; a survey needs a species list. This page compares the "
        "rules that turn one into the other, with a middle tier:\n\n"
        "- **firm** — reported without review;\n"
        "- **to check** — possibly there; an expert listens before it is reported;\n"
        "- **not found**.\n\n"
        "A rule says how many detections make each tier, above which threshold, "
        "and optionally how close together. Pick one by hand, or sweep over many "
        "and let the page find the best under your limits — then **run it on test "
        "recordings** it was not tuned on, for an honest verdict and a fresh "
        "comparison of the models.\n\n"
        "Recordings are the unit: timing inside a recording does not matter, only "
        "whether a species is on its list. Sound-event classes (*Car*, *Engine*) "
        "are never listed. The judge is the sidebar's: a detection it would hide "
        "does not count towards a list.",
        level="##")

    ready = [r for r in complete_runs(dataset) if r in picked]
    if not ready:
        st.info("No included model has scored every recording in this dataset yet — "
                "see the Models page.")
        return
    labels = {run_label[r]: r for r in sorted(ready, key=lambda r: model_style[r]["rank"])}
    order = list(labels)
    colour = {m: model_style[r]["colour"] for m, r in labels.items()}
    shape = {m: model_style[r]["shape"] for m, r in labels.items()}
    model_colour = alt.Scale(domain=order, range=[colour[m] for m in order])
    model_shape = alt.Scale(domain=order, range=[shape[m] for m in order])
    nm = lambda k: views.display_name(k, get_names())
    hours_of = {r: d / 3600 for r, d in zip(recordings["recording_id"],
                                             recordings["duration_s"])}

    if annotations is None:
        unlabelled(ctx, labels, order, colour, nm)
        return
    label_set = set(annotations["species_key"].unique().to_list())
    # A species list is for a whole recording (P2), so only recordings labelled
    # end to end can judge one; a closed recording with no birds counts too.
    all_ids = sorted(set(ctx.truth_obj.complete) & set(recordings["recording_id"].to_list()))
    if len(all_ids) < 2:
        st.info(f"A survey protocol is judged recording by recording — did the list "
                f"name the right species? — so it needs recordings labelled from start "
                f"to finish, and *{ctx.truth_obj.label}* has {len(all_ids)}. Add a few "
                "whole recordings to the labelling sample (6 or more allows a "
                "tuning/test split), or switch to *All recordings, unscored* to see the "
                "lists without scores.")
        from bex import labels as label_sets   # `labels` here is the model dict
        name, version = label_sets.parse_ref(ctx.truth_ref)
        if name != label_sets.IMPORTED and version is None:
            k = st.number_input("Whole recordings to add", 1, len(recordings), 2,
                                key="sv_whole_n")
            if st.button(f"Add {int(k)} whole recording{'s' * (k != 1)} to the "
                         f"labelling sample of {name}", type="primary", key="sv_whole"):
                ls = label_sets.load_set(cfg.store_dir, dataset, name)
                ls.sample = label_sets.draw_sample(recordings, 0, seed=len(ls.sample),
                                                   whole_recordings=int(k),
                                                   existing=ls.sample)
                label_sets.save_set(cfg.store_dir, ls)
                get_truth.clear()
                st.switch_page("ui/pages/label.py")
        st.page_link("ui/pages/label.py", label="Go to 1 · Label", icon="🏷️")
        apply_setup(ctx, lambda k: views.display_name(k, get_names()))
        return

    unaligned = [r for r in ready
                 if not has_aligned(r, dataset, truth_ref)]
    if unaligned:
        st.info(f"{len(unaligned)} run(s) need aligning against the annotations first "
                "— thresholds are fitted on the alignment. It takes a minute and is "
                "cached.")
        if st.button(f"Align {len(unaligned)} run(s)", key="sv_align", type="primary"):
            for rid in unaligned:
                build_alignment(rid, dataset, "native", truth_ref)
            st.rerun()
        return

    # A rule loaded from a sweep cell or a report: applied before the widgets
    # exist, and it switches the rule to "by hand" so it stays put.
    pending = st.session_state.pop("sv_pending_rule", None)
    if pending:
        st.session_state.update(sv_check_floor=pending.get("check_floor") or pending["floor"],
                                sv_firm_floor=pending["floor"],
                                sv_firm_k=int(pending["k"]), sv_mode=HAND,
                                _sv_source=pending.get("source", "loaded"))
        st.session_state["sv_check_k"] = min(int(st.session_state.get("sv_check_k", 1)),
                                             int(pending["k"]))

    # ---- tuning and test recordings (P13): set once, out of the way --------- #
    seed = int(st.session_state.get("sv_seed", 0))
    # Redrawn when the seed or the recordings change, and whenever the list is
    # missing — leaving the page used to drop it, which silently tuned on every
    # recording and held none back.
    if (st.session_state.get("_sv_seed_seen") != (seed, tuple(all_ids))
            or "sv_test" not in st.session_state):
        st.session_state["sv_test"] = survey.split(all_ids, seed)[1]
        st.session_state["_sv_seed_seen"] = (seed, tuple(all_ids))
    explore_all = bool(st.session_state.get("sv_all", False))
    n_test = len(st.session_state.get("sv_test") or [])
    with st.expander(
            "Recordings: " + ("exploring and testing on all of them — the split is off"
                              if explore_all else
                              f"tuning on {len(all_ids) - n_test}, holding back {n_test} "
                              "to test on")):
        st.markdown(
            "Choosing the best of many rules on a set of recordings, then scoring it on "
            "the same recordings, flatters it — and so does fitting the thresholds on "
            "them. So the recordings are split: everything on this page is worked out "
            "on the **tuning** recordings, and the **test** recordings are held back "
            "until you run a rule on them. The split is random but reproducible (the "
            "same seed draws the same split); edit it freely. With a few dozen "
            "recordings a half-split is noisy — *Full assessment* uses every recording "
            "as a test recording once.")
        c1, c2, c3 = st.columns([1, 3, 1.4], vertical_alignment="bottom")
        c1.number_input("Split seed", 0, 9999, 0, 1, key="sv_seed", persist_state="session")
        explore_all = c3.toggle("Explore on all recordings", key="sv_all",
                                persist_state="session")
        test_pick = c2.multiselect("Test recordings", all_ids, key="sv_test",
                                   disabled=explore_all, persist_state="session")
    if explore_all:
        tune, test = all_ids, all_ids
    else:
        test = sorted(test_pick)
        tune = [r for r in all_ids if r not in set(test)]
    if not tune:
        st.warning("Every recording is a test recording — leave some to tune on.")
        return

    # ---- 1 · choose your rule ------------------------------------------------ #
    heading(
        "Choose your rule",
        "**Your goal** is what 'best' means, everywhere on this page: one thing to "
        "optimise, within limits. Firm calls are never checked, so their precision "
        "is always a limit; checks cost an expert's time, so you can cap them; and "
        "you can ask for a minimum share of birds found.\n\n"
        "**Optimise automatically** searches every combination of score bar "
        "(precision floor) and firm count on the tuning recordings and takes the "
        "best for your goal — for **each model at its own best**, side by side, or "
        "for one model, whose rule the others then use too. The rest of the rule "
        "(the check count, *where*) is held as set under *More rule settings*.\n\n"
        "**Set by hand** gives you every setting, starting from the automatic "
        "choice.\n\n"
        "**A rule** has two conditions. A species is **firm** in a recording when "
        "the firm condition holds, **to check** when only the check condition does, "
        "and **not found** otherwise. Each condition: a **score bar** (*precision ≥ "
        "p* fits each species' threshold for that precision on the tuning "
        "recordings; *the sidebar's thresholds* were fitted on every recording, so "
        "they know the test ones), a number of **detections**, and **where** — "
        "anywhere in the recording, or that many within a span (counted by window "
        "start: three 3 s windows in a row fall within 10 s).\n\n"
        "**Outside the label set.** The annotations only mark the species they "
        "cover; a listing of any other species cannot be judged right or wrong and "
        "is counted apart, unless you count it as wrong.",
        level="###")

    st.markdown("**Your goal**")
    g1, g2 = st.columns([1.2, 1])
    objective = g1.selectbox("Optimise for", list(survey.OBJECTIVES), key="sv_objective",
                             format_func=lambda o: survey.OBJECTIVES[o][0],
                             persist_state="session")
    min_prec = g2.slider("Firm calls right, at least", 0.50, 0.99, 0.95, 0.01,
                         key="sv_min_prec", persist_state="session",
                         help="Nobody checks a firm call, so this is always a limit.")
    h1, h2, h3, h4 = st.columns([0.9, 1, 0.9, 1], vertical_alignment="bottom")
    limit = h1.toggle("Cap checks", value=True, key="sv_limit", persist_state="session")
    max_checks = h2.number_input("Checks per recording, at most", 0.0, 500.0, 10.0, 1.0,
                                 key="sv_max_checks", persist_state="session",
                                 disabled=not limit)
    floor_on = h3.toggle("Require birds found", value=False, key="sv_floor_on",
                         persist_state="session")
    min_found = h4.slider("Birds found after checking, at least", 0.0, 1.0, 0.7, 0.05,
                          key="sv_min_found", persist_state="session",
                          disabled=not floor_on)
    cons = survey.Constraints(min_prec, float(max_checks) if limit else None, objective,
                              float(min_found) if floor_on else None)

    def to_hand() -> None:
        """Switching to by hand starts from the rule that was just chosen (with
        each model at its own best, the first model's)."""
        auto = st.session_state.get("_sv_auto")
        if st.session_state.get("sv_mode") == HAND and auto:
            st.session_state.update(sv_firm_floor=auto["floor"],
                                    sv_check_floor=auto.get("check_floor") or auto["floor"],
                                    sv_firm_k=int(auto["k"]),
                                    _sv_source=f"the automatic choice for {auto['who']} — "
                                               "now yours to edit, for every model")

    def edited() -> None:
        st.session_state["_sv_source"] = "set by hand"

    st.markdown("**The rule**")
    m1, m2 = st.columns([1.2, 1], vertical_alignment="bottom")
    mode = m1.radio("How to set it", [AUTO, HAND], horizontal=True, key="sv_mode",
                    persist_state="session", on_change=to_hand,
                    label_visibility="collapsed")
    target = None
    if mode == AUTO:
        if st.session_state.get("sv_target") not in [EACH, *order]:
            st.session_state.pop("sv_target", None)   # e.g. a model since removed
        target = m2.selectbox("Optimise it for", [EACH, *order], key="sv_target",
                              persist_state="session",
                              help="Each model at its own best: every model gets the rule "
                                   "that suits it, shown side by side. One model: its "
                                   "best rule, applied to every model.")

    def condition(col, name: str, default_k: int, full: bool):
        # Defaults go into session state, not the widgets: a sweep click or the
        # optimiser writes these keys, and a widget may not have both.
        st.session_state.setdefault(f"sv_{name}_floor", 0.95)
        st.session_state.setdefault(f"sv_{name}_k", default_k)
        with col.container(border=True):
            st.markdown(f"**{name.capitalize()}**")
            if full:
                fl = st.selectbox("Score bar", FLOOR_OPTS, key=f"sv_{name}_floor",
                                  format_func=floor_label, persist_state="session",
                                  on_change=edited)
            else:
                fl = None
            k = (st.number_input("Detections", min_value=1, max_value=200, step=1,
                                 key=f"sv_{name}_k", persist_state="session",
                                 on_change=edited)
                 if full or name == "check" else default_k)
            w = st.selectbox("Where", SPAN_OPTS, key=f"sv_{name}_within",
                             format_func=span_label, persist_state="session",
                             on_change=edited)
        return fl, int(k), w

    unjudged_key = dict(key="sv_unjudged", persist_state="session",
                        help=f"The annotations cover {len(label_set)} species. Off: "
                             "listings of any other species are 'can't judge' and left "
                             "out of the precision. On: they count as mistakes, as "
                             "Compare does.")
    if mode == HAND:
        a, b = st.columns(2)
        ff, fk, fw = condition(a, "firm", 3, True)
        cf, ck, cw = condition(b, "check", 1, True)
        unjudged_wrong = st.checkbox("Count species outside the label set as wrong",
                                     **unjudged_key)
    else:
        with st.expander("More rule settings — held fixed while optimising"):
            st.caption("The optimiser picks the score bar (for both tiers) and the firm "
                       "count; these stay as you set them.")
            a, b = st.columns(2)
            _, _, fw = condition(a, "firm", 3, False)
            _, ck, cw = condition(b, "check", 1, False)
            unjudged_wrong = st.checkbox("Count species outside the label set as wrong",
                                         **unjudged_key)
        ff = cf = st.session_state.get("sv_firm_floor", 0.95)
        fk = int(st.session_state.get("sv_firm_k", 3))

    spans = tuple(sorted({s for s in (fw, cw) if s is not None}))
    with st.spinner("fitting thresholds on the tuning recordings…"):
        E = {m: get_survey_evidence(rid, dataset, judge, profile_name, spec,
                                    th.identity(manifest_of[rid]), tuple(tune), spans,
                                    resolved_key(resolved[rid]), truth_ref)
             for m, rid in labels.items()}
    hand_rule = survey.SurveyRule(check=survey.Condition(cf, ck, cw),
                                  firm=survey.Condition(ff, fk, fw))
    with st.spinner("sweeping…"):
        SW = {m: survey.sweep(E[m], hand_rule, annotations, tune, hours_of, label_set,
                              unjudged_wrong) for m in order}
    best_own = {m: survey.best(SW[m], cons) for m in order}

    # One rule per model. By hand, and when optimising for one model, every model
    # gets the same rule; "each model at its own best" gives each its own.
    if mode == AUTO:
        targets = order if target == EACH else [target]
        chosen, sources, problems = {}, {}, {}
        for m in targets:
            cell = best_own[m]
            if cell is not None:
                chosen[m] = survey.swept(hand_rule, cell["floor"], cell["k"],
                                         cell.get("check_floor"))
                sources[m] = "its best for your goal"
                st.session_state.setdefault("_sv_auto_tmp", {"who": m, **cell})
            else:
                # Nothing meets every limit for this model. Say which limit is out
                # of reach and what is reachable, and use the rule that comes
                # closest — never an unrelated rule, which would read as the
                # answer to the goal.
                chosen[m], sources[m], lines = infeasible(
                    lambda c, m=m: survey.best(SW[m], c), cons, hand_rule, m)
                problems[m] = lines
        auto = st.session_state.pop("_sv_auto_tmp", None)
        if auto:
            st.session_state["_sv_auto"] = auto
        rules = (chosen if target == EACH else {m: chosen[target] for m in order})
        source = (f"each model at its own best for your goal: {cons.describe()}"
                  if target == EACH else
                  f"chosen automatically for {target} ({sources[target]}): "
                  f"{cons.describe()}")
    else:
        rules = {m: hand_rule for m in order}
        source = st.session_state.get("_sv_source", "set by hand")
        sources, problems = {}, {}

    distinct = list(dict.fromkeys(rules.values()))
    c = brand()
    # A model with no rule inside the limits is flagged in the banner, not in a
    # separate error box: a ⚠️, its closest rule, and the reason one click away.
    amber = ("background:rgba(212,160,23,0.22); color:inherit; padding:0.05rem "
             "0.4rem; border-radius:4px; font-weight:600")
    flag = lambda m: (f" <span style='{amber}'>⚠️ {sources[m]}</span>"
                      if m in problems else "")
    if len(distinct) == 1:
        body = (f"<b>Your rule: <span style='color:{c['deep']}'>"
                f"{distinct[0].describe()}</span></b>"
                + "".join(f"<div style='margin-top:0.3rem'><b>{m}</b>:{flag(m)}</div>"
                          for m in problems))
    else:
        body = "<b>Your rules:</b>" + "".join(
            f"<div style='margin-top:0.2rem'>"
            + ("⚠️ " if m in problems else f"<span style='color:{colour[m]}'>●</span> ")
            + f"<b>{m}</b>: <span style='color:{c['deep']}'><b>{rules[m].describe()}"
            f"</b></span>{flag(m)}</div>" for m in order)
    st.markdown(
        f"<div style='border-left:4px solid {c['b']}; background:{c['tint']};"
        "padding:0.7rem 1rem; border-radius:6px; margin:0.6rem 0 0.4rem'>"
        f"<div style='font-size:1.05rem'>{body}</div>"
        f"<div style='margin-top:0.3rem; font-size:0.9rem; opacity:0.8'>"
        f"{source[0].upper() + source[1:]} — worked out on the {len(tune)} tuning "
        "recordings.</div></div>", unsafe_allow_html=True)
    if problems:
        why = st.columns(min(len(problems), 3) + 1)
        for col, (m, lines) in zip(why, problems.items()):
            with col.popover(f"⚠️ Why {m}?", width="stretch"):
                st.markdown("\n\n".join(lines))

    for r in distinct:
        if (r.check.k > r.firm.k and r.check.floor == r.firm.floor
                and r.check.within_s == r.firm.within_s):
            st.caption("⚠️ The check condition is stricter than the firm one, so "
                       "nothing will be 'to check'.")
        if None in (r.firm.floor, r.check.floor):
            st.caption("⚠️ The sidebar's thresholds were fitted on every recording, "
                       "test ones included — the report will flatter this rule.")

    def run(m: str, r: survey.SurveyRule, recs: list[str]):
        out = survey.outcomes(survey.tiers(E[m], r, recs), annotations, recs, label_set)
        per_rec = survey.per_recording(out, recs)
        return out, per_rec, survey.score(per_rec, hours_of, unjudged_wrong)

    T = {m: run(m, rules[m], tune) for m in order}

    # ---- 2 · what each model's lists would hold (tuning recordings) -------- #
    heading(
        "What each model's lists would hold at your rule — on the tuning recordings"
        if len(distinct) == 1 else
        "What each model's lists would hold at its own rule — on the tuning recordings",
        "The average tuning recording. The green bar is how many species are "
        "annotated in it. Each model's bar stacks what its lists would make of "
        "them: up to the dashed line, the species that are there — firm (blue), to "
        "check (pale blue), hidden by the location filter (purple), not found "
        "(grey) — so those always add up to the green bar. Above the line, the "
        "species it would list that are not there: firm (orange), to check (pale "
        "orange), and listings outside the label set (khaki).\n\n"
        "**Birds found after checking** assumes the expert's check is perfect: every "
        "real bird in the check tier is confirmed, every false one rejected.")
    n_t = max(1, len(tune))
    avg_there = T[order[0]][2]["annotated"] / n_t
    bars = [{"column": "annotated", "part": "annotated", "y0": 0.0, "y1": avg_there,
             "value": avg_there, "what": "species annotated in the recording"}]
    for m in order:
        c, y = T[m][2]["counts"], 0.0
        for o in BELOW + ABOVE:
            v = c[o] / n_t
            bars.append({"column": m, "part": o, "y0": y, "y1": y + v, "value": v,
                         "what": WORDS[o]})
            y += v
    bars = pd.DataFrame(bars)
    bars["mid"] = (bars["y0"] + bars["y1"]) / 2
    bars["label"] = [f"{v:.1f}" if v >= 0.6 else "" for v in bars["value"]]
    columns = ["annotated", *order]
    palette = {**views.SURVEY_COLOURS, "annotated": views.TRUTH_COLOUR}
    x = alt.X("column:N", sort=columns, title=None,
              axis=alt.Axis(labelAngle=0, labelLimit=160, labelFontSize=12))
    y_scale = alt.Scale(domain=[0, float(bars["y1"].max()) * 1.08], nice=False)
    stacked = alt.Chart(bars).mark_bar(width=alt.RelativeBandSize(0.62), stroke="#ffffff",
                                       strokeWidth=1).encode(
        x=x, y=alt.Y("y0:Q", title="species per recording, on average", scale=y_scale),
        y2="y1:Q",
        color=alt.Color("part:N", legend=None, scale=alt.Scale(
            domain=list(palette), range=list(palette.values()))),
        tooltip=[alt.Tooltip("column:N", title="model"), alt.Tooltip("what:N"),
                 alt.Tooltip("value:Q", title="species per recording", format=".1f")])
    text = alt.Chart(bars).mark_text(fontSize=11, fontWeight="bold").encode(
        x=x, y=alt.Y("mid:Q", scale=y_scale), text="label:N",
        color=alt.condition(alt.FieldOneOfPredicate(
            "part", ["firm, right", "filter hid a real bird", "missed", "annotated",
                     "firm, wrong"]), alt.value("#ffffff"), alt.value("#1d1b17")))
    line = alt.Chart(pd.DataFrame({"y": [avg_there]})).mark_rule(
        strokeDash=[5, 3], strokeWidth=1.5, color=ink("title")).encode(
        y=alt.Y("y:Q", scale=y_scale))
    st.altair_chart(style_chart(alt.layer(stacked, text, line).properties(height=380)),
                    width="stretch")
    st.markdown("<div style='font-size:0.85rem'>Below the dashed line, species that are "
                f"there ({avg_there:.1f} per recording): " + key_html(BELOW)
                + "<br>Above it, species listed that are not: " + key_html(ABOVE)
                + "</div>", unsafe_allow_html=True)

    # ---- 3 · every rule searched (P5) --------------------------------------- #
    heading(
        "Every rule searched — birds found against checks needed, on the tuning "
        "recordings",
        "Each dot is one rule the optimiser searched: a firm score bar, a firm "
        "count and a check score bar, with the rest of your rule as it is. Up is "
        "more birds found after checking; right is more checks for the expert.\n\n"
        "Faded dots break one of your limits. The line is each model's frontier: "
        "the most birds any rule finds for that many checks, with firm calls within "
        "your limit. The **large marker** is each model's best for your goal; the "
        "**ring** is your rule. The dashed line is your checks cap.")

    def flags(sw: pl.DataFrame, m: str) -> pl.DataFrame:
        bo = best_own[m]
        same = lambda c: ((pl.col("floor") == c["floor"]) & (pl.col("k") == c["k"])
                          & (pl.col("check_floor") == c["check_floor"]))
        r = rules[m]
        here = {"floor": r.firm.floor if r.firm.floor is not None else -1, "k": r.firm.k,
                "check_floor": r.check.floor if r.check.floor is not None else -1}
        return sw.with_columns(model=pl.lit(m), feasible=cons.feasible(sw),
                               is_best=pl.lit(False) if bo is None else same(bo),
                               is_current=same(here))

    pts = pl.concat([flags(SW[m], m) for m in order]).to_pandas()
    pts["opacity"] = [0.55 if f else 0.12 for f in pts["feasible"]]
    front = []
    for m in order:
        f = pts[(pts["model"] == m) & (pts["firm_precision"] >= min_prec)].sort_values(
            ["checks_per_recording", "recall_reviewed"], ascending=[True, False])
        best_so_far = -1.0
        for _, r in f.iterrows():
            if r["recall_reviewed"] > best_so_far:
                best_so_far = r["recall_reviewed"]
                front.append(r)
    front = pd.DataFrame(front)
    # Start the axis just below the lowest rule rather than at 0%: the rules
    # bunch towards the top, and that is where they differ.
    y_lo = max(0.0, math.floor(float(pts["recall_reviewed"].min()) * 10) / 10 - 0.05)
    enc = dict(x=alt.X("checks_per_recording:Q", title="checks per recording"),
               y=alt.Y("recall_reviewed:Q", title="birds found after checking",
                       scale=alt.Scale(domain=[y_lo, 1], nice=False),
                       axis=alt.Axis(format=".0%", tickCount=10)),
               color=alt.Color("model:N", scale=model_colour, title=None,
                               legend=alt.Legend(orient="bottom")),
               shape=alt.Shape("model:N", scale=model_shape, legend=None))
    tips = [alt.Tooltip("model:N"),
            alt.Tooltip("floor:Q", title="firm precision floor"),
            alt.Tooltip("k:Q", title="firm after"),
            alt.Tooltip("check_floor:Q", title="check precision floor"),
            alt.Tooltip("recall_reviewed:Q", title="birds found after checking",
                        format=".0%"),
            alt.Tooltip("firm_precision:Q", title="firm calls right", format=".1%"),
            alt.Tooltip("checks_per_recording:Q", title="checks per recording",
                        format=".1f")]
    dots = alt.Chart(pts).mark_point(filled=True, size=30).encode(
        **enc, opacity=alt.Opacity("opacity:Q", legend=None, scale=None), tooltip=tips)
    layers = [dots]
    if len(front):
        layers.append(alt.Chart(front).mark_line(strokeWidth=2, interpolate="step-after"
                                                 ).encode(x=enc["x"], y=enc["y"],
                                                          color=enc["color"]))
    if cons.max_checks_per_recording is not None:
        layers.append(alt.Chart(pd.DataFrame({"x": [cons.max_checks_per_recording]}))
                      .mark_rule(strokeDash=[4, 3], color=ink("label")).encode(x="x:Q"))
    layers.append(alt.Chart(pts[pts["is_best"]]).mark_point(
        filled=True, size=260, stroke=ink("text"), strokeWidth=1.5, opacity=1
    ).encode(**enc, tooltip=tips))
    layers.append(alt.Chart(pts[pts["is_current"]]).mark_point(
        filled=False, shape="circle", size=650, strokeWidth=2.5, color=ink("text")
    ).encode(x=enc["x"], y=enc["y"], tooltip=tips))
    st.altair_chart(style_chart(alt.layer(*layers).properties(height=520)), width="stretch")
    ink_t, ink_l = ink("text"), ink("label")

    def icon(svg: str, label: str) -> str:
        return (f"<span style='white-space:nowrap; margin-right:1.1rem'>"
                f"<svg width='22' height='16' style='vertical-align:middle'>{svg}</svg> "
                f"{label}</span>")
    st.markdown(
        "<div style='font-size:0.85rem; line-height:2'>"
        + icon(f"<circle cx='11' cy='8' r='6' fill='#8a8880' stroke='{ink_t}' "
               "stroke-width='1.5'/>", "each model's best for your goal")
        + icon(f"<circle cx='11' cy='8' r='7' fill='none' stroke='{ink_t}' "
               "stroke-width='2'/>", "your current rule")
        + icon("<circle cx='11' cy='8' r='4' fill='#8a8880' opacity='0.6'/>",
               "a rule within your limits")
        + icon("<circle cx='11' cy='8' r='4' fill='#8a8880' opacity='0.25'/>",
               "a rule that breaks a limit")
        + icon("<path d='M1 13 H8 V7 H15 V3 H21' fill='none' stroke='#8a8880' "
               "stroke-width='2'/>", "best birds found for that many checks")
        + (icon(f"<line x1='11' y1='1' x2='11' y2='15' stroke='{ink_l}' "
                "stroke-width='1.5' stroke-dasharray='3 2'/>", "your checks cap")
           if cons.max_checks_per_recording is not None else "")
        + "<br>Colour and shape are the model, as in the legend above.</div>",
        unsafe_allow_html=True)

    # ---- 4 · the rule report (P14) ------------------------------------------ #
    heading(
        "The rule report — on the test recordings",
        "**Run on the test recordings** scores your rule (or each model's) on the "
        "test recordings, which "
        "neither the thresholds nor the sweep have seen. The report stays as it is "
        "while you keep exploring, until you run it again.\n\n"
        "**Two comparisons**, because 'which model is best' depends on which you "
        "mean:\n"
        "1. **Same rule for every model** — the rule above, applied to each. Simple "
        "and directly comparable, but it suits whichever model it was tuned for.\n"
        "2. **Each model at its own best** — the sweep's best for each model under "
        "your limits, on the tuning recordings, then scored on the test ones. The "
        "fair version: each model read the way that suits it.\n\n"
        "Each number has a **95% interval** from resampling test recordings, and its "
        "**tuning** value beside it: a big drop from tuning to test means the rule "
        "was fitted to the tuning recordings. **Today's list** is Compare's species "
        "list: one tier, one detection at the sidebar's thresholds.\n\n"
        "**Full assessment** does better than one split: every recording is a test "
        "recording once (leave-one-recording-out). It tests *the way of choosing* a "
        "rule — your limits — because the best rule is re-chosen in every round.")
    fp = json.dumps({"rules": {m: rules[m].to_dict() for m in order}, "cons": cons.describe(), "tune": tune, "test": test,
                     "models": order, "judge": judge, "profile": profile_name,
                     "spec": spec.to_dict(), "unjudged": unjudged_wrong}, default=str)
    fp_full = json.dumps({"rule": hand_rule.to_dict(), "cons": cons.describe(), "models": order, "judge": judge,
                          "profile": profile_name, "spec": spec.to_dict(),
                          "unjudged": unjudged_wrong, "recs": all_ids}, default=str)
    r1, r2 = st.columns(2)
    if r1.button("Run on the test recordings", type="primary", key="sv_run",
                 icon="▶️", width="stretch"):
        with st.spinner("scoring on the test recordings…"):
            st.session_state["sv_report"] = build_report(
                fp, order, labels, E, SW, rules, hand_rule, cons, tune, test, explore_all,
                annotations, hours_of, label_set, unjudged_wrong, run)
    if r2.button("Full assessment (every recording in turn)", key="sv_full_go",
                 icon="🔁", width="stretch"):
        bar = st.progress(0.0, text="leaving out the first recording…")
        inputs = [survey.ModelInputs(m, th.identity(manifest_of[rid]),
                                     get_survey_detections(rid, judge, profile_name),
                                     get_curve_source(rid, dataset, truth_ref))
                  for m, rid in labels.items()]
        res = survey.full_assessment(
            inputs, hand_rule, spec, annotations, all_ids, hours_of, cons, label_set,
            unjudged_wrong,
            progress=lambda i, n, r: bar.progress(i / n, text=f"round {i + 1} of {n}: "
                                                  f"leaving out {r}"))
        bar.empty()
        res["fingerprint"] = fp_full
        res["cons"] = cons.describe()
        res["cons_obj"] = cons
        res["rule"] = hand_rule
        st.session_state["sv_full"] = res

    report = st.session_state.get("sv_report")
    full = st.session_state.get("sv_full")
    if report is None and full is None:
        st.caption("Nothing run yet.")
    if report is not None:
        show_report(report, fp, order, colour, model_colour, model_shape, rep_cons(report, cons), nm)
    if full is not None:
        show_full(full, fp_full, order, colour, model_colour, model_shape, full.get("cons_obj", cons))

    # ---- 5 · recording cards ------------------------------------------------- #
    heading(
        "One recording, species by species",
        "The lists each model would hand you for one recording, at its rule above: "
        "a row per species annotated in it or listed by any model. For an expert, "
        "the pale cells are the to-check list. Hover a cell for the detections "
        "behind it.")
    which = {r: ("test" if r in set(test) and not explore_all else "tuning") for r in all_ids}
    n_ann = dict(annotations.group_by("recording_id")
                 .agg(n=pl.col("species_key").n_unique()).iter_rows())
    q1, q2 = st.columns([3, 1], vertical_alignment="bottom")
    rec = q1.selectbox("Recording", all_ids, key="sv_rec",
                       format_func=lambda r: f"{r} — {n_ann.get(r, 0)} species annotated · "
                                             f"{which[r]}")
    recording_card(rec, order, colour, run, rules, nm)
    if q2.button("Open in the Explorer", icon="🔎", key="sv_go"):
        st.session_state.update(explorer_rec=rec, _explorer_rec=rec,
                                viewport_start=0.0, inspect_t=0.5, nav_seen={})
        st.switch_page("ui/pages/explorer.py")

    # ---- 6 · species that never go firm ------------------------------------- #
    heading(
        "Species that rarely go firm — on the tuning recordings",
        "For each annotated species, in how many of the tuning recordings it is in "
        "each model's lists firm, only to check, or not at all. Species that are "
        "mostly *to check* or *missed* are where one rule for every species "
        "fits worst — candidates for a rule of their own, later.")
    sp_rows = []
    for m in order:
        out = T[m][0].filter(pl.col("annotated"))
        g = (out.group_by("species_key")
                .agg(recordings=pl.len(),
                     firm=(pl.col("outcome") == "firm, right").sum(),
                     check=(pl.col("outcome") == "check, real").sum())
                .with_columns(model=pl.lit(m)))
        sp_rows.append(g)
    spt = pl.concat(sp_rows).with_columns(
        missed=pl.col("recordings") - pl.col("firm") - pl.col("check"))
    wide = (spt.with_columns(cell=pl.format("{} / {} / {}", "firm", "check", "missed"))
               .pivot(on="model", index=["species_key", "recordings"], values="cell")
               .join(spt.group_by("species_key").agg(firm_share=(
                   pl.col("firm").sum() / pl.col("recordings").sum())), on="species_key")
               .sort(["firm_share", "recordings"], descending=[False, True])
               .with_columns(species=pl.col("species_key").map_elements(
                   nm, return_dtype=pl.Utf8)))
    st.dataframe(wide.select("species", "recordings", *order).rename(
                     {"recordings": "in recordings"}),
                 width="stretch", hide_index=True, height=320)
    st.caption("Each model's cell: recordings where it is firm / to check / not found.")

    # ---- 7 · all the numbers ------------------------------------------------- #
    with st.expander("All the numbers — tuning recordings, at the rule above"):
        per_rec = pl.concat([T[m][1].with_columns(model=pl.lit(m)) for m in order])
        st.dataframe(per_rec, width="stretch", hide_index=True, height=300)
        st.download_button("Per-recording outcomes (CSV)", per_rec.write_csv(),
                           "survey_tuning_per_recording.csv", "text/csv",
                           key="sv_dl_tune")
        swl = pl.concat([SW[m].with_columns(model=pl.lit(m)) for m in order])
        st.download_button("The sweep (CSV)", swl.write_csv(), "survey_sweep.csv",
                           "text/csv", key="sv_dl_sweep")

    # ---- 8 · save the setup, apply it to the whole dataset (V1.2 F4) ------- #
    save_setup(ctx, labels, order, rules, E, tune, test, explore_all, annotations,
               hours_of, label_set, unjudged_wrong)
    apply_setup(ctx, nm)


def save_setup(ctx, arms, order, rules, E, tune, test, explore_all, annotations,
               hours_of, label_set, unjudged_wrong) -> None:
    """Freeze a model's rule, its thresholds and what it measured on the test
    recordings, so it can be run over recordings nobody labelled."""
    heading(
        "Save this setup",
        "A **setup** is one model read one way: the rule above, each species' "
        "threshold at the score bars it uses (fitted on the tuning recordings), the "
        "judge, and **what it measured on the test recordings**, with 95% intervals. "
        "Saved, it can be applied to every recording in this dataset — or to another "
        "dataset the same model has scored — and every list it produces carries "
        "that measurement as its claim.",
        level="###")
    a, b = st.columns([1, 1.4])
    m = a.selectbox("Model", order, key="sv_save_model")
    rid = arms[m]
    rule = rules[m]
    on = list(test) if not explore_all else list(tune)
    res = survey.assess(E[m], rule, annotations, on, hours_of, label_set, unjudged_wrong)
    bars = survey.fit_bars(get_curve_source(rid, ctx.dataset, ctx.truth_ref),
                           th.identity(ctx.manifest_of[rid]), ctx.spec, list(tune))
    bars[survey.SIDEBAR] = ctx.resolved[rid]
    s = setups.Setup(
        name="", model=th.identity(ctx.manifest_of[rid]), model_label=m,
        rule=rule.to_dict(),
        bars=setups.bars_payload(bars, {rule.check.bar, rule.firm.bar}),
        judge=ctx.judge, profile=ctx.profile_name,
        fitted_on={"dataset": ctx.dataset, "truth_ref": ctx.truth_ref, "recordings": list(tune),
                   "run_id": rid},
        performance={"score": {k: float(v) for k, v in res["score"].items()
                               if isinstance(v, (int, float))},
                     "intervals": {k: list(v) for k, v in res["intervals"].items()},
                     "recordings": len(on),
                     "hours": float(sum(hours_of.get(r, 0) for r in on)),
                     "on": "all (not held out)" if explore_all else "test"},
        created_by=st.session_state.get("labeller", ""))
    st.markdown(f"> {s.claim()}")
    if explore_all:
        st.warning("The split is off, so this was measured on the recordings it was "
                   "tuned on, which flatters it. Turn the split on before saving a setup "
                   "you mean to rely on.")
    name = b.text_input("Name", key="sv_save_name", placeholder=f"{ctx.dataset}-{m}")
    if b.button("Save setup", type="primary", key="sv_save_go", disabled=not name):
        s.name = name.strip()
        try:
            setups.save(cfg.store_dir, s)
        except ValueError as e:
            st.error(str(e))
        else:
            st.success(f"Saved *{s.name}*. Apply it below.")


def apply_setup(ctx, nm) -> None:
    """V1.2 F4: run a saved setup over every recording in the dataset — the
    labelled ones keep their labels, the rest get the model's lists, and the
    export says which is which and what the model's lists were measured to be."""
    heading(
        "Apply a setup to the whole dataset",
        "Runs a saved setup over **every** recording in this dataset. Where a "
        "recording is labelled end to end, its list is the labels (an expert's list "
        "beats a model's) with the model's verdict alongside; everywhere else, the "
        "list is the model's output, marked as such. The export carries the setup's "
        "measured performance: that is what a reader can trust the model-output rows "
        "to mean.",
        level="###")
    usable = {}
    for name in setups.list_setups(cfg.store_dir):
        su = setups.load(cfg.store_dir, name)
        runs = [m for m in ctx.included if th.identity(m) == su.model
                and m.run_id in complete_runs(ctx.dataset)]
        if runs:
            usable[name] = (su, runs[0])
    if not usable:
        st.caption("No saved setup matches a model that has scored every recording "
                   "here. Save one above (on a labelled dataset), with the same model "
                   "run over this one.")
        return
    pick = st.selectbox("Setup", list(usable), key="sv_apply_pick")
    su, run = usable[pick]
    st.markdown(f"> {su.claim()}")
    profiles = list_profiles(cfg.profiles_dir)
    profile = su.profile if su.profile in profiles else ctx.profile_name
    if profile != su.profile:
        st.warning(f"Profile *{su.profile}* is not here; using *{profile}* to judge.")
    if st.button("Apply to every recording", type="primary", key="sv_apply_go"):
        recs = ctx.recordings["recording_id"].to_list()
        with st.spinner("listing every recording…"):
            out = setups.apply(su, get_survey_detections(run.run_id, su.judge, profile),
                               recs, ctx.truth_obj)
        out = out.with_columns(species=pl.col("species_key").map_elements(
            nm, return_dtype=pl.Utf8))
        d = Path(cfg.store_dir) / "datasets" / ctx.dataset / "results"
        d.mkdir(parents=True, exist_ok=True)
        out.write_csv(d / f"{su.name}.csv")
        (d / f"{su.name}.txt").write_text(su.claim() + "\n")
        st.session_state["sv_applied"] = (su.name, out)
    applied = st.session_state.get("sv_applied")
    if applied and applied[0] == pick:
        out = applied[1]
        by = out.group_by("source").agg(recordings=pl.col("recording_id").n_unique(),
                                        rows=pl.len()).sort("source")
        st.dataframe(by, hide_index=True)
        st.dataframe(out.select("recording_id", "species", "listed", "source",
                                "model_tier", "firm_bar_detections",
                                "check_bar_detections"),
                     hide_index=True, width="stretch", height=320)
        c1, c2 = st.columns(2)
        c1.download_button("Species lists (CSV)", out.write_csv(),
                           f"{ctx.dataset}_{pick}_lists.csv", "text/csv",
                           key="sv_apply_dl")
        c2.download_button("What the model-output rows mean (TXT)", su.claim() + "\n",
                           f"{ctx.dataset}_{pick}_claim.txt", "text/plain",
                           key="sv_apply_claim")


def infeasible(pick, cons: survey.Constraints, hand_rule: survey.SurveyRule, who: str):
    """No rule meets every limit: explain what is reachable, and return the rule
    that comes closest (dropping the minimum birds found, then the checks cap),
    or the hand rule if even the firm-precision limit cannot be met."""
    from dataclasses import replace
    lines = [f"**No rule meets all your limits for {who}** on the tuning recordings."]
    closest, dropped = None, None
    if cons.min_recall_reviewed is not None:
        within = pick(replace(cons, min_recall_reviewed=None, objective="recall_reviewed"))
        if within is not None:
            lines.append(
                f"Within your other limits, the most birds any rule finds after checking "
                f"is **{within['recall_reviewed']:.0%}** ({cell_text(within, hand_rule)}) — short "
                f"of the {cons.min_recall_reviewed:.0%} you asked for.")
            closest, dropped = within, "below your minimum birds found"
        if cons.max_checks_per_recording is not None:
            uncapped = pick(replace(cons, max_checks_per_recording=None,
                                    objective="checks_per_recording"))
            if uncapped is not None:
                lines.append(
                    f"Reaching {cons.min_recall_reviewed:.0%} with firm calls "
                    f"≥ {cons.min_firm_precision:.0%} right takes at least "
                    f"**{uncapped['checks_per_recording']:.1f} checks per recording** "
                    f"({cell_text(uncapped, hand_rule)}) — "
                    f"over your cap of {cons.max_checks_per_recording:g}.")
            else:
                lines.append(f"No rule reaches {cons.min_recall_reviewed:.0%} with firm "
                             f"calls ≥ {cons.min_firm_precision:.0%} right, however many "
                             "checks you allow.")
    if closest is None and cons.max_checks_per_recording is not None:
        c = pick(replace(cons, max_checks_per_recording=None, min_recall_reviewed=None,
                         objective="checks_per_recording"))
        if c is not None:
            lines.append(f"Keeping firm calls ≥ {cons.min_firm_precision:.0%} right needs at "
                         f"least **{c['checks_per_recording']:.1f} checks per recording** — "
                         f"over your cap of {cons.max_checks_per_recording:g}.")
            closest, dropped = c, "over your checks cap"
    if closest is None:
        lines.append(f"No rule keeps firm calls {cons.min_firm_precision:.0%} right. Lower "
                     "that limit.")
    lines.append("Loosen a limit to get a rule that meets them all."
                 + (" Meanwhile the page shows the closest rule." if closest else
                    " Meanwhile the page shows your last rule set by hand."))
    if closest is None:
        return hand_rule, "no rule within your limits: your last rule set by hand", lines
    return (survey.swept(hand_rule, closest["floor"], closest["k"],
                         closest.get("check_floor")),
            f"closest rule: {dropped}", lines)


# --------------------------------------------------------------------------- #
# The report (P14) and the full assessment (P15)
# --------------------------------------------------------------------------- #

def build_report(fp, order, labels, E, SW, rules, hand_rule, cons, tune, test,
                 explore_all, annotations, hours_of, label_set, unjudged_wrong, run) -> dict:
    """Score your rules, each model's own best and today's list on the test
    recordings."""
    if not test:
        return {"fingerprint": fp, "empty": True}
    rep = {"fingerprint": fp, "rules": dict(rules),
           "one_rule": len(set(rules.values())) == 1, "cons": cons.describe(),
           "cons_obj": cons, "tune": tune, "test": test, "explore_all": explore_all,
           "same": {}, "own": {}, "baseline": {},
           "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    baseline = survey.SurveyRule(survey.Condition(None, 1), survey.Condition(None, 1))
    for m in order:
        res = survey.assess(E[m], rules[m], annotations, test, hours_of, label_set,
                            unjudged_wrong)
        rep["same"][m] = {**res, "tuning": run(m, rules[m], tune)[2], "rule": rules[m]}
        cell = survey.best(SW[m], cons)
        if cell is None:
            rep["own"][m] = None
        else:
            r = survey.swept(hand_rule, cell["floor"], cell["k"], cell.get("check_floor"))
            res = survey.assess(E[m], r, annotations, test, hours_of, label_set,
                                unjudged_wrong)
            rep["own"][m] = {**res, "tuning": run(m, r, tune)[2], "rule": r}
        rep["baseline"][m] = survey.assess(E[m], baseline, annotations, test, hours_of,
                                           label_set, unjudged_wrong, n_boot=0)["score"]
    return rep


def rep_cons(rep: dict, cons: survey.Constraints) -> survey.Constraints:
    """The goal a report was run with — it is judged by that, not today's."""
    return rep.get("cons_obj", cons)


def score_table(entries: dict, order: list[str], baseline: dict | None = None) -> pd.DataFrame:
    """A row per headline number, a column per model: test value (interval) · tuning."""
    rows = []
    for stat, (label, _) in survey.STATS.items():
        row = {"": label}
        for m in order:
            e = entries.get(m)
            if e is None:
                row[m] = "no rule meets the limits"
                continue
            v, (lo, hi) = e["score"][stat], e["intervals"][stat]
            cell = f"{fmt(stat, v)}"
            if not (math.isnan(lo) or math.isnan(hi)):
                cell += f"  ({fmt(stat, lo)}–{fmt(stat, hi)})"
            if e.get("tuning") is not None:
                cell += f"  · tuning {fmt(stat, e['tuning'][stat])}"
            if baseline is not None and stat in ("firm_precision", "firm_recall"):
                cell += f"  · today {fmt(stat, baseline[m][stat])}"
            row[m] = cell
        rows.append(row)
    return pd.DataFrame(rows)


def winner(entries: dict, order: list[str], cons: survey.Constraints) -> str:
    """Which model does best on the goal's objective while meeting its limits on
    these recordings — or that there is no clear winner, when the intervals
    overlap. A model that does better but broke a limit here is named."""
    obj = cons.objective
    higher = survey.OBJECTIVES[obj][1]
    verb = VERBS[obj]

    def meets(s: dict) -> bool:
        ok = s["firm_precision"] >= cons.min_firm_precision
        if cons.max_checks_per_recording is not None:
            ok = ok and s["checks_per_recording"] <= cons.max_checks_per_recording
        if cons.min_recall_reviewed is not None:
            ok = ok and s["recall_reviewed"] >= cons.min_recall_reviewed
        return bool(ok)

    have = [m for m in order if entries.get(m) is not None]
    ok = [m for m in have if meets(entries[m]["score"])]
    sign = -1 if higher else 1
    better = lambda a, b: sign * a < sign * b
    if not ok:
        return f"No model meets your limits on these recordings ({cons.describe()})."
    ranked = sorted(ok, key=lambda m: sign * entries[m]["score"][obj])
    top = ranked[0]
    v = entries[top]["score"][obj]
    over = [m for m in have if m not in ok and better(entries[m]["score"][obj], v)]
    note = "".join(
        f" {m} does better ({fmt(obj, entries[m]['score'][obj])}), but breaks one "
        f"of your limits here — firm calls {entries[m]['score']['firm_precision']:.0%} "
        f"right, {entries[m]['score']['checks_per_recording']:.1f} checks per recording."
        for m in over)
    if len(ranked) == 1:
        return (f"**{top}** is the only model within your limits here; it {verb} "
                f"({fmt(obj, v)})." + note)
    # Paired (V1.2 E2): both models on the same resampled recordings, so a hard
    # recording counts against both and only the gap between them is tested.
    diff, dlo, dhi = survey.paired(entries[top], entries[ranked[1]])[obj]
    clear = (dlo > 0) if higher else (dhi < 0)
    gap = (f"by {fmt(obj, abs(diff))} (95% interval of the difference "
           f"{fmt(obj, dlo)} to {fmt(obj, dhi)})" if not math.isnan(dlo) else "")
    if not any(math.isnan(x) for x in (dlo, dhi)) and clear:
        return (f"**{top}** wins: it {verb} ({fmt(obj, v)}), ahead of {ranked[1]} "
                f"{gap}, measured on the same recordings." + note)
    return (f"**No clear winner.** Of the models within your limits, {top} {verb} "
            f"({fmt(obj, v)}), but compared on the same recordings its lead over "
            f"{ranked[1]} could go either way{' — ' + gap if gap else ''}. More "
            "labelled recordings would narrow it." + note)


def stat_bars(entries: dict, order: list[str], model_colour) -> alt.Chart | None:
    """One small bar chart per headline number, most important first: a bar per
    model at its value, a whisker for its 95% interval, a grey tick at its
    tuning value. Shares run 0–100%; counts run from 0 to the largest value."""
    charts = []
    for stat, (label, higher) in survey.STATS.items():
        if stat == "checks_per_hour":
            continue    # with hour-long recordings it repeats checks per recording
        rows = []
        for m in order:
            e = entries.get(m)
            if e is None:
                continue
            v = e["score"][stat]
            lo, hi = e["intervals"][stat]
            tune = e["tuning"][stat] if e.get("tuning") is not None else float("nan")
            ok = lambda x: x is not None and not math.isnan(x)
            rows.append({"model": m, "v": v if ok(v) else None,
                         "lo": lo if ok(lo) else None, "hi": hi if ok(hi) else None,
                         "tune": tune if ok(tune) else None, "text": fmt(stat, v),
                         "tx": max(x for x in (v, hi, tune, 0.0) if ok(x))})
        d = pd.DataFrame(rows)
        if d.empty:
            continue
        pct = stat in PCT
        if pct:
            top, ticks = 1.0, [i / 5 for i in range(6)]
        else:
            # Counts end exactly on a round tick, as shares end at 100%: an
            # axis ending between ticks stops its gridlines short, and the
            # chart reads narrower than its neighbours.
            want = max(0.5, float(d["tx"].max()) * 1.18)
            step = next((st_ for st_ in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25,
                                         50, 100, 200, 250, 500, 1000) if want / st_ <= 5),
                        want / 5)
            top = math.ceil(want / step) * step
            ticks = [round(i * step, 6) for i in range(int(round(top / step)) + 1)]
        scale = alt.Scale(domain=[0, top], nice=False)
        x = alt.X("v:Q", title=None, scale=scale,
                  axis=alt.Axis(format=".0%" if pct else "~g", values=ticks))
        y = alt.Y("model:N", sort=order, title=None,
                  axis=alt.Axis(labelLimit=150, labelFontSize=11))
        tip = [alt.Tooltip("model:N"), alt.Tooltip("text:N", title=label)]
        bar = alt.Chart(d).mark_bar(height=16, cornerRadiusEnd=2).encode(
            x=x, y=y, color=alt.Color("model:N", scale=model_colour, legend=None),
            tooltip=tip)
        whisk = alt.Chart(d.dropna(subset=["lo", "hi"])).mark_rule(
            strokeWidth=1.5, color=ink("text")).encode(
            x=alt.X("lo:Q", scale=scale), x2="hi:Q", y=y)
        tick = alt.Chart(d.dropna(subset=["tune"])).mark_tick(
            thickness=2.5, size=20, color=ink("label")).encode(
            x=alt.X("tune:Q", scale=scale), y=y)
        text = alt.Chart(d).mark_text(align="left", dx=6, fontSize=12,
                                      fontWeight="bold", color=ink("text")).encode(
            x=alt.X("tx:Q", scale=scale), y=y, text="text:N")
        title = f"{label[0].upper() + label[1:]} — {'higher' if higher else 'lower'} is better"
        charts.append(alt.layer(bar, whisk, tick, text).properties(
            width=620, height=26 * len(d),
            title=alt.Title(title, anchor="start", fontSize=13, fontWeight="bold")))
    return alt.vconcat(*charts, spacing=18) if charts else None


def show_scores(title_rule, entries, order, colour, model_colour, model_shape, cons,
                baseline=None):
    st.markdown(winner(entries, order, cons))
    if title_rule is not None:
        lines = []
        for m in order:
            e = entries.get(m)
            lines.append(f"<span style='color:{colour[m]}'>●</span> <b>{m}</b>: "
                         + ("no rule meets the limits on the tuning recordings" if e is None
                            else title_rule(m, e)))
        st.markdown("<div style='font-size:0.88rem; line-height:1.8'>" + "<br>".join(lines)
                    + "</div>", unsafe_allow_html=True)
    chart = stat_bars(entries, order, model_colour)
    if chart is not None:
        st.altair_chart(style_chart(chart), width="content")
        st.caption("Bar: the value on these recordings · black line: its 95% interval · "
                   "grey tick: the same number on the tuning recordings, where there is "
                   "one. All the numbers are in the table below.")
    st.dataframe(score_table(entries, order, baseline), hide_index=True, width="stretch")


def show_report(rep, fp, order, colour, model_colour, model_shape, cons, nm):
    with st.container(border=True):
        if rep.get("empty"):
            st.warning("There are no test recordings — move some into the test set.")
            return
        stale = rep["fingerprint"] != fp
        st.markdown(f"##### Rule report · {rep['created']} · {len(rep['test'])} test "
                    f"recording(s)" + (" · :orange-badge[out of date — run it again]"
                                       if stale else ""))
        if rep["explore_all"]:
            st.warning("Scored on the same recordings it was tuned on (the split is "
                       "off) — these numbers flatter the rule.")
        st.caption(f"Limits: {rep['cons']}. Each number: test value (95% interval) · "
                   "its tuning value. 'today' is Compare's species list: one detection "
                   "at the sidebar's thresholds.")
        own_line = lambda m, e: f"<i>{e['rule'].describe()}</i>"
        if rep.get("one_rule", True):
            t1, t2 = st.tabs(["Same rule for every model",
                              "Each model at its own best rule"])
            with t1:
                st.markdown(f"Rule: *{next(iter(rep['rules'].values())).describe()}*.")
                show_scores(None, rep["same"], order, colour, model_colour, model_shape,
                            cons, rep["baseline"])
            with t2:
                show_scores(own_line, rep["own"], order, colour, model_colour,
                            model_shape, cons, rep["baseline"])
        else:
            # Your rules already are each model's own best: one view, not two
            # identical tabs.
            st.markdown("**Each model at its own best rule**")
            show_scores(own_line, rep["same"], order, colour, model_colour, model_shape,
                        cons, rep["baseline"])
        per_rec = pl.concat([rep["same"][m]["per_recording"].with_columns(model=pl.lit(m))
                             for m in order])
        per_sp = pl.concat([
            rep["same"][m]["outcomes"].group_by("species_key", "outcome").len()
            .pivot(on="outcome", index="species_key", values="len")
            .with_columns(model=pl.lit(m)) for m in order], how="diagonal")
        with st.expander("Per recording and per species — same rule"):
            st.dataframe(per_rec, hide_index=True, width="stretch", height=280)
            st.dataframe(per_sp.with_columns(species=pl.col("species_key").map_elements(
                nm, return_dtype=pl.Utf8)).fill_null(0), hide_index=True,
                width="stretch", height=280)
        snapshot = {
            "format": "bex-survey-report/1", "created": rep["created"],
            "rules": {m: r.to_dict() for m, r in rep["rules"].items()},
            "limits": rep["cons"],
            "tuning_recordings": rep["tune"], "test_recordings": rep["test"],
            "same_rule": {m: rep["same"][m]["score"] | {"intervals": rep["same"][m]["intervals"]}
                          for m in order},
            "own_best": {m: (None if rep["own"][m] is None else
                             {"rule": rep["own"][m]["rule"].to_dict(),
                              **rep["own"][m]["score"],
                              "intervals": rep["own"][m]["intervals"]}) for m in order},
            "today": rep["baseline"],
        }
        d1, d2 = st.columns(2)
        d1.download_button("Report snapshot (JSON)", json.dumps(snapshot, indent=2,
                                                                default=float),
                           "survey_report.json", "application/json", key="sv_dl_rep")
        d2.download_button("Per-recording outcomes (CSV)", per_rec.write_csv(),
                           "survey_test_per_recording.csv", "text/csv", key="sv_dl_rep_csv")


def show_full(res, fp, order, colour, model_colour, model_shape, cons):
    with st.container(border=True):
        stale = res["fingerprint"] != fp
        st.markdown(f"##### Full assessment · {res['n_rounds']} rounds, each recording "
                    "left out once" + (" · :orange-badge[out of date — run it again]"
                                       if stale else ""))
        st.caption(f"Limits: {res['cons']}. Every number is pooled over the left-out "
                   "recordings, each scored by a rule chosen without it — so it measures "
                   "the way of choosing a rule, honestly. Intervals resample recordings.")
        t1, t2 = st.tabs(["Each model at its own best rule", "One shared rule"])
        entries_own = {m: (res["models"][m]["own"] | {"tuning": None}) for m in order}
        entries_sh = {m: (res["models"][m]["shared"] | {"tuning": None}) for m in order}

        def stab(m):
            s, f = res["stability"][m], res["final_own"][m]
            if f is None:
                return "no rule meets the limits on all recordings"
            return (f"use <b>{cell_text(f, res['rule'])}</b> — "
                    f"{s['same']} of {s['rounds']} rounds chose it"
                    + ("" if s['same'] == s['rounds'] else
                       f" (firm floors {s['floors'][0]:g}–{s['floors'][1]:g}, "
                       f"firm after {s['ks'][0]}–{s['ks'][1]}, check floors "
                       f"{s['check_floors'][0]:g}–{s['check_floors'][1]:g})")
                    + (f"; {s['none']} round(s) found no rule" if s["none"] else ""))
        with t1:
            show_scores(lambda m, e: stab(m), entries_own, order, colour, model_colour,
                        model_shape, cons)
            cols = st.columns(len(order))
            for col, m in zip(cols, order):
                f = res["final_own"][m]
                if f is not None and col.button(f"Use {m}'s rule", key=f"sv_full_use_{m}"):
                    st.session_state["sv_pending_rule"] = {
                        **f, "source": f"the full assessment's rule for {m}"}
                    st.rerun()
        with t2:
            f, s = res["final_shared"], res["stability_shared"]
            st.markdown(
                "One rule for every model, chosen to meet the limits for all of them. "
                + ("No such rule on all recordings." if f is None else
                   f"Use **{cell_text(f, res['rule'])}** — "
                   f"{s['same']} of {s['rounds']} rounds chose it."))
            show_scores(None,
                        entries_sh, order, colour, model_colour, model_shape, cons)
            if f is not None and st.button("Use the shared rule", key="sv_full_use_shared"):
                st.session_state["sv_pending_rule"] = {
                    **f, "source": "the full assessment's shared rule"}
                st.rerun()


# --------------------------------------------------------------------------- #
# One recording
# --------------------------------------------------------------------------- #

def recording_card(rec, order, colour, run, rules, nm) -> None:
    frames = []
    for m in order:
        out, _, _ = run(m, rules[m], [rec])
        frames.append(out.to_pandas().assign(model=m))
    grid = pd.concat(frames, ignore_index=True)
    if grid.empty:
        st.caption("Nothing annotated or listed in this recording.")
        return
    grid["species"] = [nm(k) for k in grid["species_key"]]
    grid["detections"] = [f"firm count {int(a)}, check count {int(b)}"
                          for a, b in zip(grid["firm_shown"].fillna(0),
                                          grid["check_shown"].fillna(0))]
    real = set(grid.loc[grid["annotated"], "species"])
    truth_rows = pd.DataFrame({"species": sorted(real), "model": "annotated",
                               "outcome": "annotated", "detections": ""})
    cells = pd.concat([truth_rows, grid[["species", "model", "outcome", "detections"]]],
                      ignore_index=True)
    firm_n = grid[grid["outcome"].str.startswith("firm")].groupby("species").size()
    sp_order = sorted(set(cells["species"]),
                      key=lambda sp: (sp not in real, -firm_n.get(sp, 0), sp))
    columns = ["annotated", *order]
    palette = {**views.SURVEY_COLOURS, "annotated": views.TRUTH_COLOUR}
    chart = alt.Chart(cells).mark_rect(stroke="#ffffff", strokeWidth=2,
                                       cornerRadius=3).encode(
        x=alt.X("model:N", sort=columns, title=None,
                axis=alt.Axis(orient="top", labelAngle=0, labelLimit=160)),
        y=alt.Y("species:N", sort=sp_order, title=None,
                axis=alt.Axis(labelLimit=240, labelOverlap=False)),
        color=alt.Color("outcome:N", legend=None, scale=alt.Scale(
            domain=list(palette), range=list(palette.values()))),
        tooltip=[alt.Tooltip("species:N"), alt.Tooltip("model:N"),
                 alt.Tooltip("outcome:N"), alt.Tooltip("detections:N")])
    st.altair_chart(style_chart(chart.properties(
        width=min(760, 150 * len(columns)), height=alt.Step(18))), width="content")
    st.markdown("<div style='font-size:0.85rem'><span style='color:"
                f"{views.TRUTH_COLOUR}'>■</span> annotated &nbsp; "
                + key_html(BELOW + ABOVE) + "</div>", unsafe_allow_html=True)
    lines = []
    for m in order:
        g = grid[grid["model"] == m]["outcome"].value_counts()
        n = lambda *os: int(sum(g.get(o, 0) for o in os))
        lines.append(
            f"<span style='color:{colour[m]}'>●</span> <b>{m}</b>: firm "
            f"{n(*survey.FIRM)} ({n('firm, right')} right, {n('firm, wrong')} wrong) · "
            f"to check {n(*survey.CHECK)} ({n('check, real')} real) · misses "
            f"{n('missed', 'filter hid a real bird')} of {len(real)}")
    st.markdown("<div style='font-size:0.85rem; line-height:1.7'>" + "<br>".join(lines)
                + "</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# No annotations (P7): the lists, unscored
# --------------------------------------------------------------------------- #

def unlabelled(ctx, labels, order, colour, nm) -> None:
    st.info(f"**{ctx.dataset}** has no annotations, so rules cannot be scored or swept "
            "here. The lists below use the sidebar's thresholds — load a threshold set "
            "fitted on labelled data to make them meaningful.")
    a, b = st.columns(2)
    firm_k = int(a.number_input("Firm after this many detections", 1, 200, 3, 1,
                                key="sv_u_firm"))
    check_k = int(b.number_input("To check after this many", 1, 200, 1, 1,
                                 key="sv_u_check"))
    rule = survey.SurveyRule(survey.Condition(None, check_k), survey.Condition(None, firm_k))
    rec = st.selectbox("Recording", ctx.recordings["recording_id"].to_list(), key="sv_u_rec")
    rows = []
    for m, rid in labels.items():
        ev = get_survey_evidence(rid, ctx.dataset, ctx.judge, ctx.profile_name, ctx.spec,
                                 th.identity(ctx.manifest_of[rid]), None, (),
                                 resolved_key(ctx.resolved[rid]), ctx.truth_ref)
        t = survey.tiers(ev, rule, [rec]).filter(pl.col("tier") != "not found")
        for sp, tier, n in t.select("species_key", "tier", "firm_shown").iter_rows():
            rows.append({"species": nm(sp), "model": m, "tier": tier, "detections": n})
    if not rows:
        st.caption("No model lists anything in this recording.")
        apply_setup(ctx, nm)
        return
    df = pd.DataFrame(rows).pivot_table(index="species", columns="model", values="tier",
                                        aggfunc="first").fillna("—").reset_index()
    st.dataframe(df, hide_index=True, width="stretch")
    apply_setup(ctx, nm)


render(sidebar.render())
