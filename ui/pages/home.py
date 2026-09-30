"""BEX · Home — the landing page: what BEX is, what is loaded, where to go."""
from ui.common import *  # noqa: F403 — the shared app toolkit

def masthead() -> None:
    """The logo — B and EX in their two colours — over the name it abbreviates,
    with the same letters picked out, and one line on what BEX is for."""
    c = brand()
    st.markdown(
        f"<div style='background:linear-gradient(120deg, {c['wash']}, transparent 70%);"
        "border-radius:14px; padding:1.6rem 1.8rem 1.4rem; margin:0.4rem 0 1.2rem'>"
        "<div style='font-size:4.2rem; font-weight:800; letter-spacing:0.04em; "
        "line-height:1'>"
        f"<span style='color:{c['b']}'>B</span><span style='color:{c['ex']}'>EX</span>"
        "</div>"
        "<div style='font-size:1.45rem; font-weight:600; margin-top:0.35rem; "
        "opacity:0.92'>"
        f"<span style='color:{c['b']}'>B</span>ioacoustics "
        f"<span style='color:{c['ex']}'>Ex</span>plorer</div>"
        "<div style='margin-top:0.8rem; max-width:46rem; opacity:0.85'>"
        "Run your recordings through several bird-ID models side by side, measure "
        "each against the same ground truth, and choose how to process the rest of "
        "your data on the evidence rather than on defaults.</div>"
        "</div>", unsafe_allow_html=True)


def primary_button_style() -> None:
    """The Explorer button in the brand teal, not Streamlit's default red."""
    c = brand()
    st.markdown(
        "<style>.st-key-home_explorer button{"
        f"background:{c['b']}; border-color:{c['b']}; color:{c['on']}}}"
        ".st-key-home_explorer button:hover{"
        f"background:{c['ex']}; border-color:{c['ex']}; color:{c['on']}}}</style>",
        unsafe_allow_html=True)


def card(title: str, icon: str, blurb: str) -> None:
    st.markdown(f"#### {icon} {title}")
    st.caption(blurb)


def render() -> None:
    masthead()
    datasets = ingest.list_datasets(cfg.store_dir)
    runs = views.list_runs(cfg.store_dir)

    if not datasets or not runs:
        # A fresh install: say exactly what to do next, in order.
        st.markdown("### Getting started")
        steps = [
            (bool(datasets), "**Add recordings**",
             "Point BEX at a folder of audio.", "ui/pages/recordings.py"),
            (bool(runs), "**Run a model over them**",
             "BirdNET and Perch runners ship with BEX.", "ui/pages/models.py"),
            (False, "**Explore**", "Every model's detections, window by window.",
             "ui/pages/explorer.py"),
        ]
        for done, title, detail, page in steps:
            st.markdown(f"{'✅' if done else '⬜️'} {title} — {detail}")
        st.page_link("ui/pages/recordings.py", label="How to add recordings", icon="📁")
        return

    # The one primary action on the page (page links cannot be styled as one).
    primary_button_style()
    if st.button("Open the Explorer", icon="🔎", type="primary", key="home_explorer"):
        st.switch_page("ui/pages/explorer.py")

    # ---- what is loaded, and where to add more ------------------------------ #
    st.markdown("### Data and models")
    n_recs, hours, labelled = 0, 0.0, 0
    for d in datasets:
        recs, _ = get_dataset(d)
        n_recs += len(recs)
        hours += float(recs["duration_s"].sum()) / 3600
        labelled += int(recs["labelled"].sum())
    left, right = st.columns(2)
    with left.container(border=True):
        card("Recordings", "📁", "The audio BEX has scanned, and the annotations that "
                                 "make it possible to score a model.")
        m = st.columns(3)
        m[0].metric("Datasets", len(datasets))
        m[1].metric("Recordings", f"{n_recs:,}")
        m[2].metric("Hours of audio", f"{hours:,.1f}")
        st.caption(f"{labelled:,} of {n_recs:,} recordings annotated")
        st.page_link("ui/pages/recordings.py", label="Add recordings", icon="➕")

    with right.container(border=True):
        card("Models", "🧠", "The model runs over those recordings, and the "
                             "per-species thresholds that turn scores into detections.")
        style = views.model_styles(runs)
        labels = views.arm_labels(runs)
        m = st.columns(3)
        m[0].metric("Model runs", len(runs))
        m[1].metric("Models", len({m_.model_name for m_ in runs}))
        m[2].metric("Threshold sets", len(th.list_sets(cfg.thresholds_dir)),
                    help="Saved, named sets of per-species thresholds.")
        st.markdown(
            "<div style='font-size:0.85rem; line-height:1.7'>" + " &nbsp; ".join(
                f"<span style='white-space:nowrap'><span style='color:"
                f"{style[r.run_id]['colour']}'>●</span> {lbl}</span>"
                for r, lbl in ((r, labels[r.run_id]) for r in
                               sorted(runs, key=lambda r: style[r.run_id]["rank"])))
            + "</div>", unsafe_allow_html=True)
        l1, l2 = st.columns(2)
        l1.page_link("ui/pages/models.py", label="Add a model", icon="➕")
        l2.page_link("ui/pages/thresholds.py", label="Set thresholds", icon="🎚️")

    # ---- analysis ------------------------------------------------------------ #
    st.markdown("### Analysis")
    cards = st.columns(3)
    for col, (page, icon, title, text) in zip(cards, [
        ("ui/pages/compare.py", "📊", "Compare models",
         "Which model is better overall, on equal terms."),
        ("ui/pages/scorecard.py", "🐦", "Species scorecard",
         "One bird at a time: what each model found, missed and confused it with."),
        ("ui/pages/forensics.py", "🗺️", "Geofilter forensics",
         "What a location filter hides, and what that costs each model."),
    ]):
        with col.container(border=True):
            st.page_link(page, label=title, icon=icon)
            st.caption(text)

    st.page_link("ui/pages/about.py", label="About BEX — what it measures and why",
                 icon="ℹ️")


render()
