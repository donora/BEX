"""BEX (Bioacoustics Explorer) — compare bioacoustic ID methods, with the
geographic filter made visible.

The entry point: page setup and navigation only. The pages live in `ui/pages/`,
the settings sidebar in `ui/sidebar.py`, and everything they share in
`ui/common.py`. Every number shown is computed by pure functions in the `bex`
library over the on-disk store, reproducible headlessly.

    .venv/bin/streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="BEX — Bioacoustics Explorer", page_icon="🐦",
                   layout="wide")
from bex import theme as _theme  # noqa: E402

st.markdown(_theme.css(), unsafe_allow_html=True)

def app_mode() -> str:
    """Simple (the guided path: recordings to checked detections, exported as a
    file or sent to EcoLabel) or Research (everything else BEX does).
    A fresh install opens Simple; a store that already has datasets opens
    Research, so existing users find BEX as they left it. The choice is kept."""
    import os
    from bex import prefs
    from ui.common import cfg, ingest
    saved = os.environ.get("BEX_MODE") or prefs.get(cfg.store_dir, "mode")
    if saved in ("simple", "research"):
        return saved
    return "research" if ingest.list_datasets(cfg.store_dir) else "simple"


def set_mode(mode: str) -> None:
    from bex import prefs
    from ui.common import cfg
    prefs.put(cfg.store_dir, "mode", mode)


MODE = app_mode()
PAGES = {
    "home": st.Page("ui/pages/home.py", title="Home", icon="🏠", default=MODE == "research"),
    "guided": st.Page("ui/pages/guided.py", title="Simple mode", icon="🌿", default=MODE == "simple"),
    "label": st.Page("ui/pages/label.py", title="Label", icon="🏷️"),
    "explorer": st.Page("ui/pages/explorer.py", title="Explorer", icon="🔎"),
    "compare": st.Page("ui/pages/compare.py", title="Compare models"),
    "scorecard": st.Page("ui/pages/scorecard.py", title="Species scorecard"),
    "forensics": st.Page("ui/pages/forensics.py", title="Geofilter forensics"),
    "survey": st.Page("ui/pages/survey.py", title="Survey protocol"),
    "recordings": st.Page("ui/pages/recordings.py", title="Recordings"),
    "models": st.Page("ui/pages/models.py", title="Models"),
    "thresholds": st.Page("ui/pages/thresholds.py", title="Thresholds"),
    "about": st.Page("ui/pages/about.py", title="About", icon="ℹ️"),
}
# Streamlit's own top bar puts single pages before every menu and cannot colour
# or right-align an item, so it is hidden and the bar below is drawn instead.
# Every page still has its own address (e.g. localhost:8501/survey).
page = st.navigation(list(PAGES.values()), position="hidden")


def simple_bar() -> None:
    """Simple mode: one guided page, About, and the way to the rest of BEX."""
    st.markdown("<style>[data-testid='stMainBlockContainer']{padding-top:3.2rem}</style>",
                unsafe_allow_html=True)
    from ui.common import logo_html
    with st.container(key="bex_nav"):
        a, _, b, c = st.columns([3.2, 3, 1.4, 0.8], vertical_alignment="center")
        a.markdown(f"<a href='./' target='_self' style='text-decoration:none'>"
                   f"{logo_html(2.6, 'Simple mode')}</a>", unsafe_allow_html=True)
        if b.button("Research mode ›", help="All of BEX: labelling, the Explorer, comparing models…"):
            set_mode("research")
            st.switch_page(PAGES["home"])
        c.page_link(PAGES["about"], label="About")


def nav_bar() -> None:
    """The flow for one dataset, numbered left to right — 1 label, 2 look,
    3 compare, 4 decide — the steps in the accent colour, the current dataset
    named beside them; setting up and About to the right (V1.2 F2)."""
    if MODE == "simple":
        return simple_bar()
    from ui.common import brand
    c = brand()
    # The steps in EcoLabel's primary green, with its own text colour on it.
    pal = _theme.colours()
    accent, accent_deep, on_accent = pal["primary"], pal["primary_deep"], pal["on_primary"]
    st.markdown(
        "<style>"
        # The bar replaces Streamlit's header nav, so less space above it.
        "[data-testid='stMainBlockContainer']{padding-top:3.2rem}"
        # Every item reads as a link in the bar, not as a boxed button.
        ".st-key-bex_nav [data-testid='stPageLink'] a,"
        ".st-key-bex_nav [data-testid='stPopover'] button{"
        "border:none; background:transparent; box-shadow:none; font-weight:600;"
        "padding:0.3rem 0.6rem; border-radius:8px; min-height:0}"
        ".st-key-bex_nav [data-testid='stPopover'] button:hover,"
        ".st-key-bex_nav [data-testid='stPageLink'] a:hover{"
        f"background:{c['tint']}}}"
        # The three steps — the things you do — in the accent colour.
        + "".join(
            f".st-key-nav_step{i} [data-testid='stPageLink'] a,"
            f".st-key-nav_step{i} [data-testid='stPopover'] button{{"
            f"background:{accent}; border-radius:8px}}"
            f".st-key-nav_step{i} [data-testid='stPageLink'] a *,"
            f".st-key-nav_step{i} [data-testid='stPopover'] button *"
            f"{{color:{on_accent} !important}}"
            f".st-key-nav_step{i} [data-testid='stPageLink'] a:hover,"
            f".st-key-nav_step{i} [data-testid='stPopover'] button:hover{{"
            f"background:{accent_deep}}}"
            for i in (1, 2, 3, 4)) +
        ".st-key-nav_view button{font-weight:500; font-size:0.82rem; opacity:0.85;"
        "white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:100%}"
        ".st-key-bex_nav{border-bottom:1px solid rgba(128,128,128,0.25);"
        "padding-bottom:0.35rem; margin-bottom:0.6rem}"
        "</style>", unsafe_allow_html=True)
    with st.container(key="bex_nav"):
        cols = st.columns([0.75, 0.9, 1.05, 1.1, 1.5, 1.85, 0.8, 0.75],
                          vertical_alignment="center", gap="small")
        cols[0].page_link(PAGES["home"], label="Home")
        with cols[1].container(key="nav_step1"):
            st.page_link(PAGES["label"], label="1 · Label")
        with cols[2].container(key="nav_step2"):
            st.page_link(PAGES["explorer"], label="2 · Explorer")
        with cols[3].container(key="nav_step3"), st.popover("3 · Analysis"):
            st.page_link(PAGES["compare"], label="Compare models")
            st.page_link(PAGES["scorecard"], label="Species scorecard")
            st.page_link(PAGES["forensics"], label="Geofilter forensics",
                         help="A deep dive into what a location filter hides.")
        with cols[4].container(key="nav_step4"):
            st.page_link(PAGES["survey"], label="4 · Survey protocol")
        current = st.session_state.get("dataset")
        if not current:
            from ui.common import cfg, ingest
            from ui.common import opening_dataset, ordered_datasets
            names = ordered_datasets()
            # The same choice the sidebar will make, so the two agree on first load.
            current = opening_dataset(names)
        if current:
            # The view, said everywhere and switchable from anywhere (V1.2 F0).
            from ui import sidebar
            view = sidebar.current_view(current)
            other = (sidebar.VIEW_ALL if view == sidebar.VIEW_LABELLED
                     else sidebar.VIEW_LABELLED)
            short = "scored" if view == sidebar.VIEW_LABELLED else "unscored"
            with cols[5].container(key="nav_view"):
                st.button(f"{current} · {short}", key="nav_view_switch",
                          on_click=sidebar.set_view, args=(current, other),
                          help=f"Viewing {view.lower()}. Click to switch to "
                               f"{other.lower()}.")
        with cols[6].popover("Set up"):
            st.page_link(PAGES["recordings"], label="Recordings")
            st.page_link(PAGES["models"], label="Models")
            st.page_link(PAGES["thresholds"], label="Thresholds")
            if st.button("🌿 Simple mode", help="The guided path: recordings, model, species, check, "
                         "recall, export (a file, or EcoLabel volunteers)"):
                set_mode("simple")
                st.switch_page(PAGES["guided"])
        cols[7].page_link(PAGES["about"], label="About")


nav_bar()
page.run()
