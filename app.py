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

PAGES = {
    "home": st.Page("ui/pages/home.py", title="Home", icon="🏠", default=True),
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


def nav_bar() -> None:
    """The workflow, numbered left to right — 1 look, 2 compare, 3 decide — the
    three steps in the accent colour; setting up and About to the right."""
    from ui.common import brand
    c = brand()
    # The accent pill is the light theme's petrol teal with white text in both
    # themes: the dark theme's lighter teal would need dark text, which reads as
    # disabled on a dark page.
    accent, accent_deep = "#3e6a6b", "#2f5657"
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
            "{color:#ffffff !important}"
            f".st-key-nav_step{i} [data-testid='stPageLink'] a:hover,"
            f".st-key-nav_step{i} [data-testid='stPopover'] button:hover{{"
            f"background:{accent_deep}}}"
            for i in (1, 2, 3)) +
        ".st-key-bex_nav{border-bottom:1px solid rgba(128,128,128,0.25);"
        "padding-bottom:0.35rem; margin-bottom:0.6rem}"
        "</style>", unsafe_allow_html=True)
    with st.container(key="bex_nav"):
        cols = st.columns([0.85, 1.15, 1.25, 1.6, 1.7, 1.0, 0.85],
                          vertical_alignment="center", gap="small")
        cols[0].page_link(PAGES["home"], label="Home")
        with cols[1].container(key="nav_step1"):
            st.page_link(PAGES["explorer"], label="1 · Explorer")
        with cols[2].container(key="nav_step2"), st.popover("2 · Analysis"):
            st.page_link(PAGES["compare"], label="Compare models")
            st.page_link(PAGES["scorecard"], label="Species scorecard")
            st.page_link(PAGES["forensics"], label="Geofilter forensics",
                         help="A deep dive into what a location filter hides.")
        with cols[3].container(key="nav_step3"):
            st.page_link(PAGES["survey"], label="3 · Survey protocol")
        with cols[5].popover("Set up"):
            st.page_link(PAGES["recordings"], label="Recordings")
            st.page_link(PAGES["models"], label="Models")
            st.page_link(PAGES["thresholds"], label="Thresholds")
        cols[6].page_link(PAGES["about"], label="About")


nav_bar()
page.run()
