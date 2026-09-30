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

page = st.navigation(
    {
        # Unsectioned pages come first in the top bar. The Explorer is the main
        # event, so it is one click from anywhere rather than inside a menu.
        "": [
            st.Page("ui/pages/home.py", title="Home", icon="🏠", default=True),
            st.Page("ui/pages/explorer.py", title="Explorer", icon="🔎"),
            st.Page("ui/pages/about.py", title="About", icon="ℹ️"),
        ],
        "Analysis": [
            st.Page("ui/pages/compare.py", title="Compare models"),
            st.Page("ui/pages/scorecard.py", title="Species scorecard"),
            st.Page("ui/pages/forensics.py", title="Geofilter forensics"),
        ],
        "Set up": [
            st.Page("ui/pages/recordings.py", title="Recordings"),
            st.Page("ui/pages/models.py", title="Models"),
            st.Page("ui/pages/thresholds.py", title="Thresholds"),
        ],
    },
    position="top",
)
page.run()
