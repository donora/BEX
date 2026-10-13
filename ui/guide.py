"""Guide me: a step-by-step walk through each page (V1.2 tutorial, plan item 2).

Every step names the part of the page it is about by a CSS selector — almost
always a keyed Streamlit element, `.st-key-<key>` — which is outlined, pulsing,
and scrolled into view, while a small panel fixed to the corner of the window
says what it is for, with Back / Next / Done.

The steps for every page live here, in one table, and follow the numbered
points of TOUR.md, so the in-app guide and the tour say the same things.

    from ui import guide
    guide.button("explorer")     # beside the page title
    guide.show("explorer")       # once, anywhere on the page (draws nothing
                                 # unless the guide is running)
"""
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

# (selector, title, text) — text is Markdown.
GUIDES: dict[str, list[tuple[str, str, str]]] = {
    "home": [
        (".st-key-bex_nav", "The flow",
         "Every dataset goes through the same steps, left to right: **1 · Label** "
         "makes ground truth, **2 · Explorer** is for looking and listening, "
         "**3 · Analysis** compares the models, **4 · Survey protocol** decides how "
         "to turn detections into species lists."),
        (".st-key-nav_view", "What you are looking at",
         "The current dataset and its **view**: *scored* (only the labelled data, "
         "judged against the labels) or *unscored* (every recording, what the "
         "models report). Click to switch — neither closes off the other."),
        (".st-key-card_0", "A dataset",
         "Its size, how much is labelled, and its progress through the steps: a "
         "tick is done, a half circle under way, and the outlined step is next. "
         "Hover a step for detail."),
        (".st-key-card_0 [class*='st-key-h_']", "What to do next",
         "*Continue* goes to the next step. *Label* and *Explore all recordings* "
         "are always there."),
        (".st-key-g_add", "Your own recordings",
         "Add a folder of recordings: a name, a site, its location, and the "
         "**plausibility profile** — the list of birds plausible there (BEX "
         "suggests one from the location). Then build its spectrograms and run "
         "the models, from its card."),
        (".st-key-g_profile", "Your own species list",
         "If no profile fits your site, upload one: a CSV of the birds plausible "
         "there."),
    ],
    "label": [
        (".st-key-g_lab_set", "Your label set",
         "Labels live in a named **set**, apart from the recordings. Your name is "
         "recorded on every box. New to this? The *Practise* box below lets you "
         "try on minutes the experts have already labelled."),
        (".st-key-label_view", "Five views",
         "**Plan and sample** says how many minutes to label; **Label** is where "
         "you do it; **Progress** shows what you have found; **Publish and share** "
         "freezes and exports; **How to** is the short manual."),
        (".st-key-g_lab_move", "Moving around",
         "*Next chunk in the sample* goes to the next random minute. ◀ ▶ step to "
         "the previous or next minute of this recording — to finish a call that "
         "runs over the boundary, say."),
        (".st-key-g_lab_nav", "The whole recording",
         "Grey bars: where the models hear *something* bird-like — a loose guide "
         "to busy and quiet stretches, naming no species. Below them every "
         "minute, green when closed. Click one to open it."),
        (".st-key-g_lab_spec", "Drawing a box",
         "Click one corner of a call, then the opposite corner. The red ✕ throws a "
         "box away. The shaded strips either side are the neighbouring minutes."),
        (".st-key-g_lab_player", "Listening",
         "The player is on the same time axis as the spectrogram. *▶ box* plays "
         "just the box you have drawn."),
        (".st-key-g_lab_name", "Naming the bird",
         "Pick the species. Stuck? **💡 Suggest birds** lists what the models "
         "heard inside your box, with recordings to compare — the box is then "
         "marked as labelled with suggestions. A bird you cannot name is *Unknown "
         "bird*."),
        (".st-key-g_lab_close", "Closing the minute",
         "When every bird you heard has a box, **close** the minute. Only closed "
         "minutes count as ground truth; a closed minute with no boxes means "
         "*listened, no birds*."),
    ],
    "explorer": [
        (".st-key-g_view", "Looking at",
         "**Labelled data, scored**: every detection judged against the labels. "
         "**All recordings, unscored**: what each model reported, with no right "
         "or wrong."),
        (".st-key-g_status", "What every number rests on",
         "The dataset, and how much of it is labelled."),
        (".st-key-rule_kind", "Thresholds",
         "A model's score becomes a detection above a **threshold**, set per "
         "species by this rule. Every page uses it."),
        (".st-key-explorer_rec", "One recording at a time",
         "Choose the recording, how much to see at once, and optionally one bird "
         "to follow."),
        (".st-key-nav", "Where the birdsong is",
         "The whole recording, one row per model. Above the line, what it reported "
         "— blue right, orange wrong, purple a real bird its filter would hide. "
         "Below, the labelled birds it missed. Click to jump there."),
        ("[class*='st-key-spec_click_']", "The spectrogram and score lanes",
         "The labelled boxes, then a lane per model: a bar for each detection, as "
         "high as it clears its threshold. Click anywhere to inspect that moment."),
    ],
    "compare": [
        (".st-key-g_cmp_cards", "What each model would give you",
         "How often it is right when it reports a bird, false detections per hour, "
         "and the share of songs it finds. Figures in brackets are **95% "
         "intervals**: how far the number could move on other recordings like "
         "these."),
        (".st-key-g_cmp_paired", "Is one model better here?",
         "Each pair compared on the *same* recordings. *Better* means the interval "
         "for the difference leaves out zero; otherwise there is no clear "
         "difference yet."),
        (".st-key-g_cmp_pr", "Precision and recall",
         "Each model swept across every threshold, and where your rule puts it."),
    ],
    "survey": [
        (".st-key-sv_objective", "Your goal",
         "What 'best' means: the thing to optimise, within your limits — firm calls "
         "are never checked, so their precision is always a limit."),
        (".st-key-sv_mode", "The rule",
         "Let BEX search for the best rule, or set it by hand. A species is "
         "**firm** (reported), **to check** (an expert listens first) or **not "
         "found**."),
        (".st-key-sv_run", "Test it",
         "Rules are chosen on *tuning* recordings and scored on held-back *test* "
         "ones, so a rule cannot flatter itself."),
        (".st-key-sv_save_model", "Save a setup",
         "A model read by your rule, with what it measured on the test recordings."),
        (".st-key-g_sv_apply", "Apply it everywhere",
         "Run the setup over every recording: labelled ones keep their labels, "
         "the rest get the model's lists, carrying the measured accuracy."),
    ],
}

_STYLE = """<style>
@keyframes bexpulse { 0% {box-shadow:0 0 0 0 rgba(62,106,107,.55)}
                      70% {box-shadow:0 0 0 12px rgba(62,106,107,0)}
                      100% {box-shadow:0 0 0 0 rgba(62,106,107,0)} }
@@TARGET@@ { outline:3px solid #3e6a6b !important; outline-offset:5px;
             border-radius:6px; animation:bexpulse 1.6s ease-out infinite; }
.st-key-bex_guide { position:fixed; right:22px; bottom:22px; width:360px;
                    z-index:999999; background:@@BG@@;
                    border:2px solid #3e6a6b; border-radius:12px;
                    box-shadow:0 8px 28px rgba(0,0,0,.18);
                    padding:0.9rem 1rem 0.6rem; }
</style>"""


def _key(page: str) -> str:
    return f"_guide_{page}"


def start(page: str) -> None:
    st.session_state[_key(page)] = 0


def _move(page: str, by: int) -> None:
    st.session_state[_key(page)] = st.session_state.get(_key(page), 0) + by


def _stop(page: str) -> None:
    st.session_state.pop(_key(page), None)


def button(page: str) -> None:
    """The 🧭 button that starts a page's guide."""
    if page in GUIDES:
        st.button("🧭 Guide me", key=f"guide_btn_{page}", on_click=start, args=(page,),
                  help="A step-by-step walk through this page.")


def show(page: str) -> None:
    """Draw the running guide's panel and highlight — nothing when not running."""
    steps = GUIDES.get(page, [])
    i = st.session_state.get(_key(page))
    if i is None or not steps:
        return
    i = max(0, min(i, len(steps) - 1))
    selector, title, text = steps[i]
    dark = (getattr(st.context.theme, "type", None) or "light") == "dark"
    st.markdown(_STYLE.replace("@@TARGET@@", selector)
                      .replace("@@BG@@", "#0e1117" if dark else "#ffffff"),
                unsafe_allow_html=True)
    with st.container(key="bex_guide"):
        st.caption(f"🧭 Guide · {i + 1} of {len(steps)}")
        st.markdown(f"**{title}**\n\n{text}")
        a, b, c = st.columns(3)
        a.button("◀ Back", key="guide_back", on_click=_move, args=(page, -1),
                 disabled=i == 0, width="stretch")
        if i < len(steps) - 1:
            b.button("Next ▶", key="guide_next", on_click=_move, args=(page, 1),
                     type="primary", width="stretch")
        else:
            b.button("Done ✓", key="guide_done", on_click=_stop, args=(page,),
                     type="primary", width="stretch")
        c.button("Close", key="guide_close", on_click=_stop, args=(page,),
                 width="stretch")
    # Bring the highlighted part into view: the snippet runs in an iframe on the
    # same origin, so it can reach the page. Missing targets are skipped quietly
    # (a step about something the page is not showing right now).
    components.html(
        "<script>const el = window.parent.document.querySelector("
        f"{json.dumps(selector)});"
        "if (el) el.scrollIntoView({behavior:'smooth', block:'center'});</script>",
        height=0)
