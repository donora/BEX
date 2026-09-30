"""Smoke-run the whole app headlessly with Streamlit's AppTest.

Needs the local store (datasets + a run + mel cache), so it skips on a bare
clone — on a working checkout it catches anything that would greet the user
with a red traceback. The app is multipage (V1 N2): each test opens the page it
is about.
"""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PAGES = ["home", "explorer", "about", "compare", "scorecard", "forensics",
         "recordings", "models", "thresholds"]


def open_page(name: str, timeout: int = 300):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    if not (REPO / "store" / "datasets").exists():
        pytest.skip("no local store — run ingest + a model run first")
    at = st_testing.AppTest.from_file(str(REPO / "app.py"), default_timeout=timeout)
    at.run()
    if name != "home":
        at.switch_page(f"ui/pages/{name}.py").run()
    return at


@pytest.fixture(scope="module")
def explorer():
    return open_page("explorer")


@pytest.fixture(scope="module")
def compare():
    return open_page("compare")


@pytest.fixture(scope="module")
def scorecard():
    return open_page("scorecard")


# --------------------------------------------------------------------------- #
# Navigation (V1 N2)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    assert not open_page(page).exception


def test_home_is_the_landing_page_and_leads_to_the_explorer():
    at = open_page("home")
    assert any("ioacoustics" in str(m.value) and "plorer" in str(m.value)
               for m in at.markdown)
    go = next(b for b in at.button if b.label == "Open the Explorer")
    go.click().run()
    assert not at.exception
    assert any(sb.label == "Recording" for sb in at.selectbox), "did not reach the Explorer"


@pytest.mark.parametrize("page", ["home", "about"])
def test_pages_without_settings_have_no_sidebar(page):
    at = open_page(page)
    assert not at.sidebar.selectbox and not at.sidebar.slider


def test_settings_survive_a_visit_to_a_page_without_the_sidebar():
    """Widget state is dropped when a widget is not rendered — unless it is
    persisted. Visiting Home must not reset the experiment."""
    at = open_page("explorer")
    at.sidebar.selectbox(key="rule_kind").set_value("max F1").run()
    at.sidebar.slider(key="min_labelled").set_value(25).run()
    at.switch_page("ui/pages/home.py").run()
    at.switch_page("ui/pages/explorer.py").run()
    assert at.sidebar.selectbox(key="rule_kind").value == "max F1"
    assert at.sidebar.slider(key="min_labelled").value == 25


def test_setup_pages_explain_how_to_add_data():
    recs = " ".join(str(m.value) for m in open_page("recordings").markdown)
    assert "bex scan" in recs and "bex melcache" in recs
    models = " ".join(str(m.value) for m in open_page("models").markdown)
    assert "run.py --dataset" in models


# --------------------------------------------------------------------------- #
# Sidebar and Explorer
# --------------------------------------------------------------------------- #

def test_sidebar_has_core_controls(explorer):
    """The sidebar holds the experiment definition (V1 S1–S4): no single-run
    picker and no global θ slider — a θ shared by every model is not one."""
    labels = [sb.label for sb in explorer.sidebar.selectbox]
    assert "Dataset" in labels and "Threshold rule" in labels
    assert "Threshold set" in labels
    assert "Model run" not in labels
    assert not any("Detection threshold" in s.label for s in explorer.sidebar.slider)
    assert any(r.label == "Implausibility judge" for r in explorer.sidebar.radio)
    assert any(s.label.startswith("Δ") for s in explorer.sidebar.slider)


def test_every_available_model_is_included_by_default(explorer):
    """The app compares models; including them all is the starting point."""
    picker = next((m for m in explorer.sidebar.multiselect if m.label == "Models"), None)
    assert picker is not None, "sidebar has no model filter"
    assert len(picker.value) >= 1
    assert len(picker.value) == len(picker.options)   # options are formatted labels


def test_judge_defaults_to_profile(explorer):
    """Only the profile judge applies to every model — Perch ships no geofilter."""
    judge = next(r for r in explorer.sidebar.radio if r.label == "Implausibility judge")
    assert judge.value == "profile"


def test_sliders_are_gone_from_the_explorer(explorer):
    """V1 E3: clicking and the ◀ ▶ buttons replace both sliders."""
    assert not any("Inspect" in s.label for s in explorer.select_slider)
    assert not any(s.label.startswith("Start") for s in explorer.slider)
    assert any("window ▶" in b.label for b in explorer.button)


def test_spectrogram_click_moves_the_inspector():
    """A click on the spectrogram sets `inspect_t`; the inspector must follow it
    and every model must answer for its own window around that instant."""
    at = open_page("explorer")
    at.session_state["inspect_t"] = 12.3      # as a click would leave it
    at.run()
    assert not at.exception
    text = " ".join(str(m.value) for m in at.markdown)
    assert "Window inspector — 12.3 s" in text


@pytest.mark.parametrize("rule", ["max F2", "fixed"])
def test_every_rule_renders(rule):
    at = open_page("explorer")
    at.sidebar.selectbox(key="rule_kind").set_value(rule).run()
    assert not at.exception


def test_focusing_a_species_narrows_the_navigator():
    """The navigator is what you use to find where to look, so it has to follow
    the focus."""
    at = open_page("explorer")
    focus = next(sb for sb in at.selectbox if sb.label.startswith("Focus species"))
    assert focus.options[0].startswith("—"), "the 'all species' option must lead"
    if len(focus.options) < 2:
        pytest.skip("no species in the default recording")

    # AppTest's `options` are the *formatted* labels, and `select_index` feeds one
    # back through format_func — which only round-trips for an idempotent
    # formatter. So set the raw key, taken from the annotations of whichever
    # recording the app is showing.
    import polars as pl

    from bex import ingest
    rec_id = next(sb for sb in at.selectbox if sb.label == "Recording").value
    _, ann = ingest.read_dataset(REPO / "store", "sne")
    here = ann.filter(pl.col("recording_id") == rec_id)["species_key"].unique()
    if not len(here):
        pytest.skip("no annotated species in the default recording")
    target = here.to_list()[0]

    # set_value raises if the key is not among the picker's options, so an
    # annotated species being selectable is exactly what passing here proves.
    focus.set_value(target).run()
    assert not at.exception
    text = " ".join(str(m.value) for m in at.markdown)
    assert "only" in text and "Where the birdsong is" in text

    # And the picker must still offer every species, not just the chosen one.
    focus_after = next(sb for sb in at.selectbox if sb.label.startswith("Focus species"))
    assert len(focus_after.options) == len(focus.options), \
        "filtering the navigator collapsed its own picker's options"


# --------------------------------------------------------------------------- #
# Analysis pages
# --------------------------------------------------------------------------- #

def test_compare_is_a_dashboard_at_the_sidebars_thresholds(compare):
    """V1 C1: curves, one card per model, species panels, coverage — every
    section written only once its numbers exist."""
    text = " ".join(str(m.value) for m in compare.markdown)
    for section in ("Precision and recall", "What each model would give you",
                    "Where the models differ", "Which species each model finds"):
        assert section in text, f"missing section: {section}"
    assert "false detections per hour" in text          # one per model card
    assert "precision ≥ 0.95" in " ".join(str(c.value) for c in compare.caption)


def test_compare_keeps_every_number_and_an_export(compare):
    """The tables moved into "All the numbers", not away."""
    assert "Window grid" in [r.label for r in compare.radio]
    assert any("Download per-species table" in b.label
               for b in compare.download_button)


def test_compare_switches_to_average_precision():
    at = open_page("compare")
    at.radio(key="cmp_metric").set_value("average precision (all thresholds)").run()
    assert not at.exception


def test_matched_detection_league_lives_on_forensics():
    at = open_page("forensics")
    assert any("matched detection counts" in str(m.value) for m in at.markdown)


def test_forensics_shows_every_model_under_each_judge():
    """One card per included model; under the geofilter judge a model with no
    filter of its own says so rather than showing zeros."""
    at = open_page("forensics")
    text = " ".join(str(m.value) for m in at.markdown)
    assert "of its detections" in text and "suppresses most" in text
    assert "Its rivals, up close" not in text          # muddiness is the scorecard's
    at.radio(key="judge").set_value("geofilter").run()
    assert not at.exception
    text = " ".join(str(m.value) for m in at.markdown)
    if "Perch" in text or "perch" in text:
        assert "Ships no geofilter" in text


def test_switching_to_the_shared_grid_does_not_break_the_sweep():
    """The shared 1 s grid is a different alignment, a different cache file and a
    different denominator — and it is one click away from the default."""
    if not (REPO / "store" / "truth").exists():
        pytest.skip("no cached alignments — run scripts/evaluate.py first")
    at = open_page("compare")
    next(r for r in at.radio if r.label == "Window grid").set_value("shared-1s").run()
    assert not at.exception


def test_species_scorecard_renders(scorecard):
    """V1 S2–S6: a card per model, recordings, confusion — and the sections that
    moved elsewhere are gone."""
    assert "Species" in [sb.label for sb in scorecard.selectbox]
    text = " ".join(str(m.value) for m in scorecard.markdown)
    assert "annotated songs** across" in text
    for section in ("How each model handles it", "Which recordings each model",
                    "What this bird gets confused with"):
        assert section in text, f"missing section: {section}"
    assert "How much of this is the threshold?" not in text
    assert "Window by window" not in text


def test_scorecard_names_its_threshold(scorecard):
    """Every number is at the threshold rule, and the page says so up front."""
    text = " ".join(str(m.value) for m in scorecard.markdown)
    assert "Everything on this page is at one threshold rule" in text


def test_scorecard_opens_the_explorer_on_the_bird():
    at = open_page("scorecard")
    bird = next(sb for sb in at.selectbox if sb.label == "Species").value
    at.button(key="sc_listen_go").click().run()
    assert not at.exception
    assert at.session_state["explorer_focus"] == bird
    assert any(sb.label == "Focus species" for sb in at.selectbox)


def test_scorecard_walks_species_under_a_strict_rule():
    """Not every species can reach 95% precision on every arm. Those fall back,
    visibly, rather than breaking the tab."""
    if not (REPO / "store" / "truth").exists():
        pytest.skip("no cached alignments")
    at = open_page("scorecard")
    at.sidebar.selectbox(key="rule_kind").set_value("precision floor").run()
    assert not at.exception
    species = lambda: next(sb for sb in at.selectbox if sb.label == "Species")
    for option in species().options[:6]:
        # Fetched afresh each time: an element from an earlier run replays that
        # run's page, recording picker and all.
        species().set_value(option).run()
        assert not at.exception, f"species {option} broke the tab"


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #

def test_thresholds_page_takes_an_override():
    """H1: the overview renders, and an override is applied and counted."""
    if not (REPO / "store" / "truth").exists():
        pytest.skip("no cached alignments")
    at = open_page("thresholds")
    assert any("Every threshold" in str(m.value) for m in at.markdown)
    species = next(sb for sb in at.selectbox if sb.label == "Look up a species")

    from bex import thresholds, views
    ident = thresholds.identity(views.list_runs(REPO / "store")[0])
    at.session_state["overrides"] = {(ident, species.value): 0.5}
    at.run()
    assert not at.exception
    caps = " ".join(str(c.value) for c in at.sidebar.caption)
    assert "1 override(s)" in caps and "unsaved changes" in caps
