"""BEX · About."""
from ui.common import *  # noqa: F403 — the shared app toolkit


def render() -> None:
    st.header("Why this exists")
    st.markdown(
        "Bird-ID models such as BirdNET and Perch make it possible to survey birds "
        "from thousands of hours of recordings, but they report scores, not species "
        "lists. Between the two sit choices — which model, which threshold, whether "
        "to trust its location filter, how many detections make a species present — "
        "that change the answer and are usually left at their defaults.\n\n"
        "**BEX is for ecologists and bioacoustics researchers running acoustic "
        "surveys.** It tests those choices against annotated recordings, so you can "
        "choose how to process the rest of your data, and know how often it will be "
        "wrong, before you rely on it.")

    st.header("A processing pipeline for your bioacoustic data")
    st.markdown(
        "Every bioacoustic project makes processing decisions: which model, which "
        "confidence threshold, whether to trust the model's location filter. "
        "**This platform helps you make those choices.** Upload your own recordings, "
        "run them through several models side by side, score every model against "
        "the ground truth, and create a processing pipeline with a quantified precision and "
        "recall, species by species — with error bars that say how much labelled "
        "data each number rests on."
    )
    st.image(str(REPO_DIR / "assets" / "mission_flow.svg"), width="stretch")

    st.markdown(
        """
#### How the app follows that flow

Every dataset goes through the same steps, from the **Home** page:

1. **Bring your data and run the models** — add a folder of recordings, build
   its spectrograms, and run off-the-shelf models (BirdNET and Perch today) over
   it, each on its own window grid, with nothing filtered away (*Models*). Your
   own models join the same way: a runner that writes BEX's score format (see
   `envs/`).
2. **Label a sample** (*1 · Label*) — BEX picks a random, stratified sample of
   60-second chunks and you label them, a minute at a time, by ear and from the
   spectrogram. The models only show a coarse, species-free guide to where the
   recording is busy, so the labels do not flatter them. Each chunk is closed
   when every bird in it is labelled; only closed chunks count as ground truth.
   Or skip labelling and **just explore** what the models report, unscored —
   you can start labelling at any time.
3. **Explore** (*2 · Explorer*) — window by window, every model's detections
   against your labels, with the audio.
4. **Analyse** (*3 · Analysis*) — precision and recall for every species,
   at a threshold chosen per species by a rule you set (*Thresholds*), with 95%
   intervals from resampling recordings, and a paired test of whether one model
   is really better than another on your recordings.
5. **Decide** (*4 · Survey protocol*) — choose the rule that turns detections
   into a species list for every recording: species reported firmly, and species
   an expert should check first. Tune it, test it on recordings it was not tuned
   on, and **save it as a setup**.
6. **Apply** — run the setup over every recording. Labelled recordings keep
   their labels; the rest get the model's lists, carrying the precision and
   recall you measured, so a reader knows what they can trust them to mean.
        """
    )

    st.markdown("#### Seeing what the location filter hides")
    st.markdown(
        "Off-the-shelf tools apply a **geographic filter** after the classifier, "
        "silently removing species judged implausible for the location. BEX "
        "keeps the unfiltered scores and records what the filter *would* have "
        "hidden — because a confident score for an absent species says the model "
        "could not tell it from something that is actually there, and a filter "
        "that hides a bird you recorded is an error you would never see. The "
        "Explorer marks every real bird the filter hid, and *Geofilter forensics* "
        "measures how often it happens."
    )
    with st.expander("Definitions used throughout (full versions in PLAN.md §4)"):
        st.markdown(
            """
- **Suppression event** — a detection above its threshold for a species the
  location filter excludes (occurrence below the cut-off recorded in the run
  manifest).
- **Shadowed detection** — a plausible species detected with an implausible
  congener scoring nearly as high in the same window: the map, not the model,
  made the identification.
- **Muddiness** — per model or per species, how much confident probability mass
  lands on species that are almost certainly absent.
            """
        )

    st.markdown("#### Numbers you can check")
    st.markdown(
        "Every table traces to a **run manifest** recording the configuration that "
        "actually executed — model version, window, readout, filter settings "
        "(bottom of the *Models* page). Wherever thresholds were fitted on the same "
        "recordings they are scored on, the app says so, because that flatters "
        "every model. Everything shown is computed by the `bex` library and "
        "reproducible from the command line."
    )


def welcome_again() -> None:
    from bex import prefs
    st.divider()
    if st.button("Show the welcome again", icon="👋"):
        prefs.put(cfg.store_dir, "welcome_seen", False)
        st.session_state.pop("_welcome_shown", None)
        st.switch_page("ui/pages/home.py")


render()
welcome_again()
