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
        "recall, species by species. \nUse the **Explorer** page to view and listen to your recordings, "
        "and see the ground truth and model predictions."
    )
    st.image(str(REPO_DIR / "assets" / "mission_flow.svg"), width="stretch")

    st.markdown(
        """
#### How the app follows that flow

1. **Bring your data** — ingest a dataset (*Recordings*), and annotate a sample
   of it so there is ground truth to measure against.
2. **Run models** — off-the-shelf models (BirdNET and Perch today) run on the
   same audio, each on its own window grid, with nothing filtered away
   (*Models*). Your own models join the same way: a runner that writes BEX's
   score format (see `envs/`). Region-trained models are next on the roadmap.
3. **Score each** — precision and recall for every species against the
   annotations, at a threshold chosen per species by a rule you set and can
   override (*Thresholds*); window by window in the *Explorer*.
4. **Compare** — the same audio, the same truth and the same metrics for every
   model, so a difference between them is a difference in the models (*Compare
   models*, *Species scorecard*).
5. **Decide** — choose the rule that turns each model's detections into a
   species list for every recording: species it reports firmly, and species an
   expert should check first. Tune it to your goal and limits, then test it on
   recordings it was not tuned on (*Survey protocol*).
6. **Apply** — save the thresholds you settle on as a named set, and use them to
   process recordings that have no annotations, with the error rates you
   measured.
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


render()
