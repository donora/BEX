"""Plain-English meanings for the words BEX uses (V1.2 tutorial, plan item 4).

One table: `term()` puts a word on a page with its meaning on hover (dotted
underline), and the About page lists the lot, so a word means the same thing
everywhere it appears.
"""
from __future__ import annotations

import html

TERMS: dict[str, tuple[str, str]] = {
    # key: (word as shown, meaning)
    "precision": ("precision",
                  "When the model reports a bird, how often it is right. 95% precision: "
                  "1 report in 20 is a bird that was not there."),
    "recall": ("recall",
               "Of the birds that were really there, the share the model reported. "
               "High recall: few birds missed."),
    "threshold": ("threshold (θ)",
                  "The score a detection must reach to count. Each model has its own "
                  "per species: a higher threshold means fewer, surer detections."),
    "false_per_hour": ("false detections per hour",
                       "Wrong detections you would actually see, per hour of audio — one "
                       "bird wrongly reported across three windows counts once."),
    "songs_found": ("songs found",
                    "Of the labelled songs and calls, the share the model detected at "
                    "least once. Counted per song, so 3 s and 5 s models compare fairly."),
    "interval": ("95% interval",
                 "How far a number could move on another set of recordings like these: "
                 "resample the recordings many times and 95% of the results fall inside."),
    "plausibility": ("plausibility profile",
                     "The list of birds plausible where you recorded. A confident "
                     "detection of a bird not on it is a mistake the model made."),
    "geofilter": ("geofilter",
                  "A model's own location filter (BirdNET has one): it hides species it "
                  "does not expect at your site and time of year."),
    "closed": ("closed minute",
               "A minute you have signed off: every bird you heard in it has a box. Only "
               "closed minutes count as ground truth; one closed with no boxes means "
               "'listened, no birds'."),
    "ground_truth": ("ground truth",
                     "What was really there, as labelled by a person — what every model "
                     "is scored against."),
    "scored": ("labelled data, scored",
               "Look at only what has been labelled, with every model output judged "
               "right or wrong against the labels."),
    "unscored": ("all recordings, unscored",
                 "Look at every recording: what the models report, with no right or wrong."),
    "window": ("window",
               "The stretch of audio a model scores at once: 3 s for BirdNET, 5 s for "
               "Perch. Each window gets a score for every species."),
    "firm_check": ("firm / to check",
                   "On a survey's species list: firm calls are reported as they are; "
                   "'to check' ones are possible birds an expert listens to first."),
    "setup": ("setup",
              "A model read one way — its rule and thresholds — saved with how well it "
              "did on held-back recordings, ready to run over everything else."),
}


def term(key: str, text: str | None = None) -> str:
    """HTML: the word, dotted-underlined, with its meaning on hover. For
    st.markdown(..., unsafe_allow_html=True)."""
    word, meaning = TERMS[key]
    return (f"<span title='{html.escape(meaning, quote=True)}' style='text-decoration:"
            f"underline dotted; text-underline-offset:3px; cursor:help'>"
            f"{html.escape(text or word)}</span>")


def glossary_markdown() -> str:
    return "\n".join(f"- **{w}** — {m}" for w, m in
                     sorted(TERMS.values(), key=lambda t: t[0].lower()))
