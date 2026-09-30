"""Pure view-building functions for the app (PLAN.md §8 rule 1: thin app,
thick library). Everything the app renders is computed here, headlessly
testable; app.py only draws.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from .profiles import Profile
from .schemas import RunManifest
from .taxonomy import normalise_key, xc_species_url

# The dataviz categorical order (validated, fixed assignment — never cycled).
# Top-8 species in view get these; everything else folds into "Other" gray.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_COLOR = "#9a9992"

# Sequential ramp (one hue, light -> dark) for the activity navigator.
ACTIVITY_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]

TIER_LABELS = {0: "listed", 1: "regular", 2: "vagrant", 3: "implausible"}


def list_runs(store_dir: str | Path) -> list[RunManifest]:
    runs_dir = Path(store_dir) / "runs"
    if not runs_dir.exists():
        return []
    out = []
    for p in sorted(runs_dir.glob("*/manifest.json")):
        out.append(RunManifest.from_json(p.read_text()))
    return sorted(out, key=lambda m: m.created_utc, reverse=True)


def load_common_names(store_dir: str | Path, run_id: str) -> dict[str, str]:
    """species_key -> display common name, from the run's original label file
    ('Genus species_Common Name' lines). Missing file -> empty dict; the app
    falls back to scientific names."""
    p = Path(store_dir) / "runs" / run_id / "labels.txt"
    if not p.exists():
        return {}
    names = {}
    for line in p.read_text().splitlines():
        if "_" in line:
            sci, common = line.split("_", 1)
            names[normalise_key(sci)] = common
    return names


def readout_token(score_transform: str) -> str:
    """The one-word name of how logits became scores: 'sigmoid', 'softmax', ''.

    Reads the leading word of a manifest's `score_transform`, which the runners
    write as a documented prefix. Returns '' rather than guessing — a caller that
    needs to distinguish two runs should fall back to the run id, not to a wrong
    name.
    """
    t = score_transform.lower().lstrip()
    for name in ("softmax", "sigmoid", "identity"):
        if t.startswith(name) or t.startswith(f"per-class {name}"):
            return name
    return ""


def arm_labels(manifests: list) -> dict[str, str]:
    """run_id -> a display label that is unique within this set of runs.

    `model-version` alone stopped being unique the moment the same model was run
    twice with different readouts: two Perch runs both call themselves 'perch-v2',
    and a comparison keyed on that label silently keeps one and drops the other —
    no error, just an arm missing from the table. So collisions are resolved by
    what actually differs (the readout), and by the run id if that is not enough.
    """
    base = {m.run_id: f"{m.model_name}-{m.model_version}" for m in manifests}
    clashing = {lab for lab in base.values()
                if sum(1 for v in base.values() if v == lab) > 1}

    out: dict[str, str] = {}
    for m in manifests:
        label = base[m.run_id]
        if label in clashing:
            token = readout_token(m.score_transform)
            label = f"{label} ({token})" if token else f"{label} [{m.run_id[-8:]}]"
        out[m.run_id] = label

    # Belt and braces: two runs of the same model *and* the same readout would
    # still collide, so anything left over falls back to the run id.
    seen: dict[str, int] = {}
    for rid, label in list(out.items()):
        seen[label] = seen.get(label, 0) + 1
    for rid, label in list(out.items()):
        if seen[label] > 1:
            out[rid] = f"{label} [{rid[-8:]}]"
    return out


def merge_common_names(store_dir: str | Path, run_ids: list[str]) -> dict[str, str]:
    """One species_key -> common name map across several runs.

    A cross-arm table names each species once, but the arms disagree about how
    much they know: BirdNET's label file carries 6,522 real common names, while
    Perch's is bare scientific names and yields only ~75 entries — all of them
    artefacts of splitting its AudioSet classes ('Accelerating_and_revving') on
    the underscore. So the richest map wins and the rest only fill gaps, which
    keeps a good common name from being overwritten by a worse one.
    """
    maps = sorted((load_common_names(store_dir, r) for r in run_ids),
                  key=len, reverse=True)
    merged: dict[str, str] = {}
    for m in maps:
        for k, v in m.items():
            merged.setdefault(k, v)
    return merged


def _hit(theta: float | None) -> pl.Expr:
    """"At or above threshold": one θ for every row, or — when theta is None —
    each row's own, already resolved into the `above` column by
    `thresholds.attach`. Per-species thresholds are the app's normal case; the
    scalar form remains for scripts and tests."""
    return pl.col("above") if theta is None else pl.col("score_raw") >= theta


def common_name_index(store_dir: str | Path,
                      labels_dir: str | Path | None = None) -> dict[str, str]:
    """species_key -> English common name, from everything the store knows.

    Every model's names used to come from its own label file, so a Perch row read
    "Turdus migratorius" beside BirdNET's "American Robin" — Perch's labels are
    scientific names only. The fix is to name a species the same way whichever
    model reported it: pool every run's label file (the richest first, see
    `merge_common_names`) and the dataset's own species list.

    Only `Genus species_Common Name` entries count. Perch's AudioSet classes
    ("Keys_jangling") split on the underscore into nonsense pairs, which this
    drops; `display_name` tidies those classes instead.
    """
    names = {k: v for k, v in
             merge_common_names(store_dir, [m.run_id for m in list_runs(store_dir)]).items()
             if " " in k}
    if labels_dir is not None and (Path(labels_dir) / "species.csv").exists():
        from .taxonomy import canonical_from_sne
        try:
            for k, v in canonical_from_sne(labels_dir).select(
                    "species_key", "common_name").iter_rows():
                names.setdefault(k, v)
        except (KeyError, pl.exceptions.PolarsError):
            pass  # a species list in some other shape: the label files still apply
    return names


def display_name(key: str, names: dict[str, str]) -> str:
    """The name to show: common name if known, else the key made readable."""
    return names.get(key) or key.replace("_", " ")


def lane_blocks(
    det: pl.DataFrame,
    recording_id: str,
    t0: float,
    t1: float,
    theta: float | None,
    per_window: int = 3,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The explorer's model lane: per window in [t0, t1), the top `per_window`
    species at or above theta — split into (visible, suppressed) frames.

    Showing only the single best species per window hid real co-detections:
    at theta 0.5 more than half the non-empty windows in a dawn-chorus hour
    carry two or more species at once, and a lane that draws one of them makes
    the model look more decisive than it is. Rows come back ordered by window
    then descending score, which is the order `pack_rows` stacks them in.
    Columns: start_s, end_s, species_key, score_raw.
    """
    span = det.filter(
        (pl.col("recording_id") == recording_id)
        & (pl.col("end_s") > t0)
        & (pl.col("start_s") < t1)
        & _hit(theta)
    )

    def top_n(frame: pl.DataFrame) -> pl.DataFrame:
        cols = ["start_s", "end_s", "species_key", "score_raw"]
        if frame.is_empty():
            return frame.select(cols)
        return (
            frame.sort(["start_s", "score_raw"], descending=[False, True])
            .group_by("start_s", maintain_order=True)
            .head(per_window)
            .select(cols)
        )

    return top_n(span.filter(~pl.col("suppressed"))), top_n(span.filter(pl.col("suppressed")))


def window_grid(t0: float, t1: float, hop_s: float) -> list[float]:
    """The model's window starts *overlapping* [t0, t1) — what the inspector can
    address, since a score exists per window and not per second.

    Overlap, not containment, on both edges: a window half in view is still one
    you can see and therefore one you must be able to select. (Rounding up at
    the left while letting the last window run past the right would drop a
    visible window at the start of every off-grid viewport.)
    """
    first = math.floor(round(t0 / hop_s, 6)) * hop_s
    out, t = [], first
    while t < t1 - 1e-9:
        out.append(round(t, 6))
        t += hop_s
    return out


def click_to_time(
    x_frac: float,
    axes_x0: float,
    axes_x1: float,
    t0: float,
    t1: float,
) -> float | None:
    """A click on the rendered figure -> the instant (s) it landed on, or None.

    `x_frac` is the click's position as a fraction of the whole image width, and
    `axes_x0`/`axes_x1` are where the plotting area sits in that same fraction
    (matplotlib's axes rectangle) — the figure's left margin carries the lane
    labels, so image fraction and time fraction are not the same thing. Returns
    None for clicks in the margins rather than clamping them to an edge, because
    a click on a label is not a request to inspect second zero.

    An instant, not a window: each model is then asked for *its own* window
    containing it (`window_at`), since a 3 s and a 5 s grid do not line up.
    """
    if axes_x1 <= axes_x0 or t1 <= t0:
        return None
    axes_frac = (x_frac - axes_x0) / (axes_x1 - axes_x0)
    if not 0.0 <= axes_frac <= 1.0:
        return None
    return t0 + axes_frac * (t1 - t0)


def new_click(selection: dict, names: list[str], seen: dict
              ) -> tuple[float | None, dict]:
    """Which navigator click to act on, and the selections now seen.

    The navigator is several charts, each with its own click selection, and a
    click on one chart does not clear another's. So "the first non-empty
    selection" is wrong: after a click on the truth chart, its stale selection
    shadowed every later click on a model's chart. The click to act on is the
    selection that *changed* since the last run.
    """
    def as_time(sel) -> float | None:
        if not sel:
            return None
        if isinstance(sel, dict):
            values = sel.get("bin_start_s") or []
            return float(values[0]) if values else None
        first = sel[0]
        if isinstance(first, dict):
            value = first.get("bin_start_s")
            return float(value) if value is not None else None
        return float(first)

    seen = dict(seen)
    target = None
    for name in names:
        t = as_time(selection.get(name))
        if t is not None and t != seen.get(name):
            target = t
        seen[name] = t
    return target, seen


def window_at(det: pl.DataFrame, recording_id: str, t: float) -> tuple[float, float] | None:
    """This model's window containing instant t, as (start_s, end_s), or None.

    Read from the detections index rather than recomputed from the hop, because
    the index holds the windows the model actually produced — every window keeps
    its top-k rows, so every window is present.
    """
    w = det.filter((pl.col("recording_id") == recording_id)
                   & (pl.col("start_s") <= t) & (pl.col("end_s") > t))
    if w.is_empty():
        return None
    start = float(w["start_s"].max())
    return start, float(w.filter(pl.col("start_s") == start)["end_s"][0])


def first_detection(
    det: pl.DataFrame, recording_id: str, t0: float, t1: float, theta: float | None
) -> float | None:
    """Start of the earliest window in the span holding any detection at or
    above theta. The inspector opens here rather than at the span's first
    second, which at theta 0.9 is an empty window more often than not."""
    span = det.filter(
        (pl.col("recording_id") == recording_id)
        & (pl.col("end_s") > t0)
        & (pl.col("start_s") < t1)
        & _hit(theta)
    )
    return None if span.is_empty() else float(span["start_s"].min())


def merge_runs(blocks: pl.DataFrame, gap_tol: float = 1e-6) -> pl.DataFrame:
    """Collapse consecutive same-species windows into single spans.

    The lanes keep per-window blocks (the model's granularity is the point of
    D5), but the spectrogram overlay merges: eight adjacent windows of one
    species should read as one detection, not eight boxes of noise.
    """
    if blocks.is_empty():
        return blocks
    rows = blocks.sort("start_s").to_dicts()
    out = [dict(rows[0])]
    for r in rows[1:]:
        prev = out[-1]
        if r["species_key"] == prev["species_key"] and r["start_s"] <= prev["end_s"] + gap_tol:
            prev["end_s"] = max(prev["end_s"], r["end_s"])
            prev["score_raw"] = max(prev["score_raw"], r["score_raw"])
        else:
            out.append(dict(r))
    return pl.DataFrame(out, schema=blocks.schema)


def pack_rows(blocks: pl.DataFrame) -> pl.DataFrame:
    """Greedy interval packing: add a `row` column so simultaneous blocks stack
    instead of overdrawing each other.

    The truth lane needs this — several species sing at once in a dawn chorus,
    and painting them in one row means the last one drawn silently erases the
    rest, which is exactly the kind of quiet loss this project is about.
    """
    if blocks.is_empty():
        return blocks.with_columns(row=pl.lit(0, dtype=pl.Int32))
    # Sort on start_s with an explicit index tiebreak so the caller's ordering
    # inside a window (best score first) survives — otherwise which species
    # lands on the top row would depend on sort stability.
    ordered = (
        blocks.with_row_index("_i")
        .sort(["start_s", "_i"])
        .drop("_i")
    )
    row_ends: list[float] = []
    assigned: list[int] = []
    for b in ordered.iter_rows(named=True):
        for i, end in enumerate(row_ends):
            if b["start_s"] >= end - 1e-9:
                row_ends[i] = b["end_s"]
                assigned.append(i)
                break
        else:
            row_ends.append(b["end_s"])
            assigned.append(len(row_ends) - 1)
    return ordered.with_columns(row=pl.Series(assigned, dtype=pl.Int32))


def label_anchors(blocks: pl.DataFrame) -> pl.DataFrame:
    """One label per species per view — its widest block.

    Selective direct labelling: repeating "Hermit Warbler" over fourteen boxes
    is noise, and one label placed on the biggest instance reads as the same
    information.
    """
    if blocks.is_empty():
        return blocks
    return (
        blocks.with_columns(width=pl.col("end_s") - pl.col("start_s"))
        .sort("width", descending=True)
        .group_by("species_key", maintain_order=True)
        .first()
    )


def species_spans(
    det: pl.DataFrame,
    recording_id: str,
    t0: float,
    t1: float,
    theta: float | None,
    species_key: str,
    suppressed: bool | None = None,
) -> pl.DataFrame:
    """Every window in the span where one species scores >= theta, merged.

    `suppressed=None` takes both layers; True/False restricts to one. This is
    what the focus-species overlay draws, so a researcher can put one bird's
    model detections directly against its ground-truth boxes.
    """
    q = det.filter(
        (pl.col("recording_id") == recording_id)
        & (pl.col("end_s") > t0)
        & (pl.col("start_s") < t1)
        & _hit(theta)
        & (pl.col("species_key") == species_key)
    )
    if suppressed is not None:
        q = q.filter(pl.col("suppressed") == suppressed)
    return merge_runs(q.select("start_s", "end_s", "species_key", "score_raw"))


def focus_options(
    det_by_model: dict[str, pl.DataFrame],
    recording_id: str,
    theta: float | None,
    ann: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Which species the explorer's focus picker should offer, and why each is there.

    Emphatically **not** "what the primary model detected above the current θ".
    That was the first version and it was three kinds of too narrow: one model,
    one threshold, and only species that were the *loudest* in some bin — so
    raising θ, or picking the arm that happens to be worst at a bird, silently
    removed that bird from the list. The species you most want to inspect is
    precisely the one a model is failing to detect, and that was the one you
    could not choose.

    So the menu is the union of every species annotated in this recording and
    every species any arm detects in it, with `in_truth` and `n_models` saying
    which side each came from. A bird that is annotated but detected by nothing
    is the most interesting row in the app; it has to be selectable.
    """
    rows: dict[str, dict] = {}
    if ann is not None:
        for key in (ann.filter(pl.col("recording_id") == recording_id)["species_key"]
                    .unique().to_list()):
            rows[key] = {"species_key": key, "in_truth": True, "n_models": 0}

    for det in det_by_model.values():
        hit = det.filter((pl.col("recording_id") == recording_id) & _hit(theta))
        for key in hit["species_key"].unique().to_list():
            r = rows.setdefault(key, {"species_key": key, "in_truth": False,
                                      "n_models": 0})
            r["n_models"] += 1

    if not rows:
        return pl.DataFrame(schema={"species_key": pl.Utf8, "in_truth": pl.Boolean,
                                    "n_models": pl.Int64})
    return (pl.DataFrame(list(rows.values()))
            .sort(["in_truth", "n_models", "species_key"],
                  descending=[True, True, False]))


#: The navigator's outcome vocabulary (V1 E1), in stacking order from the axis
#: up, with colours checked for distinguishability under protan, deutan and
#: tritan simulation (min ΔE ≈ 35 among the four). "Missed" hangs below the axis.
OUTCOMES = ("correct", "filter hid a real bird", "wrong", "filter caught a mistake")
UNLABELLED_OUTCOMES = ("detected", "filter would hide")
OUTCOME_COLOURS = {
    "correct": "#2a78d6",
    "filter hid a real bird": "#4a1a6b",
    "wrong": "#eb6834",
    "filter caught a mistake": "#c9c7bf",
    "missed": "#6b6a63",
    "detected": "#2a78d6",
    "filter would hide": "#c9c7bf",
}
# Its own hue: it once shared "missed"'s grey, and not blue either — blue means
# "correct" in the model charts, and truth is what correct is measured against.
# Checked with the five above under protan / deutan / tritan (min ΔE ≈ 19).
TRUTH_COLOUR = "#1baf7a"

# Models get their own colours, distinct from the outcome colours above — a
# model is never "blue" when blue means "correct". Checked under protan / deutan
# / tritan (min ΔE ≈ 21), and always paired with a shape so colour is never the
# only cue.
MODEL_COLOURS = ["#c2185b", "#0f8b8d", "#d4a017", "#6d4c9f"]
MODEL_SHAPES = ["circle", "square", "triangle-up", "diamond"]


def model_styles(manifests: list) -> dict[str, dict[str, str]]:
    """run_id -> {"colour", "shape", "rank"}: one look per model, app-wide.

    Assigned once, from the dataset's runs, oldest first — so every page draws a
    model the same way, and a new run joins at the end instead of shifting the
    colours of the ones already there. Pages that assigned colours by their own
    list order drew the same model in two colours on two pages.
    """
    ordered = sorted(manifests, key=lambda m: (m.created_utc, m.run_id))
    return {m.run_id: {"colour": MODEL_COLOURS[i % len(MODEL_COLOURS)],
                       "shape": MODEL_SHAPES[i % len(MODEL_SHAPES)], "rank": i}
            for i, m in enumerate(ordered)}


def mark_correct(
    det: pl.DataFrame,
    ann: pl.DataFrame,
    recording_id: str,
    min_overlap_s: float = 0.5,
) -> pl.DataFrame:
    """One recording's detections, with `correct`: is this species annotated in
    this window?

    Matched per detection window, with the same rule alignment labels windows by
    (`windows.label_starts`): a box of the same species overlapping the window by
    at least min(min_overlap_s, box duration). So the navigator calls a detection
    correct exactly when the scorecard would count it as a true positive.
    """
    d = det.filter(pl.col("recording_id") == recording_id).with_row_index("_row")
    boxes = ann.filter(pl.col("recording_id") == recording_id).select(
        "species_key", pl.col("start_s").alias("_b0"), pl.col("end_s").alias("_b1"))
    if d.is_empty() or boxes.is_empty():
        return d.drop("_row").with_columns(correct=pl.lit(False))
    overlap = (pl.min_horizontal("end_s", "_b1") - pl.max_horizontal("start_s", "_b0"))
    floor = pl.min_horizontal(pl.lit(min_overlap_s), pl.col("_b1") - pl.col("_b0"))
    hits = (d.select("_row", "species_key", "start_s", "end_s")
             .join(boxes, on="species_key", how="inner")
             .filter((overlap > 0) & (overlap >= floor - 1e-9))
             .select("_row").unique())
    return (d.with_columns(correct=pl.col("_row").is_in(hits["_row"].implode()))
             .drop("_row"))


def navigator_bins(
    det: pl.DataFrame,
    ann: pl.DataFrame | None,
    recording_id: str,
    duration_s: float,
    n_bins: int = 180,
    species_key: str | None = None,
    min_overlap_s: float = 0.5,
) -> pl.DataFrame:
    """One model's whole-recording navigator: per time bin, species by outcome.

    `det` is a thresholded, judged detections frame (`above` from
    `thresholds.attach`, `implausible` from `stats.mark_implausible`). Counts are
    **distinct species**, not windows: a 3 s grid produces more windows than a
    5 s grid for the same song, so window counts would show a difference between
    models that is not there. A species counts once per bin, in its best outcome
    (correct, then filter-hid-a-real-bird, then wrong, then filter-caught-a-
    mistake). `missed` is the annotated species in the bin that the model neither
    found nor had hidden.

    With no annotations the outcomes are just `detected` / `filter would hide`,
    and there is no `missed`.

    Returns bin_start_s, bin_end_s, then n_<outcome> and names_<outcome> (species
    keys, "|"-joined) for every outcome, plus n_truth / names_truth.
    """
    bin_s = duration_s / n_bins
    labelled = ann is not None
    outcomes = list(OUTCOMES) if labelled else list(UNLABELLED_OUTCOMES)

    d = det.filter((pl.col("recording_id") == recording_id) & pl.col("above"))
    boxes = (ann.filter(pl.col("recording_id") == recording_id)
             if labelled else pl.DataFrame(schema={"species_key": pl.Utf8,
                                                   "start_s": pl.Float64,
                                                   "end_s": pl.Float64}))
    if species_key is not None:
        d = d.filter(pl.col("species_key") == species_key)
        boxes = boxes.filter(pl.col("species_key") == species_key)

    def spread(frame: pl.DataFrame) -> pl.DataFrame:
        """One row per (row, bin it overlaps)."""
        return frame.with_columns(
            bin=pl.int_ranges(
                (pl.col("start_s") / bin_s).floor().cast(pl.Int64).clip(0, n_bins - 1),
                ((pl.col("end_s") - 1e-9) / bin_s).floor().cast(pl.Int64)
                .clip(0, n_bins - 1) + 1,
            )
        ).explode("bin", empty_as_null=False)

    per_bin: dict[str, dict[int, list[str]]] = {o: {} for o in outcomes}
    if not d.is_empty():
        if labelled:
            d = mark_correct(d, ann, recording_id, min_overlap_s)
            d = d.with_columns(
                outcome=pl.when(pl.col("correct") & ~pl.col("implausible"))
                .then(pl.lit(0))
                .when(pl.col("correct"))
                .then(pl.lit(1))
                .when(~pl.col("implausible"))
                .then(pl.lit(2))
                .otherwise(pl.lit(3)))
        else:
            d = d.with_columns(outcome=pl.when(pl.col("implausible"))
                               .then(pl.lit(1)).otherwise(pl.lit(0)))
        best = (spread(d.select("species_key", "start_s", "end_s", "outcome"))
                .group_by("bin", "species_key").agg(pl.col("outcome").min()))
        for b, sp, o in best.sort("species_key").iter_rows():
            per_bin[outcomes[o]].setdefault(b, []).append(sp)

    truth: dict[int, list[str]] = {}
    if labelled and not boxes.is_empty():
        tb = (spread(boxes.select("species_key", "start_s", "end_s"))
              .select("bin", "species_key").unique().sort("species_key"))
        for b, sp in tb.iter_rows():
            truth.setdefault(b, []).append(sp)

    cols: dict[str, list] = {
        "bin_start_s": list(np.arange(n_bins) * bin_s),
        "bin_end_s": list(np.arange(1, n_bins + 1) * bin_s),
    }
    for o in outcomes:
        cols[f"n_{o}"] = [len(per_bin[o].get(b, [])) for b in range(n_bins)]
        cols[f"names_{o}"] = ["|".join(per_bin[o].get(b, [])) for b in range(n_bins)]
    if labelled:
        found = [set(per_bin["correct"].get(b, []))
                 | set(per_bin["filter hid a real bird"].get(b, []))
                 for b in range(n_bins)]
        missed = [[sp for sp in truth.get(b, []) if sp not in found[b]]
                  for b in range(n_bins)]
        cols["n_missed"] = [len(m) for m in missed]
        cols["names_missed"] = ["|".join(m) for m in missed]
        cols["n_truth"] = [len(truth.get(b, [])) for b in range(n_bins)]
        cols["names_truth"] = ["|".join(truth.get(b, [])) for b in range(n_bins)]
    return pl.DataFrame(cols)


def lane_lines(
    det: pl.DataFrame, recording_id: str, t0: float, t1: float,
    ann: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """The score lane's marks (V1 E5): every above-threshold detection in the span.

    `det` must already carry `strength` (`thresholds.strength`, computed over the
    model's whole frame so heights do not change as the view pans). With `ann`,
    `correct` says whether the species is annotated in that window — the same
    match the navigator uses (`mark_correct`) — and is null without annotations.
    Colour alone cannot say this: most wrong detections are of birds annotated
    elsewhere in the same recording, so they wear a truth colour.
    """
    span = det.filter((pl.col("recording_id") == recording_id)
                      & (pl.col("end_s") > t0) & (pl.col("start_s") < t1)
                      & pl.col("above"))
    span = (mark_correct(span, ann, recording_id) if ann is not None
            else span.with_columns(correct=pl.lit(None, dtype=pl.Boolean)))
    return (span.select("start_s", "end_s", "species_key", "score_raw", "theta",
                        "strength", "implausible", "correct")
                .sort("strength"))

def truth_blocks(ann: pl.DataFrame, recording_id: str, t0: float, t1: float) -> pl.DataFrame:
    """Ground-truth boxes intersecting the span (full boxes, not clipped)."""
    return ann.filter(
        (pl.col("recording_id") == recording_id)
        & (pl.col("end_s") > t0)
        & (pl.col("start_s") < t1)
    ).sort("start_s")


def window_table(
    det: pl.DataFrame,
    recording_id: str,
    t: float,
    profile: Profile,
    common_names: dict[str, str],
    top_n: int = 12,
) -> pl.DataFrame:
    """The inspector: top species in the window containing time t, with the
    columns §6 promises — score, occurrence, tier, suppression, XC link."""
    w = det.filter(
        (pl.col("recording_id") == recording_id)
        & (pl.col("start_s") <= t)
        & (pl.col("end_s") > t)
    ).sort("score_raw", descending=True).head(top_n)
    if w.is_empty():
        return w
    keys = w["species_key"].to_list()
    return w.with_columns(
        species=pl.Series([display_name(k, common_names) for k in keys], dtype=pl.Utf8),
        tier=pl.Series([TIER_LABELS[profile.tier_of(k)] for k in keys], dtype=pl.Utf8),
        xeno_canto=pl.Series([xc_species_url(k) for k in keys], dtype=pl.Utf8),
    ).select("species", "species_key", "score_raw",
             # Present once thresholds are resolved (thresholds.attach): the θ
             # this species has for this model, and whether it cleared it.
             *[c for c in ("theta", "above") if c in w.columns],
             "occ_score", "suppressed", "tier", "xeno_canto")


def multi_window_table(
    det_by_model: dict[str, pl.DataFrame],
    recording_id: str,
    t: float,
    profile: Profile,
    names_by_model: dict[str, dict[str, str]],
    top_n: int = 6,
) -> pl.DataFrame:
    """What every model says about the same instant.

    Each model is asked for *its own* window containing time t, and the window
    is reported alongside the scores — BirdNET's 3 s and Perch's 5 s grids do
    not line up, so "the window" is model-specific and pretending otherwise
    would quietly compare different audio.
    """
    frames = []
    for model, det in det_by_model.items():
        one = window_table(det, recording_id, t, profile,
                           names_by_model.get(model, {}), top_n=top_n)
        if one.is_empty():
            continue
        span = det.filter(
            (pl.col("recording_id") == recording_id)
            & (pl.col("start_s") <= t) & (pl.col("end_s") > t)
        )
        label = (f"{span['start_s'][0]:.0f}–{span['end_s'][0]:.0f}s"
                 if len(span) else "—")
        frames.append(one.with_columns(
            model=pl.lit(model, dtype=pl.Utf8),
            window=pl.lit(label, dtype=pl.Utf8),
        ).select("model", "window", *one.columns))
    if not frames:
        return pl.DataFrame(schema={
            "model": pl.Utf8, "window": pl.Utf8, "species": pl.Utf8,
            "species_key": pl.Utf8, "score_raw": pl.Float32, "occ_score": pl.Float32,
            "suppressed": pl.Boolean, "tier": pl.Utf8, "xeno_canto": pl.Utf8})
    return pl.concat(frames, how="diagonal_relaxed")


def species_palette(*frames: pl.DataFrame,
                    priority: pl.DataFrame | None = None) -> dict[str, str]:
    """Stable color per species for the current view: the 8 most-present species
    (by block count across the given frames) take the categorical slots in that
    order; the rest are 'Other' gray. Within one rendered view every lane and
    the legend share this mapping, so color follows the entity.

    Species in `priority` (the recording's annotations) are coloured first, by
    their own count, and only then the most-detected. Ranking by detections
    alone greyed out exactly the annotated birds the models were missing —
    the ones a viewer is most likely checking.
    """
    def tally(fs) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in fs:
            if f is None or f.is_empty():
                continue
            for k, n in f.group_by("species_key").len().iter_rows():
                counts[k] = counts.get(k, 0) + n
        return counts

    first, counts = tally([priority]), tally(frames)
    ranked = (sorted(first, key=lambda k: (-first[k], k))
              + sorted((k for k in counts if k not in first),
                       key=lambda k: (-counts[k], k)))
    palette = {k: CATEGORICAL[i] for i, k in enumerate(ranked[:len(CATEGORICAL)])}
    for k in ranked[len(CATEGORICAL):]:
        palette[k] = OTHER_COLOR
    return palette


def hz_to_mel_row(hz: float, meta: dict) -> float:
    """Frequency -> fractional mel-row for drawing truth boxes on the cached
    spectrogram. Display-grade: interpolates against the filterbank's centre
    frequencies (the sibling project's hard-won lesson is that exact overlay
    needs the renderer's own grid; scoring never uses this)."""
    import librosa

    centres = librosa.mel_frequencies(
        n_mels=int(meta["n_mels"]),
        fmin=float(meta["fmin"]),
        fmax=min(float(meta["fmax"]), meta["sr"] / 2),
    )
    return float(np.interp(hz, centres, np.arange(len(centres))))


def dataset_summary(store_dir: str | Path, name: str, cache_dir: str | Path) -> dict:
    """One row for the recordings tab: sizes, coverage, cache state."""
    from . import ingest, store as _store
    from .melcache import mel_path

    recordings, ann = ingest.read_dataset(store_dir, name)
    cached = sum(
        mel_path(cache_dir, name, rid).exists()
        for rid in recordings["recording_id"]
    )
    runs = [m for m in list_runs(store_dir) if m.dataset == name]
    return {
        "dataset": name,
        "recordings": len(recordings),
        "hours": round(recordings["duration_s"].sum() / 3600, 1),
        "labelled": bool(recordings["labelled"].any()),
        "annotations": len(ann) if ann is not None else 0,
        "mel_cached": f"{cached}/{len(recordings)}",
        "runs": ", ".join(
            f"{m.run_id} ({len(_store.list_scores(store_dir, m.run_id))} scored)"
            for m in runs
        ) or "—",
    }
