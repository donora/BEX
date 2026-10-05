"""What every BEX page shares: config, cached data access, small helpers.

Pages import this with `from ui.common import *` — it is the app's toolkit,
kept in one place so a cached result is computed once for every page.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from bex import ingest, metrics, scorecard, stats, store, truth, views
from bex import benchmark, geofilter, survey
from bex import thresholds as th
from bex.audio import load_clip
from bex.player import encode_mp3, player_html
from bex.config import load_config
from bex.melcache import display_range, load_mel, slice_mel
from bex.profiles import list_profiles, load_profile
from bex.taxonomy import xc_species_url

cfg = load_config()
REPO_DIR = Path(__file__).resolve().parent.parent


@st.cache_data(show_spinner=False)
def get_dataset(name: str):
    return ingest.read_dataset(cfg.store_dir, name)


@st.cache_data(show_spinner=False)
def get_detections(run_id: str) -> pl.DataFrame:
    return store.read_detections(cfg.store_dir, run_id)


@st.cache_data(show_spinner=False)
def get_mel(dataset: str, recording_id: str):
    return load_mel(cfg.cache_dir, dataset, recording_id)


@st.cache_data(show_spinner=False)
def get_names() -> dict[str, str]:
    """One common name per species, whichever model reported it — Perch's own
    labels are scientific names only (views.common_name_index)."""
    return views.common_name_index(cfg.store_dir, cfg.labels_dir)


@st.cache_data(show_spinner=False)
def get_vocabulary_size(run_id: str) -> int | None:
    """Classes the model can output — read from one stored score matrix."""
    scored = store.list_scores(cfg.store_dir, run_id)
    if not scored:
        return None
    return len(store.read_scores(cfg.store_dir, run_id, scored[0]).species_keys)


@st.cache_data(show_spinner=False)
def get_marked(run_id: str, mode: str, profile_name: str) -> pl.DataFrame:
    """Detections with the implausibility judge applied (bex.stats)."""
    return stats.mark_implausible(
        get_detections(run_id), load_profile(cfg.profiles_dir / profile_name), mode=mode
    )


@st.cache_data(show_spinner=False, max_entries=12)
def get_judged(run_id: str, mode: str, profile_name: str, resolved_json: str) -> pl.DataFrame:
    """Judged detections with each species' own θ applied and lane strength ranked.

    Keyed on the resolved thresholds as JSON, so any change to the rule, the set
    or an override recomputes it, and nothing else does.
    """
    resolved = resolved_from_key(resolved_json)
    return th.strength(th.attach(get_marked(run_id, mode, profile_name), resolved))


def resolved_from_key(resolved_json: str) -> th.Resolved:
    r = json.loads(resolved_json)
    return th.Resolved(r["model"], pl.DataFrame(r["table"], schema={
        "species_key": pl.Utf8, "theta": pl.Float64, "source": pl.Utf8,
        "reason": pl.Utf8}), r["default_theta"], r["default_source"])


def resolved_key(r: th.Resolved) -> str:
    return json.dumps({"model": r.model, "table": r.table.to_dicts(),
                       "default_theta": r.default_theta,
                       "default_source": r.default_source})


@st.cache_data(show_spinner=False)
def complete_runs(dataset_name: str) -> list[str]:
    """Run ids that have scored every recording in the dataset — the only ones
    it is fair to put in a comparison."""
    recs, _ = get_dataset(dataset_name)
    return [
        m.run_id for m in views.list_runs(cfg.store_dir)
        if m.dataset == dataset_name
        and len(store.list_scores(cfg.store_dir, m.run_id)) == len(recs)
    ]


@st.cache_data(show_spinner=False)
def get_comparison(run_ids: tuple[str, ...], profile_name: str, delta: float,
                   dataset_name: str) -> dict:
    """Matched-operating-point league across arms, judged by the profile."""
    _, ann = get_dataset(dataset_name)
    prof = load_profile(cfg.profiles_dir / profile_name)
    manifests = {m.run_id: m for m in views.list_runs(cfg.store_dir)}
    # Unique labels, not model-version: two readouts of one model share that name
    # and a dict keyed on it would silently drop an arm.
    label_of = views.arm_labels([manifests[r] for r in run_ids])
    arms, thetas, names = {}, {}, {}
    for rid in run_ids:
        label = label_of[rid]
        arms[label] = stats.mark_implausible(get_detections(rid), prof, mode="profile")
        names[label] = get_names()
    table = stats.matched_league(arms, None, delta, ann)
    for label, row in zip(arms, table.iter_rows(named=True)):
        thetas[label] = row["theta"]
    reported = {
        label: set(det.filter(pl.col("score_raw") >= thetas[label])["species_key"])
        for label, det in arms.items()
    }
    return {"table": table, "reported": reported, "thetas": thetas, "names": names,
            "truth": set(ann["species_key"].unique()) if ann is not None else set(),
            "labels": list(arms)}


@st.cache_data(show_spinner=False)
def get_aligned(run_id: str, dataset: str, grid_name: str):
    """The cached (window x species) truth/score frame — see `bex.truth`."""
    return truth.read_aligned(cfg.store_dir, dataset, run_id, grid_name)


@st.cache_data(show_spinner=False)
def get_scorecard(run_id: str, dataset: str, grid_name: str, precision_target: float,
                  min_support: int, min_positives: int):
    """One arm's full sweep: curves, per-species scorecard, league row.

    The curves are computed once and handed to both consumers — the per-species
    pass walks every stored row, and the table and the league row would otherwise
    each pay for it.
    """
    aligned, meta = get_aligned(run_id, dataset, grid_name)
    rules = (metrics.Rule("precision", precision_target, min_support),
             metrics.Rule("fbeta", 1.0, min_support),
             metrics.Rule("fbeta", 2.0, min_support))
    curves = metrics.species_curves(aligned)
    table = metrics.per_species_table(aligned, rules, curves)
    row = metrics.summary(aligned, rules, min_positives, curves)
    micro = metrics.pr_curve(aligned["y_true"].to_numpy(), aligned["score"].to_numpy())
    return {"meta": meta, "rules": rules, "curves": curves, "table": table,
            "row": row, "micro": micro,
            "resolution": truth.rank_resolution(aligned)}


def build_alignment(run_id: str, dataset: str, grid_name: str) -> None:
    """Align one run against truth and cache it. Minutes of npz reads, once."""
    grid = None if grid_name == "native" else truth.SHARED_GRID
    bar = st.progress(0.0, text=f"{run_id}: reading score matrices…")
    def tick(rid: str, i: int, n: int) -> None:
        bar.progress(i / n, text=f"{run_id}: {rid} ({i + 1}/{n})")
    aligned, meta = truth.align_run(cfg.store_dir, dataset, run_id, grid=grid,
                                    progress=tick)
    truth.write_aligned(cfg.store_dir, dataset, run_id, aligned, meta)
    bar.empty()


INKS = {
    "light": {"text": "#1d1b17", "label": "#5a5a54", "title": "#3a3a35",
              "grid": "#ececea", "domain": "#d8d7d2"},
    "dark": {"text": "#f2f1ec", "label": "#b9b8b0", "title": "#d6d5ce",
             "grid": "#2c2d33", "domain": "#46474e"},
}


# The BEX palette (V1 HM1): muted, earthy blue-greens. B is a deep petrol teal —
# also the app's accent colour (.streamlit/config.toml) — EX a soft sage. Each has
# a lighter twin for a dark theme.
BRAND = {
    "light": {"b": "#3e6a6b", "ex": "#6f9a86", "deep": "#2f5657",
              "wash": "rgba(111,154,134,0.10)", "tint": "rgba(62,106,107,0.08)",
              "on": "#ffffff"},
    "dark": {"b": "#7fb0b6", "ex": "#b5d2a6", "deep": "#a7cdd1",
             "wash": "rgba(127,176,182,0.08)", "tint": "rgba(127,176,182,0.10)",
             "on": "#10231f"},
}


def brand() -> dict:
    kind = getattr(st.context.theme, "type", None) or "light"
    return BRAND.get(kind, BRAND["light"])


def ink(name: str) -> str:
    """A chart ink colour for the viewer's theme — text, labels and gridlines
    fixed for a light page glare or vanish on a dark one."""
    kind = getattr(st.context.theme, "type", None) or "light"
    return INKS.get(kind, INKS["light"])[name]


def style_chart(chart: alt.Chart) -> alt.Chart:
    """Recessive grid and axes — the data should be the most prominent thing on
    the page, on a light or a dark theme."""
    return (chart.configure_view(strokeWidth=0)
                 .configure_axis(gridColor=ink("grid"), domainColor=ink("domain"),
                                 tickColor=ink("domain"), labelColor=ink("label"),
                                 titleColor=ink("title"), labelFontSize=11)
                 .configure_legend(labelColor=ink("title"), titleColor=ink("title"))
                 .configure_title(color=ink("title")))


def heading(text: str, explainer: str, level: str = "####") -> None:
    """A heading with its explanation behind an ⓘ popover (V1 E4): the chart
    stays on the page, the paragraph about it stays one click away."""
    c1, c2 = st.columns([0.92, 0.08], vertical_alignment="bottom")
    c1.markdown(f"{level} {text}")
    with c2.popover("ⓘ", width="stretch"):
        st.markdown(explainer)


def rule_banner(spec: th.RuleSpec, note: str | None = None) -> str:
    """The threshold rule, stated up front on every page whose numbers it decides
    (V1 C1) — in words, not left to a caption or an ⓘ. Returns the inline tag the
    page's headings repeat."""
    note = note or "Change it in the sidebar, or species by species on the Thresholds page."
    c = brand()
    st.markdown(
        f"<div style='border-left:4px solid {c['b']}; background:{c['tint']};"
        "padding:0.7rem 1rem; border-radius:6px; margin:0.2rem 0 1rem'>"
        f"<div style='font-size:1.05rem'><b>Everything on this page is at one "
        f"threshold rule: <span style='color:{c['deep']}'>{spec.label}</span></b></div>"
        f"<div style='margin-top:0.3rem; font-size:0.9rem; opacity:0.8'>{note}</div>"
        "</div>", unsafe_allow_html=True)
    return f":primary-background[{spec.label}]"


# ---- Thresholds page -------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_curve_points(run_id: str, dataset_name: str, species_key: str,
                     precision_floor: float, min_support: int) -> dict | None:
    """One species' PR curve (thinned for drawing) and its candidate θ."""
    card = get_scorecard(run_id, dataset_name, "native", precision_floor,
                         min_support, 1)
    curve = card["curves"].get(species_key)
    if curve is None or not len(curve):
        return None
    cands = {}
    for name, rule in (("max F1", metrics.Rule("fbeta", 1.0, min_support)),
                       (f"precision ≥ {precision_floor:g}",
                        metrics.Rule("precision", precision_floor, min_support)),
                       ("max F2", metrics.Rule("fbeta", 2.0, min_support))):
        op = metrics.apply_rule(curve, rule)
        if op["feasible"]:
            cands[name] = op["theta"]
    return {"frame": metrics.curve_frame(metrics.thin(curve, 300)).to_pandas(),
            "candidates": cands, "n_pos": curve.n_pos}


# ---- Explorer page ---------------------------------------------------- #
def shift_view(d_view: float = 0.0, d_inspect: float = 0.0) -> None:
    """◀ ▶ buttons: move the visible span and/or the inspected instant, keeping
    the instant inside the span (V1 E3)."""
    ss = st.session_state
    span, max_start = ss["_span"], ss["_max_start"]
    v = min(max(0.0, ss.get("viewport_start", 0.0) + d_view), max_start)
    t = ss.get("inspect_t")
    t = (v if t is None else t) + d_view + d_inspect
    t = min(max(0.0, t), ss["_duration"] - 1e-6)
    if not v <= t < v + span:
        v = min(max(0.0, t - span / 4), max_start)
    ss["viewport_start"], ss["inspect_t"] = v, t


def names_of(keys: str, name_of, limit: int = 6) -> str:
    ks = [k for k in keys.split("|") if k]
    shown = ", ".join(name_of(k) for k in ks[:limit])
    return shown + (f" +{len(ks) - limit}" if len(ks) > limit else "") or "—"


# ---- Compare and Species scorecard pages ------------------------------- #
@st.cache_data(show_spinner=False, max_entries=8)
def get_lab(run_id: str, dataset_name: str, mode: str, profile_name: str,
            resolved_json: str) -> tuple[pl.DataFrame, pl.DataFrame, float]:
    """The aligned frame labelled at the thresholds in force (hit, hidden), every
    annotated vocalisation's outcome, and the hours of audio scored."""
    resolved = resolved_from_key(resolved_json)
    aligned, _ = get_aligned(run_id, dataset_name, "native")
    by_table = resolved.as_dict()
    thetas = {sp: by_table.get(sp, resolved.default_theta)
              for sp in aligned["species_key"].unique().to_list()}
    hidden = (get_marked(run_id, mode, profile_name)
              .filter(pl.col("implausible"))
              .select("recording_id", "start_s", "species_key"))
    lab = benchmark.label_windows(aligned, thetas, hidden)
    recs, ann = get_dataset(dataset_name)
    boxes = benchmark.vocalisation_outcomes(lab, ann)
    scored = aligned["recording_id"].unique().to_list()
    hours = float(recs.filter(pl.col("recording_id").is_in(scored))["duration_s"].sum()) / 3600
    return lab, boxes, hours


@st.cache_data(show_spinner=False, max_entries=48)
def get_species_detail(run_id: str, dataset_name: str, mode: str, profile_name: str,
                       resolved_json: str, species_key: str, delta: float) -> dict:
    """One model on one species, at the thresholds in force: the scorecard's card,
    its recordings, what it is mistaken for, and its crowd."""
    lab, boxes, hours = get_lab(run_id, dataset_name, mode, profile_name, resolved_json)
    lab_sp = lab.filter(pl.col("species_key") == species_key)
    boxes_sp = boxes.filter(pl.col("species_key") == species_key)
    judged = get_judged(run_id, mode, profile_name, resolved_json)
    crowd, crowd_summary = benchmark.crowd(judged, species_key, delta)
    return {
        "summary": benchmark.summary(lab_sp, boxes_sp, hours),
        "windows": {"reported": int(lab_sp["hit"].sum()),
                    "caught": int((lab_sp["hit"] & (lab_sp["y_true"] == 1)).sum()),
                    "labelled": int((lab_sp["y_true"] == 1).sum())},
        "recordings": benchmark.by_recording(lab_sp, boxes_sp),
        "mistaken_for": benchmark.mistaken_for(lab, species_key),
        "crowd": crowd, "crowd_summary": crowd_summary,
    }


@st.cache_data(show_spinner=False, max_entries=24)
def get_benchmark(run_id: str, dataset_name: str, mode: str, profile_name: str,
                  resolved_json: str, precision_floor: float, min_support: int,
                  min_positives: int) -> dict:
    """One model at the thresholds in force: the Compare dashboard's numbers.

    Keyed on the resolved thresholds and the judge, so the dashboard always
    describes exactly what the sidebar describes (bex.benchmark).
    """
    lab, boxes, hours = get_lab(run_id, dataset_name, mode, profile_name, resolved_json)
    card = get_scorecard(run_id, dataset_name, "native", precision_floor,
                         min_support, min_positives)
    per_species = benchmark.per_species(lab, boxes).join(
        card["table"].select("species_key", "ap"), on="species_key", how="left")
    return {
        "summary": benchmark.summary(lab, boxes, hours),
        "per_species": per_species,
        "micro": metrics.curve_frame(metrics.thin(card["micro"], 400)).to_pandas(),
        "cmap": card["row"]["cmap"],
        "micro_ap": card["row"]["micro_ap"],
        "resolution": card["resolution"],
    }


# ---- Outcome bars (Compare, Species scorecard) ------------------------ #
def outcome_bar(shares: dict[str, float], order: list[str]) -> alt.Chart:
    """One 100% bar, split by outcome, in the navigator's colours.

    Large segments carry their percentage; the tooltip always does.
    """
    rows, x0 = [], 0.0
    for o in order:
        v = shares.get(o, 0.0)
        rows.append({"outcome": o, "x0": x0, "x1": x0 + v, "share": v,
                     "mid": x0 + v / 2, "label": f"{v:.0%}" if v >= 0.08 else ""})
        x0 += v
    df = pd.DataFrame(rows)
    x = alt.X("x0:Q", scale=alt.Scale(domain=[0, 1]), axis=None)
    bar = alt.Chart(df).mark_bar(height=22, cornerRadius=2).encode(
        x=x, x2="x1:Q",
        color=alt.Color("outcome:N", legend=None, scale=alt.Scale(
            domain=list(views.OUTCOME_COLOURS), range=list(views.OUTCOME_COLOURS.values()))),
        tooltip=[alt.Tooltip("outcome:N"), alt.Tooltip("share:Q", format=".1%")])
    text = alt.Chart(df).mark_text(fontSize=11, fontWeight="bold").encode(
        x=alt.X("mid:Q", scale=alt.Scale(domain=[0, 1])), text="label:N",
        color=alt.condition(alt.FieldOneOfPredicate("outcome", ["filter caught a mistake"]),
                            alt.value("#1d1b17"), alt.value("#ffffff")))
    return (bar + text).properties(height=26)


def outcome_key(order: list[str]) -> str:
    return " &nbsp; ".join(
        f"<span style='white-space:nowrap'><span style='color:"
        f"{views.OUTCOME_COLOURS[o]}'>■</span> {o}</span>" for o in order)


# ---- Geofilter page --------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=24)
def get_geo(run_id: str, dataset_name: str, mode: str, profile_name: str,
            resolved_json: str) -> dict:
    """What the filter does for one model at the thresholds in force (bex.geofilter)."""
    judged = get_judged(run_id, mode, profile_name, resolved_json)
    _, ann = get_dataset(dataset_name)
    truthy = geofilter.with_truth(judged, ann)
    return {
        "reported": len(truthy),
        "outcomes": geofilter.outcomes(truthy),
        "muddy": geofilter.muddy_windows(judged),
        "impossible": geofilter.impossible_species(truthy),
        "unreportable": geofilter.unreportable(judged, ann),
        "circular": geofilter.circular(judged, ann),
        "annotated_species": geofilter.annotated_species(ann),
        # Songs found needs the alignment against truth; without it, no recall.
        "songs": (benchmark.singing_outcomes(
                      get_lab(run_id, dataset_name, mode, profile_name, resolved_json)[1])
                  if ann is not None and truth.aligned_path(
                      cfg.store_dir, dataset_name, run_id, "native").exists() else None),
    }


@st.cache_data(show_spinner=False, max_entries=24)
def get_species_lists(run_id: str, dataset_name: str, mode: str, profile_name: str,
                      resolved_json: str, min_detections: int):
    """Each recording's species list for one model, at the thresholds in force,
    against the annotations (bex.benchmark.species_lists)."""
    _, ann = get_dataset(dataset_name)
    lists = benchmark.species_lists(get_judged(run_id, mode, profile_name, resolved_json),
                                    ann, min_detections)
    return lists, benchmark.species_list_summary(lists)


# ---- Survey protocol page (V1.1) ------------------------------------------ #
@st.cache_data(show_spinner=False, max_entries=12)
def get_survey_detections(run_id: str, mode: str, profile_name: str) -> pl.DataFrame:
    """Species detections with the judge's flag — what a species list reads."""
    return survey.species_only(get_marked(run_id, mode, profile_name))


@st.cache_resource(show_spinner=False, max_entries=8)
def get_curve_source(run_id: str, dataset: str) -> survey.CurveSource:
    """The aligned frame split by species once, for refitting θ on any subset."""
    aligned, _ = get_aligned(run_id, dataset, "native")
    return survey.CurveSource(aligned)


@st.cache_data(show_spinner=False, max_entries=32)
def get_survey_evidence(run_id: str, dataset: str, mode: str, profile_name: str,
                        spec: th.RuleSpec, identity: str, fit_on: tuple[str, ...] | None,
                        spans: tuple[float, ...], sidebar_json: str) -> pl.DataFrame:
    """Evidence at every precision floor (θ fitted on `fit_on`; every recording
    when None) and at the sidebar's thresholds."""
    bars = ({} if not truth.aligned_path(cfg.store_dir, dataset, run_id, "native").exists()
            else survey.fit_bars(get_curve_source(run_id, dataset), identity, spec,
                                 list(fit_on) if fit_on is not None else None))
    bars[survey.SIDEBAR] = resolved_from_key(sidebar_json)
    return survey.evidence(get_survey_detections(run_id, mode, profile_name), bars, spans)
