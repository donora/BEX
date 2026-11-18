"""Prepare a BEX dataset for EcoLabel, where volunteers check the detections.

The simple, guided path (the app's "EcoLabel" page, and `bex export-ecolabel`
/ `bex upload` on the command line):

1. a dataset and a model run (Perch by default)
2. the **candidate pool**: each window's top species (permissive), with a
   plausibility filter from the dataset's profile, and a species list the
   researcher can trim and name
3. a **guided check**: per species, the researcher marks detections Yes or No
   until it has 5 confirmed Yes and 3 No (fewer when EcoLabel's shared
   library already has some). Each listener hears 3 of the Yeses as examples
   and is tested on the others, shuffled per listener. The answers also
   calibrate the model's scores
4. a **recall target**: the calibrated scores turn "aim for 80% recall" into a
   score floor, with the false detections per 100 that costs
5. the project's profile (who, what, where; a photo)
6. a **survey protocol** (presence / activity / every call) with its cost
7. an **EcoLabel bundle** (the format in EcoLabel's BUNDLE.md), zipped, and a
   resumable upload to an EcoLabel server

Everything is kept in `store/datasets/<dataset>/ecolabel/`. The bundle holds
short clips around each candidate, never whole recordings, and no exact
coordinates. Pure functions over the store, like the rest of BEX; the app is a
thin layer on top.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf

FORMAT, VERSION = "ecolabel-bundle", "0.1"
CONTEXT_S = 0.5          # audio either side of a candidate window in its clip
CLIP_FORMAT = "ogg"      # Ogg Vorbis: a third the size of FLAC, same sample rate; plenty for listening
REF_PAD_S = 0.5          # and either side of a reference call
MIN_YES, MIN_NO = 5, 3   # per species (here + the shared library), for the crowd to be asked about it
EXAMPLES = 3             # of the Yeses, each listener hears this many as examples; the rest test them
MIN_REFS = EXAMPLES      # (older name)
LIBRARY_TERMS = "ecolabel-library-1"
MIN_CALIBRATION = 20     # checked detections before scores are turned into recall
SPECIES_STRENGTH = 10.0  # pseudo-answers pulling a species' calibration to the overall one
POOL_RANK = 3            # each window's top-k species form the candidate pool
POOL_FLOOR = {"birdnet": 0.05}        # plus a score floor, for models whose scores mean something
OCC_FLOOR, UNEXPECTED_SCORE = 0.005, 0.5   # permissive geofilter (models with one), and its exception
ANSWERS_PER_CHECK = (5.6, 8.4)        # crowd answers per checked detection: a range (EcoLabel's experience)
KEY = ["recording_id", "start_s", "species_key"]
CONTACT = "mara@melusine.uk"          # EcoLabel questions and early access


def is_open() -> bool:
    """EcoLabel isn't public yet: BEX shows sending to it as coming soon, and
    never contacts a server, unless BEX_ECOLABEL=1 (early-access users)."""
    import os
    return os.environ.get("BEX_ECOLABEL") == "1"

PROTOCOLS = {
    "presence": "Which species are there: each species in each recording, confirmed once, best detections first.",
    "activity": "Activity through time: the same, for every block of minutes.",
    "every_call": "Every detection checked.",
}
DEFAULT_PROTOCOL = {"mode": "presence", "block_s": 300, "confirmations": 1, "give_up_after": 5}
DEFAULT_ABOUT = {
    "group": {"name": "", "kind": "", "bio": "", "website": ""},
    "project": {"title": "", "summary": "", "why": ""},
    "site": {"name": "", "habitat": "", "description": "", "when": ""},
    "photo": {"caption": "", "credit": ""},
}


# =================================================================== state ===
def folder(store_dir, dataset: str) -> Path:
    d = Path(store_dir) / "datasets" / dataset / "ecolabel"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_settings(store_dir, dataset: str) -> dict:
    p = folder(store_dir, dataset) / "settings.json"
    s = json.loads(p.read_text()) if p.exists() else {}
    about = json.loads(json.dumps(DEFAULT_ABOUT))
    for k, v in (s.get("about") or {}).items():
        if k in about and isinstance(v, dict):
            about[k].update({f: v.get(f, "") for f in about[k]})
    return {"run_id": "", "included": None, "names": {}, "researcher_only": [], "target_recall": 0.8, "floor": None,
            "label_ref": "", **s, "about": about, "protocol": {**DEFAULT_PROTOCOL, **(s.get("protocol") or {})}}


def save_settings(store_dir, dataset: str, settings: dict) -> None:
    (folder(store_dir, dataset) / "settings.json").write_text(json.dumps(settings, indent=2, default=str))


def photo_path(store_dir, dataset: str) -> Path:
    return folder(store_dir, dataset) / "photo.jpg"


# ==================================================================== runs ===
def runs_for(store_dir, dataset: str) -> list[dict]:
    """Model runs over this dataset, Perch first (EcoLabel's default model)."""
    out = []
    for m in sorted((Path(store_dir) / "runs").glob("*/manifest.json")):
        man = json.loads(m.read_text())
        if man.get("dataset") == dataset and (Path(store_dir) / "detections" / f"{man['run_id']}.parquet").exists():
            out.append(man)
    return sorted(out, key=lambda m: (m.get("model_name") != "perch", m.get("created_utc", "")), reverse=False)


def model_licence(model_name: str) -> str | None:
    """A licence warning to show, or None."""
    if model_name == "birdnet":
        return ("BirdNET's model weights are CC BY-NC-SA 4.0: non-commercial use only. Your own research "
                "is fine, but EcoLabel is a paid service: check that your use of BirdNET's detections there "
                "is covered, or use Perch (Apache-2.0).")
    return None


# ================================================================== pool ===
def candidate_pool(store_dir, dataset: str, run_id: str, rank: int = POOL_RANK) -> pl.DataFrame:
    """Each window's top species: the permissive list everything else narrows.
    Models with a geo model keep their occurrence score for the geofilter."""
    model = run_id.split("-")[0]
    floor = POOL_FLOOR.get(model, 0.0)
    d = (pl.scan_parquet(Path(store_dir) / "detections" / f"{run_id}.parquet")
         .filter(pl.col("rank_in_window") <= rank, pl.col("score_raw") >= floor)
         .select("recording_id", "start_s", "end_s", "species_key",
                 pl.col("score_raw").cast(pl.Float64).alias("score"), pl.col("rank_in_window").alias("rank"),
                 pl.col("occ_score").cast(pl.Float64).fill_nan(None).alias("occ"))
         .collect())
    return d.sort("recording_id", "start_s", "rank")


def saturated(pool: pl.DataFrame) -> bool:
    """Do this run's scores pile up at the top? (Perch's per-class sigmoid
    readout does: most top detections score ~1, so a score floor can't
    separate good from bad. Its softmax readout doesn't.)"""
    return pool.height > 0 and float((pool["score"] >= 0.9999).mean()) > 0.2


#: How the simple path runs each model: Perch with its softmax readout, whose
#: scores spread out (the sigmoid one saturates; see `saturated`).
RUN_ARGS = {"perch": ["--readout", "softmax"]}


def common_names(store_dir, profile_dir: Path | None = None) -> dict[str, str]:
    """Common names, best source first: the dataset's profile, then any BirdNET
    label file in the store. (Both are fine for preparing a bundle; the names a
    product shows publicly need a licensed taxonomy, EcoLabel's LICENSES.md.)"""
    names: dict[str, str] = {}
    for lab in sorted((Path(store_dir) / "runs").glob("birdnet-*/labels.txt")):
        for line in lab.read_text().splitlines():
            sci, _, name = line.partition("_")
            if name:
                names.setdefault(sci.strip(), name.strip())
    if profile_dir and (profile_dir / "tiers.csv").exists():
        t = pl.read_csv(profile_dir / "tiers.csv")
        if "common_name" in t.columns:
            names.update({k: v for k, v in zip(t["species_key"], t["common_name"]) if v})
    return names


def species_table(pool: pl.DataFrame, tiers: dict[str, int] | None, names: dict[str, str],
                  settings: dict) -> pl.DataFrame:
    """One row per species in the pool, with what decides whether listeners
    are asked about it: plausibility (profile tier, or the model's geo score),
    a name, and the researcher's choice."""
    has_occ = pool["occ"].is_not_null().any()
    sp = (pool.group_by("species_key").agg(
        detections=pl.len(), best=pl.col("score").max(),
        occ=pl.col("occ").max() if has_occ else pl.lit(None, pl.Float64))
          .sort("detections", descending=True))
    rows = []
    chosen = settings.get("included")
    own = settings.get("names") or {}
    mine = set(settings.get("researcher_only") or [])
    for r in sp.iter_rows(named=True):
        k = r["species_key"]
        tier = (tiers or {}).get(k, 3) if tiers is not None else None
        plausible = (tier is None or tier <= 2) and (not has_occ or (r["occ"] or 0) >= OCC_FLOOR)
        # Implausible here, but the model was confident: worth the researcher's
        # look, never the crowd's. Shown separately and off by default.
        unexpected = not plausible and r["best"] >= UNEXPECTED_SCORE
        default = plausible
        rows.append({**r, "tier": tier, "plausible": plausible, "unexpected": unexpected,
                     "name": own.get(k) or names.get(k, ""),
                     "include": (k in chosen) if chosen is not None else default,
                     "researcher": k in mine})
    return pl.DataFrame(rows, schema={"species_key": pl.Utf8, "detections": pl.UInt32, "best": pl.Float64,
                                      "occ": pl.Float64, "tier": pl.Int64, "plausible": pl.Boolean,
                                      "unexpected": pl.Boolean, "name": pl.Utf8, "include": pl.Boolean,
                                      "researcher": pl.Boolean})


# ================================================================== checks ===
CHECKS_SCHEMA = {"recording_id": pl.Utf8, "start_s": pl.Float64, "species_key": pl.Utf8,
                 "verdict": pl.Utf8, "reference": pl.Boolean, "labeller": pl.Utf8, "at": pl.Utf8}


def load_checks(store_dir, dataset: str) -> pl.DataFrame:
    p = folder(store_dir, dataset) / "checks.parquet"
    return pl.read_parquet(p) if p.exists() else pl.DataFrame(schema=CHECKS_SCHEMA)


def add_check(store_dir, dataset: str, recording_id: str, start_s: float, species_key: str,
              verdict: str, reference: bool = False, labeller: str = "") -> pl.DataFrame:
    """Record (or replace) a verdict: yes / no / unsure. A reference is a clean
    call to play listeners as an example; it implies yes."""
    if verdict not in ("yes", "no", "unsure"):
        raise ValueError(verdict)
    c = load_checks(store_dir, dataset).filter(
        ~((pl.col("recording_id") == recording_id) & (pl.col("start_s") == start_s) & (pl.col("species_key") == species_key)))
    row = pl.DataFrame([{"recording_id": recording_id, "start_s": float(start_s), "species_key": species_key,
                         "verdict": "yes" if reference else verdict, "reference": bool(reference),
                         "labeller": labeller, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}],
                       schema=CHECKS_SCHEMA)
    c = pl.concat([c, row])
    c.write_parquet(folder(store_dir, dataset) / "checks.parquet")
    return c


def remove_check(store_dir, dataset: str, recording_id: str, start_s: float, species_key: str) -> pl.DataFrame:
    """Take an answer back (it returns to the queue)."""
    c = load_checks(store_dir, dataset).filter(
        ~((pl.col("recording_id") == recording_id) & (pl.col("start_s") == start_s) & (pl.col("species_key") == species_key)))
    c.write_parquet(folder(store_dir, dataset) / "checks.parquet")
    return c


def progress(r: dict) -> float:
    """How far a readiness row is towards ready, 0 to 1 (library answers count)."""
    y = min(r["known_yes"] + r.get("lib_yes", 0), MIN_YES)
    n = min(r["known_no"] + r.get("lib_no", 0), MIN_NO)
    return (y + n) / (MIN_YES + MIN_NO)


def readiness(checks: pl.DataFrame, species: list[str], library: dict | None = None) -> pl.DataFrame:
    """Per species: confirmed Yes and No here, the same in EcoLabel's shared
    library (`library_coverage`), and whether together that's enough for the
    crowd to be asked: MIN_YES Yes (each listener hears EXAMPLES of them as
    examples and is tested on the rest) and MIN_NO No."""
    c = checks.filter(pl.col("species_key").is_in(species))
    agg = c.group_by("species_key").agg(
        known_yes=(pl.col("verdict") == "yes").sum(),
        known_no=(pl.col("verdict") == "no").sum(),
        unsure=(pl.col("verdict") == "unsure").sum())
    lib = library or {}
    base = pl.DataFrame({"species_key": species,
                         "lib_yes": [int(lib.get(k, {}).get("yes", 0)) for k in species],
                         "lib_no": [int(lib.get(k, {}).get("no", 0)) for k in species]},
                        schema={"species_key": pl.Utf8, "lib_yes": pl.Int64, "lib_no": pl.Int64})
    return (base.join(agg, on="species_key", how="left")
            .with_columns(pl.col("known_yes", "known_no", "unsure").fill_null(0).cast(pl.Int64))
            .with_columns(references=pl.col("known_yes"),
                          need_yes=(MIN_YES - pl.col("known_yes") - pl.col("lib_yes")).clip(0),
                          need_no=(MIN_NO - pl.col("known_no") - pl.col("lib_no")).clip(0))
            .with_columns(ready=(pl.col("need_yes") == 0) & (pl.col("need_no") == 0),
                          covered=(pl.col("lib_yes") >= MIN_YES) & (pl.col("lib_no") >= MIN_NO))
            .select("species_key", "known_yes", "known_no", "unsure", "references", "lib_yes", "lib_no",
                    "need_yes", "need_no", "ready", "covered"))


def next_to_check(pool: pl.DataFrame, checks: pl.DataFrame, species_key: str,
                  library: dict | None = None) -> dict | None:
    """The next detection of this species to show the researcher.

    Until it has its Yeses: the best-scoring unchecked one
    (most likely a real, clean call). Then its known No: the lowest-scoring
    (likeliest wrong). Then, for calibration, walk the score range (a
    different fifth each time), so the answers cover every score the recall
    estimate needs."""
    sp = pool.filter(pl.col("species_key") == species_key)
    if checks.height:
        sp = sp.join(checks.select(KEY), on=KEY, how="anti")
    if sp.height == 0:
        return None
    r = readiness(checks, [species_key], library).row(0, named=True)
    if r["need_yes"]:
        return sp.sort("score", descending=True).row(0, named=True)
    if r["need_no"]:
        # Wrong ones are likeliest at the bottom (most species are mostly right on top).
        return sp.sort("score").row(0, named=True)
    n = int(checks.filter(pl.col("species_key") == species_key).height)
    lo, hi = [0.8, 0.6, 0.4, 0.2, 0.0][n % 5], [1.0, 0.8, 0.6, 0.4, 0.2][n % 5]
    q = sp["score"].rank("ordinal").to_numpy() / sp.height
    pick = np.flatnonzero((q > lo) & (q <= hi + 1e-9))
    i = int(pick[n % len(pick)]) if len(pick) else int(np.argmax(sp["score"].to_numpy()))
    return sp.row(i, named=True)


# ============================================================= calibration ===
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _platt(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, iters: int = 100,
           ridge: float = 1e-3) -> tuple[float, float, np.ndarray]:
    """Weighted logistic fit of y on x (y may be fractional): (a, b, the
    posterior covariance of [a, b]). Newton with step-halving, so perfectly
    separated answers (all the Nos below all the Yeses, which the guided check
    tends to produce) can't run away."""
    w = np.ones_like(y, dtype=float) if w is None else w
    X = np.c_[x, np.ones_like(x)]
    th = np.array([1.0, 0.0])

    def nll(t):
        z = X @ t
        return float(np.sum(w * (np.logaddexp(0, z) - y * z)) + ridge * t @ t / 2)
    H = np.eye(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ th)))
        g = X.T @ (w * (p - y)) + ridge * th
        H = (X * (w * p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        f0, k = nll(th), 1.0
        while nll(th - k * step) > f0 and k > 1e-6:
            k /= 2
        th = th - k * step
        if np.abs(k * step).max() < 1e-9:
            break
    return float(th[0]), float(th[1]), np.linalg.inv(H)


#: How each model's score typically turns into "is it real", measured against
#: expert labels: SNE (Kahl et al. 2022; 33 h, every call boxed), each window's
#: top 3 on the Sierra Nevada species list (BEX profile us-ca-sierra). Logistic
#: on logit(score): p = sigmoid(a * logit(score) + b). Reproduce with
#: scripts/typical_calibration.py.
#:   Perch v2, softmax readout:  27,043 candidates, 60% real (top 82%, 2nd 45%, 3rd 28%)
#:   BirdNET 2.4:                95,747 candidates, 34% real (top 58%, 2nd 24%, 3rd 11%)
TYPICAL = {"perch-softmax": (0.977, 2.459, "Perch (softmax), on expert-labelled Sierra Nevada recordings"),
           "birdnet": (1.137, -0.527, "BirdNET, on expert-labelled Sierra Nevada recordings")}
PRIOR_ANSWERS = 30       # the typical curve counts as this many answers; yours outweigh it beyond that
GENERIC = (1.0, 0.0, "a generic guess (score = chance it's real)")
GENERIC_ANSWERS = 8      # ...a much weaker one, for models BEX knows nothing about
SECONDS_PER_ANSWER = 8   # a volunteer hearing a 4 s clip and answering


def typical_for(store_dir, run_id: str) -> tuple[float, float, str] | None:
    """The typical calibration for this run's model, or None (an unknown model,
    or Perch's saturated sigmoid readout)."""
    p = Path(store_dir) / "runs" / run_id / "manifest.json"
    man = json.loads(p.read_text()) if p.exists() else {}
    name, transform = man.get("model_name", run_id.split("-")[0]), str(man.get("score_transform", ""))
    if name == "perch":
        return TYPICAL["perch-softmax"] if transform.startswith("softmax") else None
    return TYPICAL.get(name)


def calibration(pool: pl.DataFrame, checks: pl.DataFrame, typical: tuple | None = None) -> dict | None:
    """Chance each pool detection is real: a logistic fit of the researcher's
    answers on the model's score, starting from how the model typically does
    (`typical`, counted as PRIOR_ANSWERS answers spread over this pool's
    scores), then a per-species correction (shrunk towards none). With no
    typical curve, a weak generic one, and at least MIN_CALIBRATION answers.

    Returns {p, a, b, cov, offsets, answers, prior_answers, typical}, or None."""
    known = pool.join(checks.filter(pl.col("verdict") != "unsure").select(*KEY, "verdict"), on=KEY)
    if typical is None and known.height < MIN_CALIBRATION:
        return None
    a0, b0, label = typical or GENERIC
    k = PRIOR_ANSWERS if typical else GENERIC_ANSWERS
    qs = np.quantile(pool["score"].to_numpy(), (np.arange(k) + 0.5) / k) if pool.height else np.array([])
    x_prior = _logit(qs)
    y_prior = 1 / (1 + np.exp(-(a0 * x_prior + b0)))
    y = (known["verdict"] == "yes").to_numpy().astype(float)
    x = np.r_[_logit(known["score"].to_numpy()), x_prior]
    a, b, cov = _platt(x, np.r_[y, y_prior])
    p_known = 1 / (1 + np.exp(-(a * _logit(known["score"].to_numpy()) + b)))
    offsets = {}
    for s in known["species_key"].unique():
        m = (known["species_key"] == s).to_numpy()
        n, exp_yes, obs = m.sum(), p_known[m].sum(), y[m].sum()
        rate = np.clip(exp_yes / n, 1e-3, 1 - 1e-3)
        kk = SPECIES_STRENGTH
        off = np.log((obs + kk * rate) / (n - obs + kk * (1 - rate))) - np.log(rate / (1 - rate))
        offsets[s] = float(np.clip(off, -4, 4))
    off_all = pool["species_key"].replace_strict(offsets, default=0.0, return_dtype=pl.Float64).to_numpy()
    p = 1 / (1 + np.exp(-(a * _logit(pool["score"].to_numpy()) + b + off_all)))
    return {"p": p, "a": a, "b": b, "cov": cov, "offsets": off_all, "answers": int(known.height),
            "yes": int(y.sum()), "no": int(len(y) - y.sum()), "prior_answers": k, "typical": label,
            "known_typical": typical is not None}


def calibrate(pool: pl.DataFrame, checks: pl.DataFrame, typical: tuple | None = None) -> np.ndarray | None:
    """`calibration(...)["p"]`: the chance each pool detection is real."""
    c = calibration(pool, checks, typical)
    return None if c is None else c["p"]


def recall_band(pool: pl.DataFrame, cal: dict, recalls: np.ndarray, draws: int = 200,
                seed: int = 0) -> pl.DataFrame:
    """How sure the estimate is: for each recall target, the detections that
    would need checking (and the false ones among them) across plausible
    calibrations (draws from the fit's uncertainty). Columns: recall, kept_lo,
    kept_mid, kept_hi, false_lo, false_hi (10th, 50th, 90th percentiles)."""
    score = pool["score"].to_numpy()
    order = np.argsort(-score)
    x = _logit(score[order])
    off = cal["offsets"][order]
    rng = np.random.default_rng(seed)
    ab = rng.multivariate_normal([cal["a"], cal["b"]], cal["cov"], size=draws)
    kept = np.empty((draws, len(recalls)))
    false = np.empty((draws, len(recalls)))
    for i, (a, b) in enumerate(ab):
        pp = 1 / (1 + np.exp(-(a * x + b + off)))
        cum = np.cumsum(pp)
        r = cum / cum[-1]
        j = np.minimum(np.searchsorted(r, recalls), len(r) - 1)
        kept[i] = j + 1
        false[i] = (j + 1) - cum[j]
    q = lambda m, v: np.percentile(m, v, axis=0)
    return pl.DataFrame({"recall": recalls, "kept_lo": q(kept, 10), "kept_mid": q(kept, 50), "kept_hi": q(kept, 90),
                         "false_lo": q(false, 10), "false_hi": q(false, 90)})


def volunteer_hours(checks: float) -> tuple[float, float]:
    """Volunteer time to check this many detections one by one (every call):
    answers per check (ANSWERS_PER_CHECK) at SECONDS_PER_ANSWER each."""
    return tuple(checks * a * SECONDS_PER_ANSWER / 3600 for a in ANSWERS_PER_CHECK)


def recall_table(pool: pl.DataFrame, p: np.ndarray, points: int = 120) -> pl.DataFrame:
    """For score floors across the range: detections kept, expected real calls
    among them, recall (of the real calls in the pool), false per 100 kept."""
    score = pool["score"].to_numpy()
    order = np.argsort(-score)
    s, pp = score[order], p[order]
    cum_true = np.cumsum(pp)
    total = cum_true[-1] if len(cum_true) else 0.0
    idx = np.unique(np.linspace(0, len(s) - 1, min(points, len(s))).astype(int))
    rows = []
    for i in idx:
        kept = i + 1
        rows.append({"floor": float(s[i]), "kept": kept, "expected_real": float(cum_true[i]),
                     "recall": float(cum_true[i] / total) if total else 0.0,
                     "false_per_100": float(100 * (1 - cum_true[i] / kept))})
    return pl.DataFrame(rows)


def floor_for_recall(table: pl.DataFrame, target: float) -> dict:
    """The highest floor that still reaches the recall target."""
    ok = table.filter(pl.col("recall") >= target)
    return (ok.sort("kept").row(0, named=True) if ok.height else table.sort("kept").row(-1, named=True))


# ================================================================ protocol ===
def _expected_checks(p: np.ndarray, x: int) -> float:
    dist = np.zeros(x + 1)
    dist[0] = 1.0
    checks = 0.0
    for pk in p:
        need = dist[:x].sum()
        if need < 1e-9:
            break
        checks += need
        nxt = np.zeros(x + 1)
        nxt[:x] += dist[:x] * (1 - pk)
        nxt[1:x + 1] += dist[:x] * pk
        nxt[x] += dist[x]
        dist = nxt
    return checks


def estimate_protocol(kept: pl.DataFrame, p: np.ndarray | None, protocol: dict) -> dict:
    """Detections the crowd would check, and crowd answers, under a protocol:
    presence / activity work through each species' best detections until
    confirmed (or the give-up rule pauses it for you), every call checks all."""
    mode = protocol.get("mode", "presence")
    if mode == "every_call" or p is None:
        checks, units = float(kept.height), kept.height
    else:
        block = (pl.lit(0) if mode == "presence"
                 else (pl.col("start_s") // float(protocol.get("block_s", 300))).cast(pl.Int64))
        df = kept.with_columns(p=pl.Series(p), block=block).sort("score", descending=True)
        x = int(protocol.get("confirmations", 1))
        k = int(protocol.get("give_up_after", 0)) or None
        checks, units = 0.0, 0
        for _, g in df.group_by("recording_id", "species_key", "block"):
            checks += _expected_checks(g["p"].to_numpy()[:k], x)
            units += 1
    lo, hi = ANSWERS_PER_CHECK
    return {"units": units, "checks": round(checks), "answers": [int(checks * lo), int(checks * hi * 1.2)]}


# ================================================================== bundle ===
def _cut(src: Path, t0: float, t1: float, dst: Path) -> None:
    """Cut [t0, t1] to a mono clip (Ogg Vorbis, or FLAC by extension)."""
    info = sf.info(src)
    a, b = max(0, int(t0 * info.samplerate)), min(info.frames, int(t1 * info.samplerate))
    x, sr = sf.read(src, start=a, stop=b, dtype="float32", always_2d=True)
    mono = x.mean(axis=1)
    if dst.suffix == ".ogg":
        sf.write(dst, mono, sr, format="OGG", subtype="VORBIS")
    else:
        sf.write(dst, mono, sr, format="FLAC", subtype="PCM_16")


def clip_bytes(src: Path, t0: float, t1: float) -> bytes:
    """A short WAV for the app's player."""
    info = sf.info(src)
    a, b = max(0, int(t0 * info.samplerate)), min(info.frames, int(t1 * info.samplerate))
    x, sr = sf.read(src, start=a, stop=b, dtype="float32", always_2d=True)
    y = x.mean(axis=1)
    peak = float(np.percentile(np.abs(y), 99.5)) or 1e-6
    y = np.clip(y * min(0.6 / peak, 10 ** (36 / 20)), -0.95, 0.95)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def spectrogram_png(src: Path, t0: float, t1: float, hi0: float, hi1: float) -> bytes:
    """A spectrogram of [t0, t1] with [hi0, hi1] (recording time) highlighted:
    each frequency band whitened against its own noise floor, so faint calls show."""
    from PIL import Image
    info = sf.info(src)
    a, b = max(0, int(t0 * info.samplerate)), min(info.frames, int(t1 * info.samplerate))
    x, sr = sf.read(src, start=a, stop=b, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    n_fft, hop = 1024, 256
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
    mag = np.abs(np.fft.rfft(frames, axis=1)).T
    hz = np.fft.rfftfreq(n_fft, 1 / sr)
    db = 20 * np.log10(mag[(hz >= 300) & (hz <= min(11000, sr / 2))] + 1e-7)
    db -= np.median(db, axis=1, keepdims=True)
    v = (np.clip((db - 2) / 16, 0, 1) ** 0.75)[::-1]
    dur = len(x) / sr
    w = max(240, int(dur * 110))
    img = np.asarray(Image.fromarray((v * 255).astype(np.uint8)).resize((w, 200)))
    stops = np.array([[0, 14, 26, 31], [.35, 22, 72, 74], [.6, 64, 140, 110], [.82, 196, 206, 140], [1, 250, 244, 222]])
    lut = np.stack([np.interp(np.linspace(0, 1, 256), stops[:, 0], stops[:, i]) for i in (1, 2, 3)], 1).astype(np.uint8)
    rgb = lut[img]
    i0, i1 = int((hi0 - t0) / dur * w), int((hi1 - t0) / dur * w)
    mask = np.ones(w, bool)
    mask[max(0, i0):min(w, i1)] = False
    rgb[:, mask] = (rgb[:, mask] * 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _readable(store_dir, dataset: str, recording_id: str, path: Path) -> Path:
    """The file to cut from: a repaired copy if there is one already."""
    from .ingest import repairs_dir
    fixed = repairs_dir(store_dir, dataset) / f"{recording_id}.flac"
    return fixed if fixed.exists() else path


def _repair(store_dir, dataset: str, recording_id: str, path: Path) -> Path:
    """Some FLACs decode fine but libsndfile loses sync seeking in them; BEX's
    answer is a one-off transcode into the dataset's repairs folder (repair.py)."""
    from .ingest import repairs_dir
    from .repair import repair_recording
    dst = repairs_dir(store_dir, dataset) / f"{recording_id}.flac"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        repair_recording(path, dst)
    return dst


def audio_path(store_dir, dataset: str, recording_id: str) -> Path:
    from .ingest import dataset_audio_root, read_dataset, resolve_audio
    recs, _ = read_dataset(store_dir, dataset)
    rel = recs.filter(pl.col("recording_id") == recording_id)["path"][0]
    return resolve_audio(store_dir, dataset, dataset_audio_root(store_dir, dataset), recording_id, rel)


def kept_candidates(pool: pl.DataFrame, species: pl.DataFrame, checks: pl.DataFrame, floor: float | None) -> pl.DataFrame:
    """What goes in the bundle: included, named species at or above the floor
    (researcher-only ones need no name: they never reach volunteers), plus
    every detection the researcher checked (those become known answers)."""
    inc = species.filter(pl.col("include"), (pl.col("name").str.len_chars() > 0) | pl.col("researcher"))
    d = pool.join(inc.select("species_key", "unexpected", "researcher"), on="species_key")
    checked = d.join(checks.select(KEY), on=KEY, how="semi")
    above = d if floor is None else d.filter(pl.col("score") >= floor)
    return pl.concat([above, checked]).unique(KEY, maintain_order=True).sort("recording_id", "start_s", "species_key")


def build_bundle(cfg, dataset: str, out_dir: Path, progress=None) -> dict:
    """Write an EcoLabel bundle for this dataset from its saved settings and
    checks (BUNDLE.md in the EcoLabel repository): short clips around each
    candidate, the known answers, reference calls, names, the profile and the
    protocol. Returns a summary, with the folder."""
    from .ingest import read_dataset
    from .labels import load_truth
    store = cfg.store_dir
    s = load_settings(store, dataset)
    run_id = s["run_id"] or (runs_for(store, dataset) or [{}])[0].get("run_id", "")
    if not run_id:
        raise ValueError("no model run for this dataset yet")
    man = json.loads((Path(store) / "runs" / run_id / "manifest.json").read_text())
    pool = candidate_pool(store, dataset, run_id)
    tiers, profile_dir = _tiers(cfg, dataset)
    names = common_names(store, profile_dir)
    species = species_table(pool, tiers, names, s)
    checks = load_checks(store, dataset)
    cand = kept_candidates(pool, species, checks, s.get("floor"))
    if cand.height == 0:
        raise ValueError("no candidates: include some named species, and lower the floor")

    out = Path(out_dir) / f"{dataset}.ecolabel"
    if out.exists():
        shutil.rmtree(out)
    (out / "clips").mkdir(parents=True)
    (out / "references").mkdir()
    window = float(man.get("window_s", 3.0))
    cand = (cand.with_row_index("candidate_id").with_columns(pl.col("candidate_id").cast(pl.Int32),
                                                              model=pl.lit(run_id)))
    win = cand.select("recording_id", "start_s").unique(maintain_order=True).with_row_index("w")
    cand = cand.join(win, on=["recording_id", "start_s"]).sort("candidate_id").with_columns(
        clip=pl.format("clips/w{}." + CLIP_FORMAT, pl.col("w").cast(pl.Utf8).str.zfill(6))).drop("w")
    paths = {r: _readable(store, dataset, r, audio_path(store, dataset, r)) for r in cand["recording_id"].unique()}

    def cut(rec: str, t0: float, t1: float, dst: Path) -> None:
        try:
            _cut(paths[rec], t0, t1, dst)
        except sf.LibsndfileError:
            if progress:
                progress(0.0, f"Repairing {rec} (a one-off: libsndfile can't seek in it)…")
            paths[rec] = _repair(store, dataset, rec, paths[rec])
            _cut(paths[rec], t0, t1, dst)
    todo = cand.unique("clip", maintain_order=True)
    for i, r in enumerate(todo.iter_rows(named=True)):
        cut(r["recording_id"], r["start_s"] - CONTEXT_S, r["end_s"] + CONTEXT_S, out / r["clip"])
        if progress and i % 200 == 0:
            progress(i / max(1, todo.height), f"Cutting clips: {i:,} of {todo.height:,}")
    cand.select("candidate_id", "recording_id", "start_s", "end_s", "species_key", "score", "rank", "model",
                "clip", occ_score="occ", unexpected="unexpected",
                researcher_only="researcher").write_parquet(out / "candidates.parquet")

    # Known answers: the guided check, plus a label set's closed chunks if chosen.
    ck = (cand.join(checks.filter(pl.col("verdict") != "unsure"), on=KEY)
          .select("candidate_id", "verdict", "reference", "labeller"))
    ck.write_parquet(out / "checked.parquet")
    cols = ["recording_id", "kind", "start_s", "end_s", "low_hz", "high_hz", "species_key", "labeller"]
    labels = pl.DataFrame(schema={"recording_id": pl.Utf8, "kind": pl.Utf8, "start_s": pl.Float64, "end_s": pl.Float64,
                                  "low_hz": pl.Float64, "high_hz": pl.Float64, "species_key": pl.Utf8, "labeller": pl.Utf8})
    boxes = None
    if s.get("label_ref"):
        t = load_truth(store, dataset, s["label_ref"])
        if t is not None:
            boxes = t.annotations.with_columns(kind=pl.lit("box"), labeller=pl.lit(t.label))
            segs = t.coverage.with_columns(kind=pl.lit("segment"), labeller=pl.lit(t.label),
                                           low_hz=pl.lit(None, pl.Float64), high_hz=pl.lit(None, pl.Float64),
                                           species_key=pl.lit(None, pl.Utf8))
            labels = pl.concat([labels, boxes.select(cols), segs.select(cols)], how="vertical_relaxed")
    labels.write_parquet(out / "labels.parquet")

    # Examples: every confirmed Yes (EcoLabel shows each listener EXAMPLES of
    # them, and tests them on the rest, so each keeps its candidate_id), then
    # label boxes (examples only).
    refs, i = [], 0
    for r in cand.join(checks.filter(pl.col("verdict") == "yes"), on=KEY).sort("candidate_id").iter_rows(named=True):
        f = f"references/r{i:04d}.{CLIP_FORMAT}"
        t0 = max(0.0, r["start_s"] - REF_PAD_S)
        cut(r["recording_id"], t0, r["end_s"] + REF_PAD_S, out / f)
        refs.append({"file": f, "species_key": r["species_key"], "call_start_s": r["start_s"] - t0,
                     "call_end_s": r["end_s"] - t0, "source": r["recording_id"],
                     "source_start_s": t0, "source_end_s": r["end_s"] + REF_PAD_S,
                     "candidate_id": r["candidate_id"]})
        i += 1
    if boxes is not None and boxes.height:
        have = {}
        for x in refs:
            have[x["species_key"]] = have.get(x["species_key"], 0) + 1
        extra = (boxes.filter(pl.col("species_key").is_in(cand["species_key"].unique().to_list()))
                 .with_columns(fit=((pl.col("end_s") - pl.col("start_s")) - 2.5).abs()).sort("species_key", "fit"))
        for r in extra.iter_rows(named=True):
            if have.get(r["species_key"], 0) >= MIN_YES:
                continue
            if r["recording_id"] not in paths:
                paths[r["recording_id"]] = _readable(store, dataset, r["recording_id"],
                                                     audio_path(store, dataset, r["recording_id"]))
            f = f"references/r{i:04d}.{CLIP_FORMAT}"
            t0 = max(0.0, r["start_s"] - REF_PAD_S)
            cut(r["recording_id"], t0, r["end_s"] + REF_PAD_S, out / f)
            refs.append({"file": f, "species_key": r["species_key"], "call_start_s": r["start_s"] - t0,
                         "call_end_s": r["end_s"] - t0, "source": r["recording_id"],
                         "source_start_s": t0, "source_end_s": r["end_s"] + REF_PAD_S, "candidate_id": None})
            have[r["species_key"]] = have.get(r["species_key"], 0) + 1
            i += 1
    pl.DataFrame(refs, schema={"file": pl.Utf8, "species_key": pl.Utf8, "call_start_s": pl.Float64,
                               "call_end_s": pl.Float64, "source": pl.Utf8, "source_start_s": pl.Float64,
                               "source_end_s": pl.Float64, "candidate_id": pl.Int32}).write_csv(out / "references.csv")

    inc = species.filter(pl.col("species_key").is_in(cand["species_key"].unique().to_list()))
    inc.select("species_key", common_name="name", group=pl.lit("")).write_csv(out / "species.csv")

    recs, _ = read_dataset(store, dataset)
    recs = recs.filter(pl.col("recording_id").is_in(cand["recording_id"].unique().to_list())).sort("recording_id")
    site = s["about"]["site"]
    recs.select("recording_id", "duration_s", pl.col("start_time").fill_null(""), pl.col("site").fill_null(""),
                region=pl.lit(site.get("name") or cfg.site_name or ""),
                habitat=pl.lit(site.get("habitat") or "")).write_csv(out / "recordings.csv")

    if photo_path(store, dataset).exists():
        shutil.copyfile(photo_path(store, dataset), out / "photo.jpg")
    from . import __version__
    about = json.loads(json.dumps(s["about"]))
    about["project"]["title"] = about["project"]["title"] or dataset
    manifest = {
        "format": FORMAT, "version": VERSION, "project": about["project"]["title"],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": {"name": "BEX", "version": __version__},
        "models": [{k: man[k] for k in ("run_id", "model_name", "model_version", "score_transform", "window_s")
                    if k in man} | {"name": man.get("model_name", run_id.split("-")[0])}],
        "candidates": {"policy": "top_k_per_window", "k": POOL_RANK, "score_floor": s.get("floor"),
                       "target_recall": s.get("target_recall"),
                       "plausibility": "profile" if tiers is not None else ("geofilter" if pool["occ"].is_not_null().any() else "none")},
        "clips": {"format": CLIP_FORMAT, "context_s": CONTEXT_S},
        "location_precision": "region",
        "about": about,
        "protocol": s["protocol"],
        "examples": {"per_listener": EXAMPLES, "shuffled": True},
    }
    share = s.get("library") or {}
    if share.get("share"):
        manifest["library"] = {"share": True, "terms": share.get("terms", LIBRARY_TERMS),
                               "credit": share.get("credit", ""), "agreed": share.get("agreed", "")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ready = readiness(checks, inc["species_key"].to_list(), s.get("library_coverage"))
    if progress:
        progress(1.0, "Bundle ready")
    return {"folder": str(out), "candidates": cand.height, "clips": todo.height, "species": inc.height,
            "ready_species": int(ready["ready"].sum()), "references": len(refs), "known_answers": ck.height}


def _tiers(cfg, dataset: str) -> tuple[dict[str, int] | None, Path | None]:
    meta = Path(cfg.store_dir) / "datasets" / dataset / "meta.json"
    name = json.loads(meta.read_text()).get("profile", "") if meta.exists() else ""
    # As Research mode does: the dataset's own list, then the one its runs were
    # made with, then the config's (so the SNE demo never gets the UK list).
    name = (name or next((r["profile"] for r in runs_for(cfg.store_dir, dataset) if r.get("profile")), "")
            or cfg.default_profile)
    d = Path(cfg.profiles_dir) / name if name else None
    if not d or not (d / "tiers.csv").exists():
        return None, None
    t = pl.read_csv(d / "tiers.csv")
    return dict(zip(t["species_key"], t["tier"].cast(pl.Int64))), d


def zip_bundle(folder_path: Path) -> Path:
    """One file to upload. Audio is already compressed (FLAC), so stored, not deflated."""
    folder_path = Path(folder_path)
    z = folder_path.with_suffix(".ecolabel.zip") if folder_path.suffix != ".ecolabel" else folder_path.parent / (folder_path.name + ".zip")
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in sorted(folder_path.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(folder_path).as_posix())
    return z


# ================================================================== upload ===
CHUNK = 8 * 1024 * 1024


def _request(url: str, key: str, method: str = "GET", body: bytes | None = None,
             content_type: str = "application/json") -> dict:
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Authorization": f"Bearer {key}", "content-type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"{e.code} from the EcoLabel server: {detail}") from None


def _get_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def library_coverage(server: str) -> dict[str, dict]:
    """What EcoLabel's shared library already holds, per species: confirmed
    Yes and No from other projects that agreed to share. {} when the server
    can't be reached (then everything is checked here, as before)."""
    try:
        return _get_json(server.rstrip("/") + "/api/library/coverage").get("species", {})
    except (OSError, ValueError):
        return {}


def library_examples(server: str, species_key: str, n: int = 3) -> list[dict]:
    """Example calls of a species from EcoLabel's shared library, with full
    URLs to play (audio, image) and each one's credit; [] when the server
    can't be reached or has none."""
    from urllib.parse import quote
    base = server.rstrip("/")
    try:
        got = _get_json(f"{base}/api/library/examples?species={quote(species_key)}&n={n}")
    except (OSError, ValueError):
        return []
    return [{**e, "audio": base + e["audio"], "image": base + e["image"]} for e in got.get("examples", [])]


def reference_links(species_key: str) -> dict[str, str]:
    """Where to hear a species elsewhere, for any animal: Xeno-canto (birds,
    frogs, grasshoppers, bats and more) and iNaturalist's sound observations."""
    from urllib.parse import quote
    return {"Xeno-canto": "https://xeno-canto.org/species/" + species_key.replace(" ", "-"),
            "iNaturalist sounds": f"https://www.inaturalist.org/observations?taxon_name={quote(species_key)}&sounds=true"}


def library_terms(server: str) -> dict | None:
    """The library's current terms ({version, summary, text}), or None."""
    try:
        return _get_json(server.rstrip("/") + "/api/library/terms")
    except (OSError, ValueError):
        return None


def upload(zip_path: Path, server: str, key: str, progress=None) -> dict:
    """Send a bundle to an EcoLabel server, in chunks, picking up where a
    broken upload stopped (the same file, the same upload). Returns the
    server's report: the new project, and what it found in the bundle."""
    zip_path = Path(zip_path)
    server = server.rstrip("/")
    size = zip_path.stat().st_size
    h = hashlib.sha256()
    with open(zip_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    up = _request(f"{server}/api/uploads", key, "POST",
                  json.dumps({"filename": zip_path.name, "size": size, "sha256": digest}).encode())
    uid, offset = up["upload_id"], int(up.get("received", 0))
    with open(zip_path, "rb") as f:
        f.seek(offset)
        while offset < size:
            chunk = f.read(CHUNK)
            r = _request(f"{server}/api/uploads/{uid}/chunk?offset={offset}", key, "PUT", chunk,
                         "application/octet-stream")
            offset = int(r["received"])
            if progress:
                progress(offset / size, f"Uploading: {offset / 1e6:,.0f} of {size / 1e6:,.0f} MB")
    if progress:
        progress(1.0, "Checking the bundle on the server…")
    return _request(f"{server}/api/uploads/{uid}/complete", key, "POST", b"{}")
