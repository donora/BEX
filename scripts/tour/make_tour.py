"""Regenerate TOUR.md's screenshots: capture the app in Chrome, crop, and draw
the numbered markers the text refers to.

    .venv/bin/python scripts/tour/make_tour.py           # all of docs/tour/*.png it makes

Needs the SNE demo dataset with its model runs and alignments, Node.js,
`npm install playwright-core` (in this folder), and Google Chrome. It starts its
own app server, makes a temporary demo label set (from SNE's expert labels) and a
temporary survey setup, and removes them afterwards, together with anything they
created; your welcome-screen preference is put back as it was.

Markers and crops are placed by element (SHOTS below): "beside the species
picker", "from the top bar to the last card" — measured in the browser at
capture time, so they follow the layout as it changes. A marker whose element
is missing is reported and left out.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from bex import labels as L, views  # noqa: E402
from bex.config import load_config  # noqa: E402
from bex.ingest import read_dataset  # noqa: E402
from bex.melcache import load_mel  # noqa: E402

PORT = 8599
RAW = HERE / "raw"
DST = REPO / "docs" / "tour"
SET, SETUP, DATASET = "tour-demo", "tour-birdnet", "sne"
TEAL = (62, 106, 107)

def M(sel, at="left", ysel=None, fx=None, fy=None, dx=0, dy=0):
    """A marker beside an element: `at` "left" or "right" of `sel`, level with
    the middle of `ysel` (default `sel`) — or at fraction (fx, fy) of its box."""
    return dict(sel=sel, at=at, ysel=ysel, fx=fx, fy=fy, dx=dx, dy=dy)


PTS = "pts"   # fx/fy taken from the demo box's click points

# raw capture -> [(output, crop: selectors whose union (padded) is kept, markers)]
SHOTS = {
    "home.png": [("home.png", dict(sels=[".st-key-bex_nav", ".st-key-card_0",
                                         ".st-key-g_profile"], pad=26), [
        M(".st-key-nav_step1"), M(".st-key-nav_view"), M(".st-key-picto_0"),
        M(".st-key-strip_0"), M(".st-key-card_0 [class*='st-key-h_next_']"),
        M(".st-key-picto_1"), M(".st-key-card_1 [data-testid='stExpander']"),
        M(".st-key-g_add"), M(".st-key-g_profile")])],
    "label-chunk.png": [
        ("label-chunk.png", dict(sels=[".st-key-g_lab_move", "text=New box ·"], pad=26), [
            M(".st-key-g_lab_move"), M("text=/sampled chunks closed/", at="right"),
            M(".st-key-g_lab_nav", fy=0.3, fx=0, dx=-18),
            M(".st-key-g_lab_spec iframe", fx=PTS, fy=PTS, dx=-24, dy=-22),
            M(".st-key-g_lab_spec iframe", fx=0.075, fy=0.62),
            M(".st-key-g_lab_player"), M("text=New box ·")]),
        ("label-suggest.png", dict(sels=["text=New box ·", ".st-key-lab_use_3"], pad=26), [
            M(".st-key-lab_species"), M("text=What the models heard inside your box"),
            M("text=What the models heard inside your box", ysel=".st-key-lab_use_1"),
            M("a[href*='xeno-canto']"), M(".st-key-lab_use_1")])],
    "label-plan.png": [("label-plan.png", dict(sels=["text=What to label",
                                                     ".st-key-lab_proj_run"],
                                               pad=26, extra_bottom=90), [
        M(".st-key-lab_how"), M(".st-key-lab_minutes"),
        M("[data-testid='stProgress']"), M(".st-key-lab_proj_run")])],
    "label-progress.png": [("label-progress.png", dict(
        sels=["[data-testid='stMetric']", "text=What has been found"], pad=26,
        extra_bottom=330, full_width=True), [
        M("[data-testid='stMetric']"), M("text=were labelled after viewing"),
        M("text=Species found against minutes labelled"),
        M("text=Coverage: closed minutes"), M("text=What has been found")])],
    "explorer-navigator.png": [("explorer-navigator.png", dict(
        sels=[".st-key-bex_nav", "button:has-text('window ▶')"], pad=20, x0=0), [
        M(".st-key-g_view", at="right"), M("[class*='st-key-truth_']", at="right"),
        M(".st-key-g_status", at="right"), M("[class*='st-key-models_']", at="right"),
        M("text=Labelled data, scored —"), M(".st-key-nav_view"),
        M(".st-key-nav", fx=0, fy=0.1, dx=-18), M(".st-key-nav", fx=0, fy=0.33, dx=-18),
        M("button:has-text('⏮')")])],
    "compare-full.png": [
        ("compare-cards.png", dict(sels=[".st-key-g_cmp_cards", "text=Figures in brackets"],
                                   pad=26), [
            M("text=When it reports a bird", ysel="span[title^='When the model reports']"),
            M("text=When it reports a bird", ysel="span[title^='Of the labelled songs']"),
            M("text=Figures in brackets")]),
        ("compare-paired.png", dict(sels=[".st-key-g_cmp_paired", ".st-key-g_cmp_paired_df"],
                                    pad=26), [
            M(".st-key-g_cmp_paired_df", fx=0.605, fy=0, dy=18),
            M(".st-key-g_cmp_paired_df", fx=0.77, fy=0, dy=18)])],
    "survey-full.png": [("survey-save.png", dict(sels=["text=Save this setup",
                                                       ".st-key-sv_apply_dl"], pad=26), [
        M(".st-key-sv_save_model"), M("blockquote"), M(".st-key-sv_apply_go"),
        M(".st-key-sv_apply_summary"), M(".st-key-sv_apply_table"),
        M(".st-key-sv_apply_dl")])],
    # Simple mode (a second app server, BEX_MODE=simple; sne staged by simple_demo)
    "simple-new.png": [("simple-start.png", dict(sels=[".st-key-bex_nav", "[data-testid='stForm']"], pad=26), [
        M(".st-key-eco_stepper"), M(".st-key-eco_pick"), M("[data-testid='stForm']"),
        M("button:has-text('Research mode')")])],
    "simple-species.png": [
        ("simple-species.png", dict(sels=[".st-key-eco_stepper", "text=Researcher checks"], pad=26,
                                    extra_bottom=430), [
            M("text=On this region's species list"), M("text=Researcher checks")]),
        ("simple-unexpected.png", dict(sels=["text=/Unexpected here:/"], pad=26, extra_bottom=330,
                                       full_width=True), [
            M("text=/Unexpected here:/")])],
    "simple-check.png": [("simple-check.png", dict(
        sels=[".st-key-eco_stepper", "text=Already have expert labels"], pad=26), [
        M("[data-testid='stMetric']"), M("button:has-text('checking ›')"), M(".pip-legend"),
        M("[class*='st-key-eco_row_']"), M("text=Already have expert labels")])],
    "simple-checking.png": [("simple-checking.png", dict(
        sels=[".st-key-eco_stepper", "button:has-text('Not sure')", "text=Compare with"], pad=26,
        extra_bottom=60), [
        M("[data-testid='stImage']"), M("text=/^Is this a /"), M("button:has-text('Undo last')", at="right"),
        M("text=Compare with")])],
    "simple-recall.png": [("simple-recall.png", dict(
        sels=[".st-key-eco_stepper", "button:has-text('Use this')"], pad=26), [
        M("[data-testid='stSlider']"), M("[data-testid='stMetric']"), M("text=Dashed lines"),
        M("text=/Model score floor/"), M("button:has-text('Use this')")])],
    "simple-export.png": [("simple-export.png", dict(
        sels=[".st-key-eco_stepper", "button:has-text('detections (CSV)')"], pad=26), [
        M("text=Detections file"), M("[data-testid='stDataFrame']"),
        M("button:has-text('detections (CSV)')"), M("text=EcoLabel volunteers · coming soon", at="right")])],
    "simple-soon.png": [("simple-soon.png", dict(
        sels=["text=EcoLabel is coming soon", ".st-key-eco_soon"], pad=26), [
        M("text=EcoLabel is coming soon"), M(".st-key-eco_soon")])],
}


def measure_spec() -> dict:
    """Every selector each capture needs measured, for capture.mjs."""
    out = {}
    for raw, outs in SHOTS.items():
        sels = set()
        for _, crop, marks in outs:
            sels.update(crop["sels"])
            for m in marks:
                sels.add(m["sel"])
                if m["ysel"]:
                    sels.add(m["ysel"])
        out[raw] = sorted(sels)
    return out


def demo_set(cfg) -> dict:
    """A label set made of SNE's expert labels for 9 sampled minutes, closed —
    so Progress has something to show — and the click points for a box over a
    real call in the next minute."""
    recs, ann = read_dataset(cfg.store_dir, DATASET)
    ls = L.create_set(cfg.store_dir, DATASET, SET, "Mara")
    ls.sample = L.draw_sample(recs, 20, seed=11)
    for r in ls.sample.head(9).iter_rows(named=True):
        a = ann.filter((pl.col("recording_id") == r["recording_id"])
                       & (pl.col("start_s") >= r["start_s"]) & (pl.col("start_s") < r["end_s"]))
        for j, b in enumerate(a.iter_rows(named=True)):
            L.add_box(ls, r["recording_id"], b["start_s"], b["end_s"], b["low_hz"],
                      b["high_hz"], b["species_key"], "Mara", assisted=(j % 4 == 0))
        L.close_chunk(ls, r["recording_id"], r["start_s"], r["end_s"], "Mara")
    L.save_set(cfg.store_dir, ls)
    nxt = L.next_chunk(ls)
    b = (ann.filter((pl.col("recording_id") == nxt["recording_id"])
                    & (pl.col("start_s") >= nxt["start_s"] + 6)
                    & (pl.col("end_s") < nxt["end_s"] - 6))
            .with_columns(d=pl.col("end_s") - pl.col("start_s"))
            .filter(pl.col("d").is_between(1, 6)).sort("d", descending=True).row(0, named=True))
    mel, meta = load_mel(cfg.cache_dir, DATASET, nxt["recording_id"])
    t0, t1, n = max(0, nxt["start_s"] - 5), nxt["end_s"] + 5, mel.shape[0]
    fx = lambda t: 0.06 + (t - t0) / (t1 - t0) * (0.985 - 0.06)              # noqa: E731
    fy = lambda hz: 1 - (0.12 + views.hz_to_mel_row(hz, meta) / n * (0.93 - 0.12))  # noqa: E731
    return {"x1": fx(b["start_s"]), "y1": fy(b["high_hz"]), "x2": fx(b["end_s"]),
            "y2": fy(b["low_hz"]), "set": SET, "setup": SETUP}


def simple_demo(cfg) -> None:
    """Simple mode part-way through on sne: the on-list species ticked and
    named, the busiest eight answered (5 Yes, 3 No each) and three more begun,
    every answer taken from SNE's expert labels, and a project description.
    sne's ecolabel folder is backed up first and put back by cleanup()."""
    from bex import ecolabel as E
    src = E.folder(cfg.store_dir, DATASET)
    shutil.rmtree(RAW / "ecolabel-backup", ignore_errors=True)
    shutil.copytree(src, RAW / "ecolabel-backup")
    s = E.load_settings(cfg.store_dir, DATASET)
    run = s["run_id"] or E.runs_for(cfg.store_dir, DATASET)[0]["run_id"]
    pool = E.candidate_pool(cfg.store_dir, DATASET, run)
    tiers, pdir = E._tiers(cfg, DATASET)
    sp = E.species_table(pool, tiers, E.common_names(cfg.store_dir, pdir), {**s, "included": None})
    keep = sp.filter(pl.col("plausible"), pl.col("name").str.len_chars() > 0)
    _, ann = read_dataset(cfg.store_dir, DATASET)
    real = (pool.join(ann.select("recording_id", "species_key", b0="start_s", b1="end_s"),
                      on=["recording_id", "species_key"])
            .filter(pl.min_horizontal("end_s", "b1") - pl.max_horizontal("start_s", "b0") >= 0.5)
            .select(E.KEY).unique().with_columns(real=pl.lit(True)))
    cand = pool.join(real, on=E.KEY, how="left").with_columns(pl.col("real").fill_null(False))
    (src / "checks.parquet").unlink(missing_ok=True)
    for i, k in enumerate(keep["species_key"].head(11)):
        need = {"yes": 5, "no": 3} if i < 8 else {"yes": 2, "no": 1}
        for r in cand.filter(pl.col("species_key") == k).sort("score", descending=True).iter_rows(named=True):
            v = "yes" if r["real"] else "no"
            if need[v]:
                E.add_check(cfg.store_dir, DATASET, r["recording_id"], r["start_s"], k, v, labeller="Mara")
                need[v] -= 1
            if not any(need.values()):
                break
    about = s["about"]
    about["project"].update(title="Dawn chorus, Sierra Nevada (BEX demo)",
                            summary="Which birds sing at dawn in the Sierra Nevada mountains of California.",
                            why="To see how bird life changes up the mountain.")
    about["group"].update(name="BEX demo", kind="Example project")
    about["site"].update(name="Sierra Nevada, California", habitat="mixed conifer forest",
                         when="Mornings in May and June 2018")
    E.save_settings(cfg.store_dir, DATASET, {**s, "run_id": run, "floor": None, "target_recall": 0.8,
                                             "included": keep["species_key"].to_list(),
                                             "names": dict(zip(keep["species_key"], keep["name"])),
                                             "about": about})
    print(f"simple demo: {keep.height} species, {E.load_checks(cfg.store_dir, DATASET).height} answers")


def annotate(pts: dict) -> None:
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    for raw, outs in SHOTS.items():
        boxes = json.loads((RAW / f"{raw}.boxes.json").read_text())
        for out, crop, marks in outs:
            im = Image.open(RAW / raw).convert("RGB")
            d = ImageDraw.Draw(im)
            for n, m in enumerate(marks, start=1):
                b = boxes.get(m["sel"])
                if b is None:
                    print(f"  {out}: marker {n} not placed — {m['sel']!r} not on the page")
                    continue
                if m["fx"] is not None:
                    fx = pts["x1"] if m["fx"] == PTS else m["fx"]
                    fy = pts["y1"] if m["fy"] == PTS else m["fy"]
                    x, y = b["x"] + b["width"] * fx, b["y"] + b["height"] * fy
                else:
                    yb = boxes.get(m["ysel"]) if m["ysel"] else b
                    yb = yb or b
                    y = yb["y"] + yb["height"] / 2
                    x = b["x"] - 18 if m["at"] == "left" else b["x"] + b["width"] + 18
                x, y = x + m["dx"], y + m["dy"]
                d.ellipse((x - 14, y - 14, x + 14, y + 14), fill=TEAL,
                          outline=(255, 255, 255), width=2)
                d.text((x, y + 1), str(n), fill=(255, 255, 255), font=font, anchor="mm")
            got = [boxes[s_] for s_ in crop["sels"] if boxes.get(s_)]
            pad = crop.get("pad", 20)
            x0 = crop.get("x0", min(g["x"] for g in got) - pad - 20)
            y0 = min(g["y"] for g in got) - pad
            x1 = max(g["x"] + g["width"] for g in got) + pad
            y1 = max(g["y"] + g["height"] for g in got) + pad + crop.get("extra_bottom", 0)
            if crop.get("full_width"):
                x1 = im.width - 40
            box = (max(0, int(x0)), max(0, int(y0)), min(im.width, int(x1)),
                   min(im.height, int(y1)))
            im.crop(box).save(DST / out, optimize=True)
            print("wrote", DST / out, box)


def cleanup(cfg) -> None:
    backup = RAW / "ecolabel-backup"
    if backup.exists():
        dst = Path(cfg.store_dir) / "datasets" / DATASET / "ecolabel"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(backup, dst)
        shutil.rmtree(backup)
    shutil.rmtree(L.set_dir(cfg.store_dir, DATASET, SET), ignore_errors=True)
    for p in (Path(cfg.store_dir) / "truth" / DATASET / "sets").glob(f"{SET}@*"):
        shutil.rmtree(p, ignore_errors=True)
    (Path(cfg.store_dir) / "setups" / f"{SETUP}.json").unlink(missing_ok=True)
    for p in (Path(cfg.store_dir) / "datasets" / DATASET / "results").glob(f"{SETUP}.*"):
        p.unlink()
    for d in (L.labels_dir(cfg.store_dir, DATASET), Path(cfg.store_dir) / "setups",
              Path(cfg.store_dir) / "datasets" / DATASET / "results"):
        if d.exists() and not any(d.iterdir()):
            d.rmdir()


def main() -> int:
    cfg = load_config()
    RAW.mkdir(exist_ok=True)
    prefs = Path(cfg.store_dir) / "ui_prefs.json"
    prefs_before = prefs.read_text() if prefs.exists() else None
    def serve(port: int, mode: str) -> subprocess.Popen:
        return subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(REPO / "app.py"),
                                 "--server.port", str(port), "--server.headless", "true"],
                                cwd=REPO, env={**os.environ, "BEX_MODE": mode},
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    server, simple = serve(PORT, "research"), None
    try:
        time.sleep(6)
        (RAW / "measure.json").write_text(json.dumps(measure_spec(), indent=1))
        node = ["node", str(HERE / "capture.mjs")]
        subprocess.run([*node, "home", str(PORT), str(RAW)], cwd=HERE, check=True)
        pts = demo_set(cfg)
        (RAW / "pts.json").write_text(json.dumps(pts))
        subprocess.run([*node, "rest", str(PORT), str(RAW)], cwd=HERE, check=True)
        simple_demo(cfg)
        simple = serve(PORT + 1, "simple")
        time.sleep(6)
        subprocess.run([*node, "simple", str(PORT + 1), str(RAW)], cwd=HERE, check=True)
        annotate(pts)
    finally:
        server.terminate()
        if simple:
            simple.terminate()
        cleanup(cfg)
        if prefs_before is None:
            prefs.unlink(missing_ok=True)
        else:
            prefs.write_text(prefs_before)
    return 0


if __name__ == "__main__":
    sys.exit(main())
