"""Survey setups: saved, reloaded, and applied to unlabelled recordings (V1.2 F4)."""
import polars as pl

from bex import labels as L
from bex import setups as S
from bex import thresholds as th

ROBIN, WREN, OWL = "Turdus migratorius", "Troglodytes troglodytes", "Strix varia"


def det(rows):
    return pl.DataFrame([{"recording_id": r, "start_s": float(t), "species_key": k,
                          "score_raw": s, "implausible": False} for r, t, k, s in rows])


def setup():
    bars = {"p0.9": th.Resolved("toy", pl.DataFrame(
        {"species_key": [ROBIN, WREN, OWL], "theta": [0.5, 0.5, 0.5],
         "source": ["rule"] * 3, "reason": [""] * 3}), None, "rule")}
    return S.Setup(name="toy-setup", model="toy", model_label="Toy",
                   rule={"check": {"floor": 0.9, "k": 1, "within_s": None},
                         "firm": {"floor": 0.9, "k": 2, "within_s": None}},
                   bars=S.bars_payload(bars, {"p0.9"}), judge="profile", profile="p",
                   fitted_on={"dataset": "sne", "truth_ref": "imported"},
                   performance={"score": {"firm_precision": 0.93, "recall_reviewed": 0.7},
                                "intervals": {"firm_precision": [0.88, 0.97],
                                              "recall_reviewed": [0.6, 0.8]},
                                "recordings": 12, "hours": 12.0, "on": "test"})


def test_round_trip_and_claim(tmp_path):
    S.save(tmp_path, setup())
    assert S.list_setups(tmp_path) == ["toy-setup"]
    s = S.load(tmp_path, "toy-setup")
    assert s.survey_rule().firm.k == 2
    assert "93% (88%–97%)" in s.claim() and "12 test recording" in s.claim()


def test_apply_lists_model_output_and_prefers_labels_where_complete():
    d = det([("u1", 0, ROBIN, 0.9), ("u1", 3, ROBIN, 0.8), ("u1", 6, WREN, 0.6),
             ("l1", 0, ROBIN, 0.9), ("l1", 3, ROBIN, 0.9), ("l1", 0, OWL, 0.2)])
    ann = pl.DataFrame({"recording_id": ["l1"], "start_s": [1.0], "end_s": [2.0],
                        "low_hz": [1.0], "high_hz": [2.0], "species_key": [OWL]})
    truth = L.Truth("x", ann, pl.DataFrame(), ["l1"], {"l1": 60.0}, False)
    out = S.apply(setup(), d, ["u1", "l1"], truth)
    u = {r["species_key"]: r for r in out.filter(pl.col("recording_id") == "u1").to_dicts()}
    assert u[ROBIN]["listed"] == "firm" and u[WREN]["listed"] == "check"
    assert {r["source"] for r in u.values()} == {"model output"}
    lab = out.filter(pl.col("recording_id") == "l1").to_dicts()
    # The labels are the list for a labelled recording: the owl the model scored
    # too low is present, and the robin it was sure of is not on the list.
    assert [(r["species_key"], r["listed"], r["source"], r["model_tier"]) for r in lab] == \
        [(OWL, "present", "labelled", "not found")]
