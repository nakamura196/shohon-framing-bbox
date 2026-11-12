"""The rules reproduce the released Configurations 1–5 from the stored raw detector outputs."""
import csv
import json
import os

import pytest

from shohon_bbox import rules

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW = json.load(open(os.path.join(DATA, "raw", "raw_detector_outputs.json"), encoding="utf-8"))


def clip(box, w, h):
    return [round(min(max(v, 0.0), lim), 1) + 0.0 for v, lim in zip(box, (w, h, w, h))]


def released(name):
    d = {}
    for r in csv.DictReader(open(os.path.join(DATA, "predictions", name), encoding="utf-8")):
        d.setdefault(r["key"], []).append((r["class"], float(r["score"]), [float(r[c]) for c in ("x1", "y1", "x2", "y2")]))
    return {k: sorted(v) for k, v in d.items()}


@pytest.mark.parametrize("num,name", [(1, "gdino"), (2, "owlv2"), (3, "ndl_layout"), (4, "ensemble_raw"), (5, "proposed")])
def test_configuration_matches_release(num, name):
    ref = released(f"config{num}_{name}.csv")
    for k, e in RAW.items():
        got = sorted((b["cls"], b["score"], clip(b["box"], e["w"], e["h"])) for b in rules.all_configs(e, None)[name])
        assert got == ref.get(k, []), k


def test_proposed_has_at_most_one_book_and_chart():
    for e in RAW.values():
        out = rules.all_configs(e, None)["proposed"]
        assert sum(b["cls"] == "book" for b in out) <= 1
        assert sum(b["cls"] == "colour_chart" for b in out) <= 1
        assert sum(b["cls"] == "ruler" for b in out) <= 2


def test_presence_false_removes_low_score_targets():
    for e in RAW.values():
        out = rules.all_configs(e, {"ruler": False, "colour_chart": False})["proposed_meta"]
        assert all(b["score"] >= 0.85 for b in out if b["cls"] == "ruler")
        assert all(b["score"] >= 0.80 for b in out if b["cls"] == "colour_chart")


def test_geometry():
    assert rules.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert rules.iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert rules.ioa([0, 0, 10, 10], [0, 0, 20, 20]) == 1.0
    assert rules.aspect([0, 0, 40, 10]) == 4.0
