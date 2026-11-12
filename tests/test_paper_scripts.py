import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paper"))
DATA = os.path.join(ROOT, "data")

import negatives  # noqa: E402
import yolo_cv  # noqa: E402


def test_folds_follow_from_ground_truth():
    folds = yolo_cv.make_folds(yolo_cv.read_gold(DATA))
    assert folds == yolo_cv.released_folds(DATA)
    assert sorted(set(folds.values())) == [0, 1, 2, 3, 4]


def test_negatives_table_equals_paper():
    raw = json.load(open(os.path.join(DATA, "negatives", "raw_detector_outputs.json"), encoding="utf-8"))
    keys = [r["key"] for r in csv.DictReader(open(os.path.join(DATA, "negatives", "images.csv"), encoding="utf-8"))]
    assert len(keys) == 74
    want = open(os.path.join(ROOT, "paper", "expected_negatives.md"), encoding="utf-8").read()
    assert negatives.table(raw, keys).strip() == want.strip()
