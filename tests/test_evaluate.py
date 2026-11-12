import csv
import json
import os

from shohon_bbox import evaluate


def test_perfect_prediction(tmp_path):
    gold = {"a-0001": [{"cls": "book", "box": [0, 0, 100, 100]}, {"cls": "ruler", "box": [0, 100, 100, 110]}]}
    pred = {"proposed": {"a-0001": {"boxes": [dict(b, score=0.9) for b in gold["a-0001"]]}}}
    json.dump(gold, open(tmp_path / "g.json", "w"))
    json.dump(pred, open(tmp_path / "p.json", "w"))
    evaluate.main(["--pred", str(tmp_path / "p.json"), "--gold", str(tmp_path / "g.json"), "--out", str(tmp_path), "--bootstrap", "10"])
    rows = {(r["method"], r["class"]): r for r in csv.DictReader(open(os.path.join(tmp_path, "summary.tsv")), delimiter="\t")}
    assert float(rows[("proposed", "ALL_MICRO")]["f1_50"]) == 1.0


def test_missing_and_extra(tmp_path):
    gold = {"a-0001": [{"cls": "book", "box": [0, 0, 100, 100]}, {"cls": "ruler", "box": [0, 100, 100, 110]}]}
    pred = {"proposed": {"a-0001": {"boxes": [{"cls": "book", "box": [0, 0, 100, 100], "score": 0.9},
                                              {"cls": "colour_chart", "box": [200, 200, 220, 220], "score": 0.5}]}}}
    json.dump(gold, open(tmp_path / "g.json", "w"))
    json.dump(pred, open(tmp_path / "p.json", "w"))
    evaluate.main(["--pred", str(tmp_path / "p.json"), "--gold", str(tmp_path / "g.json"), "--out", str(tmp_path), "--bootstrap", "10"])
    rows = {(r["method"], r["class"]): r for r in csv.DictReader(open(os.path.join(tmp_path, "summary.tsv")), delimiter="\t")}
    assert float(rows[("proposed", "ALL_MICRO")]["f1_50"]) == 0.5
