#!/usr/bin/env python3
"""Reproduce the evaluation tables of the paper from the released data.

Steps:
  1. Recompute Configurations 1–6 from the stored raw detector outputs
     (data/raw/raw_detector_outputs.json) with the rules in shohon_bbox.rules, and check that they
     equal the released predictions (data/predictions/config1–6_*.csv) box for box.
     Configuration 6 uses the page-level presence labels of the companion dataset
     (canvases.tsv, Zenodo version 1.2, https://doi.org/10.5281/zenodo.24152375), which is
     downloaded once into paper/.cache/ and checked against its MD5.
  2. Score Configurations 1–7 against the ground truth (data/gold/gold102.csv) and check that
     every value in the summary equals paper/expected_summary.tsv (the values in the paper) to
     three decimals. The mean IoU of matched boxes is not compared (see SKIPPED).
  3. Section 4.3: redraw the 74 images without a ruler or a chart from canvases.tsv and check that
     they equal data/negatives/images.csv; count the false detections of Configurations 1–6 on
     them from data/negatives/raw_detector_outputs.json and check the table against
     paper/expected_negatives.md. Also check that the cross-validation folds of Configuration 7
     follow from the ground truth (the cv_fold column of data/images.csv).

Usage:
  python paper/reproduce.py [--data data] [--canvases path/to/canvases.tsv] [--out results/]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from shohon_bbox import evaluate, rules  # noqa: E402

CANVASES_URL = "https://zenodo.org/api/records/24152375/files/canvases.tsv/content"
CANVASES_MD5 = "e7042f5d69bec0484ed8c08ce1e6529a"
CONFIGS = [(1, "gdino"), (2, "owlv2"), (3, "ndl_layout"), (4, "ensemble_raw"), (5, "proposed"),
           (6, "proposed_meta"), (7, "yolo11x")]


def read_boxes(path):
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        d.setdefault(r["key"], []).append({"cls": r["class"], "label": r["class"],
                                           "box": [float(r[c]) for c in ("x1", "y1", "x2", "y2")],
                                           "score": float(r["score"]) if r["score"] else 1.0})
    return d


def canvases_file(path):
    if not path:
        path = os.path.join(HERE, ".cache", "canvases.tsv")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print("downloading canvases.tsv (26 MB) from Zenodo ...", flush=True)
            urllib.request.urlretrieve(CANVASES_URL, path)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    if md5 != CANVASES_MD5:
        raise SystemExit(f"{path}: md5 {md5} is not that of Zenodo version 1.2 ({CANVASES_MD5})")
    return path


def presence_labels(path, keys):
    def flag(v):
        return {"true": True, "false": False}.get(str(v).strip().lower())
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
        k = f"{r['bid']}-{int(r['canvas_index']):04d}"
        if k in keys:
            out[k] = {"ruler": flag(r["ruler"]), "colour_chart": flag(r["colorchart"])}
    return out


def clip(box, w, h):
    """The released CSV files clip boxes to the image; the paper scored the unclipped boxes."""
    x1, y1, x2, y2 = box
    return [round(min(max(x1, 0.0), w), 1) + 0.0, round(min(max(y1, 0.0), h), 1) + 0.0,
            round(min(max(x2, 0.0), w), 1) + 0.0, round(min(max(y2, 0.0), h), 1) + 0.0]


def check_configs(data, canvases):
    raw = json.load(open(os.path.join(data, "raw", "raw_detector_outputs.json"), encoding="utf-8"))
    presence = presence_labels(canvases, set(raw))
    bad = 0
    recomputed = {}
    for num, name in CONFIGS[:6]:
        released = read_boxes(os.path.join(data, "predictions", f"config{num}_{name}.csv"))
        for k, e in raw.items():
            boxes = rules.all_configs(e, presence.get(k))[name]
            recomputed.setdefault(name, {})[k] = {"boxes": boxes}
            mine = sorted((b["cls"], round(b["score"], 4), clip(b["box"], e["w"], e["h"])) for b in boxes)
            theirs = sorted((b["cls"], round(b["score"], 4), [round(v, 1) for v in b["box"]]) for b in released.get(k, []))
            if mine != theirs:
                bad += 1
                print(f"  config {num} differs on {k}")
    print(f"step 1: configurations 1–6 recomputed for {len(raw)} images: "
          + ("all equal to the released predictions" if not bad else f"{bad} differences"))
    return bad == 0, recomputed


# The released ground truth clips 24 boxes (on 21 images) that extend a few pixels beyond the image edge (the paper
# scored the unclipped boxes). This changes the mean IoU of matched boxes in the fourth decimal and
# nothing that the paper reports, so mean_iou_matched is not compared.
SKIPPED = ("mean_iou_matched",)


def _r3(v):
    """Values as printed in the paper (three decimals)."""
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return v


def check_scores(data, recomputed, out):
    gold = read_boxes(os.path.join(data, "gold", "gold102.csv"))
    keys = list(recomputed["proposed"])
    pred = dict(recomputed)  # Configurations 1–6 as recomputed (unclipped), as scored in the paper
    d = read_boxes(os.path.join(data, "predictions", "config7_yolo11x.csv"))
    pred["yolo11x"] = {k: {"boxes": d.get(k, [])} for k in keys}
    with tempfile.TemporaryDirectory() as tmp:
        gp, pp = os.path.join(tmp, "gold.json"), os.path.join(tmp, "pred.json")
        json.dump(gold, open(gp, "w"))
        json.dump(pred, open(pp, "w"))
        stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            evaluate.main(["--pred", pp, "--gold", gp, "--out", out])
        finally:
            sys.stdout.close()
            sys.stdout = stdout
    got = {(r["method"], r["class"]): r for r in csv.DictReader(open(os.path.join(out, "summary.tsv")), delimiter="\t")}
    diff = 0
    for r in csv.DictReader(open(os.path.join(HERE, "expected_summary.tsv")), delimiter="\t"):
        g = got.get((r["method"], r["class"]))
        for c, v in r.items():
            if c in ("method", "class"):
                continue
            if c == "mean_iou_matched":  # not reported in the paper; see SKIPPED below
                continue
            if g is None or _r3(g.get(c)) != _r3(v):
                diff += 1
                print(f"  {r['method']} {r['class']} {c}: paper {v}, now {g.get(c) if g else None}")
    print("step 2: scores of configurations 1–7: "
          + ("every value equals the paper's (to three decimals)" if not diff else f"{diff} values differ"))
    for name in [n for _, n in CONFIGS]:
        print(f"  {name:14s} F1@0.50 {got[(name, 'ALL_MICRO')]['f1_50']}")
    return diff == 0


def check_negatives(data, canvases):
    import negatives
    import yolo_cv
    neg = os.path.join(data, "negatives")
    pick, _, _ = negatives.draw(canvases, negatives.excluded_books())
    released = [[r[c] for c in negatives.COLUMNS] for r in csv.DictReader(open(os.path.join(neg, "images.csv"), encoding="utf-8"))]
    ok_list = pick == released
    raw = json.load(open(os.path.join(neg, "raw_detector_outputs.json"), encoding="utf-8"))
    got = negatives.table(raw, [r[0] for r in released])
    ok_table = got.strip() == open(os.path.join(HERE, "expected_negatives.md"), encoding="utf-8").read().strip()
    ok_folds = yolo_cv.make_folds(yolo_cv.read_gold(data)) == yolo_cv.released_folds(data)
    print("step 3: images without a ruler or a chart: list "
          + ("equals data/negatives/images.csv" if ok_list else "DIFFERS")
          + "; table " + ("equals the paper's" if ok_table else "DIFFERS")
          + "; cross-validation folds " + ("equal data/images.csv" if ok_folds else "DIFFER"))
    if not ok_table:
        print(got)
    return ok_list and ok_table and ok_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--canvases", help="canvases.tsv of Zenodo version 1.2 (downloaded if omitted)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    canvases = canvases_file(a.canvases)
    ok1, recomputed = check_configs(a.data, canvases)
    ok2 = check_scores(a.data, recomputed, a.out)
    ok3 = check_negatives(a.data, canvases)
    print(f"report: {os.path.join(a.out, 'report.md')}")
    sys.exit(0 if ok1 and ok2 and ok3 else 1)


if __name__ == "__main__":
    main()
