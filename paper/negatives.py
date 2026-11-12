#!/usr/bin/env python3
"""False detections on images that contain neither a ruler nor a colour chart (the paper's Section 4.3).

The 102 evaluation images all contain a ruler or a chart. To see what the configurations output on
images without either, 74 images were drawn from the companion dataset (canvases.tsv of Zenodo
version 1.2, https://doi.org/10.5281/zenodo.24152375): images whose `ruler` and `colorchart` are both
`false` with source `human` (checked by the author), excluding the 300 books of the paper's sample
(data/negatives/excluded_books.txt), one image per book (seed 20261112). The list, with the IIIF
image URLs, is data/negatives/images.csv; the raw detector outputs on the images, reduced to a
longer side of 1,400 pixels, are data/negatives/raw_detector_outputs.json.

  python paper/negatives.py select --canvases paper/.cache/canvases.tsv --out results/negatives.csv
  python paper/negatives.py fetch --dir neg_images/
  python paper/negatives.py detect --dir neg_images/ --out results/negatives_raw.json
  python paper/negatives.py count [--raw data/negatives/raw_detector_outputs.json]

`select` redraws the list (it equals data/negatives/images.csv); `count` prints the table of the
paper from the raw outputs. paper/reproduce.py runs `select` and `count` and compares them with the
released list and with paper/expected_negatives.md.
"""
import argparse
import csv
import io
import json
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from shohon_bbox import rules  # noqa: E402

NEG = os.path.join(ROOT, "data", "negatives")
SEED = 20261112
COLUMNS = ["key", "bid", "canvas_index", "work", "page_type", "image_url"]
NAMES = [("(1)", "gdino"), ("(2)", "owlv2"), ("(3)", "ndl_layout"), ("(4)", "ensemble_raw"), ("(5)", "proposed"),
         ("(6)", "proposed_meta")]


def draw(canvases, excluded):
    rows = [r for r in csv.DictReader(open(canvases, encoding="utf-8"), delimiter="\t")
            if r["ruler"] == "false" and r["colorchart"] == "false"
            and r["ruler_source"] == "human" and r["colorchart_source"] == "human" and r["bid"] not in excluded]
    by_book = {}
    for r in rows:
        by_book.setdefault(r["bid"], []).append(r)
    rng = random.Random(SEED)
    pick = [rng.choice(sorted(v, key=lambda r: int(r["canvas_index"]))) for _, v in sorted(by_book.items())]
    return [[f"{r['bid']}-{int(r['canvas_index']):04d}", r["bid"], r["canvas_index"], r["work"], r["page_type"],
             r["image_url"]] for r in pick], len(rows), len(by_book)


def excluded_books():
    return set(open(os.path.join(NEG, "excluded_books.txt"), encoding="utf-8").read().split())


def select(a):
    pick, n_rows, n_books = draw(a.canvases, excluded_books())
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(pick)
    print(f"{n_rows} candidate images in {n_books} books; wrote {len(pick)} to {a.out}")


def fetch(a):
    from PIL import Image
    from shohon_bbox.iiif import USER_AGENT, reduce
    os.makedirs(a.dir, exist_ok=True)
    for r in csv.DictReader(open(a.list, encoding="utf-8")):
        dst = os.path.join(a.dir, f"{r['key']}.jpg")
        if os.path.exists(dst):
            continue
        req = urllib.request.Request(r["image_url"], headers={"User-Agent": USER_AGENT})
        im = Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=120).read())).convert("RGB")
        reduce(im, 1400).save(dst, quality=95)
        print(r["key"], im.size, flush=True)
        time.sleep(1.0)


def detect(a):
    from PIL import Image
    from shohon_bbox.detectors import Detectors
    det = Detectors()
    out = {}
    for r in csv.DictReader(open(a.list, encoding="utf-8")):
        out[r["key"]] = det.run(Image.open(os.path.join(a.dir, f"{r['key']}.jpg")).convert("RGB"))
        print(r["key"], flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False)


def table(raw, keys):
    absent = {"ruler": False, "colour_chart": False}
    lines = [f"{len(keys)} images without a ruler or a chart (author-checked)", "",
             "| config | images with a chart | with a ruler | with a slip | with any of the three | "
             "boxes of the three | images with one book box |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    detail = {}
    for lab, n in NAMES:
        c = {"colour_chart": 0, "ruler": 0, "imaging_slip": 0}
        anyimg = boxes = onebook = 0
        for k in keys:
            cfg = rules.all_configs(raw[k], absent)[n]
            got = {x["cls"] for x in cfg}
            for cl in c:
                c[cl] += cl in got
            t = [x for x in cfg if x["cls"] != "book"]
            anyimg += bool(t)
            boxes += len(t)
            onebook += sum(x["cls"] == "book" for x in cfg) == 1
            if n == "proposed" and t:
                detail[k] = [(x["cls"], x["score"], x["source"]) for x in t]
        N = len(keys)
        lines.append(f"| {lab} {n} | {c['colour_chart']} ({c['colour_chart'] / N:.0%}) | {c['ruler']} ({c['ruler'] / N:.0%}) | "
                     f"{c['imaging_slip']} ({c['imaging_slip'] / N:.0%}) | {anyimg} ({anyimg / N:.0%}) | {boxes} | {onebook} |")
    lines += ["", "Configuration 5, images with a chart, ruler or slip box:", ""]
    for k, v in detail.items():
        lines.append(f"- {k}: " + ", ".join(f"{c} {s:.2f} ({src})" for c, s, src in v))
    return "\n".join(lines) + "\n"


def count(a):
    raw = json.load(open(a.raw, encoding="utf-8"))
    keys = [r["key"] for r in csv.DictReader(open(a.list, encoding="utf-8"))]
    print(table(raw, keys), end="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "fetch", "detect", "count"])
    ap.add_argument("--canvases", help="canvases.tsv of Zenodo version 1.2")
    ap.add_argument("--out")
    ap.add_argument("--list", default=os.path.join(NEG, "images.csv"))
    ap.add_argument("--dir", help="folder of the reduced images")
    ap.add_argument("--raw", default=os.path.join(NEG, "raw_detector_outputs.json"))
    a = ap.parse_args()
    {"select": select, "fetch": fetch, "detect": detect, "count": count}[a.step](a)


if __name__ == "__main__":
    main()
