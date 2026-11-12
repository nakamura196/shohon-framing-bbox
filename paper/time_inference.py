#!/usr/bin/env python3
"""Inference time per image of the three detectors (the paper's Section 4.4).

Runs Grounding DINO, OWLv2 and YOLOv8-NDL-Layout with the weights, prompts and thresholds of
shohon_bbox.detectors on the 102 evaluation images, one image at a time, on the CPU with 8 torch
threads, and records the time of each detector. The first three images are warm-up runs and are not
measured. The script also counts the images on which the output equals the stored raw output
(data/raw/raw_detector_outputs.json; same labels, coordinates within 1 pixel).

The paper's run (2026-11-11, Apple M2 Max, CPU): Python 3.12.13, torch 2.15.0, transformers 5.22.1
for Grounding DINO and OWLv2 (mean 2.16 s and 1.78 s); ultralytics 8.4.7 for YOLOv8-NDL-Layout.

Usage:
  python paper/fetch_images.py --out images/
  python paper/time_inference.py --images-dir images/ [--out results/timing.json]
"""
import argparse
import csv
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
WARMUP = 3


def same(a, b, tol=1.0):
    if len(a) != len(b):
        return False
    sa = sorted(a, key=lambda x: (x["label"], x["box"]))
    sb = sorted(b, key=lambda x: (x["label"], x["box"]))
    return all(x["label"] == y["label"] and max(abs(p - q) for p, q in zip(x["box"], y["box"])) <= tol
               for x, y in zip(sa, sb))


def summary(ts):
    ts = sorted(ts)
    n = len(ts)
    return {"n": n, "mean": sum(ts) / n, "median": ts[n // 2], "min": ts[0], "max": ts[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="folder of <key>.jpg (paper/fetch_images.py)")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "timing.json"))
    a = ap.parse_args()

    import torch
    import transformers
    import ultralytics
    from PIL import Image
    from shohon_bbox.detectors import Detectors

    keys = [r["key"] for r in csv.DictReader(open(os.path.join(a.data, "images.csv"), encoding="utf-8"))]
    stored = json.load(open(os.path.join(a.data, "raw", "raw_detector_outputs.json"), encoding="utf-8"))
    det = Detectors(threads=a.threads)
    names = ("gdino", "owlv2", "ndl_layout")
    times = {n: [] for n in names}
    match = {n: 0 for n in names}
    for i, k in enumerate(keys):
        im = Image.open(os.path.join(a.images_dir, f"{k}.jpg")).convert("RGB")
        line = []
        for n in names:
            t0 = time.perf_counter()
            boxes = getattr(det, n)(im)
            dt = time.perf_counter() - t0
            match[n] += same(boxes, stored[k][n])
            if i >= WARMUP:
                times[n].append(dt)
            line.append(f"{n}={dt:.2f}s")
        print(f"{i + 1}/{len(keys)} {k} " + " ".join(line), flush=True)
    res = {n: summary(times[n]) for n in names}
    res.update({"same_as_stored": dict(match, of=len(keys)), "device": "cpu", "threads": torch.get_num_threads(),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "ultralytics": ultralytics.__version__, "python": platform.python_version(),
                "machine": platform.machine(), "processor": platform.processor(), "images": len(keys),
                "warmup_excluded": WARMUP, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
