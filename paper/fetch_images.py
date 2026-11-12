#!/usr/bin/env python3
"""fetch_images.py — download the 102 evaluation images from their IIIF servers.

For each row of data/images.csv: download `image_url`, check it against `image_sha256`
(or, for the books whose files change byte-wise between deliveries, against `pixel_sha256`,
the SHA-256 of the decoded RGB pixels), and write <out>/<key>.jpg resized to
`eval_width` x `eval_height`, the size used for detection and evaluation.

The author's evaluation images were reduced from the same downloads with other software, so the
resized files are not byte-identical; detector outputs on them can differ slightly from
data/raw/raw_detector_outputs.json. Licences differ per image (column `license`); the images are not
redistributed with this repository.

Usage:
  python paper/fetch_images.py --out images/ [--images data/images.csv]
  shohon-bbox detect images/<key>.jpg --raw   # already at evaluation size, so not reduced further
"""
import argparse
import csv
import hashlib
import io
import os
import time
import urllib.request

from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(HERE, "data", "images.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bad = []
    for r in csv.DictReader(open(a.images, encoding="utf-8")):
        dst = os.path.join(a.out, f"{r['key']}.jpg")
        if os.path.exists(dst):
            continue
        req = urllib.request.Request(r["image_url"], headers={"User-Agent": "shohon-framing-bbox (https://github.com/nakamura196/shohon-framing-bbox)"})
        data = urllib.request.urlopen(req, timeout=120).read()
        im = Image.open(io.BytesIO(data)).convert("RGB")
        ok = (hashlib.sha256(data).hexdigest() == r["image_sha256"]
              or hashlib.sha256(im.tobytes()).hexdigest() == r["pixel_sha256"])
        if not ok:
            bad.append(r["key"])
        im.resize((int(r["eval_width"]), int(r["eval_height"])), Image.LANCZOS).save(dst, quality=95)
        print(r["key"], "ok" if ok else "HASH MISMATCH", flush=True)
        time.sleep(a.sleep)
    if bad:
        print(f"{len(bad)} images differ from the files used in the paper:", *bad)


if __name__ == "__main__":
    main()
