#!/usr/bin/env python3
"""Configuration 7 of the paper: YOLO11x trained on the 102 verified images, 5-fold cross-validation.

The 102 images come from 102 different books, so the folds are disjoint by book. Images are
stratified by the presence of an imaging slip, a colour chart and a ruler in the ground truth and
dealt to the folds in turn (seed 20261111); the folds are the `cv_fold` column of data/images.csv.
The hyperparameters (HP below) were fixed before training and not tuned on the test folds; no
validation split is used.

Steps:
  python paper/yolo_cv.py folds                       # recompute the folds; check them against data/images.csv
  python paper/fetch_images.py --out images/
  python paper/yolo_cv.py prepare --images-dir images/ --kit kit/   # images, YOLO labels, folds.tsv
  # on a GPU machine (pip install "ultralytics>=8.4,<9"):
  python paper/yolo_cv.py train --kit kit/ --weights yolo11x.pt --tag yolo11x --device 0
  shohon-bbox evaluate --pred kit/oof_yolo11x.json --gold data/gold/gold102.csv --out results/yolo

The paper's run (2026-11-12) used an NVIDIA A100-SXM4-40GB, ultralytics 8.4.119 and torch
2.13.0+cu130 (data/predictions/config7_yolo11x_meta.json); training is not bit-reproducible on
other hardware or library versions. Its out-of-fold predictions are data/predictions/config7_yolo11x.csv.
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CLASSES = ["book", "colour_chart", "ruler", "imaging_slip"]
SEED = 20261111
N_FOLDS = 5
# Fixed before training. nbs=8 updates the weights after every batch (the default nbs=64 accumulates 64
# images, which leaves about 140 updates for 82 training images; a first run with it under-fitted).
HP = dict(epochs=200, imgsz=1024, batch=8, nbs=8, conf_op=0.25, conf_raw=0.05, nms_iou=0.50)


def read_gold(data):
    gold = {}
    for r in csv.DictReader(open(os.path.join(data, "gold", "gold102.csv"), encoding="utf-8")):
        gold.setdefault(r["key"], []).append({"cls": r["class"], "box": [float(r[c]) for c in ("x1", "y1", "x2", "y2")]})
    return gold


def make_folds(gold):
    """{key: fold} from the presence of (imaging_slip, colour_chart, ruler) in the ground truth."""
    strata = {}
    for k, boxes in gold.items():
        has = tuple(int(any(b["cls"] == c for b in boxes)) for c in ("imaging_slip", "colour_chart", "ruler"))
        strata.setdefault(has, []).append(k)
    rng = random.Random(SEED)
    order = []
    for s in sorted(strata, reverse=True):
        ks = sorted(strata[s])
        rng.shuffle(ks)
        order += ks
    return {k: i % N_FOLDS for i, k in enumerate(order)}


def released_folds(data):
    return {r["key"]: int(r["cv_fold"]) for r in csv.DictReader(open(os.path.join(data, "images.csv"), encoding="utf-8"))}


def cmd_folds(a):
    gold = read_gold(a.data)
    folds = make_folds(gold)
    rel = released_folds(a.data)
    print("fold sizes", sorted(Counter(folds.values()).items()))
    for fk in range(N_FOLDS):
        print(fk, dict(Counter(b["cls"] for k in folds if folds[k] == fk for b in gold[k])))
    ok = folds == rel
    print("folds equal the cv_fold column of data/images.csv" if ok else "folds DIFFER from data/images.csv")
    sys.exit(0 if ok else 1)


def cmd_prepare(a):
    from PIL import Image
    gold = read_gold(a.data)
    folds = make_folds(gold)
    assert folds == released_folds(a.data), "folds differ from data/images.csv"
    os.makedirs(os.path.join(a.kit, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.kit, "labels"), exist_ok=True)
    for k, boxes in gold.items():
        src = os.path.join(a.images_dir, f"{k}.jpg")
        w, h = Image.open(src).size
        shutil.copy2(src, os.path.join(a.kit, "images", f"{k}.jpg"))
        lines = []
        for b in boxes:
            x1, y1, x2, y2 = b["box"]
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            lines.append(f"{CLASSES.index(b['cls'])} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        open(os.path.join(a.kit, "labels", f"{k}.txt"), "w").write("\n".join(lines) + "\n")
    with open(os.path.join(a.kit, "folds.tsv"), "w") as f:
        f.write("key\tfold\n")
        for k in sorted(folds):
            f.write(f"{k}\t{folds[k]}\n")
    print(f"{len(gold)} images, {sum(map(len, gold.values()))} boxes -> {a.kit}")


def cmd_train(a):
    import torch
    import ultralytics
    from ultralytics import YOLO
    kit = os.path.abspath(a.kit)
    folds = {}
    for line in open(os.path.join(kit, "folds.tsv")).read().splitlines()[1:]:
        k, f = line.split("\t")
        folds[k] = int(f)
    keys = sorted(folds)
    runs = os.path.join(kit, "runs")
    cuda = torch.cuda.is_available() and a.device not in ("cpu", "mps")
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    oof_op, oof_raw, train_sec = {}, {}, []
    for fk in range(N_FOLDS):
        tr = [os.path.join(kit, "images", f"{k}.jpg") for k in keys if folds[k] != fk]
        te = [k for k in keys if folds[k] == fk]
        lst = os.path.join(kit, f"train_fold{fk}.txt")
        open(lst, "w").write("\n".join(tr) + "\n")
        yaml = os.path.join(kit, f"fold{fk}.yaml")
        # `val` points at the training list only because ultralytics requires the field; val=False below.
        open(yaml, "w").write(f"train: {lst}\nval: {lst}\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
        for c in ("labels.cache", f"train_fold{fk}.cache"):
            p = os.path.join(kit, c)
            if os.path.exists(p):
                os.unlink(p)
        print(f"=== fold {fk}: train {len(tr)} test {len(te)} weights {a.weights}", flush=True)
        t0 = time.perf_counter()
        m = YOLO(a.weights)
        m.train(data=yaml, epochs=HP["epochs"], imgsz=HP["imgsz"], batch=HP["batch"], nbs=HP["nbs"], device=a.device,
                seed=SEED + fk, deterministic=True, cos_lr=True, val=False, plots=False,
                project=runs, name=f"{a.tag}_fold{fk}", exist_ok=True, verbose=False, workers=4)
        sync()
        train_sec.append(time.perf_counter() - t0)
        wdir = os.path.join(runs, f"{a.tag}_fold{fk}", "weights")
        em = YOLO(os.path.join(wdir, "last.pt"))
        for k in te:
            r = em.predict(os.path.join(kit, "images", f"{k}.jpg"), conf=HP["conf_raw"], iou=HP["nms_iou"],
                           imgsz=HP["imgsz"], device=a.device, verbose=False)[0]
            raw = [{"box": [round(float(x), 1) for x in b.xyxy[0].tolist()], "cls": CLASSES[int(b.cls[0])],
                    "label": CLASSES[int(b.cls[0])], "score": round(float(b.conf[0]), 4)} for b in r.boxes]
            oof_raw[k] = raw
            oof_op[k] = [b for b in raw if b["score"] >= HP["conf_op"]]
        if fk == 0:
            shutil.copy2(os.path.join(wdir, "last.pt"), os.path.join(kit, f"{a.tag}_fold0.pt"))
        shutil.rmtree(wdir, ignore_errors=True)
    assert len(oof_op) == len(keys)
    # Inference time: weights of fold 0, one image at a time, first three images are warm-up runs.
    bm = YOLO(os.path.join(kit, f"{a.tag}_fold0.pt"))
    ts = []
    for i, k in enumerate(keys):
        t0 = time.perf_counter()
        bm.predict(os.path.join(kit, "images", f"{k}.jpg"), conf=HP["conf_op"], imgsz=HP["imgsz"], device=a.device, verbose=False)
        sync()
        if i >= 3:
            ts.append(time.perf_counter() - t0)
    meta = dict(HP, tag=a.tag, weights=os.path.basename(a.weights), device=a.device,
                gpu=torch.cuda.get_device_name(0) if cuda else None, ultralytics=ultralytics.__version__,
                torch=torch.__version__, seed=SEED, fold_train_seconds=[round(x, 1) for x in train_sec],
                infer_mean_sec=round(sum(ts) / len(ts), 4), images=len(keys))
    out = os.path.join(kit, f"oof_{a.tag}.json")
    json.dump({a.tag: oof_op, f"{a.tag}_raw005": oof_raw}, open(out, "w"), indent=1)
    json.dump(meta, open(os.path.join(kit, f"meta_{a.tag}.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["folds", "prepare", "train"])
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--images-dir", help="folder of <key>.jpg at evaluation size (paper/fetch_images.py)")
    ap.add_argument("--kit", default="kit", help="working folder for training")
    ap.add_argument("--weights", default="yolo11x.pt")
    ap.add_argument("--tag", default="yolo11x")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    {"folds": cmd_folds, "prepare": cmd_prepare, "train": cmd_train}[a.step](a)


if __name__ == "__main__":
    main()
