"""Scoring of four-class box predictions against ground truth.

Classes: book, colour_chart, ruler, imaging_slip.

Measures (Section 3.2 of the paper):
  - per-class and overall (micro / macro) precision, recall and F1 at IoU >= 0.50 and >= 0.75
  - 95% confidence interval of F1@0.50 by bootstrap over books (seed 20261110, 2,000 resamples)
  - COCO-style AP@50 and AP@50:95 (101-point interpolation) and the four-class mAP
  - mean IoU of matched pairs
  - editing effort of a draft: boxes accepted unchanged (IoU >= 0.80 / 0.85 / 0.90), resized
    (0.50 <= IoU < 0.85), added (missed), deleted (false positives), and images needing no edit
    (every box matched at IoU >= 0.85, nothing missing or extra)

Input formats (detected automatically):
  A  method -> image -> boxes   {"proposed": {"100238424-0052": {"boxes": [...]}}}
  B  image -> method -> boxes   {"100238424-0052": {"proposed": {"boxes": [...]}}}
  C  list of image records      [{"id": "100238424-0052", "proposed": [...]}]

Usage:
  shohon-bbox evaluate --pred predictions.json --gold gold.json --out results/
"""

import argparse
import csv
import json
import os
import random
import statistics as st
from collections import defaultdict
from typing import Dict, List, Tuple, Any

CLASSES = ["book", "colour_chart", "ruler", "imaging_slip"]

LABEL_ALIASES = {
    "book": "book",
    "open book": "book",
    "a book": "book",
    "an open book": "book",
    "1_overall": "book",
    "cutting board": "book",
    "colour_chart": "colour_chart",
    "color_chart": "colour_chart",
    "color chart": "colour_chart",
    "a color chart": "colour_chart",
    "color card": "colour_chart",
    "colorchecker": "colour_chart",
    "ruler": "ruler",
    "a ruler": "ruler",
    "scale": "ruler",
    "measuring ruler": "ruler",
    "imaging_slip": "imaging_slip",
    "paper label": "imaging_slip",
    "a paper label": "imaging_slip",
    "label": "imaging_slip",
    "number tag": "imaging_slip",
    "slip": "imaging_slip",
    "caption": "imaging_slip",
}

META_KEYS = {"path", "width", "height", "id", "image_id", "canvas", "book_id", "work", "page_type", "sec"}


def canonical_image_id(iid: str) -> str:
    base = os.path.basename(str(iid).strip())
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    return base


def normalize_label(lbl: str) -> str:
    if not lbl:
        return ""
    s = str(lbl).strip().lower()
    if s in CLASSES:
        return s
    if s in LABEL_ALIASES:
        return LABEL_ALIASES[s]
    if "color" in s or "chart" in s or "colour" in s:
        return "colour_chart"
    if "ruler" in s or "scale" in s:
        return "ruler"
    if "book" in s or s == "1_overall" or "cutting board" in s:
        return "book"
    if "label" in s or "paper" in s or "slip" in s or "tag" in s or "caption" in s or s == "5_stamp":
        return "imaging_slip"
    for k, v in LABEL_ALIASES.items():
        if k in s:
            return v
    return s


def to_xyxy(box: List[float], fmt: str = "xyxy") -> List[float]:
    x0, y0, a, b = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if fmt == "xywh":
        return [x0, y0, x0 + a, y0 + b]
    return [x0, y0, a, b]


def iou_xyxy(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_greedy(preds: List[Dict], golds: List[Dict], iou_thr: float) -> List[Tuple[int, int, float]]:
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            v = iou_xyxy(p["box"], g["box"])
            if v >= iou_thr and v > 0:
                pairs.append((v, p.get("score", 1.0), i, j))
    pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    used_p, used_g, matched = set(), set(), []
    for v, _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((i, j, v))
    return matched


def compute_prf(tp: int, n_pred: int, n_gold: int) -> Tuple[float, float, float]:
    p = tp / n_pred if n_pred > 0 else 0.0
    r = tp / n_gold if n_gold > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_ap_coco(frames_pred_gold: List[Tuple[List[Dict], List[Dict]]], iou_thr: float) -> float:
    dets = []
    n_gold = 0
    for fi, (preds, golds) in enumerate(frames_pred_gold):
        n_gold += len(golds)
        for p in preds:
            dets.append((float(p.get("score", 1.0)), fi, p["box"]))
    if n_gold == 0:
        return 0.0
    dets.sort(key=lambda x: -x[0])
    used = defaultdict(set)
    tps = []
    for _, fi, pbox in dets:
        golds = frames_pred_gold[fi][1]
        best_iou = iou_thr
        best_j = -1
        for j, g in enumerate(golds):
            if j in used[fi]:
                continue
            v = iou_xyxy(pbox, g["box"])
            if v >= best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0:
            used[fi].add(best_j)
            tps.append(1)
        else:
            tps.append(0)

    rec, prec = [], []
    tp_cum = 0
    for k, is_tp in enumerate(tps, 1):
        tp_cum += is_tp
        rec.append(tp_cum / n_gold)
        prec.append(tp_cum / k)
    for k in range(len(prec) - 2, -1, -1):
        prec[k] = max(prec[k], prec[k + 1])
    ap_sum = 0.0
    idx = 0
    for r_step in [i / 100.0 for i in range(101)]:
        while idx < len(rec) and rec[idx] < r_step:
            idx += 1
        ap_sum += prec[idx] if idx < len(rec) else 0.0
    return ap_sum / 101.0


def extract_book_id(image_id: str) -> str:
    base = canonical_image_id(image_id)
    if "-" in base:
        return base.split("-")[0]
    return base


def parse_box_list(val: Any, box_fmt: str = "xyxy") -> List[Dict]:
    if isinstance(val, dict):
        dets = val.get("detections", val.get("boxes", val.get("annotations", [])))
    elif isinstance(val, list):
        dets = val
    else:
        dets = []
    norm_dets = []
    for d in dets:
        if isinstance(d, dict):
            if "box" in d:
                b = d["box"]
                fmt = "xywh" if d.get("format") == "xywh" else box_fmt
            elif "bbox" in d:
                b = d["bbox"]
                fmt = "xywh" if d.get("format") == "xywh" else box_fmt
            elif all(k in d for k in ("x", "y", "w", "h")):
                b = [d["x"], d["y"], d["w"], d["h"]]
                fmt = "xywh"
            elif all(k in d for k in ("x1", "y1", "x2", "y2")):
                b = [d["x1"], d["y1"], d["x2"], d["y2"]]
                fmt = "xyxy"
            else:
                continue
            lbl = normalize_label(d.get("label", d.get("cls", d.get("class", d.get("category", "")))))
            sc = float(d.get("score", d.get("conf", 1.0)))
        elif isinstance(d, (list, tuple)) and len(d) >= 5:
            b = list(d[:4])
            sc = float(d[4])
            lbl = normalize_label(d[5]) if len(d) >= 6 else "book"
            fmt = box_fmt
        else:
            continue
        if lbl not in CLASSES:
            continue
        norm_dets.append({"box": to_xyxy(b, fmt), "label": lbl, "score": sc})
    return norm_dets


def load_gold(raw_gold: Any, box_fmt: str = "xyxy") -> Dict[str, List[Dict]]:
    if isinstance(raw_gold, dict):
        for k in ("gold", "annotations", "frames", "images"):
            if k in raw_gold and isinstance(raw_gold[k], (dict, list)):
                raw_gold = raw_gold[k]
                break
    out = {}
    if isinstance(raw_gold, list):
        for item in raw_gold:
            if not isinstance(item, dict):
                continue
            iid = canonical_image_id(item.get("id", item.get("image_id", item.get("key", ""))))
            if iid:
                out[iid] = parse_box_list(item, box_fmt)
    elif isinstance(raw_gold, dict):
        for iid, val in raw_gold.items():
            out[canonical_image_id(iid)] = parse_box_list(val, box_fmt)
    return out


def load_predictions(raw_pred: Any, box_fmt: str = "xyxy") -> Dict[str, Dict[str, List[Dict]]]:
    """
    Format A (Method -> Image -> Boxes), Format B (Image -> Method -> Boxes),
    Format C (list of records) into {"method_name": {"image_id": [boxes...]}}.
    """
    methods = defaultdict(dict)
    if isinstance(raw_pred, list):
        for item in raw_pred:
            if not isinstance(item, dict):
                continue
            iid = canonical_image_id(item.get("id", item.get("image_id", item.get("key", ""))))
            if not iid:
                continue
            for k, v in item.items():
                if k in META_KEYS:
                    continue
                if isinstance(v, (list, dict)):
                    methods[k][iid] = parse_box_list(v, box_fmt)
        return dict(methods)

    if not isinstance(raw_pred, dict) or not raw_pred:
        return {}

    first_key = next(iter(raw_pred.keys()))
    first_val = raw_pred[first_key]
    # Check if top-level keys look like image IDs (e.g. "100238424-0052" or have "path"/"width")
    is_image_top = (
        any(ch.isdigit() for ch in first_key[:5]) and "-" in first_key
    ) or (isinstance(first_val, dict) and ("width" in first_val or "path" in first_val))

    if is_image_top:
        for iid_raw, img_obj in raw_pred.items():
            iid = canonical_image_id(iid_raw)
            if not isinstance(img_obj, dict):
                continue
            for m_key, m_val in img_obj.items():
                if m_key in META_KEYS:
                    continue
                if isinstance(m_val, (list, dict)):
                    methods[m_key][iid] = parse_box_list(m_val, box_fmt)
    else:
        for m_key, m_obj in raw_pred.items():
            if not isinstance(m_obj, dict):
                continue
            for iid_raw, val in m_obj.items():
                iid = canonical_image_id(iid_raw)
                methods[m_key][iid] = parse_box_list(val, box_fmt)
    return dict(methods)


def evaluate_method(
    image_ids: List[str],
    pred_by_img: Dict[str, List[Dict]],
    gold_by_img: Dict[str, List[Dict]],
    n_boot: int = 2000,
    seed: int = 20261110,
) -> Dict[str, Any]:
    books = sorted({extract_book_id(iid) for iid in image_ids})
    per_class = {}
    ap50_list, ap50_95_list = [], []
    f1_50_list, f1_75_list = [], []

    micro_counts = {
        0.5: {"tp": 0, "pred": 0, "gold": 0},
        0.75: {"tp": 0, "pred": 0, "gold": 0},
    }
    micro_book_counts = {
        0.5: defaultdict(lambda: [0, 0, 0]),
        0.75: defaultdict(lambda: [0, 0, 0]),
    }
    all_matched_ious = []
    all_gold_best_ious = []

    edit_unmod_80 = 0
    edit_unmod_85 = 0
    edit_unmod_90 = 0
    edit_minor = 0
    edit_fn = 0
    edit_fp = 0
    total_pred_all = 0
    total_gold_all = 0

    for cls in CLASSES:
        frames_pg = []
        tp50 = np50 = ng50 = 0
        tp75 = np75 = ng75 = 0
        cls_matched_ious = []
        cls_gold_best_ious = []
        cls_unmod_80 = cls_unmod_85 = cls_unmod_90 = cls_minor = cls_fn = cls_fp = 0

        for iid in image_ids:
            b_id = extract_book_id(iid)
            P = [x for x in pred_by_img.get(iid, []) if x["label"] == cls]
            G = [x for x in gold_by_img.get(iid, []) if x["label"] == cls]
            frames_pg.append((P, G))

            m50 = match_greedy(P, G, 0.50)
            m75 = match_greedy(P, G, 0.75)

            tp50 += len(m50)
            np50 += len(P)
            ng50 += len(G)
            tp75 += len(m75)
            np75 += len(P)
            ng75 += len(G)

            micro_book_counts[0.5][b_id][0] += len(m50)
            micro_book_counts[0.5][b_id][1] += len(P)
            micro_book_counts[0.5][b_id][2] += len(G)
            micro_book_counts[0.75][b_id][0] += len(m75)
            micro_book_counts[0.75][b_id][1] += len(P)
            micro_book_counts[0.75][b_id][2] += len(G)

            for _, _, v in m50:
                cls_matched_ious.append(v)
                all_matched_ious.append(v)
                if v >= 0.80:
                    cls_unmod_80 += 1
                if v >= 0.90:
                    cls_unmod_90 += 1
                if v >= 0.85:
                    cls_unmod_85 += 1
                else:
                    cls_minor += 1

            cls_fn += len(G) - len(m50)
            cls_fp += len(P) - len(m50)

            for g in G:
                best_v = max((iou_xyxy(p["box"], g["box"]) for p in P), default=0.0)
                cls_gold_best_ious.append(best_v)
                all_gold_best_ious.append(best_v)

        micro_counts[0.5]["tp"] += tp50
        micro_counts[0.5]["pred"] += np50
        micro_counts[0.5]["gold"] += ng50
        micro_counts[0.75]["tp"] += tp75
        micro_counts[0.75]["pred"] += np75
        micro_counts[0.75]["gold"] += ng75

        edit_unmod_80 += cls_unmod_80
        edit_unmod_85 += cls_unmod_85
        edit_unmod_90 += cls_unmod_90
        edit_minor += cls_minor
        edit_fn += cls_fn
        edit_fp += cls_fp
        total_pred_all += np50
        total_gold_all += ng50

        p50, r50, f50 = compute_prf(tp50, np50, ng50)
        p75, r75, f75 = compute_prf(tp75, np75, ng75)
        ap50 = compute_ap_coco(frames_pg, 0.50)
        ap50_95 = st.mean(compute_ap_coco(frames_pg, t / 100.0) for t in range(50, 100, 5))

        if ng50 > 0:
            ap50_list.append(ap50)
            ap50_95_list.append(ap50_95)
            f1_50_list.append(f50)
            f1_75_list.append(f75)

        per_class[cls] = {
            "gold": ng50,
            "pred": np50,
            "tp50": tp50,
            "precision50": round(p50, 4),
            "recall50": round(r50, 4),
            "f1_50": round(f50, 4),
            "tp75": tp75,
            "precision75": round(p75, 4),
            "recall75": round(r75, 4),
            "f1_75": round(f75, 4),
            "AP50": round(ap50, 4),
            "AP50_95": round(ap50_95, 4),
            "mean_iou_matched": round(st.mean(cls_matched_ious), 4) if cls_matched_ious else 0.0,
            "mean_iou_all_gold": round(st.mean(cls_gold_best_ious), 4) if cls_gold_best_ious else 0.0,
            "preannot_edit": {
                "unmodified_iou80_rate": round(cls_unmod_80 / ng50, 4) if ng50 else 0.0,
                "unmodified_iou85_rate": round(cls_unmod_85 / ng50, 4) if ng50 else 0.0,
                "unmodified_iou90_rate": round(cls_unmod_90 / ng50, 4) if ng50 else 0.0,
                "minor_adjust_rate": round(cls_minor / ng50, 4) if ng50 else 0.0,
                "missed_fn_rate": round(cls_fn / ng50, 4) if ng50 else 0.0,
                "deleted_fp_rate": round(cls_fp / np50, 4) if np50 else 0.0,
            },
        }

    zero_touch_imgs = 0
    for iid in image_ids:
        ok = True
        for cls in CLASSES:
            P = [x for x in pred_by_img.get(iid, []) if x["label"] == cls]
            G = [x for x in gold_by_img.get(iid, []) if x["label"] == cls]
            if len(P) != len(G):
                ok = False
                break
            m85 = match_greedy(P, G, 0.85)
            if len(m85) != len(G):
                ok = False
                break
        if ok:
            zero_touch_imgs += 1

    rng = random.Random(seed)
    boot_f1s = []
    if books:
        for _ in range(n_boot):
            sampled_books = [rng.choice(books) for _ in books]
            tp_b = sum(micro_book_counts[0.5][b][0] for b in sampled_books)
            np_b = sum(micro_book_counts[0.5][b][1] for b in sampled_books)
            ng_b = sum(micro_book_counts[0.5][b][2] for b in sampled_books)
            boot_f1s.append(compute_prf(tp_b, np_b, ng_b)[2])
        boot_f1s.sort()
        ci95 = [
            round(boot_f1s[int(0.025 * len(boot_f1s))], 4),
            round(boot_f1s[int(0.975 * len(boot_f1s)) - 1], 4),
        ]
    else:
        ci95 = [0.0, 0.0]

    mp50, mr50, mf50 = compute_prf(
        micro_counts[0.5]["tp"], micro_counts[0.5]["pred"], micro_counts[0.5]["gold"]
    )
    mp75, mr75, mf75 = compute_prf(
        micro_counts[0.75]["tp"], micro_counts[0.75]["pred"], micro_counts[0.75]["gold"]
    )

    return {
        "images": len(image_ids),
        "books": len(books),
        "total_gold": total_gold_all,
        "total_pred": total_pred_all,
        "micro": {
            "precision50": round(mp50, 4),
            "recall50": round(mr50, 4),
            "f1_50": round(mf50, 4),
            "f1_50_ci95": ci95,
            "precision75": round(mp75, 4),
            "recall75": round(mr75, 4),
            "f1_75": round(mf75, 4),
            "mean_iou_matched": round(st.mean(all_matched_ious), 4) if all_matched_ious else 0.0,
            "mean_iou_all_gold": round(st.mean(all_gold_best_ious), 4) if all_gold_best_ious else 0.0,
        },
        "macro": {
            "f1_50": round(st.mean(f1_50_list), 4) if f1_50_list else 0.0,
            "f1_75": round(st.mean(f1_75_list), 4) if f1_75_list else 0.0,
            "mAP50": round(st.mean(ap50_list), 4) if ap50_list else 0.0,
            "mAP50_95": round(st.mean(ap50_95_list), 4) if ap50_95_list else 0.0,
        },
        "preannot_summary": {
            "unmodified_iou80_rate": round(edit_unmod_80 / total_gold_all, 4) if total_gold_all else 0.0,
            "unmodified_iou85_rate": round(edit_unmod_85 / total_gold_all, 4) if total_gold_all else 0.0,
            "unmodified_iou90_rate": round(edit_unmod_90 / total_gold_all, 4) if total_gold_all else 0.0,
            "minor_adjust_rate": round(edit_minor / total_gold_all, 4) if total_gold_all else 0.0,
            "missed_fn_rate": round(edit_fn / total_gold_all, 4) if total_gold_all else 0.0,
            "deleted_fp_rate": round(edit_fp / total_pred_all, 4) if total_pred_all else 0.0,
            "zero_touch_image_rate": round(zero_touch_imgs / len(image_ids), 4) if image_ids else 0.0,
        },
        "per_class": per_class,
    }


def format_markdown_tables(results: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Evaluation report\n")
    lines.append("## Table A: overall performance and editing effort (micro / macro)\n")
    lines.append(
        "| Method | Predicted boxes | P@0.50 | R@0.50 | F1@0.50 (95% CI) | F1@0.75 | mAP@50 | mAP@50:95 | Mean IoU | Unchanged (IoU≥0.80 / ≥0.85) | Deleted (FP/Pred) | Images needing no edit |"
    )
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for m_name, res in results.items():
        mi = res["micro"]
        ma = res["macro"]
        pa = res["preannot_summary"]
        ci_str = f"{mi['f1_50']:.3f} [{mi['f1_50_ci95'][0]:.3f}, {mi['f1_50_ci95'][1]:.3f}]"
        lines.append(
            f"| `{m_name}` | {res['total_pred']} | {mi['precision50']:.3f} | {mi['recall50']:.3f} | "
            f"**{ci_str}** | {mi['f1_75']:.3f} | {ma['mAP50']:.3f} | {ma['mAP50_95']:.3f} | "
            f"{mi['mean_iou_matched']:.3f} | {pa['unmodified_iou80_rate']*100:.1f}% / {pa['unmodified_iou85_rate']*100:.1f}% | "
            f"{pa['deleted_fp_rate']*100:.1f}% | {pa['zero_touch_image_rate']*100:.1f}% |"
        )

    lines.append("\n## Table B: per-class F1@0.50 / F1@0.75\n")
    lines.append(
        "| Method | `book` F1@50 / 75 | `colour_chart` F1@50 / 75 | `ruler` F1@50 / 75 | `imaging_slip` F1@50 / 75 |"
    )
    lines.append("|---|---|---|---|---|")
    for m_name, res in results.items():
        pc = res["per_class"]
        cols = []
        for c in CLASSES:
            ci = pc[c]
            cols.append(f"{ci['f1_50']:.3f} / {ci['f1_75']:.3f} (P={ci['precision50']:.2f}, R={ci['recall50']:.2f})")
        lines.append(f"| `{m_name}` | " + " | ".join(cols) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate 4-class BBox predictions against Gold annotations.")
    ap.add_argument("--pred", required=True, help="Path to predictions JSON")
    ap.add_argument("--gold", required=True, help="Path to Gold BBox JSON")
    ap.add_argument("--out", required=True, help="Output directory for summary.json, summary.tsv, and report.md")
    ap.add_argument("--box-format", default="xyxy", choices=["xyxy", "xywh"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)

    with open(args.pred) as f:
        raw_pred = json.load(f)
    with open(args.gold) as f:
        raw_gold = json.load(f)

    gold_by_img = load_gold(raw_gold, args.box_format)
    pred_methods = load_predictions(raw_pred, args.box_format)
    image_ids = sorted(gold_by_img.keys())

    results = {}
    for method_name, pred_by_img in pred_methods.items():
        results[method_name] = evaluate_method(
            image_ids=image_ids,
            pred_by_img=pred_by_img,
            gold_by_img=gold_by_img,
            n_boot=args.bootstrap,
        )

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out, "summary.tsv"), "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "method", "class", "gold", "pred", "tp50", "precision50", "recall50", "f1_50",
            "tp75", "precision75", "recall75", "f1_75", "AP50", "AP50_95",
            "mean_iou_matched", "unmodified_iou85_rate", "deleted_fp_rate",
        ])
        for m_name, res in results.items():
            mi = res["micro"]
            ma = res["macro"]
            pa = res["preannot_summary"]
            w.writerow([
                m_name, "ALL_MICRO", res["total_gold"], res["total_pred"],
                "", mi["precision50"], mi["recall50"], mi["f1_50"],
                "", mi["precision75"], mi["recall75"], mi["f1_75"],
                ma["mAP50"], ma["mAP50_95"], mi["mean_iou_matched"],
                pa["unmodified_iou85_rate"], pa["deleted_fp_rate"],
            ])
            for c in CLASSES:
                ci = res["per_class"][c]
                ce = ci["preannot_edit"]
                w.writerow([
                    m_name, c, ci["gold"], ci["pred"],
                    ci["tp50"], ci["precision50"], ci["recall50"], ci["f1_50"],
                    ci["tp75"], ci["precision75"], ci["recall75"], ci["f1_75"],
                    ci["AP50"], ci["AP50_95"], ci["mean_iou_matched"],
                    ce["unmodified_iou85_rate"], ce["deleted_fp_rate"],
                ])

    md_report = format_markdown_tables(results)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(md_report)
    print(md_report)


if __name__ == "__main__":
    main()
