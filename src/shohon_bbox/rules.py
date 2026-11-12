"""Configurations 4–6 of the paper, computed from the raw detector outputs.

Pure functions only: no file paths, no model loading. Input is the raw output of the three
detectors for one image (as stored in data/raw/raw_detector_outputs.json); output is a list of boxes
{"box": [x1, y1, x2, y2], "cls": <class>, "score": <float>, "source": <detector>}.

Classes: book, colour_chart, ruler, imaging_slip.

The rules follow Section 3.1 of the paper. Where the paper's wording leaves a detail open,
the choice is written next to the code as "SPEC:" and is repeated in the paper.
"""

CLASSES = ("book", "colour_chart", "ruler", "imaging_slip")

# ---------------------------------------------------------------- geometry

def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def ioa(a, b):
    """Fraction of box a's area that lies inside box b."""
    aa = area(a)
    return inter(a, b) / aa if aa > 0 else 0.0


def aspect(b):
    w, h = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
    return max(w / h, h / w)


def nms(boxes, thr):
    kept = []
    for b in sorted(boxes, key=lambda x: x["score"], reverse=True):
        if all(iou(b["box"], k["box"]) < thr for k in kept):
            kept.append(b)
    return kept


# ---------------------------------------------------------------- label mapping (Appendix)

def _box(raw):
    return [round(float(v), 1) for v in raw["box"]]


def map_gdino(raw_boxes):
    out = []
    for b in raw_boxes:
        lbl = b["label"].lower()
        if "color" in lbl or "chart" in lbl:
            cls = "colour_chart"
        elif "ruler" in lbl or "scale" in lbl:
            cls = "ruler"
        elif "book" in lbl:
            cls = "book"
        elif "label" in lbl or "paper" in lbl or "slip" in lbl or "tag" in lbl:
            cls = "imaging_slip"
        else:
            continue
        out.append({"box": _box(b), "cls": cls, "score": round(float(b["score"]), 4),
                    "source": "gdino", "raw_label": b["label"]})
    return out


def map_owlv2(raw_boxes):
    out = []
    for b in raw_boxes:
        lbl = b["label"].lower()
        if "color" in lbl or "chart" in lbl:
            cls = "colour_chart"
        elif "ruler" in lbl:
            cls = "ruler"
        elif "book" in lbl:
            cls = "book"
        elif "label" in lbl or "paper" in lbl or "slip" in lbl:
            cls = "imaging_slip"
        else:
            continue
        out.append({"box": _box(b), "cls": cls, "score": round(float(b["score"]), 4),
                    "source": "owlv2", "raw_label": b["label"]})
    return out


def map_ndl(raw_boxes):
    out = []
    for b in raw_boxes:
        if b["label"] == "1_overall":
            cls = "book"
        elif b["label"] == "5_stamp":
            cls = "imaging_slip"
        else:
            continue
        out.append({"box": _box(b), "cls": cls, "score": round(float(b["score"]), 4),
                    "source": "ndl", "raw_label": b["label"]})
    return out


# ---------------------------------------------------------------- Configuration 4

def raw_ensemble(gd, ow, nd):
    """Union of Configurations 1–3, NMS at IoU >= 0.50 within each class."""
    out = []
    for cls in CLASSES:
        out.extend(nms([dict(x) for x in gd + ow + nd if x["cls"] == cls], 0.50))
    return out


# ---------------------------------------------------------------- Configurations 5 and 6

# SPEC: a "high-confidence" ruler or colour chart in Rule 1 is a Grounding DINO or OWLv2
# candidate of that class with score >= 0.40.
TARGET_CONF = 0.40
# SPEC: in Rule 1, a YOLOv8 box "encloses one page" of a Grounding DINO spread when at least
# 90% of the YOLOv8 box lies inside the spread box and its area is below 62% of the spread's.
PAGE_INSIDE = 0.90
PAGE_SHARE = 0.62


def _select_book(gd, ow, nd, img_area):
    cands = []
    for x in nd:
        if x["cls"] == "book" and 0.03 * img_area <= area(x["box"]) <= 0.88 * img_area:
            cands.append(dict(x, score=round(x["score"] * 0.95, 4)))
    for x in gd:
        if x["cls"] == "book" and 0.03 * img_area <= area(x["box"]) <= 0.88 * img_area:
            bonus = 0.04 if "open" in x["raw_label"].lower() else 0.0
            cands.append(dict(x, score=round(x["score"] + bonus, 4)))
    if not cands:
        return None
    targets = [x["box"] for x in gd + ow
               if x["cls"] in ("ruler", "colour_chart") and x["score"] >= TARGET_CONF]

    def swallows(book):
        return any(ioa(t, book["box"]) > 0.50 for t in targets)

    cands.sort(key=lambda c: c["score"], reverse=True)
    best = cands[0]
    if best["source"] == "gdino" and swallows(best):
        tighter = [c for c in cands if c["source"] == "ndl"
                   and area(c["box"]) < area(best["box"]) and not swallows(c)]
        if tighter:
            best = tighter[0]
    elif best["source"] == "ndl":
        spreads = [c for c in cands if c["source"] == "gdino" and "open" in c["raw_label"].lower()
                   and ioa(best["box"], c["box"]) >= PAGE_INSIDE
                   and area(best["box"]) < PAGE_SHARE * area(c["box"]) and not swallows(c)]
        if spreads:
            best = spreads[0]
    return dict(best, cls="book")


def _chart_candidates(gd, ow, book, img_area):
    def ok(x, max_area):
        return area(x["box"]) < max_area * img_area and not (book and ioa(x["box"], book["box"]) > 0.35)
    c = [dict(x) for x in gd if x["cls"] == "colour_chart" and ok(x, 0.35)]
    if not c:  # OWLv2 only when Grounding DINO has no usable chart
        c = [dict(x) for x in ow if x["cls"] == "colour_chart" and ok(x, 0.30)]
    return nms(c, 0.35)


def _ruler_candidates(gd, ow, book, charts):
    """Rulers that survive the book and chart overlap tests (before the shape test)."""
    out = []
    for r in [x for x in gd + ow if x["cls"] == "ruler"]:
        bx = r["box"]
        if book and (iou(bx, book["box"]) > 0.40 or (aspect(bx) < 4.5 and ioa(bx, book["box"]) > 0.55)):
            continue
        if any(iou(bx, ch["box"]) > 0.30 or ioa(bx, ch["box"]) > 0.50 for ch in charts):
            continue
        out.append(dict(r))
    return out


def _two_axes(rulers):
    v = nms([r for r in rulers if (r["box"][3] - r["box"][1]) >= (r["box"][2] - r["box"][0])], 0.35)[:1]
    h = nms([r for r in rulers if (r["box"][2] - r["box"][0]) > (r["box"][3] - r["box"][1])], 0.35)[:1]
    return v + h


def _slips(gd, ow, book, charts, rulers, img_area):
    # SPEC: slip candidates come from Grounding DINO and OWLv2 only (YOLOv8's 5_stamp is not
    # used), as in the earlier implementation whose stored outputs contain no YOLOv8 slip.
    c = []
    for x in gd + ow:
        if x["cls"] != "imaging_slip" or x["score"] < 0.30 or area(x["box"]) > 0.15 * img_area:
            continue
        bx = x["box"]
        if book and ioa(bx, book["box"]) > 0.35:
            continue
        if any(ioa(bx, ch["box"]) > 0.35 for ch in charts):
            continue
        if any(ioa(bx, rl["box"]) > 0.50 for rl in rulers):
            continue
        c.append(dict(x))
    return nms(c, 0.35)[:2]


def proposed(gd, ow, nd, w, h, presence=None):
    """Configuration 5 (presence=None) or Configuration 6 (presence = {"ruler": bool|None,
    "colour_chart": bool|None} from the companion dataset)."""
    img_area = float(w * h)
    presence = presence or {}
    book = _select_book(gd, ow, nd, img_area)

    # Rule 3, charts. SPEC: at most one colour chart (the highest-scoring after NMS).
    chart_all = _chart_candidates(gd, ow, book, img_area)
    charts = chart_all[:1]
    if presence.get("colour_chart") is False:
        charts = [c for c in charts if c["score"] >= 0.80]
    elif presence.get("colour_chart") is True and not charts:
        # SPEC: recovery takes the highest-scoring chart candidate returned by either detector
        # at its default threshold (Grounding DINO 0.22, OWLv2 0.18) that passes the size and
        # book-overlap tests; the stored raw outputs contain nothing below those thresholds.
        rec = [dict(x) for x in gd + ow if x["cls"] == "colour_chart"
               and area(x["box"]) < 0.35 * img_area and not (book and ioa(x["box"], book["box"]) > 0.35)]
        charts = sorted(rec, key=lambda x: x["score"], reverse=True)[:1]

    # Rule 3, rulers.
    overlap_ok = _ruler_candidates(gd, ow, book, charts)
    rulers = _two_axes([r for r in overlap_ok if aspect(r["box"]) >= 2.3 or r["score"] >= 0.45])
    if presence.get("ruler") is False:
        rulers = [r for r in rulers if r["score"] >= 0.85]
    elif presence.get("ruler") is True and not rulers:
        # SPEC: same recovery as for charts, among rulers that pass the overlap tests.
        rulers = sorted(overlap_ok, key=lambda x: x["score"], reverse=True)[:1]

    # Rule 2, slips (needs the final charts and rulers).
    slips = _slips(gd, ow, book, charts, rulers, img_area)

    return ([book] if book else []) + charts + rulers + slips


def all_configs(entry, presence):
    """Configurations 1–6 for one entry of the raw detector outputs."""
    gd, ow, nd = map_gdino(entry.get("gdino", [])), map_owlv2(entry.get("owlv2", [])), map_ndl(entry.get("ndl_layout", []))
    w, h = entry["w"], entry["h"]
    return {
        "gdino": gd,
        "owlv2": ow,
        "ndl_layout": nd,
        "ensemble_raw": raw_ensemble(gd, ow, nd),
        "proposed": proposed(gd, ow, nd, w, h),
        "proposed_meta": proposed(gd, ow, nd, w, h, presence),
    }
