"""Command line: shohon-bbox detect | rules | review | evaluate."""
import argparse
import json
import sys

from . import __version__


def _flag(v):
    m = {"true": True, "false": False, "unknown": None}
    if v not in m:
        raise ValueError(v)
    return m[v]


def _presence(a):
    if a.ruler is None and a.colour_chart is None:
        return None
    return {"ruler": a.ruler, "colour_chart": a.colour_chart}


def cmd_detect(a):
    from . import annotation, iiif, rules
    from .detectors import Detectors
    det = Detectors(threads=a.threads)
    results = []
    raws = {}
    for src in a.inputs:
        im, target = iiif.load(src, a.canvas)
        small = iiif.reduce(im, a.max_side)
        raw = det.run(small)
        if a.format == "raw":
            raws[_stem(src)] = raw
            continue
        gd, ow, nd = rules.map_gdino(raw["gdino"]), rules.map_owlv2(raw["owlv2"]), rules.map_ndl(raw["ndl_layout"])
        boxes = rules.proposed(gd, ow, nd, raw["w"], raw["h"], _presence(a))
        out = iiif.to_target(boxes, small.size, target)
        if a.format == "annotation":
            results.append(annotation.annotation_page(out, target))
        else:
            rec = {"input": src, "target": target["source"], "width": target["width"], "height": target["height"],
                   "detection_size": list(small.size), "boxes": out}
            if a.raw:
                rec["raw"] = raw
            results.append(rec)
    if a.format == "raw":
        json.dump(raws, sys.stdout, ensure_ascii=False)
    else:
        json.dump(results[0] if len(results) == 1 else results, sys.stdout, ensure_ascii=False, indent=2)
    print()


def _stem(src):
    import os
    import urllib.parse
    return os.path.splitext(os.path.basename(urllib.parse.urlparse(src).path))[0] or src


def cmd_rules(a):
    from . import rules
    raw = json.load(open(a.raw_json, encoding="utf-8"))
    if a.all:
        out = {}
        for k, e in raw.items():
            d = rules.all_configs(e, None)
            d.pop("proposed_meta")
            out[k] = dict(d, infer_width=e["w"], infer_height=e["h"])
    else:
        out = {k: rules.all_configs(e, None)["proposed"] for k, e in raw.items()}
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    print()


def cmd_review(a):
    from . import review
    review.serve(a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shohon-bbox", description="Boxes for books, colour charts, rulers and "
                                 "imaging slips in digitized book images.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="detect boxes in images, image URLs or IIIF Canvases")
    d.add_argument("inputs", nargs="+", help="image file, image URL, or IIIF manifest URL (with --canvas)")
    d.add_argument("--canvas", help="Canvas of the manifest: 1-based position or Canvas id")
    d.add_argument("--format", choices=["json", "annotation", "raw"], default="json",
                   help="json (default), a IIIF/W3C Web Annotation page, or raw: the raw detector outputs of all "
                        "inputs as {file stem: outputs}, in detection-image pixels (input to rules and review)")
    d.add_argument("--ruler", type=_flag, choices=[True, False, None], metavar="{true,false,unknown}",
                   help="known presence of a ruler (Configuration 6 of the paper)")
    d.add_argument("--colour-chart", type=_flag, choices=[True, False, None], metavar="{true,false,unknown}",
                   help="known presence of a colour chart (Configuration 6 of the paper)")
    d.add_argument("--max-side", type=int, default=1400, help="long side of the image used for detection")
    d.add_argument("--threads", type=int, default=8)
    d.add_argument("--raw", action="store_true", help="also output the raw detector outputs")
    d.set_defaults(func=cmd_detect)

    r = sub.add_parser("rules", help="apply the spatial rules (Configuration 5) to stored raw detector outputs")
    r.add_argument("raw_json", help="e.g. data/raw/raw_detector_outputs.json")
    r.add_argument("--all", action="store_true", help="output Configurations 1–5 for each image (drafts for review)")
    r.set_defaults(func=cmd_rules)

    v = sub.add_parser("review", help="local web interface to correct draft boxes or draw them blind")
    v.add_argument("images_dir", help="folder of images named <key>.jpg / .png")
    v.add_argument("--drafts", help="raw detector outputs, or {key: [boxes]} / {key: {name: [boxes]}}")
    v.add_argument("--images-csv", help="optional list of images in order, with metadata (e.g. data/images.csv)")
    v.add_argument("--out", default="annotations.json", help="file for the reviewed boxes (default annotations.json)")
    v.add_argument("--blind-out", default="blind_annotations.json", help="file for boxes drawn in blind mode")
    v.add_argument("--port", type=int, default=8795)
    v.set_defaults(func=cmd_review)

    e = sub.add_parser("evaluate", help="score predictions against ground truth", add_help=False)
    e.set_defaults(func=None)

    a, rest = ap.parse_known_args(argv)
    if a.cmd == "evaluate":
        from . import evaluate
        return evaluate.main(rest)
    if rest:
        ap.error("unrecognized arguments: " + " ".join(rest))
    return a.func(a)


if __name__ == "__main__":
    main()
