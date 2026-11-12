"""A local web interface to review draft boxes, or to draw boxes blind.

`shohon-bbox review IMAGES_DIR [--drafts FILE] [--images-csv FILE] [--out FILE] [--blind-out FILE]`

- Review mode shows a draft for each image (Configuration 6, or 5, of the paper by default), which
  the annotator corrects: move, resize, delete, add or relabel boxes, then mark the image as
  verified. The original drafts of every configuration stay available in a menu.
- Blind mode (checkbox, or `?blind=1`) shows no draft and no score; boxes are drawn on a blank
  image and saved to a separate file. The paper used it to measure how much the drafts bias the
  verified boxes, and for the second annotator.

Drafts can be raw detector outputs (data/raw/raw_detector_outputs.json, or
`shohon-bbox detect images/*.jpg --format raw`), from which Configurations 1–5 are computed, or a JSON object {key: [boxes]} or
{key: {name: [boxes]}}. Box coordinates are pixels of the image the drafts were computed on
(`infer_width` x `infer_height`); without drafts, pixels of the image file shown.

The saved file is {"boxes": {key: [{"cls", "box": [x1, y1, x2, y2], ...}]}, "verified": {key: true},
"modified": {key: true}, "sizes": {key: [width, height]}}; `shohon-bbox evaluate --gold` reads it
(verified images only). The server listens on 127.0.0.1 only.
"""
import csv
import json
import os
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import rules

HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review.html")
IMAGE_EXT = (".jpg", ".jpeg", ".png")
EMPTY = {"boxes": {}, "verified": {}, "modified": {}, "sizes": {}}


def normalize_drafts(raw):
    """{key: {name: [boxes], "infer_width": w, "infer_height": h}} from any of the accepted forms."""
    out = {}
    for k, v in (raw or {}).items():
        if isinstance(v, list):
            out[k] = {"proposed": v}
        elif isinstance(v, dict) and "gdino" in v and "w" in v and not any("cls" in b for b in v["gdino"]):
            d = rules.all_configs(v, None)
            d.pop("proposed_meta")  # equals "proposed" without presence labels
            d.update(infer_width=v["w"], infer_height=v["h"])
            out[k] = d
        elif isinstance(v, dict):
            out[k] = v
    return out


def local_images(folder):
    found = {}
    if folder and os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(name)
            if ext.lower() in IMAGE_EXT:
                found[stem] = os.path.join(folder, name)
    return found


def build_frames(images, drafts, images_csv=None):
    frames = []
    if images_csv:
        for r in csv.DictReader(open(images_csv, encoding="utf-8")):
            k = r["key"]
            frames.append({"key": k, "work": r.get("work", ""), "page_type": r.get("page_type", ""),
                           "width": int(r.get("canvas_width") or 0), "height": int(r.get("canvas_height") or 0),
                           "canvas_uri": r.get("canvas_uri", ""), "has_local": k in images,
                           "img_src": f"/img/{k}" if k in images else r.get("image_url", "")})
    else:
        for k in sorted(set(images) | set(drafts)):
            if k not in images:
                continue
            frames.append({"key": k, "work": "", "page_type": "", "width": 0, "height": 0, "canvas_uri": "",
                           "has_local": True, "img_src": f"/img/{k}"})
    return frames


def load_state(path):
    if path and os.path.exists(path):
        s = json.load(open(path, encoding="utf-8"))
        return {k: s.get(k, {}) for k in EMPTY}
    return json.loads(json.dumps(EMPTY))


def save_state(path, state):
    if os.path.exists(path):
        shutil.copyfile(path, os.path.splitext(path)[0] + ".prev.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def make_handler(cfg):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, ctype):
            b = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def send_json(self, obj):
            self.send(200, json.dumps(obj, ensure_ascii=False), "application/json")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self.send(200, open(HTML, "rb").read(), "text/html; charset=utf-8")
            elif path == "/frames.json":
                self.send_json(cfg["frames"])
            elif path == "/predictions.json":
                self.send_json(cfg["drafts"])
            elif path == "/state":
                self.send_json(load_state(cfg["out"]))
            elif path == "/blind_state":
                self.send_json(load_state(cfg["blind_out"]))
            elif path.startswith("/img/"):
                p = cfg["images"].get(path[len("/img/"):])
                if p:
                    ctype = "image/png" if p.lower().endswith(".png") else "image/jpeg"
                    self.send(200, open(p, "rb").read(), ctype)
                else:
                    self.send(404, "image not found", "text/plain")
            else:
                self.send(404, "not found", "text/plain")

        def do_POST(self):
            path = self.path.split("?")[0]
            target = {"/state": cfg["out"], "/blind_state": cfg["blind_out"]}.get(path)
            if not target:
                return self.send(404, "not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n))
            save_state(target, {k: d.get(k, {}) for k in EMPTY})
            self.send(200, "ok", "text/plain")

        def log_message(self, *args):
            pass

    return Handler


def make_server(images_dir, drafts_path=None, images_csv=None, out="annotations.json",
                blind_out="blind_annotations.json", port=8795):
    images = local_images(images_dir)
    drafts = normalize_drafts(json.load(open(drafts_path, encoding="utf-8"))) if drafts_path else {}
    cfg = {"images": images, "drafts": drafts, "frames": build_frames(images, drafts, images_csv),
           "out": os.path.abspath(out), "blind_out": os.path.abspath(blind_out)}
    if not cfg["frames"]:
        raise SystemExit(f"no images in {images_dir} (.jpg, .jpeg, .png named <key>.<ext>)")
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(cfg)), cfg


def serve(a):
    httpd, cfg = make_server(a.images_dir, a.drafts, a.images_csv, a.out, a.blind_out, a.port)
    print(f"{len(cfg['frames'])} images. Open http://127.0.0.1:{httpd.server_address[1]}/")
    print(f"  review: {cfg['out']}\n  blind:  {cfg['blind_out']}\nCtrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
