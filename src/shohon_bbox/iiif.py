"""Loading an input image (file, image URL, or IIIF Canvas) and mapping boxes back to it.

Detection runs on the image reduced to `max_side` pixels on the long side (1,400 in the paper).
Boxes are then returned in the coordinates of the original image, or of the IIIF Canvas when the
input is a manifest and a Canvas, so that they can be used as `#xywh=` fragments.
"""
import io
import json
import os
import urllib.request

USER_AGENT = "shohon-bbox (https://github.com/nakamura196/shohon-framing-bbox)"
IIIF_WIDTH = 2000  # width requested from a IIIF Image service, as for the paper's images


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _open(data):
    from PIL import Image
    return Image.open(io.BytesIO(data)).convert("RGB")


def _label(v):
    if isinstance(v, dict):  # IIIF v3 language map
        for vals in v.values():
            return vals[0] if vals else ""
    return v or ""


def canvases(manifest):
    """[(canvas_id, width, height, image_url)] for a IIIF Presentation 2 or 3 manifest."""
    out = []
    if "items" in manifest:  # v3
        for c in manifest["items"]:
            body = c["items"][0]["items"][0]["body"]
            svc = body.get("service") or []
            svc = svc[0] if isinstance(svc, list) and svc else svc
            sid = svc.get("id") or svc.get("@id") if svc else None
            url = f"{sid}/full/{IIIF_WIDTH},/0/default.jpg" if sid else body["id"]
            out.append((c["id"], int(c["width"]), int(c["height"]), url))
    else:  # v2
        for c in manifest["sequences"][0]["canvases"]:
            res = c["images"][0]["resource"]
            svc = res.get("service")
            svc = svc[0] if isinstance(svc, list) and svc else svc
            sid = svc.get("@id") if svc else None
            url = f"{sid}/full/{IIIF_WIDTH},/0/default.jpg" if sid else res["@id"]
            out.append((c["@id"], int(c["width"]), int(c["height"]), url))
    return out


def load(src, canvas=None):
    """Return (PIL image, target) for a local file, an image URL, or a manifest URL with a Canvas.

    `canvas` is a 1-based position or a Canvas id. `target` is a dict with `source` (file path,
    image URL or Canvas id), `width` and `height` (size of the coordinate space of the output).
    """
    if os.path.exists(src):
        im = _open(open(src, "rb").read())
        return im, {"source": os.path.abspath(src), "width": im.width, "height": im.height, "canvas": False}
    data = _get(src)
    if canvas is None:
        try:
            im = _open(data)
            return im, {"source": src, "width": im.width, "height": im.height, "canvas": False}
        except Exception:
            raise SystemExit(f"{src} is not an image; for a IIIF manifest give --canvas")
    cs = canvases(json.loads(data))
    if str(canvas).isdigit():
        cid, cw, ch, url = cs[int(canvas) - 1]
    else:
        match = [c for c in cs if c[0] == canvas]
        if not match:
            raise SystemExit(f"Canvas {canvas} not in {src}")
        cid, cw, ch, url = match[0]
    im = _open(_get(url))
    return im, {"source": cid, "width": cw, "height": ch, "canvas": True, "manifest": src, "image_url": url}


def reduce(im, max_side=1400):
    """The image reduced to `max_side` on the long side (never enlarged)."""
    from PIL import Image
    s = max_side / max(im.size)
    if s >= 1:
        return im
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)


def to_target(boxes, det_size, target):
    """Boxes from detection-image pixels to target coordinates (integers, x, y, w, h)."""
    sx, sy = target["width"] / det_size[0], target["height"] / det_size[1]
    out = []
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        x, y = max(0, round(x1 * sx)), max(0, round(y1 * sy))
        w = min(target["width"], round(x2 * sx)) - x
        h = min(target["height"], round(y2 * sy)) - y
        out.append({"class": b["cls"], "score": b["score"], "source": b["source"], "xywh": [x, y, w, h],
                    "fragment": f"{target['source']}#xywh={x},{y},{w},{h}"})
    return out
