"""Boxes as a W3C Web Annotation page (IIIF Presentation 3 AnnotationPage).

Each box becomes one annotation with motivation "tagging": the body is the class name and the
target is the image or Canvas with a `#xywh=` fragment. The page can be loaded by IIIF viewers
that display annotations, or merged into a manifest's `annotations`.
"""

CONTEXT = "http://iiif.io/api/presentation/3/context.json"


def annotation_page(boxes, target, page_id=None):
    base = page_id or f"{target['source']}/shohon-bbox"
    items = []
    for i, b in enumerate(boxes, 1):
        items.append({
            "id": f"{base}/annotation/{i}",
            "type": "Annotation",
            "motivation": "tagging",
            "body": {"type": "TextualBody", "value": b["class"], "format": "text/plain", "language": "en"},
            "target": b["fragment"],
            "generator": {"type": "Software", "name": f"shohon-bbox ({b['source']}, score {b['score']})"},
        })
    return {"@context": CONTEXT, "id": base, "type": "AnnotationPage", "items": items}
