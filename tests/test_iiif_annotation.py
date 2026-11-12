from shohon_bbox import annotation, iiif


def test_canvases_v3():
    m = {"items": [{"id": "https://ex.org/c/1", "width": 4000, "height": 3000, "items": [{"items": [{"body": {
        "id": "https://ex.org/i/1/full/max/0/default.jpg", "service": [{"id": "https://ex.org/i/1", "type": "ImageService3"}]}}]}]}]}
    assert iiif.canvases(m) == [("https://ex.org/c/1", 4000, 3000, "https://ex.org/i/1/full/2000,/0/default.jpg")]


def test_canvases_v2():
    m = {"sequences": [{"canvases": [{"@id": "https://ex.org/c/1", "width": 4000, "height": 3000, "images": [{"resource": {
        "@id": "https://ex.org/i/1/full/full/0/default.jpg", "service": {"@id": "https://ex.org/i/1"}}}]}]}]}
    assert iiif.canvases(m) == [("https://ex.org/c/1", 4000, 3000, "https://ex.org/i/1/full/2000,/0/default.jpg")]


def test_to_target_scales_and_clips():
    boxes = [{"box": [-2.0, 10.0, 70.0, 35.0], "cls": "ruler", "score": 0.9, "source": "owlv2"}]
    target = {"source": "https://ex.org/c/1", "width": 1400, "height": 700}
    out = iiif.to_target(boxes, (140, 70), target)
    assert out[0]["xywh"] == [0, 100, 700, 250]
    assert out[0]["fragment"] == "https://ex.org/c/1#xywh=0,100,700,250"


def test_reduce_never_enlarges():
    from PIL import Image
    im = Image.new("RGB", (800, 600))
    assert iiif.reduce(im, 1400).size == (800, 600)
    assert iiif.reduce(Image.new("RGB", (2800, 1400)), 1400).size == (1400, 700)


def test_annotation_page():
    boxes = [{"class": "book", "score": 0.5, "source": "ndl", "xywh": [1, 2, 3, 4], "fragment": "https://ex.org/c/1#xywh=1,2,3,4"}]
    page = annotation.annotation_page(boxes, {"source": "https://ex.org/c/1"})
    assert page["type"] == "AnnotationPage"
    a = page["items"][0]
    assert a["target"] == "https://ex.org/c/1#xywh=1,2,3,4"
    assert a["body"]["value"] == "book"
