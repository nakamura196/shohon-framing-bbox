import json
import os
import threading
import urllib.request

from shohon_bbox import evaluate, review

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _raw():
    raw = json.load(open(os.path.join(ROOT, "data", "raw", "raw_detector_outputs.json"), encoding="utf-8"))
    k = sorted(raw)[0]
    return k, raw[k]


def test_drafts_from_raw_outputs():
    k, e = _raw()
    d = review.normalize_drafts({k: e})[k]
    assert {"gdino", "owlv2", "ndl_layout", "ensemble_raw", "proposed"} <= set(d)
    assert "proposed_meta" not in d
    assert (d["infer_width"], d["infer_height"]) == (e["w"], e["h"])
    assert all(b["cls"] in ("book", "colour_chart", "ruler", "imaging_slip") for b in d["proposed"])


def test_drafts_from_box_lists():
    boxes = [{"cls": "book", "box": [0, 0, 10, 10]}]
    assert review.normalize_drafts({"a": boxes}) == {"a": {"proposed": boxes}}


def test_server_round_trip(tmp_path):
    from PIL import Image
    imgs = tmp_path / "images"
    imgs.mkdir()
    Image.new("RGB", (40, 30), "white").save(imgs / "k1.jpg")
    out, blind = tmp_path / "ann.json", tmp_path / "blind.json"
    httpd, cfg = review.make_server(str(imgs), out=str(out), blind_out=str(blind), port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        frames = json.load(urllib.request.urlopen(base + "/frames.json"))
        assert [f["key"] for f in frames] == ["k1"] and frames[0]["img_src"] == "/img/k1"
        assert urllib.request.urlopen(base + "/img/k1").read()[:2] == b"\xff\xd8"
        assert b"shohon-bbox review" in urllib.request.urlopen(base + "/").read()
        state = {"boxes": {"k1": [{"cls": "ruler", "box": [1, 2, 30, 8]}]}, "verified": {"k1": True},
                 "modified": {"k1": True}, "sizes": {"k1": [40, 30]}}
        for path, f in (("/state", out), ("/blind_state", blind)):
            req = urllib.request.Request(base + path, data=json.dumps(state).encode(), method="POST")
            assert urllib.request.urlopen(req).read() == b"ok"
            assert json.load(open(f)) == state
        assert json.load(urllib.request.urlopen(base + "/state")) == state
        try:
            urllib.request.urlopen(base + "/img/..%2Fann")
            assert False, "unknown image served"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_evaluate_reads_review_file():
    state = {"boxes": {"a": [{"cls": "book", "box": [0, 0, 10, 10]}], "b": [{"cls": "ruler", "box": [0, 0, 5, 1]}]},
             "verified": {"a": True}, "modified": {}, "sizes": {}}
    gold = evaluate.load_gold(state)
    assert list(gold) == ["a"] and gold["a"][0]["label"] == "book"
