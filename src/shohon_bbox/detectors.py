"""Raw outputs of the three detectors (Configurations 1–3 before label mapping).

The models are downloaded from the Hugging Face Hub at the revisions used in the paper.
Prompts and thresholds are those of the paper's Section 3.1 and Appendix. Inference runs on the
CPU with 8 threads, as in the paper; results on other hardware or library versions can differ
slightly.

Requires the optional dependencies: pip install "shohon-bbox[detect]"
"""
import os

MODELS = {
    "gdino": ("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71", None),
    "owlv2": ("google/owlv2-base-patch16-ensemble", "cfd3195ba4ea9592eec887ded089f4c08eff231d", None),
    "ndl": ("nakamura196/yolov8-ndl-layout", "538ce5fa010434714a4f6d075586fd40e1b23e07", "model_- 19 may 2024 15_13.pt"),
}

GD_PROMPT = "open book . book . color chart . ruler . paper label ."
GD_THRESHOLD, GD_TEXT_THRESHOLD = 0.22, 0.20
OW_TEXTS = [["an open book", "a book", "a color chart", "a ruler", "a small paper label"]]
OW_THRESHOLD = 0.18
NDL_CONF = 0.20


def model_path(name):
    """Local path of a pinned model (downloaded on first use)."""
    from huggingface_hub import hf_hub_download, snapshot_download
    repo, rev, fname = MODELS[name]
    if fname:
        return hf_hub_download(repo, fname, revision=rev)
    return snapshot_download(repo, revision=rev)


def _round_box(b):
    return [round(float(x), 1) for x in b]


class Detectors:
    """Loads the three detectors once; call .run(image) for each PIL image."""

    def __init__(self, threads=8):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  Owlv2ForObjectDetection, Owlv2Processor)
        os.environ.setdefault("YOLO_VERBOSE", "False")
        from ultralytics import YOLO
        self.torch = torch
        torch.set_num_threads(threads)
        p = model_path("gdino")
        self.gd_proc = AutoProcessor.from_pretrained(p)
        self.gd_mod = AutoModelForZeroShotObjectDetection.from_pretrained(p).eval()
        p = model_path("owlv2")
        self.ow_proc = Owlv2Processor.from_pretrained(p)
        self.ow_mod = Owlv2ForObjectDetection.from_pretrained(p).eval()
        self.ndl = YOLO(model_path("ndl"))

    def run(self, im):
        """Raw outputs for one RGB PIL image, in the format of data/raw/raw_detector_outputs.json."""
        torch = self.torch
        w, h = im.size
        e = {"w": w, "h": h}
        inp = self.gd_proc(images=im, text=GD_PROMPT, return_tensors="pt")
        with torch.no_grad():
            out = self.gd_mod(**inp)
        res = self.gd_proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=GD_THRESHOLD, text_threshold=GD_TEXT_THRESHOLD, target_sizes=[(h, w)])[0]
        lbls = res.get("text_labels", res.get("labels", []))
        e["gdino"] = [{"label": str(l), "score": round(float(s), 4), "box": _round_box(b)}
                      for b, s, l in zip(res["boxes"].tolist(), res["scores"].tolist(), lbls)]
        inp = self.ow_proc(text=OW_TEXTS, images=im, return_tensors="pt")
        with torch.no_grad():
            out = self.ow_mod(**inp)
        res = self.ow_proc.post_process_grounded_object_detection(out, threshold=OW_THRESHOLD, target_sizes=[(h, w)])[0]
        e["owlv2"] = [{"label": OW_TEXTS[0][int(l)], "score": round(float(s), 4), "box": _round_box(b)}
                      for b, s, l in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist())]
        r = self.ndl.predict(im, conf=NDL_CONF, verbose=False)[0]
        e["ndl_layout"] = [{"label": r.names[int(b.cls[0])], "score": round(float(b.conf[0]), 4),
                            "box": _round_box(b.xyxy[0].tolist())} for b in r.boxes]
        return e
