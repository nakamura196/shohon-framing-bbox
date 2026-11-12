# Changelog

## 0.1.0 — 2026-11-13

First release, accompanying the paper submitted to *Digital Scholarship in the Humanities*.

- `shohon-bbox detect`: boxes from an image file, image URL or IIIF Canvas, as JSON or a Web Annotation page
- `shohon-bbox rules`: the spatial rules applied to stored raw detector outputs
- `shohon-bbox detect --format raw`: raw detector outputs for a set of images, as drafts for review
- `shohon-bbox rules`: with `--all`, Configurations 1–5 for each image
- `shohon-bbox review`: local web interface to correct drafts, with a blind mode (no drafts or scores shown)
- `shohon-bbox evaluate`: the scoring used in the paper; reads JSON, files saved by `review`, and the CSV files of `data/`
- `data/`: the paper's dataset (CC BY 4.0), including the 74 images without rulers or charts of Section 4.3
- `paper/reproduce.py`: reproduces the paper's tables from `data/`, including Section 4.3 and the cross-validation folds
- `paper/yolo_cv.py`, `paper/negatives.py`, `paper/time_inference.py`: Configuration 7, Section 4.3 and the inference times
