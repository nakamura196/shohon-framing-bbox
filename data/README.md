# Book, colour-chart, ruler and imaging-slip boxes for 102 Japanese classical book images

Bounding boxes in four classes for 102 IIIF images (102 different book witnesses) from the
Kokusho Database (National Institute of Japanese Literature), together with independent
blind annotations of 20 images by two annotators and the predictions of the seven detection configurations
compared in the accompanying paper.

The images themselves are **not** redistributed. Each image is referenced by its IIIF Canvas URI,
the IIIF Image API URL that was downloaded, and the SHA-256 of the downloaded file.

## Classes

| class | what is boxed |
|---|---|
| `book` | the book object: an opening, a cover, a scroll section, an attached certificate or a storage box |
| `colour_chart` | colour charts and grey scales |
| `ruler` | rulers; each arm of an L-shaped ruler is a separate box |
| `imaging_slip` | slips placed outside the book: shot labels, caption slips |

Frame-number plates (white numerals on a black plate) placed on the copy stand are not boxed in the ground truth
and belong to no class.

Ground truth: 102 images, 299 boxes (`book` 102, `colour_chart` 76, `ruler` 111, `imaging_slip` 10).

## Files

| file | content |
|---|---|
| `images.csv` | one row per image (see below) |
| `gold/gold102.csv` | ground truth, reviewed by the author starting from the drafts in `drafts/` |
| `drafts/review_drafts.csv` | the drafts shown during review: output of an earlier implementation of configuration 6, whose code was not preserved. Configurations 5 and 6 in `predictions/` come from the released code (`src/shohon_bbox/rules.py`) and differ from these drafts on 22 images |
| `blind20/author_as_drawn.csv` | 20 images drawn by the author on a blank screen without any draft (48 boxes), as drawn |
| `blind20/author_corrected.csv` | the same, after the author corrected the one disagreement with the ground truth (a colour chart drawn as `ruler`) after comparing the two |
| `blind20/second_annotator.csv` | the same 20 images drawn on a blank screen by a second annotator not involved in developing the method (52 boxes). The annotation guide given to this annotator included frame-number plates under `imaging_slip`; all four `imaging_slip` boxes in this file are such plates and were excluded when comparing with the ground truth |
| `predictions/configN_*.csv` | predictions of configurations 1–7; `predictions/configs.csv` names them |
| `raw/raw_detector_outputs.json` | raw outputs of Grounding DINO, OWLv2 and YOLOv8-NDL-Layout for the 102 images (labels as returned by each model, scores, boxes in evaluation-image pixels); `paper/reproduce.py` recomputes configurations 1–6 from them |
| `predictions/config7_yolo11x_meta.json` | training settings, software versions and timings of configuration 7 |
| `negatives/images.csv` | the 74 images without a ruler or a colour chart used in Section 4.3 of the paper (one per book, drawn from the companion dataset; key, work, page type, IIIF image URL) |
| `negatives/raw_detector_outputs.json` | raw outputs of the three detectors on these 74 images, reduced to a longer side of 1,400 pixels |
| `negatives/excluded_books.txt` | the 300 books of the paper's sample, excluded when drawing the 74 images |

`images.csv` columns: `key` (`<book id>-<4-digit canvas position>`), `order` (review order; the 102 images
are the first 102 of a 300-image sample in that order, not a random subset), `bid` (Kokusho book id),
`canvas_index` (1-based position in the manifest), `canvas_uri`, `manifest_uri`, `image_url`,
`canvas_width`/`canvas_height` (full Canvas size), `image_width`/`image_height` (downloaded image),
`image_sha256` (SHA-256 of the downloaded file), `pixel_sha256` (SHA-256 of the decoded RGB pixels,
row-major, 8 bits per channel, decoded with Pillow), `dhash` (64-bit difference hash), `downloaded` (file timestamp of the downloaded image; the download date itself was not logged),
`eval_width`/`eval_height` (the image used for detection and evaluation: the downloaded image reduced to
1,400 px (46 images) or 640 px (56 images) on the long side),
`work`, `title`, `kansha` (manuscript or printed), `institution`, `license_category`, `license`,
`page_type`, `cv_fold` (fold 0–4 used for configuration 7), `in_blind20`.

Box files, one row per box: `key`, `box_index`, `class`, `score` (empty for human boxes),
`x1 y1 x2 y2` (pixels of the evaluation image, `eval_width` × `eval_height`), `nx1 ny1 nx2 ny2` (normalised to [0, 1]),
`canvas_x canvas_y canvas_w canvas_h` (pixels of the full IIIF Canvas) and `canvas_fragment`
(`<canvas_uri>#xywh=…`, usable as a IIIF/Web Annotation target).

## Configurations

1. Grounding DINO (`IDEA-Research/grounding-dino-tiny`) alone
2. OWLv2 (`google/owlv2-base-patch16-ensemble`) alone
3. YOLOv8-NDL-Layout alone
4. Ensemble of 1–3 with NMS only
5. Proposed: ensemble with spatial rules, visual information only
6. Proposed with page-level presence labels from the companion dataset
7. YOLO11x fine-tuned on the ground truth, five-fold cross-validation; each image is predicted by the
   model that did not see it (out-of-fold). Folds are stratified by the presence of slips, charts and rulers.

Configurations 1–6 are given at the operating thresholds used in the paper; configuration 7 at confidence 0.25.

## Reproducing the paper's numbers

`python paper/reproduce.py` (in the repository root) recomputes configurations 1–6 from the raw
detector outputs, checks them against `predictions/`, re-runs the paper's evaluation and checks every
reported value (micro F1@0.5: 1 0.405, 2 0.561, 3 0.348, 4 0.488, 5 0.919, 6 0.927, 7 0.746).

## Notes

- For 12 books whose Canvas ids do not follow their position in the manifest (e.g. the first Canvas is
  `canvas/0`), `canvas_uri` was taken from the manifest by matching the downloaded image's IIIF service.
- Boxes are clipped to the image. In the ground truth, 24 boxes on 21 images (23 rulers, 1 book) extended a few pixels
  beyond the image edge when drawn; the paper scored them unclipped. This changes only the mean IoU of
  matched boxes, in the fourth decimal, and none of the values reported in the paper.
- Licences of the images differ by institution and are listed per image; they apply to the images,
  not to these annotations.

- On 2026-11-12 all 102 images were downloaded again. For 90 the file SHA-256 was unchanged. For the other
  12 the server returns different bytes on every request while the decoded pixels are identical
  (`pixel_sha256` unchanged), so compare `pixel_sha256` rather than `image_sha256`.

## Licence

Annotations and predictions: CC BY 4.0. Code: MIT License.

## Citation

See `CITATION.cff` in the repository root. The paper is under review; its reference will be added here.
