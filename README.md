# DocIntel

A semantic segmentation model that looks at a scanned-looking document image and labels every pixel as one of: background, text, table, image, signature, barcode, or form field. Think of it as the "which region is what" step that would normally sit in front of OCR or a document-extraction pipeline, so downstream steps know where to look.

The backbone is a torchvision DeepLabV3 with a ResNet-50 encoder, pretrained on COCO, with the classifier head swapped out for 7 classes. Nothing exotic — this is mostly about getting a clean, working segmentation loop (data, loss, metrics, checkpointing, dashboard) rather than chasing a novel architecture.

## The data is synthetic, and that matters

There's no real scanned-document dataset in this repo. `src/data.py` procedurally paints 224×224 "documents": white background, then randomly placed text blocks, a ruled table, an image patch, a sine-wave squiggle standing in for a signature, a black/white bar-code pattern, and boxed form fields — each stamped into a pixel mask with its class id as it's drawn. It's a cheap way to get paired image/mask data without labeling anything by hand, and it's genuinely useful for exercising the training loop, but it is not a proxy for real document layout variance (rotation, scanning noise, overlapping regions, handwriting, low-quality photos of paper). `data/README.md` points at CORD and FUNSD as real layout datasets you could drop into `data/raw/`, but nothing in the current pipeline reads from there yet — that's a genuine gap, not a hidden feature.

## Training

```
python train.py --n 500 --epochs 30 --lr 0.001 --batch-size 16 --device cuda
```

Falls back to CPU automatically if CUDA isn't available. The loss is a 50/50 mix of cross-entropy and Dice (`src/core.py`), which is the standard trick for segmentation tasks where one class (background, here) dominates the pixel count and cross-entropy alone would happily ignore everything else. Optimizer is AdamW with `ReduceLROnPlateau` watching validation mIoU, mixed precision via `torch.amp`. The best checkpoint (by val mIoU) gets written to `models/best_model.pt`, and a metrics dict — best mIoU, final val mIoU/accuracy, epoch count, sample count, device — gets written to `models/metrics.json`.

One honest note: the `models/metrics.json` and `models/model.pkl` currently sitting in this repo are leftovers from an earlier version of this project that trained tabular classifiers (logistic regression / random forest / gradient boosting / XGBoost) on hand-engineered document features instead of doing pixel-level segmentation. That code doesn't exist anymore — `src/` only has the segmentation pipeline now — so those two files are stale and don't describe the current model. I'm not going to quote numbers from them here since they'd be measuring something else entirely. Run `train.py` yourself to get metrics that actually match this architecture.

## Running the dashboard

```
streamlit run app.py
```

It regenerates a batch of synthetic documents (slider controls how many, 100–2000), loads `models/best_model.pt` if it exists, and runs inference on a validation split. If no checkpoint is present it'll happily run with the pretrained backbone and a randomly initialized head — which means the metrics you see out of the box will look bad until you actually train something. That's expected, not a rendering bug.

Four tabs:

- **Explorer** — side-by-side original image / ground-truth mask / predicted mask for a few samples, so you can eyeball where the model is going wrong.
- **Model Lab** — per-class IoU as a table and a bar chart.
- **Layout Quality** — the mIoU/pixel-accuracy formulas rendered as LaTeX (mostly for my own reference while building this), plus a breakdown of what fraction of pixels each class actually occupies in the current batch — useful context since IoU on a rare class is noisy.
- **Document Types** — currently just a placeholder describing an idea (a second head off the same backbone for document-type classification, e.g. invoice vs. receipt vs. contract) that isn't built. It's an honest stub, not a hidden feature.

## What I'd fix next

- Swap the synthetic generator for real layout data (CORD/FUNSD) — the model currently has no idea what a real scanned page looks like.
- The document-type classification tab is aspirational; either build the multi-task head or remove the tab.
- `tests/test_smoke.py` covers the data generator, dataloader shapes, model forward pass, and metric bounds, but nothing exercises `app.py` — a Streamlit smoke test (or at least importing it) would catch regressions like the one below.
- While going through this, I found `app.py` was importing `matplotlib.pyplot` *after* using it inside the "Layout Quality" tab, which would have crashed that tab on load — fixed by moving the import to the top of the file.
- `requirements.txt` still listed `pandas`, `scikit-learn`, and `scipy` from the earlier tabular-classifier version of this project, none of which the current code imports — trimmed down to what's actually used (`torch`, `torchvision`, `numpy`, `streamlit`, `matplotlib`, `tqdm`).

## Layout

```
DocIntel/
  src/data.py            synthetic document + mask generator, dataloaders
  src/model.py            DeepLabV3 build + train/validate loop
  src/core.py             CE+Dice loss, mIoU / pixel-accuracy metrics
  src/visualizations.py   matplotlib helpers for the dashboard
  src/persist.py          plain state-dict save/load helpers
  train.py                CLI training entrypoint
  app.py                  Streamlit dashboard
  tests/test_smoke.py     pytest sanity checks on data/model/metrics
  models/                 checkpoint + metrics output (gitignored)
```

MIT licensed.
