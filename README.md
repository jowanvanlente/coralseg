# 🪸 Coral Reef SegFormer Pipeline — Handover Package

This folder contains everything needed to:

1. **Train** a SegFormer model on coral reef images (in Google Colab)
2. **Generate** SAM-derived training masks from CoralNet point annotations (in Colab)
3. **Test & compare** trained models visually (local Streamlit webapp)
4. **Convert** sparse point annotations to dense COCO masks (local Streamlit webapp)

---

## 📁 Folder structure

```
handover/
├── README.md                              ← you are here
│
├── notebooks/                             ← run these in Google Colab
│   ├── segformer_train.ipynb              ← TRAIN the segmentation model
│   └── batch_export_sam.ipynb             ← GENERATE SAM masks from points
│
├── webapp_segformer_predict/              ← TEST trained models (local Streamlit)
│   ├── segformer_predict_app.py
│   └── requirements.txt
│
└── webapp_sam_pixel_to_mask/              ← CONVERT points → COCO masks (local Streamlit)
    ├── webapp.py
    ├── labelset.json
    ├── utils.py
    ├── coco_export.py
    ├── adaptive_segmentation.py
    ├── confidence_scoring.py
    ├── graph_segmentation.py
    ├── graph_first_segmentation.py
    ├── hybrid_segmentation.py
    ├── region_merging.py
    ├── superpixel_labeling.py
    └── requirements.txt
```

---

## 🚀 Quick start (the 30-second version)

| What you want to do | Where to go |
|---------------------|-------------|
| Train a new SegFormer model | Open `notebooks/segformer_train.ipynb` in Colab |
| Generate SAM masks for stage 3 training | Open `notebooks/batch_export_sam.ipynb` in Colab |
| Visually inspect a trained model | `cd webapp_segformer_predict && streamlit run segformer_predict_app.py` |
| Convert CoralNet points → dense COCO masks | `cd webapp_sam_pixel_to_mask && streamlit run webapp.py` |

---

## 📓 1. Notebooks (Google Colab)

Both notebooks are **self-contained** — they install their own dependencies, mount your Google Drive, and download model checkpoints automatically. **Don't run them locally** — open them in Colab.

### `segformer_train.ipynb` — train the segmentation model

The main training pipeline. Trains a 2-head SegFormer-B2:
- **Head A**: pretrained Coralscapes (39 classes)
- **Head B**: your custom merged class taxonomy (~35 classes)

**Drive layout it expects:**
```
/MyDrive/coral_training/
├── images/                                 ← all coral .jpg files (flat or nested)
├── annotations_coralnet.csv                ← CoralNet point export
├── gold_coco.json                          ← Roboflow expert polygons (optional)
└── sam_coco.json                           ← output of batch_export_sam.ipynb (optional)
```

**Quick Settings cell (top of notebook):**
```python
TRAINING_FLOW          = "124"          # Which stages: 1=warmstart, 2=points, 3=SAM, 4=gold
RESUME_FROM_EXPERIMENT = None           # or "exp_name" to continue a previous run
FORCE_RERENDER         = False          # True = rebuild masks (after data changes)
BATCH_SIZE             = 8              # drop to 4 if out of memory
INPUT_SIZE             = 512            # input resolution
KEEP_TOP_N_CLASSES     = 35             # None = keep all merged classes
USE_LABEL_MERGE        = True           # merge 94 → ~49 broader classes
STAGE4_LOSS            = "dice_ce"      # "ce" (original), "dice_ce" (preserves edges)
MODEL_NAME             = "nvidia/mit-b2"
```

**Cell-by-cell map:**

| Cell | Purpose |
|------|---------|
| 1 (Quick Settings) | Edit knobs without scrolling — overrides Cell 2 defaults |
| 2 (Setup) | pip installs, Google Drive mount, imports |
| 3 (Config) | All other settings (paths, advanced knobs); reads Quick Settings via `globals().get()` |
| 4 (Coralscapes) | Optional: download Coralscapes dataset for joint training |
| 5 (Class taxonomy) | Reads CSV, applies label merge, builds `YOUR_CLASSES` |
| 6 (Render masks) | Generates training PNG masks from CSV/COCO; smart image resolver; cached |
| 7 (Preview 5b) | Visual sanity check before training: shows random masks per stage |
| 8 (Dataset/loaders) | DataLoader with augmentations |
| 9 (Model) | Builds 2-head SegFormer; resumes from checkpoint if exists |
| 10 — | (blank divider) |
| 11 (Training loop) | Multi-stage training loop with stage-aware loss (Dice for points, dice_ce for polygons) |
| 12 (Validation) | mIoU, per-class IoU, confusion matrix, **point accuracy vs CoralNet 72%** |
| 13 (HTML report) | Generates report for biologist with comparison metrics & explanations |
| 14 (Visual preview) | Predictions on N test images |
| 15 (Export bundle) | Saves `bundle.pt` for the prediction webapp + `summary.json` + `README.md` |
| 16 (Compare experiments) | Side-by-side comparison of two prior runs |
| 17 (Pseudo-labels) | Optional: generate pseudo-labels for next training iteration |
| 18 (Track record) | Scans all experiments → builds `runs_overview.md` with comparison table |

**What changes between runs you might want:**
- `STAGE_CONFIGS` (in Cell 3) — epochs and LR per stage
- `STAGE4_LOSS` — `"ce"` for max mIoU, `"dice_ce"` for sharper edges
- `KEEP_TOP_N_CLASSES` — fewer = higher mIoU but less biological detail
- `_LABEL_MERGE_MAP` (in Cell 3) — which CoralNet labels get merged

**Outputs (saved to Drive):**
```
segformer_runs/experiments/<experiment_name>/
├── best.pt                  ← training checkpoint (resume only)
├── final.pt                 ← last-epoch weights
├── bundle.pt                ← self-contained model for the prediction webapp ⭐
├── summary.json             ← all settings + metrics
├── README.md                ← per-experiment human-readable summary
├── history.json             ← per-epoch loss/mIoU
├── confusion_matrix.png
├── headB_confusion_matrix.png
├── metrics_full.json        ← per-class IoU, point accuracy
└── report.html              ← shareable report for the biologist
```

`runs_overview.md` is also saved at the experiments root, summarising **all runs** in one place.

### `batch_export_sam.ipynb` — generate SAM masks (stage 3 input)

Converts CoralNet point annotations into dense SAM-generated polygons. The output `sam_coco.json` becomes the supervision for stage 3 of training.

**When to run it:** Only if you want to add stage 3 (SAM masks) to your training flow. With `TRAINING_FLOW = "14"` or `"124"` you can skip it. With `"234"` or `"1234"` you need it.

**Time cost:** ~3-5 sec/image × thousands of images = several hours on Colab T4. Has crash-safe resume, so disconnects don't lose progress.

**Cell-by-cell map:**

| Cell | Purpose |
|------|---------|
| 1 | Install deps, download SAM checkpoint, mount Drive |
| 2 | Configuration (Drive path, SAM model size, AMG params) |
| 3 | Load CSV, index images (smart resolver: handles case/double-ext/Roboflow hashes) |
| 4 | Load SAM model |
| 5 | Preview on N images — **always inspect this before running the full batch** |
| 6 | Process all images with resume support |
| 7 | Build final `sam_coco.json` with embedded settings |
| 8 | Stats & quality check |

**Settings to know (Cell 2):**
- `SAM_MODEL_TYPE = 'vit_b'` — fastest, good enough. Use `'vit_h'` for better quality (3× slower)
- `AMG_POINTS_PER_SIDE` — density of mask candidates SAM generates (32 = balanced)
- `AMG_PRED_IOU_THRESH` / `AMG_STABILITY_THRESH` — quality filters

---

## 🖥️ 2. Webapps (run locally)

Both webapps are Streamlit apps. Open them in a browser at `http://localhost:8501` after launching.

### `webapp_segformer_predict/` — test trained models

**Purpose:** Load one or two trained `bundle.pt` files and visualise predictions on coral images. Compare two models side-by-side on the same image.

**Setup (one-time):**
```bash
cd webapp_segformer_predict
python -m venv .venv
.venv\Scripts\activate                  # Windows
# source .venv/bin/activate             # macOS/Linux
pip install -r requirements.txt
```

**Run:**
```bash
streamlit run segformer_predict_app.py
```

**Features:**
- Upload `bundle.pt` from Drive (download from `experiments/<name>/bundle.pt`) — **not** `best.pt`
- Pick image: quick-load buttons (set folder once), upload, or paste path
- View original | prediction with class labels | overlay blend
- Class coverage table (which classes occupy what % of the image)
- **🆕 Comparison mode:** load a second bundle in the sidebar — both predictions render side-by-side, with a pixel-agreement metric if class lists match
- Settings: mask opacity, minimum region size for labels, color legend
- Bundle and image folder paths are remembered across sessions

**Tip — typical comparison workflow:**
1. Train a "stage 2 only" model (`TRAINING_FLOW = "12"`) → download bundle as `points_bundle.pt`
2. Train a "stage 4" fine-tune (`TRAINING_FLOW = "14"`) → download bundle as `gold_bundle.pt`
3. In the webapp: load `points_bundle.pt` as Model A, `gold_bundle.pt` as Model B
4. Cycle through test images — see whether stage 4 improved or over-smoothed predictions

### `webapp_sam_pixel_to_mask/` — convert CoralNet points → COCO masks

**Purpose:** Take CoralNet sparse point annotations (one label per pixel-grid point) and produce dense polygon masks ready for Roboflow / training. Three segmentation methods available: SLIC superpixel, adaptive density-based, and graph-based (Felzenszwalb).

This is the **same logic** as `batch_export_sam.ipynb` but runs locally and lets you tune parameters interactively. Useful when you want to understand the conversion behaviour before running the batch on thousands of images.

**Setup (one-time):**
```bash
cd webapp_sam_pixel_to_mask
python -m venv .venv
.venv\Scripts\activate                  # Windows
# source .venv/bin/activate             # macOS/Linux
pip install -r requirements.txt
```

**Run:**
```bash
streamlit run webapp.py
```

**Features:**
- **Test mode:** visualise segmentation on one image — adjust knobs, see results live
- **Export mode:** process multiple images, download a COCO JSON
- Three segmentation methods (pick whichever produces masks closest to your gold standard)
- Built-in sample image with 900 annotations for testing without your own data

**File structure (why all those .py files?):**
- `webapp.py` — main Streamlit UI
- `coco_export.py` — assembles the final COCO JSON
- `utils.py` — image/CSV loading, resize, etc.
- `superpixel_labeling.py` — SLIC method
- `adaptive_segmentation.py` — adaptive density-based
- `graph_segmentation.py`, `graph_first_segmentation.py`, `hybrid_segmentation.py` — graph-based variants
- `region_merging.py` — post-process: merge adjacent same-label regions
- `confidence_scoring.py` — quality scores per region
- `labelset.json` — class taxonomy + colours

You shouldn't normally need to edit any of these — the webapp's UI exposes all the relevant knobs.

---

## 🔄 Typical end-to-end workflow

```
1. Annotate points in CoralNet → export CSV
2. (Optional) Use webapp_sam_pixel_to_mask to generate dense masks → import into Roboflow
3. (Optional) Use Roboflow to refine polygons → export gold_coco.json
4. Run batch_export_sam.ipynb in Colab → produces sam_coco.json (only if using stage 3)
5. Run segformer_train.ipynb in Colab → produces bundle.pt
6. Test predictions in webapp_segformer_predict (local) → iterate
7. Export report.html → share with biologist
```

---

## ❓ FAQ

**Q: Why two webapps, not one?**  
The prediction webapp tests trained models. The pixel-to-mask webapp generates training data. Different purposes; different audiences (modeler vs annotator); kept separate.

**Q: Why are the notebooks not local-runnable?**  
They install heavy GPU packages and mount Google Drive — both Colab-specific. Run them in Colab; the Quick Settings cell at the top of `segformer_train.ipynb` is your only knob.

**Q: I get "Bundle is missing required keys" when loading in the predict webapp.**  
You uploaded `best.pt` instead of `bundle.pt`. `best.pt` is a training checkpoint (resume only); `bundle.pt` is the self-contained file the webapp needs. Both live in the experiment folder — pick `bundle.pt`.

**Q: I get OOM errors during training.**  
In Quick Settings, drop `BATCH_SIZE` from 8 to 4. If still OOM, drop `INPUT_SIZE` from 512 to 384.

**Q: What's a sensible mIoU target?**  
mIoU 0.30+ is competitive vs CoralNet's 72% point accuracy. The notebook computes both metrics — see Cell 12's "Point accuracy" section for the apples-to-apples comparison.

**Q: Compare mode shows "different class lists" warning.**  
That's expected if Model A and Model B were trained with different `KEEP_TOP_N_CLASSES` or merge maps. The pixel-agreement metric isn't computed but the visual side-by-side still works.

**Q: How do I revert the new Dice+CE loss to plain CE?**  
In Quick Settings, change `STAGE4_LOSS = "dice_ce"` → `STAGE4_LOSS = "ce"`. Re-train. No other code changes needed.

---

## 📞 Where to look when something breaks

- **Training crashes mid-epoch:** check Cell 12 (training loop) error trace — usually a CUDA OOM or a malformed COCO file
- **All images "not found":** Cell 6 has a smart resolver that handles `.JPG`/`.jpg`/double-ext/Roboflow hashes. If still missing, check `YOUR_IMAGES_DRIVE` path in Cell 3
- **Webapp won't load bundle:** confirm you're loading `bundle.pt`, not `best.pt`
- **mIoU stuck at 0:** check that `_current_stage` is set and the right loss is firing — Cell 11 prints the active stage at the start of each
- **Track record cell empty:** `EXPERIMENTS_DIR` doesn't exist yet — train at least one model first

---

That's the whole pipeline. Start with `notebooks/segformer_train.ipynb` if you want to train; with `webapp_segformer_predict/` if you have a `bundle.pt` ready to inspect.
