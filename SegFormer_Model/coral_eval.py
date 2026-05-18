"""
SegFormer evaluation toolkit — standalone module used by segformer_eval.ipynb.

Loads a saved bundle.pt, evaluates it on a fixed canonical SAM-masks val set,
and generates a self-contained HTML report identical in shape to Cell 9b of
segformer_train.ipynb. Multiple bundles can be evaluated and compared side-by-side.

Why this exists separately from segformer_train.ipynb
-----------------------------------------------------
- Run multiple experiments, then evaluate them all on the SAME masks.
- No model-loading / training cells in the way; just config + eval.
- All models compared on the *exact same* SAM-derived val set so mIoU
  numbers across runs are directly comparable.

Constraints (current version)
-----------------------------
- Only models trained with KEEP_TOP_N_CLASSES=None are supported (i.e.
  every model uses the full class list derived from the CSV + label merge).
- SAM categories that don't appear in the canonical class list become
  IGNORE pixels — they don't penalize the model.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead

IGNORE_INDEX = 255
DEFAULT_INPUT_SIZE = 512
DEFAULT_MODEL_NAME = "nvidia/mit-b2"
PRETRAINED_CORALSCAPES = "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"

# ─── label merge (must match segformer_train.ipynb Cell 3) ───────────────────
# Imported from the canonical config so the eval val set is rendered with the
# same class taxonomy the model was trained on.
try:
    import importlib.util
    _here = os.path.dirname(os.path.abspath(__file__))
    _cfg_path = os.path.join(_here, "..", "input", "labelset", "class_merge_config.py")
    _spec = importlib.util.spec_from_file_location("class_merge_config", _cfg_path)
    _cfg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cfg)
    _EXCLUDED_LABELS = set(_cfg._EXCLUDED_LABELS)
    _LABEL_MERGE_MAP = dict(_cfg._LABEL_MERGE_MAP)
except Exception as e:
    print(f"Warning: Could not load class_merge_config.py from {_cfg_path}: {e}")
    print("Using empty merge map (no label merging).")
    _EXCLUDED_LABELS, _LABEL_MERGE_MAP = set(), {}


def apply_label_merge(label):
    """Map a CoralNet short code to its merged class, or None if excluded."""
    if label in _EXCLUDED_LABELS or pd.isna(label):
        return None
    return _LABEL_MERGE_MAP.get(label, label)


# ─── COCO preprocessor (mirror of training notebook's render path) ──────────
GENUS_TO_CORALNET = {
    'Acanthastrea': 'Aca', 'Acropora': 'Acr', 'Alveopora': 'Alv', 'Anomastraea': 'Ano',
    'Astrea': 'Astrea', 'Astreopora': 'Astreo', 'Blastomussa': 'Bla', 'Caulastrea': 'Caula',
    'Coscinaraea': 'Cos', 'Cyphastrea': 'Cyph', 'Diploastrea': 'Diplo', 'Dipsastraea': 'Dipsa',
    'Duncanopsammia': 'Dun', 'Echinophyllia': 'Echphy', 'Echinopora': 'Echpo', 'Favites': 'Favit',
    'Fungiidae': 'Fungii', 'Galaxea': 'Gal', 'Gardineroseris': 'Gar', 'Goniastrea': 'Gonia',
    'Goniopora': 'Gonio', 'Hard coral': 'HC', 'Hydnophora': 'Hydno', 'Isopora': 'Iso',
    'Leptastrea': 'Lepta', 'Leptoria': 'Leptor', 'Leptoseris': 'Leptos', 'Lobophyllia': 'Lobophyl',
    'Merulina': 'Mer', 'Micromussa': 'Micro', 'Montipora': 'Monti', 'Mycedium': 'Myc',
    'Oulophyllia': 'Oul', 'Oxypora': 'Oxy', 'Pachyseris': 'Pachy', 'Paramontastea': 'Para',
    'Paramontastraea': 'Para', 'Pavona': 'Pav', 'Pectinia': 'Pec', 'Physogyra': 'Phy',
    'Platygyra': 'Platy', 'Plerogyra': 'Plero', 'Plesiastrea': 'Plesia', 'Pocillopora': 'Poc',
    'Podabacia': 'Pod', 'Porites (branching)': 'PorB', 'Porites branching': 'PorB',
    'Porites (massive)': 'PorM', 'Porites massive': 'PorM', 'Psammocora': 'Psa',
    'Seriatopora': 'Ser', 'Stylophora': 'Styl', 'Symphyllia': 'Sym',
    'Turbinaria (coral)': 'Turb-HC', 'Turbinaria': 'Turb-HC',
    'Aglaophenia spp.': 'Agl', 'Anemone': 'Anemone', 'Bivalve': 'Bivalve', 'Bryozoan': 'Bryo',
    'Corallimorpharia': 'Cormor', 'Didemnidae': 'Didae', 'Hydroid': 'Hydro',
    'Lobophytum': 'Lobphyt', 'Millepora': 'Mil', 'Sarcophyton': 'Sarcopydae',
    'Soft Coral': 'SC', 'Sponge': 'SP', 'Tubipora musica': 'Tubmus', 'Tunicate': 'Tun',
    'Echinoderms: sea urchin': 'Urchins ex', 'XENIIDAE': 'Xe', 'Zoanthid': 'Zoan',
    'Biofilm': 'Biofilm ex', 'Rhytisma': 'Rhy', 'Sand': 'S', 'Dead coral': 'Dead (ex)',
    'Hard Substrate': 'HS_AR', 'Rock': 'Rock', 'Cyanobacteria': 'Cya', 'Framer': 'Frame',
    'Rubble': 'R', 'Unknown': 'Unknown', 'Ochrophyta': 'BA', 'Caulerpa': 'Cau',
    'CCA (crustose coralline algae)': 'CCA', 'Dictyota': 'Dic', 'Algae (filamentous)': 'Fila (ex)',
    'Chlorophyta': 'GA', 'Halimeda': 'Hal', 'Lobophora variegata': 'Lobvar', 'Macroalgae': 'MA',
    'Padina': 'Pad', 'Rhodophyta': 'RA', 'Sargassum': 'Sar', 'Turf algae': 'TA',
    'Turbinaria (algae)': 'Turb-BA', 'Valonia spp.': 'Val', 'Seagrass': 'SG', 'REEFo': None,
}

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def _normalise_stem(name):
    name = re.sub(r'\.rf\.[A-Za-z0-9]+', '', name)
    name = re.sub(r'[-_](jpe?g|png)[-_](jpe?g|png)$', '', name, flags=re.IGNORECASE)
    stem = name
    for _ in range(3):
        base, ext = os.path.splitext(stem)
        if ext.lower() in _IMG_EXTS:
            stem = base
        else:
            break
    stem = re.sub(r'[-_](jpe?g|png)$', '', stem, flags=re.IGNORECASE)
    return stem.lower()


def build_image_index(images_dir):
    index = {}
    for fn in os.listdir(images_dir):
        if os.path.splitext(fn)[1].lower() not in _IMG_EXTS:
            continue
        index.setdefault(_normalise_stem(fn), []).append(fn)
    return index


def resolve_image(fname, images_dir, index):
    candidates = index.get(_normalise_stem(fname), [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return os.path.join(images_dir, candidates[0])
    fl = fname.lower()
    for c in candidates:
        if c.lower() == fl:
            return os.path.join(images_dir, c)
    return os.path.join(images_dir, sorted(candidates, key=lambda x: (len(x), x))[0])


def _clean_roboflow_filename(fname):
    fname = re.sub(r'\.rf\.[A-Za-z0-9]+', '', fname)
    stem, ext = os.path.splitext(fname)
    s2, e2 = os.path.splitext(stem)
    if e2.lower() in {'.jpg', '.jpeg', '.png'}:
        stem, ext = s2, e2
    stem = re.sub(r'[-_](jpe?g|png)$', '', stem, flags=re.IGNORECASE)
    return stem + ext


def preprocess_roboflow_coco(coco_path, use_label_merge=True):
    """Load a SAM/Roboflow COCO JSON and remap names to merged short codes."""
    with open(coco_path, encoding='utf-8') as f:
        coco = json.load(f)

    for img in coco['images']:
        clean = img.get('extra', {}).get('name', '') or img['file_name']
        clean = _clean_roboflow_filename(clean)
        img['file_name'] = clean

    old_cats = {c['id']: c['name'] for c in coco.get('categories', [])}
    seen, new_cats, old_to_new = {}, [], {}
    for c in coco['categories']:
        nm = GENUS_TO_CORALNET.get(c['name'], c['name'])
        if nm is not None and use_label_merge:
            nm = apply_label_merge(nm)
        if nm is None:
            old_to_new[c['id']] = None
            continue
        if nm not in seen:
            seen[nm] = len(new_cats) + 1
            new_cats.append({'id': seen[nm], 'name': nm, 'supercategory': 'coral'})
        old_to_new[c['id']] = seen[nm]
    coco['categories'] = new_cats
    new_anns = []
    for a in coco['annotations']:
        nid = old_to_new.get(a['category_id'])
        if nid is None:
            continue
        a['category_id'] = nid
        new_anns.append(a)
    coco['annotations'] = new_anns
    return coco


def render_coco_to_png(coco, images_dir, out_dir, class_to_idx, ignore=IGNORE_INDEX):
    """Rasterize SAM polygons → per-pixel class-index PNGs (uncovered = ignore)."""
    os.makedirs(out_dir, exist_ok=True)
    cat_id_to_idx = {c['id']: class_to_idx.get(c['name']) for c in coco.get('categories', [])}
    images_by_id = {im['id']: im for im in coco['images']}
    anns_by_img = defaultdict(list)
    for a in coco['annotations']:
        if a['image_id'] in images_by_id:
            anns_by_img[a['image_id']].append(a)

    img_index = build_image_index(images_dir)
    n_done = 0
    rendered = []  # list of (img_path, mask_path)
    for img_id, meta in images_by_id.items():
        img_path = resolve_image(meta['file_name'], images_dir, img_index)
        if not img_path:
            continue
        with Image.open(img_path) as im:
            actual_w, actual_h = im.size
        sx = actual_w / meta['width'] if meta['width'] else 1.0
        sy = actual_h / meta['height'] if meta['height'] else 1.0
        mask = np.full((actual_h, actual_w), ignore, dtype=np.uint8)
        for a in sorted(anns_by_img[img_id], key=lambda a: -a.get('area', 0)):
            ci = cat_id_to_idx.get(a['category_id'])
            if ci is None:
                continue
            seg = a.get('segmentation', [])
            if isinstance(seg, dict):
                try:
                    from pycocotools import mask as mu
                    rle = seg
                    if isinstance(rle.get('counts'), list):
                        rle = mu.frPyObjects([rle], rle['size'][0], rle['size'][1])[0]
                    bn = mu.decode(rle).astype(np.uint8)
                    if bn.shape != mask.shape:
                        bn = cv2.resize(bn, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask[bn > 0] = ci
                except Exception:
                    continue
            else:
                for poly in seg:
                    if len(poly) < 6:
                        continue
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    pts[:, 0] *= sx; pts[:, 1] *= sy
                    cv2.fillPoly(mask, [pts.astype(np.int32)], ci)
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')
        cv2.imwrite(out_path, mask)
        rendered.append((img_path, out_path))
        n_done += 1
    return rendered


# ─── dataset / dataloader ────────────────────────────────────────────────────
def build_val_loader(image_paths, mask_paths, input_size=DEFAULT_INPUT_SIZE, batch_size=4):
    val_tf = A.Compose([
        A.LongestMaxSize(max_size=input_size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=input_size, min_width=input_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, mask_fill_value=IGNORE_INDEX),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    class _DS(Dataset):
        def __init__(self, ip, mp, tf):
            self.ip, self.mp, self.tf = ip, mp, tf
        def __len__(self): return len(self.ip)
        def __getitem__(self, idx):
            img = cv2.cvtColor(cv2.imread(self.ip[idx]), cv2.COLOR_BGR2RGB)
            msk = cv2.imread(self.mp[idx], cv2.IMREAD_UNCHANGED)
            assert img.shape[:2] == msk.shape[:2], f"shape mismatch {self.ip[idx]}"
            t = self.tf(image=img, mask=msk)
            return t['image'], t['mask'].long()

    return DataLoader(_DS(image_paths, mask_paths, val_tf),
                      batch_size=batch_size, shuffle=False, num_workers=0)


# ─── model definition (must match segformer_train Cell 7) ────────────────────
class TwoHeadSegFormer(torch.nn.Module):
    def __init__(self, base_model_name, num_classes_a, num_classes_b):
        super().__init__()
        # Build encoder + Head A from pretrained Coralscapes weights, Head B fresh.
        # We immediately overwrite all weights when loading bundle.pt below, so
        # the pretrained download here is just to get the right architecture.
        pretrained = SegformerForSemanticSegmentation.from_pretrained(PRETRAINED_CORALSCAPES)
        self.encoder = pretrained.segformer
        cfg_a = SegformerConfig.from_pretrained(PRETRAINED_CORALSCAPES)
        # Adjust head A class count if checkpoint has different
        if num_classes_a != cfg_a.num_labels:
            cfg_a.num_labels = num_classes_a
        self.head_a = SegformerDecodeHead(cfg_a)
        cfg_b = SegformerConfig.from_pretrained(base_model_name, num_labels=num_classes_b)
        self.head_b = SegformerDecodeHead(cfg_b)
        self.num_classes_a = num_classes_a
        self.num_classes_b = num_classes_b

    def forward(self, pixel_values, dataset_id=1):
        feats = self.encoder(pixel_values, output_hidden_states=True, return_dict=True).hidden_states
        head = self.head_a if dataset_id == 0 else self.head_b
        logits = head(feats)
        return F.interpolate(logits, size=pixel_values.shape[-2:], mode='bilinear', align_corners=False)


def _infer_n_classes(sd, prefix):
    for k, v in sd.items():
        if k.startswith(prefix) and k.endswith('classifier.weight'):
            return int(v.shape[0])
    raise RuntimeError(f"no {prefix}classifier.weight in checkpoint")


def load_bundle(bundle_path, device):
    """Load a bundle.pt or best.pt → (model, classes, meta_dict)."""
    raw = torch.load(bundle_path, map_location='cpu', weights_only=False)
    sd = raw.get('model', raw)
    classes = raw.get('your_classes')
    if classes is None:
        raise RuntimeError(f"{bundle_path}: no 'your_classes' in checkpoint — re-export as bundle.pt from Cell 11")
    
    # Use metadata num_classes if available (more robust than inferring from state dict)
    n_a = raw.get('num_classes_a')
    n_b = raw.get('num_classes_b')
    
    # Fallback: infer from state dict if metadata missing
    if n_a is None:
        try:
            n_a = _infer_n_classes(sd, 'head_a.')
        except RuntimeError:
            n_a = 0  # No head_a in checkpoint
            print(f"  ⓘ No head_a found in checkpoint (single-head model)")
    if n_b is None:
        n_b = _infer_n_classes(sd, 'head_b.')
    
    if n_b != len(classes):
        raise RuntimeError(f"head_b={n_b} but len(classes)={len(classes)} — checkpoint inconsistent")

    model = TwoHeadSegFormer(raw.get('model_name', DEFAULT_MODEL_NAME), n_a, n_b)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"  ⚠ unexpected keys (ignored): {len(unexpected)}")
    if missing:
        print(f"  ⚠ missing keys: {len(missing)}")
    model.to(device).eval()
    meta = {
        'experiment_name': raw.get('experiment_name', os.path.basename(os.path.dirname(bundle_path))),
        'stage':           raw.get('stage'),
        'epoch':           raw.get('epoch'),
        'train_miou':      raw.get('miou'),
        'model_name':      raw.get('model_name', DEFAULT_MODEL_NAME),
        'input_size':      raw.get('input_size', DEFAULT_INPUT_SIZE),
        'keep_top_n':      raw.get('keep_top_n_classes'),
        'use_label_merge': raw.get('use_label_merge', True),
        'drop_classes':    raw.get('drop_classes', []),
    }
    return model, list(classes), meta


# ─── metrics ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def build_confusion(model, loader, n_classes, device, use_amp=True):
    model.eval()
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    autocast = torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda')
    for imgs, masks in loader:
        imgs = imgs.to(device)
        with autocast:
            logits = model(imgs, dataset_id=1)
        pred = logits.argmax(1).cpu().numpy()
        m = masks.numpy()
        valid = m != IGNORE_INDEX
        # Clamp predictions to valid range to handle class index mismatches
        pred_clamped = np.clip(pred[valid], 0, n_classes - 1)
        m_valid = m[valid]
        conf += np.bincount((m_valid * n_classes + pred_clamped).ravel(),
                            minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    return conf


def metrics_from_conf(conf):
    conf = conf.astype(np.float64)
    K = conf.shape[0]
    tp = np.diag(conf); fp = conf.sum(0) - tp; fn = conf.sum(1) - tp
    sup = conf.sum(1); total = conf.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        iou  = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
        dice = np.where(2*tp + fp + fn > 0, 2*tp / (2*tp + fp + fn), np.nan)
        prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        rec  = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
    freq = sup / max(total, 1)
    return {
        'per_class': {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec,
                      'support': sup.astype(np.int64)},
        'summary': {
            'mIoU':              float(np.nanmean(iou)),
            'FWIoU':             float(np.nansum(np.where(np.isnan(iou), 0, iou * freq))),
            'mDice_F1':          float(np.nanmean(dice)),
            'mPrecision_macro':  float(np.nanmean(prec)),
            'mRecall_macro':     float(np.nanmean(rec)),
            'pixel_accuracy':    float(tp.sum() / max(total, 1)),
            'n_classes_with_support': int((sup > 0).sum()),
            'n_classes_total':        int(K),
        },
    }


def confusion_pairs(conf, classes, k=20):
    c = conf.copy().astype(np.int64); np.fill_diagonal(c, 0)
    pairs = []
    for flat in np.argsort(-c, axis=None)[:k]:
        i, j = np.unravel_index(flat, c.shape)
        n = int(c[i, j])
        if n == 0: break
        pairs.append({'true': classes[i], 'predicted_as': classes[j], 'pixels': n,
                      'frac_of_true': float(n / max(conf[i].sum(), 1))})
    return pairs


# ─── point accuracy (CoralNet baseline) ──────────────────────────────────────
@torch.no_grad()
def point_accuracy(model, csv_path, images_dir, class_to_idx, device,
                   input_size=DEFAULT_INPUT_SIZE, use_amp=True, use_label_merge=True,
                   max_pts_per_image=500):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, low_memory=False)
    label_col = 'Label code' if 'Label code' in df.columns else 'Label'
    df = df.rename(columns={label_col: '_label'})
    if use_label_merge:
        df['_label'] = df['_label'].map(apply_label_merge)
        df = df[df['_label'].notna()]
    df = df[df['_label'].isin(class_to_idx.keys())][['Name', 'Row', 'Column', '_label']].dropna()
    if max_pts_per_image:
        cnts = df.groupby('Name').size()
        df = df[~df['Name'].isin(cnts[cnts > max_pts_per_image].index)]
    if len(df) == 0:
        return None

    img_index = build_image_index(images_dir)
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    autocast = torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda')

    correct = total = missing = 0
    per_cor, per_tot = defaultdict(int), defaultdict(int)

    for name, g in df.groupby('Name'):
        ip = resolve_image(name, images_dir, img_index)
        if not ip:
            missing += 1; continue
        img = cv2.imread(ip)
        if img is None:
            missing += 1; continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        scale = input_size / max(h0, w0)
        nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
        rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        pad = cv2.copyMakeBorder(rs, 0, input_size - nh, 0, input_size - nw,
                                  cv2.BORDER_CONSTANT, value=0)
        x = torch.from_numpy(norm(image=pad)['image'].transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with autocast:
            logits = model(x, dataset_id=1)
        pred = F.interpolate(logits, size=(input_size, input_size),
                              mode='bilinear', align_corners=False).argmax(1)[0].cpu().numpy()
        for _, r in g.iterrows():
            xi = int(round(float(r['Column']) * scale))
            yi = int(round(float(r['Row']) * scale))
            if 0 <= xi < input_size and 0 <= yi < input_size:
                total += 1; per_tot[r['_label']] += 1
                if int(pred[yi, xi]) == class_to_idx[r['_label']]:
                    correct += 1; per_cor[r['_label']] += 1
    if total == 0:
        return None
    return {
        'overall':        correct / total,
        'correct':        correct,
        'total':          total,
        'missing_images': missing,
        'coralnet_baseline': 0.72,
        'delta_vs_coralnet': correct / total - 0.72,
        'per_class': {c: {'correct': per_cor[c], 'total': per_tot[c],
                          'accuracy': per_cor[c] / per_tot[c] if per_tot[c] else None}
                      for c in per_tot},
    }


# ─── plotting ────────────────────────────────────────────────────────────────
def _fig_to_b64(fig, dpi=110):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight'); buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def render_confusion_png(conf, classes, title="Confusion matrix"):
    import matplotlib.pyplot as plt
    n = conf.shape[0]
    sup = conf.sum(1)
    keep = [i for i in range(n) if sup[i] > 0]
    labels = [classes[i] for i in keep]
    sub = conf[np.ix_(keep, keep)].astype(np.float32)
    sub /= np.maximum(sub.sum(1, keepdims=True), 1)
    sz = max(10, len(labels) * 0.4)
    fig, ax = plt.subplots(figsize=(sz, sz * 0.9))
    im = ax.imshow(sub, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    fs = max(7, 11 - len(labels) // 8)
    ax.set_xticklabels(labels, rotation=90, fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title, fontsize=13, fontweight='bold')
    thr = sub.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = sub[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                        fontsize=max(5, 9 - len(labels) // 10),
                        color='white' if v > thr else 'black')
    plt.tight_layout()
    return _fig_to_b64(fig)


def render_perclass_bar(metrics, title):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    pc = metrics['per_class']; s = metrics['summary']
    rows = sorted(
        [(i, pc['iou'][i], pc['precision'][i], pc['recall'][i])
         for i in range(len(pc['iou'])) if pc['support'][i] > 0],
        key=lambda r: -(0 if np.isnan(r[1]) else r[1])
    )
    return rows, _bar_chart(rows, s['mIoU'], title)


def _bar_chart(rows, miou, title):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    classes_idx = [r[0] for r in rows]
    return classes_idx  # placeholder, see new build_html below


# ─── HTML report (single model) ──────────────────────────────────────────────
_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     max-width:1100px;margin:40px auto;padding:0 20px;color:#222}
h1{font-size:1.6em;border-bottom:3px solid #2196F3;padding-bottom:8px}
h2{font-size:1.2em;margin-top:36px;color:#1565C0}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:.9em}
th{background:#1565C0;color:#fff;padding:8px 10px;text-align:left}
td{padding:6px 10px;border-bottom:1px solid #eee}
.meta{display:flex;gap:30px;flex-wrap:wrap;background:#f5f5f5;
      padding:14px 18px;border-radius:8px;margin:16px 0;font-size:.9em}
.meta div{flex:1;min-width:200px}
.meta b{display:block;color:#555;font-size:.8em;text-transform:uppercase;letter-spacing:.05em}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:12px 0}
.metric{background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;padding:12px 16px}
.metric .val{font-size:2em;font-weight:700;color:#1565C0}
.metric .lbl{font-size:.8em;font-weight:600;text-transform:uppercase;color:#666;margin-bottom:4px}
.metric .def{font-size:.78em;color:#888;margin-top:4px}
.guidance{background:#e3f2fd;border-left:4px solid #2196F3;padding:10px 14px;
          margin:8px 0;border-radius:0 6px 6px 0;font-size:.88em}
.note{font-size:.82em;color:#555;font-style:italic;margin:4px 0 10px}
img{display:block;margin:0 auto;max-width:100%}
footer{margin-top:48px;font-size:.75em;color:#aaa;border-top:1px solid #eee;padding-top:12px}
.compare-table th, .compare-table td{text-align:right}
.compare-table td:first-child, .compare-table th:first-child{text-align:left}
.best{background:#e8f5e9;font-weight:bold}
"""


def _build_perclass_bar_b64(metrics, classes, title):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    pc = metrics['per_class']
    rows = [(classes[i], pc['iou'][i], pc['precision'][i], pc['recall'][i])
            for i in range(len(classes)) if pc['support'][i] > 0]
    rows.sort(key=lambda r: -(0 if np.isnan(r[1]) else r[1]))
    names = [r[0] for r in rows]
    ious  = [0 if np.isnan(r[1]) else r[1] for r in rows]
    precs = [0 if np.isnan(r[2]) else r[2] for r in rows]
    recs  = [0 if np.isnan(r[3]) else r[3] for r in rows]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.38), 5))
    x = np.arange(len(names)); w = 0.28
    ax.bar(x - w, ious,  w, label='IoU',       color='#2196F3', alpha=0.9)
    ax.bar(x,     precs, w, label='Precision', color='#4CAF50', alpha=0.9)
    ax.bar(x + w, recs,  w, label='Recall',    color='#FF9800', alpha=0.9)
    ax.axhline(metrics['summary']['mIoU'], color='#2196F3', linestyle='--',
               linewidth=1.2, alpha=0.6, label=f"mIoU={metrics['summary']['mIoU']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha='right', fontsize=max(7, 11 - len(names) // 8))
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return _fig_to_b64(fig)


def build_single_report_html(experiment_name, classes, metrics, conf, top_confusions,
                              point_acc, eval_meta, model_meta):
    """Self-contained HTML for one model. Mirrors Cell 9b of training notebook."""
    s = metrics['summary']; pc = metrics['per_class']

    bar_b64 = _build_perclass_bar_b64(metrics, classes,
                                       f"Per-class IoU / Precision / Recall — {experiment_name}")
    cm_b64 = render_confusion_png(conf, classes,
                                   f"Confusion matrix — {experiment_name}\n"
                                   f"mIoU={s['mIoU']:.3f}  ({s['n_classes_with_support']} classes with support)")

    def pct(v): return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v*100:.1f}%"
    def bar(v, w=20):
        v = 0 if v is None or (isinstance(v, float) and np.isnan(v)) else v
        n = int(v * w); return '█' * n + '░' * (w - n)

    rows = sorted(
        [(classes[i], pc['iou'][i], pc['dice'][i], pc['precision'][i],
          pc['recall'][i], int(pc['support'][i])) for i in range(len(classes))
         if pc['support'][i] > 0],
        key=lambda r: -(0 if np.isnan(r[1]) else r[1])
    )
    table_rows = []
    for nm, iou, dice, pr, rc, sup in rows:
        ival = 0 if np.isnan(iou) else iou
        bg = 'background:#e8f5e9' if ival >= 0.65 else \
             'background:#fff9c4' if ival >= 0.40 else 'background:#ffebee'
        table_rows.append(
            f'<tr style="{bg}"><td><b>{nm}</b></td>'
            f'<td>{pct(iou)}</td><td>{pct(dice)}</td>'
            f'<td>{pct(pr)}</td><td>{pct(rc)}</td><td>{sup:,}</td>'
            f'<td style="font-family:monospace;font-size:11px;color:#555">{bar(iou)}</td></tr>'
        )
    table_rows_html = '\n'.join(table_rows)

    conf_rows = []
    for p in top_confusions[:20]:
        frac = p['frac_of_true']
        col = '#ffebee' if frac > 0.20 else ('#fff9c4' if frac > 0.10 else '')
        conf_rows.append(
            f'<tr style="background:{col}"><td><b>{p["true"]}</b></td><td>→</td>'
            f'<td><b>{p["predicted_as"]}</b></td><td>{p["pixels"]:,} px</td>'
            f'<td>{frac*100:.1f}%</td></tr>'
        )

    pa_html = ""
    if point_acc:
        delta = point_acc['delta_vs_coralnet']
        color = '#2a8a3a' if delta >= 0 else '#a83232'
        verdict = (f"BEATS CoralNet by {delta*100:+.1f} pp" if delta >= 0
                   else f"Below CoralNet by {abs(delta)*100:.1f} pp")
        pa_html = f"""
<section style='background:#fff;border:2px solid {color};padding:20px;border-radius:8px;margin:24px 0;'>
  <h2 style='margin-top:0;color:{color};'>Direct CoralNet comparison — point accuracy</h2>
  <p style='font-size:18px;margin:8px 0;'>
    <strong>This model: {point_acc['overall']*100:.2f}%</strong>
    &nbsp;|&nbsp; CoralNet baseline: 72.00%
    &nbsp;|&nbsp; <strong style='color:{color};'>{verdict}</strong>
  </p>
  <p>Evaluated on {point_acc['total']:,} CoralNet-style labeled points.
     Same metric CoralNet reports — directly comparable.</p>
</section>"""

    guidance = []
    miou = s['mIoU']
    if miou < 0.35:
        guidance.append(("mIoU &lt; 35%", "Low overall accuracy. SAM-mask training is harder than gold-mask. "
                                          "Use this as a baseline; targeted refinement typically lifts 5–15 pp."))
    elif miou < 0.55:
        guidance.append(("mIoU 35–55%", "Moderate accuracy — competitive for full segmentation."))
    else:
        guidance.append(("mIoU &gt; 55%", "Strong accuracy for full segmentation."))
    if s['FWIoU'] > miou + 0.08:
        guidance.append(("FWIoU ≫ mIoU", "Doing well on common classes; rare classes drag mIoU down."))
    if s['mRecall_macro'] > s['mPrecision_macro'] + 0.07:
        guidance.append(("Recall &gt; Precision", "Model over-predicts. Common with sparse training data."))
    elif s['mPrecision_macro'] > s['mRecall_macro'] + 0.07:
        guidance.append(("Precision &gt; Recall", "Model is conservative — misses some pixels."))
    guidance_html = '\n'.join(f'<div class="guidance"><b>{t}</b> — {d}</div>' for t, d in guidance)

    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>",
        f"<title>SegFormer Eval — {experiment_name}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>SegFormer Coral Segmentation — {experiment_name}</h1>",
        pa_html,
        '<div class="meta">',
        f'<div><b>Experiment</b>{experiment_name}</div>',
        f'<div><b>Eval source</b>{eval_meta.get("eval_source", "SAM masks")}</div>',
        f'<div><b>Val images</b>{eval_meta.get("n_val_images", "?")}</div>',
        f'<div><b>Classes evaluated</b>{s["n_classes_with_support"]} / {s["n_classes_total"]}</div>',
        f'<div><b>Train mIoU (best.pt)</b>{model_meta.get("train_miou", "—")}</div>',
        f'<div><b>Generated</b>{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>',
        '</div>',
        '<h2>Summary Metrics</h2>',
        '<div class="summary">',
        f'<div class="metric"><div class="lbl">mIoU</div><div class="val">{miou*100:.1f}%</div>'
        '<div class="def">Mean IoU across classes — segmentation headline.</div></div>',
        f'<div class="metric"><div class="lbl">Pixel Accuracy</div><div class="val">{s["pixel_accuracy"]*100:.1f}%</div>'
        '<div class="def">% of pixels correctly classified. Inflated by dominant classes.</div></div>',
        f'<div class="metric"><div class="lbl">FWIoU</div><div class="val">{s["FWIoU"]*100:.1f}%</div>'
        '<div class="def">Frequency-weighted IoU — realistic field number.</div></div>',
        f'<div class="metric"><div class="lbl">Mean Dice / F1</div><div class="val">{s["mDice_F1"]*100:.1f}%</div>'
        '<div class="def">Slightly more lenient cousin of mIoU.</div></div>',
        f'<div class="metric"><div class="lbl">Macro Recall</div><div class="val">{s["mRecall_macro"]*100:.1f}%</div>'
        '<div class="def">Per-class recall, averaged equally.</div></div>',
        f'<div class="metric"><div class="lbl">Macro Precision</div><div class="val">{s["mPrecision_macro"]*100:.1f}%</div>'
        '<div class="def">Per-class precision, averaged equally.</div></div>',
        '</div>',
        '<h2>Interpretation</h2>',
        '<p class="note">Note: SAM-mask val set is noisier than gold annotations. mIoU here is a fair *relative* '
        'comparator across runs but absolute numbers under-state model quality. For absolute quality, '
        'compare against the gold val set in segformer_train.ipynb.</p>',
        guidance_html,
        '<h2>Per-class Performance</h2>',
        '<p class="note">Green = IoU ≥ 65%, yellow = 40–65%, red = &lt;40%.</p>',
        f'<img src="data:image/png;base64,{bar_b64}">',
        '<table><thead><tr><th>Class</th><th>IoU</th><th>Dice</th><th>Precision</th>'
        '<th>Recall</th><th>Val pixels</th><th>IoU bar</th></tr></thead>',
        f'<tbody>{table_rows_html}</tbody></table>',
        '<h2>Confusion Matrix</h2>',
        '<p class="note">Rows = true class, columns = predicted. Diagonal = recall. '
        'Bright off-diagonal cells = common mix-ups.</p>',
        f'<img src="data:image/png;base64,{cm_b64}">',
        '<h2>Top Misclassifications</h2>',
        '<p class="note">Pairs &gt;20% (red) are real weaknesses worth targeted refinement.</p>',
        '<table><thead><tr><th>True class</th><th></th><th>Predicted as</th>'
        '<th>Pixels</th><th>% of true class</th></tr></thead>',
        f'<tbody>{"".join(conf_rows)}</tbody></table>',
        f'<footer>Generated by coral_eval.py — {datetime.now().strftime("%Y-%m-%d %H:%M")}</footer>',
        '</body></html>',
    ]
    return '\n'.join(parts)


# ─── side-by-side comparison report ──────────────────────────────────────────
def build_comparison_html(results, eval_meta):
    """results: list of dicts with keys: experiment_name, classes, metrics,
       point_acc, model_meta. All must share the same `classes` list."""
    cls0 = results[0]['classes']
    for r in results[1:]:
        if r['classes'] != cls0:
            raise ValueError(f"class lists differ between {results[0]['experiment_name']} "
                             f"and {r['experiment_name']} — only same-taxonomy models can be compared")
    classes = cls0

    # Summary table
    metric_keys = [('mIoU', 'mIoU'), ('FWIoU', 'FWIoU'),
                   ('Pixel Acc.', 'pixel_accuracy'), ('Mean Dice', 'mDice_F1'),
                   ('Macro Prec.', 'mPrecision_macro'), ('Macro Recall', 'mRecall_macro')]
    head = "<tr><th>Metric</th>" + "".join(f"<th>{r['experiment_name']}</th>" for r in results) + "</tr>"
    body = []
    for label, key in metric_keys:
        vals = [r['metrics']['summary'][key] for r in results]
        best = max(vals)
        cells = "".join(
            f'<td class="{ "best" if abs(v - best) < 1e-9 else ""}">{v*100:.2f}%</td>' for v in vals)
        body.append(f"<tr><td><b>{label}</b></td>{cells}</tr>")
    if any(r['point_acc'] for r in results):
        vals = [r['point_acc']['overall'] if r['point_acc'] else float('nan') for r in results]
        finite = [v for v in vals if not np.isnan(v)]
        best = max(finite) if finite else float('nan')
        cells = "".join(
            f'<td class="{ "best" if not np.isnan(v) and abs(v - best) < 1e-9 else ""}">'
            f'{("—" if np.isnan(v) else f"{v*100:.2f}%")}</td>' for v in vals)
        body.append(f"<tr><td><b>Point acc. (vs CoralNet 72%)</b></td>{cells}</tr>")
    summary_table = f"<table class='compare-table'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"

    # Per-class IoU table (sorted by best model's IoU)
    pc_rows = []
    for ci, cname in enumerate(classes):
        ious = [r['metrics']['per_class']['iou'][ci] for r in results]
        sups = [int(r['metrics']['per_class']['support'][ci]) for r in results]
        if max(sups) == 0:
            continue
        finite = [v for v in ious if not np.isnan(v)]
        best = max(finite) if finite else None
        cells = ""
        for v, sup in zip(ious, sups):
            txt = "—" if np.isnan(v) else f"{v*100:.1f}%"
            cls = "best" if best is not None and not np.isnan(v) and abs(v - best) < 1e-9 else ""
            cells += f'<td class="{cls}">{txt}<br><span style="color:#888;font-size:.8em">n={sup:,}</span></td>'
        pc_rows.append((max(finite) if finite else 0, f"<tr><td><b>{cname}</b></td>{cells}</tr>"))
    pc_rows.sort(key=lambda r: -r[0])
    pc_head = "<tr><th>Class</th>" + "".join(f"<th>{r['experiment_name']}</th>" for r in results) + "</tr>"
    pc_table = (f"<table class='compare-table'><thead>{pc_head}</thead>"
                f"<tbody>{''.join(r[1] for r in pc_rows)}</tbody></table>")

    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>",
        "<title>SegFormer eval — side-by-side</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>SegFormer Coral Segmentation — Side-by-side comparison</h1>",
        '<div class="meta">',
        f'<div><b>Models compared</b>{len(results)}</div>',
        f'<div><b>Eval source</b>{eval_meta.get("eval_source", "SAM masks")}</div>',
        f'<div><b>Val images</b>{eval_meta.get("n_val_images", "?")}</div>',
        f'<div><b>Classes</b>{len(classes)}</div>',
        f'<div><b>Generated</b>{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>',
        '</div>',
        '<p class="note">All models evaluated on the SAME masks (fixed canonical val set). '
        'Best value per row highlighted green. Class taxonomy must match — KEEP_TOP_N_CLASSES=None only.</p>',
        '<h2>Summary metrics</h2>', summary_table,
        '<h2>Per-class IoU</h2>',
        '<p class="note">Sorted by best model\'s IoU. n = validation pixels (same for all models).</p>',
        pc_table,
        f'<footer>Generated by coral_eval.py — {datetime.now().strftime("%Y-%m-%d %H:%M")}</footer>',
        '</body></html>',
    ]
    return '\n'.join(parts)


# ─── orchestration ───────────────────────────────────────────────────────────
def evaluate_bundle(bundle_path, image_paths, mask_paths, classes, csv_path,
                     images_dir, device, input_size=DEFAULT_INPUT_SIZE,
                     batch_size=4, use_amp=True, do_point_acc=True):
    """Full evaluation of one bundle. Returns dict ready for HTML report."""
    print(f"\n→ Loading {bundle_path}")
    model, model_classes, meta = load_bundle(bundle_path, device)
    if model_classes != classes:
        raise ValueError(
            f"Model class list does not match canonical class list.\n"
            f"  Model has {len(model_classes)} classes, canonical has {len(classes)}.\n"
            f"  Diff (in model only): {sorted(set(model_classes) - set(classes))[:10]}\n"
            f"  Diff (in canonical only): {sorted(set(classes) - set(model_classes))[:10]}\n"
            f"Eval requires identical taxonomy (KEEP_TOP_N_CLASSES=None for all models)."
        )
    if meta.get('keep_top_n') not in (None, 0):
        print(f"  ⚠ keep_top_n_classes={meta['keep_top_n']} — comparison may be unfair")

    n_cls = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print(f"  Building val loader ({len(image_paths)} images)…")
    loader = build_val_loader(image_paths, mask_paths,
                              input_size=meta.get('input_size') or input_size,
                              batch_size=batch_size)
    print(f"  Computing confusion matrix…")
    conf = build_confusion(model, loader, n_cls, device, use_amp=use_amp)
    metrics = metrics_from_conf(conf)
    pairs = confusion_pairs(conf, classes, k=20)
    print(f"  → mIoU = {metrics['summary']['mIoU']:.4f}  "
          f"FWIoU = {metrics['summary']['FWIoU']:.4f}")

    pa = None
    if do_point_acc and csv_path and os.path.exists(csv_path):
        print(f"  Computing CoralNet point accuracy…")
        pa = point_accuracy(model, csv_path, images_dir, class_to_idx, device,
                            input_size=meta.get('input_size') or input_size,
                            use_amp=use_amp,
                            use_label_merge=bool(meta.get('use_label_merge', True)))
        if pa:
            print(f"  → point accuracy = {pa['overall']*100:.2f}% "
                  f"(Δ vs CoralNet 72%: {pa['delta_vs_coralnet']*100:+.1f} pp)")

    # Free GPU memory before next model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'experiment_name': meta['experiment_name'],
        'classes':         classes,
        'metrics':         metrics,
        'conf':            conf,
        'top_confusions':  pairs,
        'point_acc':       pa,
        'model_meta':      meta,
    }
