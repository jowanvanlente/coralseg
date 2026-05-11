"""
SegFormer Predict — Streamlit validation app with side-by-side model comparison.

Load one or two `*_bundle.pt` files produced by segformer_train.ipynb (Cell 11)
and run the trained 2-head SegFormer on any image. When two bundles are loaded,
both predictions are shown side-by-side for visual comparison.

Run locally:
    pip install -r requirements.txt
    streamlit run segformer_predict_app.py
"""

from __future__ import annotations

import io
import json
import os
import hashlib
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SegformerDecodeHead,
)

# ════════════════════════════════════════════════════════════════════════════
# On-disk cache (survives Streamlit session timeouts)
# ════════════════════════════════════════════════════════════════════════════
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHED_BUNDLE_A = CACHE_DIR / "last_bundle_a.pt"
CACHED_BUNDLE_B = CACHE_DIR / "last_bundle_b.pt"
SETTINGS_FILE   = CACHE_DIR / "settings.json"

DEFAULT_TEST_IMAGES = [
    "DSCN2121.JPG", "G0081188.JPG", "trainingJ10.JPG",
    "DSCN2122.JPG", "DSCN2123.JPG",
    "G0081189.JPG", "G0081190.JPG",
    "trainingJ11.JPG", "trainingJ12.JPG", "trainingJ13.JPG",
]


def _load_settings() -> Dict[str, Any]:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_settings(s: Dict[str, Any]) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(s, indent=2))
    except Exception:
        pass


_settings = _load_settings()


# ════════════════════════════════════════════════════════════════════════════
# Page config
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="SegFormer Predict", layout="wide")
st.title("🧪 SegFormer Predict — coral validation")
st.caption(
    "Load one trained model bundle (*_bundle.pt) to inspect predictions, or "
    "two bundles to compare them side-by-side on the same image."
)


# ════════════════════════════════════════════════════════════════════════════
# Model architecture (must match segformer_train.ipynb)
# ════════════════════════════════════════════════════════════════════════════
class TwoHeadSegFormer(nn.Module):
    def __init__(self, base_model_name: str, num_classes_a: int, num_classes_b: int):
        super().__init__()
        base = SegformerForSemanticSegmentation.from_pretrained(
            base_model_name, num_labels=num_classes_b, ignore_mismatched_sizes=True
        )
        self.encoder = base.segformer
        cfg_a = SegformerConfig.from_pretrained(base_model_name, num_labels=num_classes_a)
        cfg_b = SegformerConfig.from_pretrained(base_model_name, num_labels=num_classes_b)
        self.head_a = SegformerDecodeHead(cfg_a)
        self.head_b = SegformerDecodeHead(cfg_b)
        self.num_classes_a = num_classes_a
        self.num_classes_b = num_classes_b

    def forward(self, pixel_values: torch.Tensor, dataset_id: int) -> torch.Tensor:
        feats = self.encoder(pixel_values, output_hidden_states=True, return_dict=True).hidden_states
        head = self.head_a if dataset_id == 0 else self.head_b
        logits = head(feats)
        return F.interpolate(
            logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
        )


# ════════════════════════════════════════════════════════════════════════════
# Bundle loading
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_bundle(bundle_digest: str, bundle_bytes: bytes, device: str) -> Dict[str, Any]:
    _ = bundle_digest  # only used for cache key
    buf = io.BytesIO(bundle_bytes)
    bundle = torch.load(buf, map_location=device, weights_only=False)
    required = {"model_state", "model_name", "input_size", "num_classes_a",
                "num_classes_b", "your_classes"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Bundle is missing required keys: {sorted(missing)}")

    model = TwoHeadSegFormer(
        base_model_name=bundle["model_name"],
        num_classes_a=int(bundle["num_classes_a"]),
        num_classes_b=int(bundle["num_classes_b"]),
    ).to(device).eval()

    missing_k, unexpected_k = model.load_state_dict(bundle["model_state"], strict=False)
    return {
        "model": model,
        "device": device,
        "input_size": int(bundle["input_size"]),
        "your_classes": list(bundle["your_classes"]),
        "stage": bundle.get("stage", "?"),
        "best_miou": float(bundle.get("best_miou", -1)),
        "epoch": int(bundle.get("epoch", -1)),
        "missing_keys": len(missing_k),
        "unexpected_keys": len(unexpected_k),
    }


def load_bundle_section(label: str, slot: str,
                        cached_path: Path, device: str) -> Dict[str, Any] | None:
    """Sidebar UI for loading one bundle. slot is 'a' or 'b' (used for widget keys)."""
    st.sidebar.markdown(f"### {label}")

    bundle_file = st.sidebar.file_uploader(
        f"Upload *_bundle.pt", type=["pt", "pth"], key=f"upload_{slot}",
    )
    bundle_path = st.sidebar.text_input(
        "…or path on disk",
        value=_settings.get(f"last_bundle_path_{slot}", ""),
        placeholder="C:\\path\\to\\bundle.pt",
        key=f"path_{slot}",
    )

    bundle_bytes: bytes | None = None
    bundle_source = ""
    if bundle_file is not None:
        bundle_bytes = bundle_file.getvalue()
        bundle_source = f"upload ({bundle_file.name})"
        try:
            cached_path.write_bytes(bundle_bytes)
        except Exception as e:
            st.sidebar.warning(f"Could not cache bundle: {e}")
    elif bundle_path and os.path.exists(bundle_path):
        bundle_bytes = Path(bundle_path).read_bytes()
        bundle_source = f"disk ({os.path.basename(bundle_path)})"
        _settings[f"last_bundle_path_{slot}"] = bundle_path
        _save_settings(_settings)
    elif cached_path.exists():
        bundle_bytes = cached_path.read_bytes()
        bundle_source = "cached from previous session"

    if bundle_bytes is None:
        return None

    digest = hashlib.md5(bundle_bytes).hexdigest()
    try:
        state = load_bundle(digest, bundle_bytes, device)
    except Exception as e:
        st.sidebar.error(f"Failed to load {label}: {e}")
        return None

    state["source"] = bundle_source
    st.sidebar.success(
        f"**{label}** loaded — {bundle_source}\n\n"
        f"Stage: `{state['stage']}`  •  "
        f"mIoU: `{state['best_miou']:.4f}` (ep {state['epoch']})  •  "
        f"Classes: **{len(state['your_classes'])}**"
    )

    if st.sidebar.button(f"🗑️ Forget {label}", key=f"forget_{slot}",
                         use_container_width=True):
        try:
            if cached_path.exists():
                cached_path.unlink()
        except Exception:
            pass
        load_bundle.clear()
        st.rerun()

    return state


# ════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ════════════════════════════════════════════════════════════════════════════
def _make_palette(n: int, seed: int = 2) -> np.ndarray:
    """Distinct per-class palette using golden-ratio hue spread."""
    import colorsys
    golden = 0.61803398875
    h0 = (seed * 0.137) % 1.0
    out = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        h = (h0 + i * golden) % 1.0
        s = 0.70 + 0.25 * ((i * 7) % 3) / 2
        v = 0.72 + 0.25 * ((i * 11) % 2)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return out


@torch.no_grad()
def predict(img_rgb: np.ndarray, model: nn.Module, input_size: int, device: str) -> np.ndarray:
    h0, w0 = img_rgb.shape[:2]
    tf = A.Compose([
        A.LongestMaxSize(max_size=input_size),
        A.PadIfNeeded(min_height=input_size, min_width=input_size,
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    x = tf(image=img_rgb)["image"].unsqueeze(0).to(device)
    logits = model(x, dataset_id=1)  # Head B = your classes
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    scale = input_size / max(h0, w0)
    nh, nw = int(h0 * scale), int(w0 * scale)
    py, px = (input_size - nh) // 2, (input_size - nw) // 2
    pred = pred[py:py + nh, px:px + nw]
    return cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)


def colorize(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for ci in np.unique(mask):
        if ci >= len(palette):
            continue
        rgb[mask == ci] = palette[ci]
    return rgb


def draw_labels(img_rgb: np.ndarray, pred: np.ndarray, classes: list[str],
                palette: np.ndarray, min_region_px: int = 1500) -> np.ndarray:
    out = img_rgb.copy()
    h, w = pred.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.9, min(w, h) / 1400.0)
    thick_fg = max(2, int(round(font_scale * 2)))
    thick_bg = thick_fg + 3
    for ci in np.unique(pred):
        if ci >= len(classes):
            continue
        binary = (pred == ci).astype(np.uint8)
        n_comp, comp, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
        name = classes[ci]
        base_col = tuple(int(c) for c in palette[ci])
        bright = tuple(min(255, c + 60) for c in base_col)
        for k in range(1, n_comp):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < min_region_px:
                continue
            cx, cy = int(cents[k][0]), int(cents[k][1])
            if comp[cy, cx] != k:
                ys, xs = np.where(comp == k)
                cy, cx = int(ys.mean()), int(xs.mean())
            (tw, th), _ = cv2.getTextSize(name, font, font_scale, thick_fg)
            tx = max(5, min(w - tw - 5, cx - tw // 2))
            ty = max(th + 5, min(h - 5, cy + th // 2))
            cv2.putText(out, name, (tx, ty), font, font_scale, (0, 0, 0), thick_bg, cv2.LINE_AA)
            cv2.putText(out, name, (tx, ty), font, font_scale, bright, thick_fg, cv2.LINE_AA)
    return out


def coverage_rows(pred: np.ndarray, classes: list[str]) -> list[dict]:
    unique, counts = np.unique(pred, return_counts=True)
    total = pred.size
    rows = []
    for ci, n in sorted(zip(unique.tolist(), counts.tolist()), key=lambda t: -t[1]):
        if ci >= len(classes):
            continue
        rows.append({
            "class": classes[ci],
            "pixels": int(n),
            "coverage %": f"{n / total * 100:.2f}",
        })
    return rows


# ════════════════════════════════════════════════════════════════════════════
# Sidebar — bundles + display options
# ════════════════════════════════════════════════════════════════════════════
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.header("📦 Model bundles")
st.sidebar.write(f"Device: **{device}**")
st.sidebar.caption(
    "Load Model A to predict. Optionally load Model B to compare predictions side-by-side."
)

state_a = load_bundle_section("Model A", "a", CACHED_BUNDLE_A, device)

with st.sidebar.expander("🆚 Add Model B (optional, for comparison)", expanded=False):
    pass  # expander only — Model B section goes outside but visually grouped

st.sidebar.markdown("---")
state_b = load_bundle_section("Model B (comparison)", "b", CACHED_BUNDLE_B, device)

if state_a is None:
    st.info(
        "👈 Upload a `*_bundle.pt` for **Model A** in the sidebar, or paste a path. "
        "Generate one by running **Cell 11** in `segformer_train.ipynb`."
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Overlay settings")
blend_alpha = st.sidebar.slider("Mask opacity", 0.0, 1.0, 0.45, 0.05)
min_region_pct = st.sidebar.slider(
    "Label regions ≥ (% of image)", 0.0, 5.0, 0.2, 0.05,
    help="Only regions bigger than this fraction of the image get a text label.",
)
show_legend = st.sidebar.checkbox("Show class legend", value=True)


# ════════════════════════════════════════════════════════════════════════════
# Main — image input
# ════════════════════════════════════════════════════════════════════════════
st.header("1. Pick an image")

_default_folder = _settings.get("images_folder", "")
images_folder = st.text_input(
    "Test-image folder (used by the quick-load buttons below)",
    value=_default_folder,
    placeholder="C:\\path\\to\\your\\images",
    help="Folder containing your coral images. Remembered across sessions.",
)
if images_folder != _default_folder:
    _settings["images_folder"] = images_folder
    _save_settings(_settings)


def _resolve_test_image(name: str, folder: str) -> str | None:
    if not folder or not os.path.isdir(folder):
        return None
    candidates = {name, name.swapcase()}
    stem, ext = os.path.splitext(name)
    for alt_ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
        candidates.add(stem + alt_ext)
    for c in candidates:
        p = os.path.join(folder, c)
        if os.path.isfile(p):
            return p
    return None


_available_buttons: list[tuple[str, str]] = []
for name in DEFAULT_TEST_IMAGES:
    p = _resolve_test_image(name, images_folder)
    if p:
        _available_buttons.append((os.path.basename(p), p))

if images_folder and os.path.isdir(images_folder) and len(_available_buttons) < 10:
    seen = {p for _, p in _available_buttons}
    try:
        extras = sorted(
            f for f in os.listdir(images_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and os.path.join(images_folder, f) not in seen
        )
    except OSError:
        extras = []
    for f in extras:
        if len(_available_buttons) >= 10:
            break
        _available_buttons.append((f, os.path.join(images_folder, f)))

if "picked_image_path" not in st.session_state:
    st.session_state.picked_image_path = None

if _available_buttons:
    st.caption(f"Quick-load ({len(_available_buttons)} images found in folder):")
    _rows = (len(_available_buttons) + 4) // 5
    _idx = 0
    for _ in range(_rows):
        cols = st.columns(5)
        for c in cols:
            if _idx >= len(_available_buttons):
                break
            label, path = _available_buttons[_idx]
            if c.button(label, key=f"qload_{_idx}", use_container_width=True):
                st.session_state.picked_image_path = path
            _idx += 1

tab_upload, tab_path = st.tabs(["📤 Upload", "📁 From disk"])

image_bytes: bytes | None = None
image_name = "image"

with tab_upload:
    up = st.file_uploader("Coral image (jpg / png)", type=["jpg", "jpeg", "png"])
    if up is not None:
        image_bytes = up.getvalue()
        image_name = up.name
        st.session_state.picked_image_path = None

with tab_path:
    p = st.text_input("Path on disk", value="", placeholder="C:\\path\\to\\image.jpg")
    if p and os.path.exists(p):
        image_bytes = Path(p).read_bytes()
        image_name = os.path.basename(p)
        st.session_state.picked_image_path = None

if image_bytes is None and st.session_state.picked_image_path \
        and os.path.exists(st.session_state.picked_image_path):
    image_bytes = Path(st.session_state.picked_image_path).read_bytes()
    image_name = os.path.basename(st.session_state.picked_image_path)

if image_bytes is None:
    st.info("Pick a quick-load image above, upload one, or enter a path.")
    st.stop()

arr = np.frombuffer(image_bytes, dtype=np.uint8)
img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("Could not decode this file as an image.")
    st.stop()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]


# ════════════════════════════════════════════════════════════════════════════
# Prediction (single or compare)
# ════════════════════════════════════════════════════════════════════════════
compare_mode = state_b is not None

st.header("2. Prediction" + (" — comparing Model A vs Model B" if compare_mode else ""))
st.write(f"Image: **{image_name}** — {w}×{h}")


def _run_and_render(state: Dict[str, Any], label: str) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray]:
    with st.spinner(f"Running {label} ..."):
        pred = predict(img_rgb, state["model"], state["input_size"], device)
    palette = _make_palette(len(state["your_classes"]), seed=2)
    pred_rgb = colorize(pred, palette)
    blend = cv2.addWeighted(img_rgb, 1 - blend_alpha, pred_rgb, blend_alpha, 0)
    min_region_px = max(1, int(w * h * (min_region_pct / 100.0)))
    labeled_blend = draw_labels(blend, pred, state["your_classes"], palette, min_region_px)
    return pred, coverage_rows(pred, state["your_classes"]), labeled_blend, palette


pred_a, rows_a, blend_a, palette_a = _run_and_render(state_a, "Model A")

if compare_mode:
    pred_b, rows_b, blend_b, palette_b = _run_and_render(state_b, "Model B")

    # Three columns: original | A | B
    c0, c1, c2 = st.columns(3)
    with c0:
        st.image(img_rgb, caption="Original", use_container_width=True)
    with c1:
        st.image(blend_a, caption=f"Model A — {state_a['stage']} (mIoU {state_a['best_miou']:.3f})",
                 use_container_width=True)
    with c2:
        st.image(blend_b, caption=f"Model B — {state_b['stage']} (mIoU {state_b['best_miou']:.3f})",
                 use_container_width=True)

    st.subheader("Class coverage — side-by-side")
    cov1, cov2 = st.columns(2)
    with cov1:
        st.markdown(f"**Model A** ({len(state_a['your_classes'])} classes)")
        st.dataframe(rows_a, use_container_width=True, hide_index=True)
    with cov2:
        st.markdown(f"**Model B** ({len(state_b['your_classes'])} classes)")
        st.dataframe(rows_b, use_container_width=True, hide_index=True)

    # Quick agreement check (only meaningful if both have same class list)
    if state_a["your_classes"] == state_b["your_classes"]:
        agreement = float((pred_a == pred_b).mean()) * 100
        st.info(f"🤝 The two models agree on **{agreement:.1f}%** of pixels.")
    else:
        st.caption(
            "Note: Model A and Model B have different class lists, so direct pixel "
            "agreement isn't computed. Compare visually instead."
        )

else:
    # Single-model layout
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Original", use_container_width=True)
        st.image(blend_a, caption="Blend + class labels", use_container_width=True)
    with col2:
        pred_rgb_only = colorize(pred_a, palette_a)
        min_region_px = max(1, int(w * h * (min_region_pct / 100.0)))
        labeled_pred = draw_labels(pred_rgb_only, pred_a,
                                   state_a["your_classes"], palette_a, min_region_px)
        st.image(labeled_pred, caption="Prediction + class labels", use_container_width=True)
        st.subheader("Class coverage")
        st.dataframe(rows_a, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# Legend
# ════════════════════════════════════════════════════════════════════════════
if show_legend:
    st.header("3. Color legend (Model A)")
    present = sorted(int(c) for c in np.unique(pred_a) if c < len(state_a["your_classes"]))
    cols = st.columns(6)
    for i, ci in enumerate(present):
        c = palette_a[ci]
        swatch = (f'<div style="width:28px;height:28px;background:rgb({c[0]},{c[1]},{c[2]});'
                  f'border:1px solid #333;display:inline-block;vertical-align:middle;margin-right:6px;"></div>')
        cols[i % 6].markdown(
            f'{swatch}<span style="vertical-align:middle;">{state_a["your_classes"][ci]}</span>',
            unsafe_allow_html=True,
        )

    if compare_mode:
        st.subheader("Color legend (Model B)")
        present_b = sorted(int(c) for c in np.unique(pred_b) if c < len(state_b["your_classes"]))
        cols = st.columns(6)
        for i, ci in enumerate(present_b):
            c = palette_b[ci]
            swatch = (f'<div style="width:28px;height:28px;background:rgb({c[0]},{c[1]},{c[2]});'
                      f'border:1px solid #333;display:inline-block;vertical-align:middle;margin-right:6px;"></div>')
            cols[i % 6].markdown(
                f'{swatch}<span style="vertical-align:middle;">{state_b["your_classes"][ci]}</span>',
                unsafe_allow_html=True,
            )

st.markdown(
    "---\n"
    "Tips: raise **Label regions ≥ (% of image)** in the sidebar if labels look busy. "
    "Load a Model B in the sidebar to compare two trained models side-by-side."
)
