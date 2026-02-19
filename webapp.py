"""
Sparse to Dense COCO - Streamlit Web App

Two modes:
1. Test Mode: Visualize segmentation on single image
2. Export Mode: Process multiple images to COCO JSON
"""

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import load_labelset_from_json, load_annotations_from_df, scale_image_and_points, normalize_image_name
from superpixel_labeling import multi_scale_labeling
from adaptive_segmentation import multi_scale_adaptive_labeling
from graph_segmentation import multi_scale_graph_labeling
from hybrid_segmentation import multi_scale_hybrid_labeling
from graph_first_segmentation import multi_scale_graph_first_labeling
from coco_export import export_to_coco_dict
from confidence_scoring import calculate_region_confidence, apply_confidence_threshold, get_confidence_summary
from region_merging import merge_regions

st.set_page_config(page_title="Annotation Segmentation", layout="wide")


def count_segments(mask):
    """Count distinct connected-component instances per class in a labeled mask.
    Returns dict with 'total' and per-class breakdown."""
    import cv2 as _cv2
    stats = {}
    total = 0
    for cid in np.unique(mask):
        if cid == 0:
            continue
        binary = (mask == cid).astype(np.uint8)
        n_comp, _ = _cv2.connectedComponents(binary)
        n_instances = n_comp - 1  # subtract background label
        stats[int(cid)] = n_instances
        total += n_instances
    stats['total'] = total
    return stats

# Compact UI styling
st.markdown("""
<style>
    .stSlider > div > div > div > div { height: 0.4rem; }
    .stSlider label { font-size: 0.85rem; }
    div[data-testid="stExpander"] details summary { font-size: 0.9rem; padding: 0.3rem 0; }
    .stButton button { padding: 0.3rem 0.8rem; font-size: 0.85rem; }
    .stMarkdown p { margin-bottom: 0.3rem; }
    .stCaption { font-size: 0.75rem; }
    h1 { font-size: 1.8rem !important; margin-bottom: 0.3rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.3rem 0.8rem; font-size: 0.85rem; }
    div[data-testid="stNumberInput"] input { padding: 0.2rem 0.4rem; font-size: 0.85rem; }
    .stSelectbox label, .stRadio label { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.title("Annotation-Based Segmentation")
st.caption("Turn sparse point annotations (CSV) into full segmentation masks, then export to COCO JSON.")

with st.expander("ℹ️ Start here (what to do)", expanded=False):
    st.markdown(
        """
        **Goal:** create a segmentation mask for each image using sparse point labels, then export to COCO for Roboflow.

        **Step-by-step**
        1. **Upload images** (left sidebar).
        2. **Upload annotations CSV** (left sidebar).
        3. Click **(Re)load images and annotations** (main page) to make sure filenames line up.
        4. Use **Test Segmentation** to verify the result on one image.
        5. Use **Export COCO** to process many images and download COCO JSON.

        **Common pitfall: filenames**
        - The CSV column `Name` must refer to the same filenames as your uploaded images.
        - This app does *normalized matching* to handle cases like double extensions:
          - `photo.jpeg.jpeg` -> `photo.jpeg`
          - `G0258570.JPG.JPG` -> `G0258570.JPG`
        """
    )

# ==================== SIDEBAR: Data Upload ====================
st.sidebar.title("📤 Upload Data")

# Load default labelset
@st.cache_data
def load_default_labelset():
    labelset_path = os.path.join(os.path.dirname(__file__), 'labelset.json')
    with open(labelset_path, 'r') as f:
        return json.load(f)

# Initialize session state
if 'labelset' not in st.session_state:
    st.session_state.labelset = load_default_labelset()
if 'images' not in st.session_state:
    st.session_state.images = {}
if 'points_dict' not in st.session_state:
    st.session_state.points_dict = {}
if 'processed' not in st.session_state:
    st.session_state.processed = {}
# Test mode result state
if 'test_result' not in st.session_state:
    st.session_state.test_result = None

# Custom defaults state
if 'custom_defaults' not in st.session_state:
    st.session_state.custom_defaults = {
        'scale_factor': 0.4,
        'num_rounds': 3,
        'superpixel': [3000, 900, 30],
        'adaptive': [1.0, 0.5, 0.25],
        'adaptive_min_dist': 10,
        'adaptive_density': 5,
        'graph': [100, 300, 1000]
    }

# ==================== SMART SCALE CALCULATION ====================
def compute_smart_superpixel_scales(processing_scale: float, num_rounds: int) -> list:
    """
    Compute smart superpixel counts based on processing scale.
    
    At full resolution (1.0), we'd use base values like [3000, 900, 30].
    At lower resolution, the image has fewer pixels (scale²), so we need
    proportionally fewer superpixels to achieve similar region sizes.
    
    Base values at scale=1.0: [5000, 1500, 500, 50] for 4 rounds
    """
    area_factor = processing_scale ** 2  # Image area scales quadratically
    
    # Base values for full resolution (scale=1.0), decreasing for gap-filling
    base_values = [5000, 1500, 500, 50]
    
    # Scale down based on image area, with minimum values
    scaled = []
    for i in range(num_rounds):
        val = int(base_values[i] * area_factor)
        # Ensure minimum sensible values
        min_vals = [100, 50, 20, 10]
        scaled.append(max(val, min_vals[i]))
    
    return scaled

def compute_smart_adaptive_scales(processing_scale: float, num_rounds: int) -> list:
    """
    Compute smart adaptive resolution multipliers.
    
    These are relative to the already-scaled image, so we use a progression
    from coarse to fine. At lower processing scales, we can use higher
    relative multipliers since the image is already small.
    """
    # Progression from 1.0 (full) down to finer scales
    # At low processing_scale, the watershed already works on small image,
    # so we can afford higher multipliers
    if num_rounds == 1:
        return [1.0]
    elif num_rounds == 2:
        return [1.0, 0.5]
    elif num_rounds == 3:
        return [1.0, 0.5, 0.25]
    else:  # 4 rounds
        return [1.0, 0.6, 0.35, 0.15]

def compute_smart_graph_scales(processing_scale: float, num_rounds: int) -> list:
    """
    Compute smart graph segmentation thresholds based on processing scale.
    
    The threshold controls how aggressively regions merge. At lower resolution,
    pixel differences are averaged over larger areas, so we may need slightly
    higher thresholds to get similar behavior.
    """
    # Base thresholds at scale=1.0, increasing for more aggressive merging
    base_values = [100, 300, 800, 2000]
    
    # At lower scales, increase thresholds slightly (fewer pixels = need looser merging)
    # But not too aggressively - factor of ~1/sqrt(scale) is reasonable
    scale_adjustment = 1.0 / (processing_scale ** 0.5)
    scale_adjustment = min(scale_adjustment, 3.0)  # Cap at 3x
    
    scaled = []
    for i in range(num_rounds):
        val = int(base_values[i] * scale_adjustment)
        scaled.append(val)
    
    return scaled

def get_smart_defaults(method: str, processing_scale: float, num_rounds: int) -> list:
    """Get smart default values for a given method, scale, and number of rounds."""
    if method == "superpixel" or method == "graph_first_sp":
        return compute_smart_superpixel_scales(processing_scale, num_rounds)
    elif method == "adaptive":
        return compute_smart_adaptive_scales(processing_scale, num_rounds)
    else:  # graph, graph_first_gr
        return compute_smart_graph_scales(processing_scale, num_rounds)

# ==================== SCALE-ADAPTIVE PARAMETER SYSTEM ====================
import math

# Log-spaced options for sliders (allows fine control at low values)
LOG_OPTIONS_10K = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000]
LOG_OPTIONS_5K = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
AD_OPTIONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def clamp_to_options(val, options):
    """Find the closest value in options list."""
    return min(options, key=lambda x: abs(x - val))

def scale_value(base_val, processing_scale, method):
    """
    Convert base value (at 100% scale) to display value for current processing scale.
    
    Superpixel: base_val * scale² (area proportional)
    Graph: base_val / sqrt(scale) (threshold adjustment)
    Adaptive: no scaling (relative multipliers)
    """
    if method in ("superpixel", "graph_first_sp"):
        return base_val * (processing_scale ** 2)
    elif method in ("graph", "graph_first_gr"):
        return base_val / (processing_scale ** 0.5)
    else:  # adaptive
        return base_val  # No scaling for adaptive (already relative)

def unscale_value(display_val, processing_scale, method):
    """
    Convert display value back to base value (at 100% scale).
    Inverse of scale_value.
    """
    if method in ("superpixel", "graph_first_sp"):
        return display_val / (processing_scale ** 2)
    elif method in ("graph", "graph_first_gr"):
        return display_val * (processing_scale ** 0.5)
    else:  # adaptive
        return display_val

def get_base_defaults(method, num_rounds):
    """Get default BASE values (at 100% scale) for a method."""
    if method in ("superpixel", "graph_first_sp"):
        return [5000, 1500, 500, 50][:num_rounds]
    elif method in ("graph", "graph_first_gr"):
        return [100, 300, 800, 2000][:num_rounds]
    else:  # adaptive
        if num_rounds == 1:
            return [1.0]
        elif num_rounds == 2:
            return [1.0, 0.5]
        elif num_rounds == 3:
            return [1.0, 0.5, 0.25]
        else:
            return [1.0, 0.6, 0.35, 0.15]

def init_base_values(prefix, method, num_rounds):
    """Initialize base values in session state if not present."""
    key = f"{prefix}_base_{method}"
    if key not in st.session_state:
        st.session_state[key] = get_base_defaults(method, num_rounds)
    # Extend if num_rounds increased
    current = st.session_state[key]
    if len(current) < num_rounds:
        defaults = get_base_defaults(method, num_rounds)
        st.session_state[key] = current + defaults[len(current):num_rounds]
    return st.session_state[key]

def get_display_values(base_values, processing_scale, method, adapt_on, options):
    """
    Derive display values from base values.
    If adapt_on: apply scaling formula
    If adapt_off: show base values as-is
    Always clamp to valid options.
    """
    display = []
    for base in base_values:
        if adapt_on:
            scaled = scale_value(base, processing_scale, method)
        else:
            scaled = base
        display.append(clamp_to_options(scaled, options))
    return display

def update_base_from_display(prefix, method, round_idx, display_val, processing_scale, adapt_on):
    """
    Update base value when user edits a display value.
    If adapt_on: invert the scale formula
    If adapt_off: store directly as base
    """
    key = f"{prefix}_base_{method}"
    if key in st.session_state:
        if adapt_on:
            base = unscale_value(display_val, processing_scale, method)
        else:
            base = display_val
        st.session_state[key][round_idx] = base

def format_settings_txt(method, scale_factor, num_rounds, scale_values, seg_params, adapt_on, conf_threshold=0, conf_enabled=False, merge_params=None):
    """Format current segmentation settings as a text string for export."""
    from datetime import datetime
    
    lines = [
        "=" * 50,
        "SEGMENTATION SETTINGS EXPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 50,
        "",
        f"Method: {method}",
        f"Processing Scale: {scale_factor * 100:.0f}%",
        f"Number of Rounds: {num_rounds}",
        f"Adapt Values: {'ON' if adapt_on else 'OFF'}",
        "",
        "--- Round Values ---",
    ]
    
    for i, val in enumerate(scale_values):
        lines.append(f"  Round {i+1}: {val}")
    
    lines.append("")
    lines.append("--- Advanced Settings ---")
    
    if "graph-first" in method.lower():
        lines.append(f"  Discovery Scale: {seg_params.get('discovery_scale', 1000)}")
        lines.append(f"  Fill Method: {seg_params.get('fill_method', 'superpixel')}")
        lines.append(f"  Allow Fill Overwrite: {seg_params.get('allow_overwrite', False)}")
    elif "hybrid" in method.lower():
        lines.append(f"  Allow Overwrite: {seg_params.get('allow_overwrite', False)}")
    elif "superpixel" in method.lower() or "slic" in method.lower():
        lines.append("  (Default SLIC settings)")
    elif "adaptive" in method.lower():
        lines.append(f"  Min Distance: {seg_params.get('min_distance', 10)}")
        lines.append(f"  Density Threshold: {seg_params.get('density_threshold', 5)}")
        lines.append(f"  Allow Overwrite: {seg_params.get('allow_overwrite', False)}")
    elif "graph" in method.lower():
        lines.append(f"  Allow Overwrite: {seg_params.get('allow_overwrite', False)}")
    
    lines.append("")
    lines.append("--- Region Merging ---")
    if merge_params is not None:
        lines.append("  Region Merging: ON")
        lines.append(f"  Min Area (speckle removal): {merge_params.get('min_area', 100)}")
        lines.append(f"  Small Region Merge: {merge_params.get('small_region_merge', 500)}")
        lines.append(f"  Color Similarity Threshold: {merge_params.get('color_threshold', 30.0)}")
        lines.append(f"  Morph Close Kernel: {merge_params.get('morph_close_ksize', 5)}")
    else:
        lines.append("  Region Merging: OFF (not applicable for this method)")
    
    lines.append("")
    lines.append("--- Confidence Filtering ---")
    if conf_enabled:
        lines.append(f"  Confidence Filtering: ON (Threshold: {conf_threshold})")
    else:
        lines.append("  Confidence Filtering: OFF")
    
    lines.append("")
    lines.append("=" * 50)
    lines.append("Use these settings to reproduce your segmentation results.")
    lines.append("=" * 50)
    
    return "\n".join(lines)

# Sample data loader
st.sidebar.markdown("### 🎯 Quick Start")
if st.sidebar.button("📂 Load Sample Data", use_container_width=True, 
    help="Load 3 sample coral reef images with ~2700 annotations to try the app"):
    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
    sample_images = ['G0258570.JPG', 'G0258574.JPG', 'G0258577.JPG']
    sample_csv_path = os.path.join(sample_dir, 'sample_annotations.csv')
    
    if os.path.exists(sample_csv_path):
        # Load sample images
        st.session_state.images = {}
        loaded_count = 0
        for img_name in sample_images:
            img_path = os.path.join(sample_dir, img_name)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                st.session_state.images[img_name] = img
                loaded_count += 1
        
        # Load sample annotations
        df = pd.read_csv(sample_csv_path, low_memory=False)
        st.session_state.points_dict = load_annotations_from_df(df)
        total_annotations = sum(len(d) for d in st.session_state.points_dict.values())
        
        st.sidebar.success(f"✓ Sample data loaded! ({loaded_count} images, {total_annotations:,} annotations)")
        st.rerun()
    else:
        st.sidebar.error("Sample data files not found")

st.sidebar.markdown("---")

# Image upload
st.sidebar.markdown("### 🖼️ Images")
uploaded_images = st.sidebar.file_uploader(
    "Upload images",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    help=("Upload the images you want to create segmentation masks for (.jpg, .jpeg, .png). "
          "You can select multiple files at once. Each image must have corresponding annotations in the CSV file. "
          "Image filenames must match the 'Name' column in your CSV. "
          "The app uses normalized matching to handle common quirks like double extensions (e.g. 'photo.JPG.JPG' matches 'photo.JPG'). "
          "Larger images take longer to process -- the Processing Scale setting lets you trade speed for detail.")
)

if uploaded_images:
    upload_errors = []
    for img_file in uploaded_images:
        if img_file.name not in st.session_state.images:
            try:
                file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is not None:
                    st.session_state.images[img_file.name] = img
                else:
                    upload_errors.append(img_file.name)
                img_file.seek(0)
            except Exception as e:
                upload_errors.append(f"{img_file.name}: {str(e)[:50]}")
    if st.session_state.images:
        st.sidebar.success(f"✓ {len(st.session_state.images)} images loaded")
    if upload_errors:
        with st.sidebar.expander("⚠️ Upload issues", expanded=True):
            st.warning("Some files failed. Try uploading fewer/smaller images at once.")
            for err in upload_errors[:5]:
                st.caption(f"• {err}")

# Annotations upload
st.sidebar.markdown("### 📍 Annotations")
uploaded_csv = st.sidebar.file_uploader(
    "Upload annotations CSV",
    type=['csv'],
    help=("Upload the CoralNet (or similar) CSV with one annotation point per row. "
          "Required columns: 'Name' (image filename), 'Row' (Y pixel coordinate), 'Column' (X pixel coordinate), "
          "and a label column (e.g. 'Label' or 'Label code') matching your labelset Short Codes. "
          "Each row represents one point annotation on one image. The segmentation algorithm uses these sparse points "
          "as seeds to grow full dense masks. More points per image generally means better segmentation quality. "
          "Extra columns are OK and will be ignored.")
)

if uploaded_csv:
    try:
        df = pd.read_csv(uploaded_csv, low_memory=False)
        st.session_state.points_dict = load_annotations_from_df(df)
        total_points = sum(len(d) for d in st.session_state.points_dict.values())
        st.sidebar.success(f"✓ {total_points:,} annotations loaded")
    except ValueError as e:
        st.sidebar.error(str(e))
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")

# Labelset viewer with edit protection
st.sidebar.markdown("### 🏷️ Labelset")

# Initialize edit mode state
if 'labelset_edit_mode' not in st.session_state:
    st.session_state.labelset_edit_mode = False
if 'labelset_edit_confirmed' not in st.session_state:
    st.session_state.labelset_edit_confirmed = False

if st.sidebar.button("📋 View Labelset", use_container_width=True):
    st.session_state.show_labelset_modal = True

# Labelset modal dialog
@st.dialog("🏷️ Labelset Viewer", width="large")
def show_labelset_dialog():
    st.markdown("### Class Definitions")
    st.caption(f"Total classes: {len(st.session_state.labelset)}")
    
    # Edit mode toggle
    col_edit, col_status = st.columns([1, 2])
    with col_edit:
        if not st.session_state.labelset_edit_confirmed:
            if st.button("✏️ Enable Edit Mode", type="secondary"):
                st.session_state.labelset_edit_mode = True
        else:
            st.success("Edit mode active")
    
    # Warning dialog for edit mode
    if st.session_state.labelset_edit_mode and not st.session_state.labelset_edit_confirmed:
        st.warning("⚠️ **Warning:** Editing the labelset can break the app if mistakes are made. Class IDs and Short Codes must remain consistent with your annotations.")
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✓ I Understand", type="primary", use_container_width=True):
                st.session_state.labelset_edit_confirmed = True
                st.rerun()
        with col_cancel:
            if st.button("✗ Cancel", use_container_width=True):
                st.session_state.labelset_edit_mode = False
                st.rerun()
    
    st.markdown("---")
    
    # Display labelset
    if st.session_state.labelset_edit_confirmed:
        # Editable text area
        labelset_json = st.text_area(
            "Edit JSON below:",
            value=json.dumps(st.session_state.labelset, indent=2),
            height=500
        )
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                try:
                    new_labelset = json.loads(labelset_json)
                    st.session_state.labelset = load_labelset_from_json(new_labelset)
                    st.session_state.labelset_edit_mode = False
                    st.session_state.labelset_edit_confirmed = False
                    st.success("✓ Labelset updated!")
                    st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
        with col_cancel:
            if st.button("✗ Cancel Edit", use_container_width=True):
                st.session_state.labelset_edit_mode = False
                st.session_state.labelset_edit_confirmed = False
                st.rerun()
    else:
        # Read-only scrollable view
        for entry in st.session_state.labelset:
            color_code = entry.get('Color Code', [200, 200, 200])
            if isinstance(color_code, list):
                color_str = f"rgb({color_code[0]},{color_code[1]},{color_code[2]})"
            else:
                color_str = "#ccc"
            name = entry.get('Name', entry.get('Short Code', ''))
            short = entry.get('Short Code', '')
            group = entry.get('Functional Group', '')
            class_id = entry.get('Count', '')
            
            st.markdown(
                f"<div style='display:flex;align-items:center;padding:8px;border-bottom:1px solid #eee;'>"
                f"<span style='display:inline-block;width:24px;height:24px;background:{color_str};border:1px solid #555;border-radius:4px;margin-right:12px;flex-shrink:0;'></span>"
                f"<div style='flex-grow:1;'>"
                f"<div style='font-weight:600;'>{name} <span style='color:#888;font-weight:normal;'>({short})</span></div>"
                f"<div style='font-size:12px;color:#666;'>{group} • ID: {class_id}</div>"
                f"</div></div>",
                unsafe_allow_html=True
            )

if 'show_labelset_modal' in st.session_state and st.session_state.show_labelset_modal:
    show_labelset_dialog()
    st.session_state.show_labelset_modal = False

# Apply metadata to labelset
st.session_state.labelset = load_labelset_from_json(st.session_state.labelset)

# Status
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Status")
st.sidebar.info(f"Images: {len(st.session_state.images)}")
st.sidebar.info(f"CSV image names: {len(st.session_state.points_dict)}")
st.sidebar.info(f"Classes: {len(st.session_state.labelset)}")
st.sidebar.caption("If matching looks wrong, open **Preview Annotations** to see exactly what matches and what doesn't.")

# Annotation Preview Button
if st.session_state.points_dict or st.session_state.images:
    if st.sidebar.button("🔍 Preview Annotations", use_container_width=True, 
                         help="View annotation details and check image/annotation matching"):
        st.session_state.show_annotation_preview = True

# ==================== ANNOTATION PREVIEW ====================
if 'show_annotation_preview' not in st.session_state:
    st.session_state.show_annotation_preview = False

if st.session_state.show_annotation_preview:
    st.header("🔍 Annotation Preview & Matching Status")
    
    # Close button
    if st.button("✖ Close Preview"):
        st.session_state.show_annotation_preview = False
        st.rerun()
    
    st.markdown("---")
    
    # Build normalized matching
    image_names = list(st.session_state.images.keys())
    annotation_names = list(st.session_state.points_dict.keys())
    
    # Create normalized mappings
    img_norm_map = {normalize_image_name(n): n for n in image_names}
    ann_norm_map = {normalize_image_name(n): n for n in annotation_names}
    
    # Find matches via normalized names
    matched_norm = set(img_norm_map.keys()) & set(ann_norm_map.keys())
    matched_images = [img_norm_map[n] for n in matched_norm]
    
    images_without_annotations = [n for n in image_names if normalize_image_name(n) not in ann_norm_map]
    annotations_without_images = [n for n in annotation_names if normalize_image_name(n) not in img_norm_map]
    
    # Matching statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", len(image_names))
    with col2:
        st.metric("Annotation Names", len(annotation_names))
    with col3:
        st.metric("✅ Matched", len(matched_images))
    with col4:
        total_annotations = sum(len(pts) for pts in st.session_state.points_dict.values())
        st.metric("Total Points", total_annotations)
    
    st.markdown("---")
    
    # Simple search
    search_query = st.text_input("🔎 Search (type to filter):", key="ann_search")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["✅ Matched", "📊 Most Annotations", "⚠️ Images without annotations", "❌ Annotations without images"])
    
    with tab1:
        st.caption(f"{len(matched_images)} images matched via normalized names")
        items = matched_images
        if search_query:
            items = [n for n in items if search_query.lower() in n.lower()]
        for img_name in sorted(items)[:100]:
            norm = normalize_image_name(img_name)
            ann_name = ann_norm_map.get(norm, "")
            ann_count = len(st.session_state.points_dict.get(ann_name, []))
            if img_name == ann_name:
                st.markdown(f"✅ `{img_name}` ({ann_count} pts)")
            else:
                st.markdown(f"✅ `{img_name}` ↔ `{ann_name}` ({ann_count} pts)")
    
    with tab2:
        st.caption("Images sorted by annotation count (most to least)")
        # Build list of (image_name, annotation_count) for matched images
        img_ann_counts = []
        for img_name in matched_images:
            norm = normalize_image_name(img_name)
            ann_name = ann_norm_map.get(norm, "")
            ann_count = len(st.session_state.points_dict.get(ann_name, []))
            img_ann_counts.append((img_name, ann_name, ann_count))
        
        # Sort by count descending
        img_ann_counts.sort(key=lambda x: x[2], reverse=True)
        
        # Filter by search
        if search_query:
            img_ann_counts = [(n, a, c) for n, a, c in img_ann_counts if search_query.lower() in n.lower()]
        
        # Show stats
        if img_ann_counts:
            counts = [c for _, _, c in img_ann_counts]
            st.markdown(f"**Max:** {max(counts)} pts | **Min:** {min(counts)} pts | **Avg:** {sum(counts)/len(counts):.1f} pts")
        
        for img_name, ann_name, ann_count in img_ann_counts[:100]:
            bar = "█" * min(ann_count // 5, 20)  # Visual bar, 1 block per 5 points
            st.markdown(f"`{ann_count:4d}` {bar} `{img_name}`")
    
    with tab3:
        st.caption(f"{len(images_without_annotations)} images have no matching annotations")
        items = images_without_annotations
        if search_query:
            items = [n for n in items if search_query.lower() in n.lower()]
        for name in sorted(items)[:100]:
            norm = normalize_image_name(name)
            st.markdown(f"⚠️ `{name}` (normalized: `{norm}`)")
    
    with tab4:
        st.caption(f"{len(annotations_without_images)} annotation entries have no matching image")
        items = annotations_without_images
        if search_query:
            items = [n for n in items if search_query.lower() in n.lower()]
        for name in sorted(items)[:100]:
            norm = normalize_image_name(name)
            ann_count = len(st.session_state.points_dict.get(name, []))
            st.markdown(f"❌ `{name}` (normalized: `{norm}`) - {ann_count} pts")
    
    st.markdown("---")
    st.info("💡 **Tip:** Normalized matching handles double extensions like `.JPG.JPG` → `.JPG`. If names still don't match, check capitalization and exact spelling.")
    
    st.stop()

# ==================== MAIN CONTENT ====================
if not st.session_state.images:
    st.info("👈 Upload images in the sidebar to get started.")
    
    st.markdown("---")
    st.markdown("### How to use this app")
    st.markdown("""
    1. **Upload images** - Your reef/coral photos (.jpg, .jpeg, .png)
    2. **Upload annotations** - CSV from CoralNet with columns: `Name, Row, Column, Label`
    3. **Test on single image** - Fine-tune parameters before batch processing
    4. **Export all** - Generate COCO JSON for Roboflow
    
    #### Annotation CSV Format (CoralNet export)
    ```
    Name,Row,Column,Label
    image001.jpg,120,340,Acr
    image001.jpg,245,512,TA
    image002.jpg,100,200,S
    ...
    ```
    """)
    st.stop()

# ==================== IMAGE MATCHING ====================
def rebuild_matching():
    """Rebuild image-annotation matching using normalized names"""
    # Build mapping: normalized_name -> original image name
    image_norm_map = {normalize_image_name(name): name for name in st.session_state.images.keys()}
    # Build mapping: normalized_name -> original annotation key
    ann_norm_map = {normalize_image_name(name): name for name in st.session_state.points_dict.keys()}
    
    # Find matches via normalized names
    annotated_images = []
    norm_to_image = {}  # normalized -> image name
    norm_to_ann = {}    # normalized -> annotation key
    for norm_name, img_name in image_norm_map.items():
        if norm_name in ann_norm_map:
            annotated_images.append(img_name)
            norm_to_image[norm_name] = img_name
            norm_to_ann[norm_name] = ann_norm_map[norm_name]
    
    # Store mappings in session state
    st.session_state.norm_to_ann = norm_to_ann
    st.session_state.image_norm_map = image_norm_map
    st.session_state.ann_norm_map = ann_norm_map
    st.session_state.annotated_images = annotated_images
    return annotated_images

# Initial matching or get cached
if 'annotated_images' not in st.session_state:
    annotated_images = rebuild_matching()
else:
    annotated_images = st.session_state.annotated_images
    # Ensure mappings exist
    if 'norm_to_ann' not in st.session_state:
        annotated_images = rebuild_matching()

# Matching controls
all_images = list(st.session_state.images.keys())
col_match1, col_match2, col_match3 = st.columns([1, 1, 2])

with col_match1:
    if st.button("(Re)load images and annotations", help="Re-runs the matching between your uploaded image filenames and the 'Name' column in your CSV. "
            "This is necessary after uploading new images or a new CSV, because the app needs to figure out which CSV rows belong to which image. "
            "The matching uses normalized filenames (e.g. 'photo.JPG.JPG' becomes 'photo.JPG') so minor naming inconsistencies are handled automatically. "
            "After clicking, check the match summary to make sure all your images have annotations."):
        annotated_images = rebuild_matching()
        st.toast(f"✅ Matching updated: {len(annotated_images)} of {len(all_images)} images have annotations")
        st.rerun()

with col_match2:
    show_all_images = st.checkbox("Show all images", value=False, 
                                   help="By default, only images that have matching annotations in your CSV are shown in the dropdown. "
                                        "Enable this to also see images that have no CSV match -- useful for debugging filename mismatches. "
                                        "You cannot segment an image without annotations, so unmatched images will show a warning if selected.")

with col_match3:
    st.caption(f"📊 **{len(annotated_images)}** matched / **{len(all_images)}** total images loaded")
    st.caption("Matched = an uploaded image filename can be linked to a CSV `Name` (using normalization).")

# Determine which images to show in selection
selectable_images = all_images if show_all_images else annotated_images

if not selectable_images:
    if show_all_images:
        st.warning("⚠️ No images loaded. Upload images in the sidebar.")
    else:
        st.warning("⚠️ No images have matching annotations. Try 'Show all images' or check the Preview Annotations panel.")
    st.stop()

# ==================== MODE SELECTION ====================
mode = st.radio(
    "🎯 Select Mode",
    ["🔬 Test Segmentation", "📦 Export COCO"],
    help="Test: visualize one image. Export: process multiple to COCO.",
    horizontal=True
)

st.markdown("---")

# ==================== TEST MODE ====================
if mode == "🔬 Test Segmentation":
    
    with st.expander("ℹ️ How to use Test Segmentation", expanded=False):
        st.markdown(
            """
            Use this mode to validate your settings on **one** image before processing everything.
 
            - Pick an image on the left.
            - Optionally toggle **Show annotations** to confirm points land on the correct objects.
            - Click **Visualize** to generate a mask overlay.
            - If the mask looks wrong, try a different method or adjust the scale settings.
            """
        )
    
    # General settings - Method selection vertical, info and scale on right
    st.subheader("🔧 General Settings")
    col_method, col_info_scale = st.columns([1.2, 2])
    
    with col_method:
        st.markdown("**Segmentation Method**")
        seg_method = st.radio(
            "Method",
            ["🔷 Superpixel (SLIC)", "🎯 Adaptive (Density-based)", "📊 Graph-based (Felzenszwalb)", "🔀 Hybrid (SLIC + Graph)", "🔍 Graph-First (Anchor + Fill)"],
            horizontal=False,
            label_visibility="collapsed"
        )
    
    with col_info_scale:
        col_info, col_scale_inner = st.columns([1.5, 1])
        
        with col_info:
            if seg_method == "🔷 Superpixel (SLIC)":
                with st.expander("ℹ️ How Superpixel works", expanded=False):
                    st.markdown("""**SLIC Superpixel Segmentation (boundary-respecting regions)**

**Basic idea:** split the image into small, coherent regions (*superpixels*) and spread each point label to its whole region.

**When to use it:**
- Good default.
- Works well when object boundaries are visible (coral vs sand vs water) and you want clean edges.

**Scale progression: 3000 → 900 → 30 (DECREASING)**

The number represents **how many superpixels** to create:
- **Higher number (3000)** = More superpixels = **Smaller regions**
- **Lower number (30)** = Fewer superpixels = **Larger regions**

**Why start with many small superpixels?**
1. **Scale 1 (3000):** Creates ~3000 tiny regions. Small superpixels precisely follow object boundaries.
2. **Scale 2 (900):** Creates ~900 medium regions. Fills gaps left by Scale 1.
3. **Scale 3 (30):** Creates ~30 large regions. Covers any remaining unlabeled areas.

**Never overwrites:** Once a pixel is labeled, it stays labeled.""")
            
            elif seg_method == "🎯 Adaptive (Density-based)":
                with st.expander("ℹ️ How Adaptive works", expanded=False):
                    st.markdown("""**Adaptive (density-based) Segmentation (gap-filling from points)**

**Basic idea:** use the sparse points as seeds and grow regions outward (watershed-style). Areas with more points tend to get more detailed boundaries.

**When to use it:**
- When annotations are uneven (some dense areas, some sparse) and you want the algorithm to fill gaps.
- Often good when superpixels under-segment or leave holes.

**Scale progression: 1.0 → 0.5 → 0.25 (DECREASING resolution)**

The number represents **image resolution multiplier**:
- **1.0** = Full resolution = **Larger segments**
- **0.25** = Quarter resolution = **Smaller, finer segments**

**Why does lower resolution create finer segments?**
At lower resolution, watershed creates segments relative to the smaller image. When upscaled back, these become many small segments.

**Can overwrite:** If enabled and a finer scale has high confidence, it can refine boundaries.""")
            
            elif seg_method == "📊 Graph-based (Felzenszwalb)":
                with st.expander("ℹ️ How Graph-based works", expanded=False):
                    st.markdown("""**Graph-based (Felzenszwalb) Segmentation (merge-by-similarity)**

**Basic idea:** build a graph of pixels, then merge neighboring pixels/regions if they look similar. This often creates natural regions without requiring a fixed grid.

**When to use it:**
- If you want segments that follow texture/color similarity.
- Useful when superpixels are too regular or adaptive watershed feels too aggressive.

**Scale progression: 100 → 300 → 1000 (INCREASING)**

The number is a **similarity threshold** for merging regions:
- **Lower number (100)** = Strict merging = **Many small segments**
- **Higher number (1000)** = Loose merging = **Few large segments**

**Can overwrite:** If enabled and a coarser scale has high confidence, it can correct over-segmentation.""")
            
            elif seg_method == "🔀 Hybrid (SLIC + Graph)":
                with st.expander("ℹ️ How Hybrid works", expanded=False):
                    st.markdown("""**Hybrid Segmentation (SLIC + Felzenszwalb combined)**

**Basic idea:** Mix both methods! Each round can be either Superpixel (SLIC) or Graph-based, giving you full control.

**When to use it:**
- When you want the best of both worlds
- Start with SLIC for clean boundaries, then use Graph for texture-based filling (or vice versa)

**How to configure:**
- For each round, pick **S** (Superpixel) or **G** (Graph)
- **S rounds:** Number = superpixel count (higher = smaller regions)
- **G rounds:** Number = merge threshold (higher = larger regions)

**Example:** S:3000 → G:500 → S:100 = Start precise with SLIC, fill gaps with Graph, then coarse SLIC pass.""")
            
            else:  # Graph-First
                with st.expander("ℹ️ How Graph-First works", expanded=False):
                    st.markdown("""**Graph-First (Anchor + Fill) Segmentation**

**Basic idea:** First, run Felzenszwalb at a **high** scale to discover the obvious, coherent objects (corals, rocks, etc.). These large regions become "anchor" labels. Then, fill in the remaining unlabeled areas with progressive multi-round segmentation.

**Why this works:**
- A high Felzenszwalb scale merges similar pixels aggressively → only genuinely distinct objects survive as separate regions.
- These discovery regions are high-confidence because they represent structures the algorithm is *sure* about.
- The fill-in rounds (Superpixel or Graph) then handle the less certain surrounding areas progressively.

**Two phases:**
1. **Discovery (Graph, high scale):** e.g. scale 1000-2000. Creates a few large regions following natural boundaries. Labels these from your points → anchors.
2. **Fill-in (1-4 rounds):** Uses Superpixel counts or Graph scales to progressively fill remaining unlabeled pixels. Anchor labels are never overwritten (by default).

**When to use it:**
- When you have distinct objects (corals) surrounded by more ambiguous background.
- When you want the algorithm to "see" the important structures first, then fill in around them.
- Works especially well for images where obvious color/texture boundaries define the key objects.""")
        
        with col_scale_inner:
            scale_factor = st.slider("Processing Scale", 0.1, 1.0, st.session_state.custom_defaults['scale_factor'], 0.05,
                help="Controls the resolution at which segmentation is computed. "
                     "The image is resized to this fraction before processing (e.g. 0.4 = 40%% of original pixels). "
                     "Lower values are much faster because the algorithm works on fewer pixels, but fine details and small objects may be lost. "
                     "Higher values preserve detail but take significantly longer. "
                     "Recommended: 0.3-0.5 for most images. Use 0.2 for very large images or quick tests, 0.6+ when you need precise boundaries on small objects.")
            st.caption(f"Processing at {scale_factor*100:.0f}% → ~{1/(scale_factor**2):.0f}x faster")
            if st.button("💾 Save", key="save_scale", help="Save this value as your default"):
                st.session_state.custom_defaults['scale_factor'] = scale_factor
                st.toast(f"✓ Saved {scale_factor} as new default")
    
    st.markdown("---")
    
    # Three columns layout
    col_left, col_mid, col_right = st.columns([1.8, 0.8, 1.8])
    
    # LEFT: Image selection and preview
    with col_left:
        st.markdown("**🖼️ Select Image**")
        test_image = st.selectbox("Image", selectable_images, label_visibility="collapsed")
        
        total_points_in_image = 0
        points_df = None
        if test_image:
            # Use normalized lookup to find annotations
            norm_name = normalize_image_name(test_image)
            ann_key = st.session_state.norm_to_ann.get(norm_name)
            if ann_key:
                points_df = st.session_state.points_dict[ann_key]
                total_points_in_image = len(points_df)
        
        show_points = st.toggle("📍 Show annotations", value=False,
            help="Overlay the original sparse point annotations on the preview image. "
                 "Each dot is one annotation from your CSV, colored by its class label. "
                 "Use this to verify that annotation points actually land on the correct objects before running segmentation. "
                 "If points appear in wrong locations, check that your CSV Row/Column values match this image.")
        
        if test_image:
            image = st.session_state.images[test_image]
            preview_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preview_h = 450
            preview_w = int(image.shape[1] * (preview_h / image.shape[0]))
            preview_resized = cv2.resize(preview_rgb, (preview_w, preview_h))
            
            # Draw points if toggle is on
            if show_points and points_df is not None:
                scale_preview = preview_h / image.shape[0]
                label_to_color = {}
                for entry in st.session_state.labelset:
                    color_code = entry['Color Code']
                    if isinstance(color_code, list):
                        label_to_color[entry['Short Code']] = tuple(color_code)
                
                for _, row in points_df.iterrows():
                    px = int(row['Column'] * scale_preview)
                    py = int(row['Row'] * scale_preview)
                    label = row['Label']
                    color = label_to_color.get(label, (255, 255, 255))
                    cv2.circle(preview_resized, (px, py), 4, color, -1)
                    cv2.circle(preview_resized, (px, py), 4, (255, 255, 255), 1)
            
            # Image + Legend side by side
            img_col, legend_col = st.columns([3, 1.2], gap="small")
            with img_col:
                st.image(preview_resized, caption=f"Original: {image.shape[1]}x{image.shape[0]} | {total_points_in_image} annotations")
            with legend_col:
                if points_df is not None and total_points_in_image > 0:
                    with st.expander("Legend", expanded=False):
                        label_counts = points_df['Label'].value_counts().to_dict()
                        for entry in st.session_state.labelset:
                            label = entry.get('Short Code', '')
                            count = label_counts.get(label, 0)
                            if count <= 0:
                                continue
                            color_code = entry.get('Color Code', [200, 200, 200])
                            if isinstance(color_code, list):
                                color_str = f"rgb({color_code[0]},{color_code[1]},{color_code[2]})"
                            else:
                                color_str = "#ccc"
                            name = entry.get('Name', label)
                            pct = (count / total_points_in_image) * 100
                            st.markdown(
                                f"<div style='display:flex;align-items:flex-start;margin-bottom:6px;'>"
                                f"<span style='display:inline-block;width:12px;height:12px;background:{color_str};border:1px solid #555;border-radius:2px;margin-right:6px;margin-top:2px;flex-shrink:0;'></span>"
                                f"<div style='line-height:1.2;'>"
                                f"<div style='font-size:11px;font-weight:600;'>{name}</div>"
                                f"<div style='font-size:10px;color:#888;'>{label} • {pct:.1f}%</div>"
                                f"</div></div>",
                                unsafe_allow_html=True
                            )
    
    # MIDDLE: Parameters
    with col_mid:
        st.markdown("**⚙️ Parameters**")
        
        # Number of rounds selector
        num_rounds = st.slider("Rounds", 1, 4, st.session_state.custom_defaults['num_rounds'], 1,
            help="How many segmentation passes to run on the image. Each round creates segments at a different scale and labels unlabeled pixels. "
                 "Round 1 typically covers the majority of the image. Rounds 2-4 progressively fill in remaining gaps with coarser/finer segments. "
                 "More rounds = better pixel coverage (closer to 100%%) but slower processing. "
                 "3 rounds is a good default. Use 1-2 for quick tests, 4 if you see gaps in the final mask.")
        
        # Adapt values toggle
        use_smart = st.toggle("Adapt values", value=True, key="test_smart",
            help="When ON, the round parameter values are automatically adjusted based on your Processing Scale. "
                 "For example, at 40%% scale the image has only 16%% of its original pixels, so superpixel counts are reduced proportionally "
                 "to keep region sizes visually similar. Without this, the same superpixel count on a smaller image would create tiny, useless regions. "
                 "When OFF, you control the exact raw values. Leave this ON unless you know exactly what values you need.")
        
        # Get method key
        if seg_method == "🔷 Superpixel (SLIC)":
            method_key = "superpixel"
            options = LOG_OPTIONS_10K
        elif seg_method == "🎯 Adaptive (Density-based)":
            method_key = "adaptive"
            options = AD_OPTIONS
        elif seg_method == "📊 Graph-based (Felzenszwalb)":
            method_key = "graph"
            options = LOG_OPTIONS_5K
        elif seg_method == "🔀 Hybrid (SLIC + Graph)":
            method_key = "hybrid"
            options = LOG_OPTIONS_10K  # Will be dynamic per round
        else:  # Graph-First
            method_key = "graph_first"
            options = LOG_OPTIONS_5K  # Discovery scale uses graph options
        
        # Initialize base values (persistent, at 100% scale)
        base_values = init_base_values("test", method_key, num_rounds)
        
        # Derive display values from base values
        display_values = get_display_values(base_values, scale_factor, method_key, use_smart, options)
        
        # Track scale changes - when adapt is ON and scale changes, update widget keys BEFORE widget creation
        test_scale_track = "test_last_scale_track"
        if test_scale_track not in st.session_state:
            st.session_state[test_scale_track] = scale_factor
        
        if use_smart and st.session_state[test_scale_track] != scale_factor and method_key not in ("hybrid", "graph_first"):
            st.session_state[test_scale_track] = scale_factor
            # Pre-set widget keys to new display values (before widgets are created)
            prefixes = {"superpixel": "test_sp_", "adaptive": "test_ad_", "graph": "test_gr_"}
            prefix = prefixes[method_key]
            for i in range(num_rounds):
                st.session_state[f"{prefix}{i}"] = display_values[i]
        
        st.caption(f"{'Adapted' if use_smart else 'Manual'} ({scale_factor*100:.0f}% scale)")
        
        if seg_method == "🔷 Superpixel (SLIC)":
            scale_values = []
            for i in range(num_rounds):
                # Create callback to update base value when slider changes
                def make_callback(idx, mkey, pscale, adapt):
                    def cb():
                        widget_key = f"test_sp_{idx}"
                        if widget_key in st.session_state:
                            update_base_from_display("test", mkey, idx, st.session_state[widget_key], pscale, adapt)
                    return cb
                
                val = st.select_slider(
                    f"Round {i+1}", 
                    options=LOG_OPTIONS_10K, 
                    value=display_values[i], 
                    key=f"test_sp_{i}",
                    on_change=make_callback(i, method_key, scale_factor, use_smart)
                )
                scale_values.append(val)
            
            seg_params = {'scales': scale_values}
            
        elif seg_method == "🎯 Adaptive (Density-based)":
            scale_values = []
            for i in range(num_rounds):
                def make_callback(idx, mkey, pscale, adapt):
                    def cb():
                        widget_key = f"test_ad_{idx}"
                        if widget_key in st.session_state:
                            update_base_from_display("test", mkey, idx, st.session_state[widget_key], pscale, adapt)
                    return cb
                
                val = st.select_slider(
                    f"Round {i+1}", 
                    options=AD_OPTIONS, 
                    value=display_values[i], 
                    key=f"test_ad_{i}",
                    format_func=lambda x: f"{x:.2f}",
                    on_change=make_callback(i, method_key, scale_factor, use_smart)
                )
                scale_values.append(val)
            
            with st.expander("⚙️ Advanced", expanded=False):
                min_dist = st.slider("Min dist", 1, 50, st.session_state.custom_defaults['adaptive_min_dist'], 1,
                    help="Minimum distance (in pixels) between watershed seed points. "
                         "The adaptive method places seeds at your annotation points and grows regions outward. "
                         "This setting prevents seeds that are very close together from creating redundant tiny regions. "
                         "Higher values = fewer seeds = larger, simpler regions. Lower values = more seeds = finer detail but potentially noisy boundaries. "
                         "Default 10 works well for most cases.")
                density_thresh = st.slider("Density", 1, 20, st.session_state.custom_defaults['adaptive_density'], 1,
                    help="Controls how many annotation points must be nearby for a region to be considered 'densely annotated'. "
                         "In areas with fewer points than this threshold, the algorithm uses larger, coarser regions to fill gaps. "
                         "In areas above this threshold, it creates finer regions. "
                         "Higher values = more of the image is treated as 'sparse' and gets coarser regions. "
                         "Lower values = the algorithm treats most areas as dense and creates finer segments everywhere. "
                         "Default 5 is usually good.")
                allow_ow = st.checkbox("Overwrite", value=False, key="ad_ow",
                    help="When OFF (default), once a pixel is labeled in an earlier round, it stays labeled -- later rounds only fill gaps. "
                         "When ON, a later round can overwrite an earlier label if the new round has higher confidence for that pixel. "
                         "This can improve accuracy in areas where the first round made a poor assignment, but may also cause instability. "
                         "Recommendation: leave OFF for your first attempt, try ON if you see obvious mislabeling.")
            
            seg_params = {
                'scales': scale_values,
                'min_distance': min_dist,
                'density_threshold': density_thresh,
                'allow_overwrite': allow_ow
            }
            
        elif seg_method == "📊 Graph-based (Felzenszwalb)":
            scale_values = []
            for i in range(num_rounds):
                def make_callback(idx, mkey, pscale, adapt):
                    def cb():
                        widget_key = f"test_gr_{idx}"
                        if widget_key in st.session_state:
                            update_base_from_display("test", mkey, idx, st.session_state[widget_key], pscale, adapt)
                    return cb
                
                val = st.select_slider(
                    f"Round {i+1}", 
                    options=LOG_OPTIONS_5K, 
                    value=display_values[i], 
                    key=f"test_gr_{i}",
                    on_change=make_callback(i, method_key, scale_factor, use_smart)
                )
                scale_values.append(val)
            
            with st.expander("⚙️ Advanced", expanded=False):
                g_allow_ow = st.checkbox("Overwrite", value=False, key="gr_ow",
                    help="When OFF (default), once a pixel is labeled in an earlier round, it stays labeled -- later rounds only fill gaps. "
                         "When ON, a later round can overwrite an earlier label if the new round has higher confidence for that pixel. "
                         "This can improve accuracy in areas where the first round made a poor assignment, but may also cause instability. "
                         "Recommendation: leave OFF for your first attempt, try ON if you see obvious mislabeling.")
            
            seg_params = {
                'scales': scale_values,
                'allow_overwrite': g_allow_ow
            }
        
        elif seg_method == "🔀 Hybrid (SLIC + Graph)":
            st.caption("**S** = Superpixel (count), **G** = Graph (threshold)")
            round_configs = []
            
            # Clear any cached values that might be invalid
            for i in range(4):
                for key in [f"test_hybrid_s_{i}", f"test_hybrid_g_{i}"]:
                    if key in st.session_state:
                        val = st.session_state[key]
                        if key.endswith(f"s_{i}") and val not in LOG_OPTIONS_10K:
                            del st.session_state[key]
                        elif key.endswith(f"g_{i}") and val not in LOG_OPTIONS_5K:
                            del st.session_state[key]
            
            for i in range(num_rounds):
                col_type, col_val = st.columns([1, 2])
                
                with col_type:
                    round_type = st.radio(
                        f"R{i+1}",
                        ["S", "G"],
                        horizontal=True,
                        key=f"test_hybrid_type_{i}",
                        label_visibility="collapsed"
                    )
                
                with col_val:
                    if round_type == "S":
                        # Superpixel: higher = smaller regions
                        # Defaults: 3000 → 1000 → 100
                        val = st.select_slider(
                            f"Round {i+1}",
                            options=LOG_OPTIONS_10K,
                            value=3000 if i == 0 else (1000 if i == 1 else 100),
                            key=f"test_hybrid_s_{i}",
                            label_visibility="collapsed"
                        )
                    else:
                        # Graph: higher = larger regions
                        # Defaults: 100 → 300 → 1000
                        val = st.select_slider(
                            f"Round {i+1}",
                            options=LOG_OPTIONS_5K,
                            value=100 if i == 0 else (300 if i == 1 else 1000),
                            key=f"test_hybrid_g_{i}",
                            label_visibility="collapsed"
                        )
                
                round_configs.append({
                    'type': 'superpixel' if round_type == 'S' else 'graph',
                    'value': val
                })
            
            with st.expander("⚙️ Advanced", expanded=False):
                h_allow_ow = st.checkbox("Overwrite", value=False, key="hybrid_ow",
                    help="When OFF (default), once a pixel is labeled in an earlier round, it stays labeled -- later rounds only fill gaps. "
                         "When ON, a later round can overwrite an earlier label if the new round has higher confidence for that pixel. "
                         "This can improve accuracy in areas where the first round made a poor assignment, but may also cause instability. "
                         "Recommendation: leave OFF for your first attempt, try ON if you see obvious mislabeling.")
            
            seg_params = {
                'round_configs': round_configs,
                'allow_overwrite': h_allow_ow
            }
        
        else:  # Graph-First (Anchor + Fill)
            st.markdown("**Phase 1: Discovery**")
            gf_discovery_scale = st.select_slider(
                "Discovery scale",
                options=LOG_OPTIONS_5K,
                value=1000,
                key="test_gf_discovery",
                help="Felzenszwalb scale for the discovery phase. Higher values create fewer, larger regions that represent "
                     "the most obvious objects in the image. At scale 1000-2000 only genuinely distinct structures (corals, rocks) "
                     "survive as separate regions. These become your 'anchor' labels that fill-in rounds respect. "
                     "Try 500 for more detailed discovery, 2000+ for only the most prominent objects."
            )
            
            st.markdown("**Phase 2: Fill-in**")
            gf_fill_method = st.radio(
                "Fill method",
                ["Superpixel", "Graph"],
                horizontal=True,
                key="test_gf_fill_method",
                help="Which algorithm to use for the fill-in rounds. "
                     "Superpixel (SLIC) creates compact, boundary-respecting regions -- good for clean edges. "
                     "Graph (Felzenszwalb) follows texture/color similarity -- good for natural boundaries."
            )
            gf_fill_key = "graph_first_sp" if gf_fill_method == "Superpixel" else "graph_first_gr"
            gf_fill_options = LOG_OPTIONS_10K if gf_fill_method == "Superpixel" else LOG_OPTIONS_5K
            
            # Initialize fill-round base values
            gf_fill_base = init_base_values("test", gf_fill_key, num_rounds)
            gf_fill_display = get_display_values(gf_fill_base, scale_factor, gf_fill_key, use_smart, gf_fill_options)
            
            gf_fill_values = []
            for i in range(num_rounds):
                def make_gf_callback(idx, mkey, pscale, adapt):
                    def cb():
                        widget_key = f"test_gf_fill_{idx}"
                        if widget_key in st.session_state:
                            update_base_from_display("test", mkey, idx, st.session_state[widget_key], pscale, adapt)
                    return cb
                
                val = st.select_slider(
                    f"Fill {i+1}",
                    options=gf_fill_options,
                    value=gf_fill_display[i],
                    key=f"test_gf_fill_{i}",
                    on_change=make_gf_callback(i, gf_fill_key, scale_factor, use_smart)
                )
                gf_fill_values.append(val)
            
            with st.expander("⚙️ Advanced", expanded=False):
                gf_allow_ow = st.checkbox("Allow fill to overwrite anchors", value=False, key="test_gf_ow",
                    help="When OFF (default), fill-in rounds can only label pixels that the discovery round left unlabeled. "
                         "The discovery 'anchor' labels are preserved. This is usually what you want -- the discovery round identifies "
                         "the obvious objects, and fill-in just handles the rest. "
                         "When ON, fill-in rounds can overwrite discovery labels. Only use this if discovery is mislabeling some areas.")
            
            seg_params = {
                'discovery_scale': gf_discovery_scale,
                'fill_method': 'superpixel' if gf_fill_method == 'Superpixel' else 'graph',
                'fill_values': gf_fill_values,
                'allow_overwrite': gf_allow_ow
            }
        
        # Region merging (graph/hybrid/graph_first only)
        merge_params = None
        if method_key in ("graph", "hybrid", "graph_first"):
            merge_enabled = st.checkbox(
                "Enable region merging", value=True, key="test_merge_enabled",
                help="Felzenszwalb graph segmentation often produces hundreds or thousands of tiny fragments per object instead of one clean region. "
                     "Region merging is a postprocessing step that consolidates those fragments into coherent, object-level masks. "
                     "This is strongly recommended when exporting to Roboflow, because without merging each tiny fragment becomes a separate annotation, "
                     "which is unusable for training. With merging enabled, you typically go from thousands of segments down to tens."
            )
            if merge_enabled:
                with st.expander("🔗 Merge settings", expanded=False):
                    merge_min_area = st.slider("Min area (speckle removal)", 10, 2000, 100, 10,
                        key="test_merge_min_area",
                        help="Any isolated region (connected group of same-class pixels) smaller than this many pixels is removed entirely and set to background. "
                             "This cleans up tiny specks and noise that Felzenszwalb leaves behind -- single pixels, small dots, sensor noise artifacts. "
                             "These specks would each become a tiny, useless annotation in Roboflow. "
                             "Higher values remove more aggressively (but may delete legitimately small objects). "
                             "At 20%% processing scale, 100 pixels here corresponds to roughly 2500 pixels in the original image. "
                             "Default 100 is conservative. Increase to 300-500 if you still see many tiny speckles.")
                    merge_small = st.slider("Small region merge", 50, 5000, 500, 50,
                        key="test_merge_small",
                        help="After speckle removal, regions of the same class that are still smaller than this pixel count get merged into their nearest same-class neighbor. "
                             "For example, if 'coral' has a large region and a small fragment 20px away, the small fragment gets spatially connected to the large one via a thin bridge. "
                             "This dramatically reduces the number of separate instances per class, turning many scattered fragments into a few coherent objects. "
                             "Higher values = more aggressive merging (larger fragments get absorbed). "
                             "Default 500 works well. Increase to 1000-2000 if you want fewer, larger objects. Decrease if small separate objects are important.")
                    merge_color = st.slider("Color similarity threshold", 0.0, 100.0, 30.0, 5.0,
                        key="test_merge_color",
                        help="After size-based merging, this step looks at adjacent regions of the same class and merges them if their average colors are similar enough. "
                             "The value is the maximum Euclidean distance in RGB color space (0-255 per channel) between two regions' mean colors for them to be merged. "
                             "For example, two 'sand' regions that are both beige (similar RGB) will be merged into one, but a light sand and dark shadow region won't be. "
                             "Higher values = merge more aggressively (even regions with somewhat different colors). "
                             "Set to 0 to disable color-based merging entirely. "
                             "Default 30 merges visually similar regions. Increase to 50-80 for more merging, or decrease to 10-15 to only merge near-identical colors.")
                    merge_morph = st.slider("Morph close kernel", 0, 51, 5, 1,
                        key="test_merge_morph",
                        help="Morphological closing is an image processing operation that fills small holes and gaps inside regions. "
                             "Imagine each class mask as a shape with tiny pin-holes and ragged edges -- closing smooths those out by expanding the shape slightly, then shrinking it back. "
                             "The kernel size controls how large the 'fill brush' is (in pixels on the processed image): larger values fill bigger gaps but may also merge regions that should stay separate. "
                             "IMPORTANT: this value is in processed-image pixels, so it depends on your Processing Scale. "
                             "At 20%% scale, kernel 5 covers ~25 original pixels. At 100%% scale, kernel 5 only covers 5 original pixels -- "
                             "so at high scales you may need kernel 15-30+ to get the same physical gap-filling effect. "
                             "This runs BEFORE speckle removal and size-based merging. Set to 0 to skip. Default 5 is good at low scales (0.2-0.4). "
                             "At higher scales (0.6+), try 11-25. Values above 30 are aggressive and may over-smooth boundaries.")
                merge_params = {
                    'min_area': merge_min_area,
                    'small_region_merge': merge_small,
                    'color_threshold': merge_color,
                    'morph_close_ksize': merge_morph
                }
        
        # Confidence filtering (optional)
        conf_enabled = st.checkbox(
            "Enable confidence filtering (slower)", value=False, key="conf_enabled",
            help="Calculates a confidence score (0-100) for each labeled region based on how well it matches the nearby annotation points. "
                 "Regions far from any annotation point, or where the label disagrees with nearby points, get low confidence. "
                 "This adds noticeable processing time because it must analyze every region's relationship to every annotation point. "
                 "Use this when you want to remove uncertain or likely-incorrect labels from your output before sending to Roboflow."
        )
        if conf_enabled:
            conf_threshold = st.slider(
                "Confidence threshold", 0, 100, 40, 5, key="conf_threshold",
                help="Regions with a confidence score below this value are removed (set to background). "
                     "A score of 0 means 'no confidence at all' and 100 means 'perfectly certain'. "
                     "Higher threshold = stricter filtering = more regions removed = less coverage but higher quality labels. "
                     "Lower threshold = keeps more regions including uncertain ones = better coverage but some labels may be wrong. "
                     "Default 40 is a moderate filter. Try 20 for lenient, 60-80 for strict."
            )
        else:
            conf_threshold = 0
            st.caption("Confidence filtering is OFF (faster). Turn it on to filter uncertain regions; it will add noticeable processing time.")

        run_viz = st.button("🎨 Visualize", type="primary", use_container_width=True)
        
        # Export settings button
        if seg_method == "🔀 Hybrid (SLIC + Graph)":
            # For hybrid, format round_configs as scale_values for export
            hybrid_scale_values = [f"{c['type'][0].upper()}:{c['value']}" for c in round_configs]
            settings_txt = format_settings_txt(seg_method, scale_factor, num_rounds, hybrid_scale_values, seg_params, use_smart, conf_threshold, conf_enabled, merge_params)
        elif seg_method == "🔍 Graph-First (Anchor + Fill)":
            gf_display_values = [f"D:{seg_params['discovery_scale']}"] + [f"F:{v}" for v in seg_params['fill_values']]
            settings_txt = format_settings_txt(seg_method, scale_factor, num_rounds, gf_display_values, seg_params, use_smart, conf_threshold, conf_enabled, merge_params)
        else:
            settings_txt = format_settings_txt(seg_method, scale_factor, num_rounds, scale_values, seg_params, use_smart, conf_threshold, conf_enabled, merge_params)
        st.download_button(
            "💾 Export Settings",
            settings_txt,
            file_name="segmentation_settings.txt",
            mime="text/plain",
            use_container_width=True,
            help="Save current settings to a .txt file"
        )
    
    # RIGHT: Result
    with col_right:
        st.markdown("**🖼️ Result**")
        
        # Check if we have a valid result
        has_result = (
            st.session_state.test_result is not None
            and st.session_state.test_result.get('image_name') == test_image
        )
        
        if has_result:
            result = st.session_state.test_result
            
            # Confidence threshold slider (only if enabled)
            conf_summary = result.get('confidence_summary', {})
            if conf_enabled:
                conf_col1, conf_col2 = st.columns([2, 1])
                with conf_col1:
                    conf_threshold = st.slider(
                        "Confidence threshold", 0, 100, conf_threshold, 5,
                        help="Hide regions with confidence below this value"
                    )
                with conf_col2:
                    if conf_summary:
                        st.caption(f"Range: {conf_summary.get('min', 0):.0f}-{conf_summary.get('max', 0):.0f} | Avg: {conf_summary.get('mean', 0):.0f}")
            else:
                conf_threshold = 0
                if conf_summary:
                    st.caption("Confidence filtering is off. Enable it in the left panel to filter uncertain regions (slower).")
            
            # Apply confidence filtering
            final_mask = result['final_mask']
            confidence_map = result.get('confidence_map')
            if conf_enabled and confidence_map is not None and conf_threshold > 0:
                filtered_mask = apply_confidence_threshold(final_mask, confidence_map, conf_threshold)
            else:
                filtered_mask = final_mask
            
            # Rebuild colored mask with filtering
            colored_mask = np.zeros((*filtered_mask.shape, 3), dtype=np.uint8)
            for entry in st.session_state.labelset:
                class_id = int(entry['Count'])
                color_code = entry['Color Code']
                if isinstance(color_code, list):
                    colored_mask[filtered_mask == class_id] = color_code
            
            # Opacity slider
            mask_opacity = st.slider("Overlay strength", 0, 100, 60, 5,
                help="Controls how transparent the colored segmentation mask is on top of the original image. "
                     "0%% = you only see the original photo (no mask visible). 100%% = you only see the colored mask (no photo visible). "
                     "60%% is a good default for inspecting results -- you can see both the class colors and the underlying image. "
                     "Lower it to 20-30%% if you want to focus on the photo, raise it to 80-100%% to focus purely on which pixels got which label.")
            mask_alpha = mask_opacity / 100.0
            
            # Compute overlay
            overlay = cv2.addWeighted(
                result['base_rgb'], 1.0 - mask_alpha,
                colored_mask, mask_alpha, 0
            )
            
            # Resize and display
            result_h = 450
            result_w = int(overlay.shape[1] * (result_h / overlay.shape[0]))
            result_display = cv2.resize(overlay, (result_w, result_h))
            
            # Image + Legend side by side
            img_col, legend_col = st.columns([3, 1.2], gap="small")
            with img_col:
                # Show filtered coverage
                filtered_coverage = (filtered_mask > 0).sum() / filtered_mask.size * 100
                if conf_threshold > 0:
                    st.image(result_display, caption=f"Coverage: {filtered_coverage:.1f}% (filtered from {result['coverage']:.1f}%)")
                else:
                    st.image(result_display, caption=f"Coverage: {result['coverage']:.1f}%")
            with legend_col:
                labeled_pixels = (filtered_mask > 0).sum()
                if labeled_pixels > 0:
                    with st.expander("Legend", expanded=False):
                        unique_ids, counts = np.unique(filtered_mask, return_counts=True)
                        mask_counts = dict(zip(unique_ids.tolist(), counts.tolist()))
                        
                        rows = []
                        for entry in st.session_state.labelset:
                            cid = int(entry.get('Count', 0))
                            cnt = mask_counts.get(cid, 0)
                            if cnt <= 0:
                                continue
                            color_code = entry.get('Color Code', [200, 200, 200])
                            if isinstance(color_code, list):
                                color_str = f"rgb({color_code[0]},{color_code[1]},{color_code[2]})"
                            else:
                                color_str = "#ccc"
                            rows.append({
                                'cnt': cnt,
                                'name': entry.get('Name', entry.get('Short Code', '')),
                                'short': entry.get('Short Code', ''),
                                'color': color_str
                            })
                        
                        for r in sorted(rows, key=lambda x: x['cnt'], reverse=True):
                            pct = (r['cnt'] / labeled_pixels) * 100
                            st.markdown(
                                f"<div style='display:flex;align-items:flex-start;margin-bottom:6px;'>"
                                f"<span style='display:inline-block;width:12px;height:12px;background:{r['color']};border:1px solid #555;border-radius:2px;margin-right:6px;margin-top:2px;flex-shrink:0;'></span>"
                                f"<div style='line-height:1.2;'>"
                                f"<div style='font-size:11px;font-weight:600;'>{r['name']}</div>"
                                f"<div style='font-size:10px;color:#888;'>{r['short']} • {pct:.1f}%</div>"
                                f"</div></div>",
                                unsafe_allow_html=True
                            )
            # --- Segment count panel ---
            with st.expander("📊 Segment Counts (for Roboflow)", expanded=True):
                pre_stats = result.get('pre_merge_seg_stats', {})
                post_stats = result.get('post_merge_seg_stats', {})
                merge_applied = result.get('merge_applied', False)
                
                if merge_applied and pre_stats.get('total', 0) > 0:
                    st.markdown(
                        f"**Before merge:** {pre_stats.get('total', '?')} segments &rarr; "
                        f"**After merge:** {post_stats.get('total', '?')} segments "
                        f"(**{pre_stats.get('total',0) - post_stats.get('total',0)}** removed)"
                    )
                else:
                    st.markdown(f"**Total segments:** {post_stats.get('total', '?')}")
                
                # Per-class breakdown table
                seg_rows = []
                for entry in st.session_state.labelset:
                    cid = int(entry.get('Count', 0))
                    post_n = post_stats.get(cid, 0)
                    if post_n <= 0:
                        continue
                    row = {
                        'Class': entry.get('Name', entry.get('Short Code', '')),
                        'Segments': post_n,
                    }
                    if merge_applied:
                        pre_n = pre_stats.get(cid, 0)
                        row['Before Merge'] = pre_n
                        row['Reduced'] = pre_n - post_n
                    seg_rows.append(row)
                
                if seg_rows:
                    seg_rows.sort(key=lambda x: x['Segments'], reverse=True)
                    import pandas as _pd
                    st.dataframe(_pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)
                
                st.caption("Each 'segment' is a separate connected-component instance that Roboflow will see as an individual annotation.")
        else:
            st.info("👈 Click 'Visualize' to see result")
    
    # Process visualization
    if run_viz:
        norm_name = normalize_image_name(test_image)
        ann_key = st.session_state.norm_to_ann.get(norm_name)
        if not ann_key:
            st.warning(f"No annotations for {test_image}")
        else:
            with st.spinner(f"Processing {test_image}..."):
                image = st.session_state.images[test_image]
                points_df = st.session_state.points_dict[ann_key]
                
                # Scale down
                scaled_image, scaled_points = scale_image_and_points(image, points_df, scale_factor)
                
                # Segment
                if seg_method == "🔷 Superpixel (SLIC)":
                    final_mask, intermediate = multi_scale_labeling(
                        scaled_image, scaled_points, st.session_state.labelset, seg_params['scales']
                    )
                elif seg_method == "🎯 Adaptive (Density-based)":
                    final_mask, intermediate = multi_scale_adaptive_labeling(
                        scaled_image, scaled_points, st.session_state.labelset, seg_params['scales'],
                        min_distance=seg_params.get('min_distance', 10),
                        density_threshold=seg_params.get('density_threshold', 5),
                        allow_overwrite=seg_params.get('allow_overwrite', False)
                    )
                elif seg_method == "📊 Graph-based (Felzenszwalb)":
                    final_mask, intermediate = multi_scale_graph_labeling(
                        scaled_image, scaled_points, st.session_state.labelset, seg_params['scales'],
                        allow_overwrite=seg_params.get('allow_overwrite', False)
                    )
                elif seg_method == "🔀 Hybrid (SLIC + Graph)":
                    final_mask, intermediate = multi_scale_hybrid_labeling(
                        scaled_image, scaled_points, st.session_state.labelset,
                        seg_params['round_configs'],
                        allow_overwrite=seg_params.get('allow_overwrite', False)
                    )
                else:  # Graph-First
                    final_mask, intermediate = multi_scale_graph_first_labeling(
                        scaled_image, scaled_points, st.session_state.labelset,
                        discovery_scale=seg_params['discovery_scale'],
                        fill_method=seg_params['fill_method'],
                        fill_values=seg_params['fill_values'],
                        allow_overwrite=seg_params.get('allow_overwrite', False)
                    )
                
                # Apply region merging for graph/hybrid methods
                pre_merge_seg_stats = count_segments(final_mask)
                if merge_params is not None:
                    final_mask = merge_regions(
                        final_mask, image=scaled_image,
                        **merge_params
                    )
                post_merge_seg_stats = count_segments(final_mask)
                
                # Calculate confidence scores only if enabled
                if conf_enabled:
                    confidence_map, region_stats = calculate_region_confidence(
                        final_mask, scaled_points, st.session_state.labelset
                    )
                    confidence_summary = get_confidence_summary(region_stats)
                else:
                    confidence_map, region_stats, confidence_summary = None, {}, {}
                
                # Create colored mask
                image_rgb = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
                colored_mask = np.zeros((*final_mask.shape, 3), dtype=np.uint8)
                for entry in st.session_state.labelset:
                    class_id = int(entry['Count'])
                    color_code = entry['Color Code']
                    if isinstance(color_code, list):
                        colored_mask[final_mask == class_id] = color_code
                
                coverage = (final_mask > 0).sum() / final_mask.size * 100
                
                # Store result
                st.session_state.test_result = {
                    'image_name': test_image,
                    'base_rgb': image_rgb,
                    'colored_mask': colored_mask,
                    'final_mask': final_mask,
                    'confidence_map': confidence_map,
                    'region_stats': region_stats,
                    'confidence_summary': confidence_summary,
                    'scaled_points': scaled_points,
                    'intermediate': intermediate,
                    'scaled_image': scaled_image,
                    'coverage': coverage,
                    'pre_merge_seg_stats': pre_merge_seg_stats,
                    'post_merge_seg_stats': post_merge_seg_stats,
                    'merge_applied': merge_params is not None
                }
                
                st.success(f"✓ Processed {test_image}")
                st.rerun()
    
    # Detailed results below
    if has_result and st.session_state.test_result.get('intermediate'):
        st.markdown("---")
        with st.expander("📊 Multi-Scale Progression", expanded=False):
            with st.expander("ℹ️ Understanding Multi-Scale Progression", expanded=False):
                st.markdown("""
**What is Multi-Scale Progression?**

Multi-scale progression is the core technique this app uses to convert sparse point annotations into dense segmentation masks. Instead of processing the image once, we run the segmentation algorithm multiple times at different "scales" (sizes/resolutions), each pass filling in more of the image.

**Why use multiple scales?**

A single segmentation pass would leave large gaps between annotation points. By running multiple passes with progressively different segment sizes, we can:
1. Capture fine details where annotations are dense
2. Fill gaps in sparsely annotated areas
3. Ensure complete coverage of the entire image

**What the visualization shows:**

- **Top row:** The actual segments/superpixels created at each scale, shown with yellow boundaries
- **Bottom row:** The cumulative labeled mask after each scale. Colors show which class each pixel is assigned to
- **Percentage values:** Show what fraction of the image is labeled after each scale (should approach 100%)

**How to interpret results:**

- If coverage jumps mostly at Scale 1 → your annotations are dense and well-distributed
- If coverage jumps mostly at Scale 2-3 → your annotations are sparse and the algorithm is extrapolating
- If final coverage is below 95% → consider adjusting Scale 3 parameters for more aggressive gap-filling
""")
            
            intermediate = st.session_state.test_result['intermediate']
            scaled_image = st.session_state.test_result['scaled_image']
            
            # Coverage metrics
            cols = st.columns(len(intermediate))
            for i, result in enumerate(intermediate):
                with cols[i]:
                    cov = (result['cumulative_mask'] > 0).sum() / result['cumulative_mask'].size * 100
                    if 'n_superpixels' in result:
                        label = f"{result['n_superpixels']} SP"
                    else:
                        label = f"Scale {result.get('scale', i+1)}"
                    st.metric(f"Scale {i+1}", f"{cov:.1f}%", label)
            
            # Visualization
            from skimage.segmentation import mark_boundaries
            
            fig, axes = plt.subplots(2, len(intermediate), figsize=(14, 7))
            image_rgb = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
            
            for i, result in enumerate(intermediate):
                # Top: segments
                if 'superpixels' in result:
                    marked = mark_boundaries(image_rgb, result['superpixels'], color=(1, 1, 0), mode='thick')
                    title = f"{result['n_superpixels']} SP"
                elif 'segments' in result:
                    marked = mark_boundaries(image_rgb, result['segments'], color=(1, 1, 0), mode='thick')
                    title = f"{result.get('scale', 'Scale')}x"
                else:
                    marked = image_rgb.copy()
                    title = f"Scale {i+1}"
                axes[0, i].imshow(marked)
                axes[0, i].set_title(title, fontsize=10)
                axes[0, i].axis('off')
                
                # Bottom: cumulative mask
                colored = np.zeros((*result['cumulative_mask'].shape, 3), dtype=np.uint8)
                for entry in st.session_state.labelset:
                    class_id = int(entry['Count'])
                    color_code = entry['Color Code']
                    if isinstance(color_code, list):
                        colored[result['cumulative_mask'] == class_id] = color_code
                
                overlay = cv2.addWeighted(image_rgb, 0.5, colored, 0.5, 0)
                axes[1, i].imshow(overlay)
                cov = (result['cumulative_mask'] > 0).sum() / result['cumulative_mask'].size * 100
                axes[1, i].set_title(f"{cov:.1f}%", fontsize=10)
                axes[1, i].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ==================== EXPORT MODE ====================
else:
    st.subheader("📦 Export to COCO")
    
    with st.expander("ℹ️ How to use Export COCO", expanded=False):
        st.markdown(
            """
            Use this mode when you're happy with the method/settings and want to process **many images**.
 
            1. Select images (use **Select All** if you want everything).
            2. Pick a segmentation method and scale settings.
            3. Click **Process All Images**.
            4. Download **COCO JSON** and upload it to Roboflow.
 
            Tip: if you see only a few selectable images, enable **Show all images** above, or click **Match images and annotations**.
            """
        )
    
    # Settings
    col_select, col_method, col_params = st.columns([1, 1.2, 1.2])
    
    with col_select:
        st.markdown("**Image Selection**")
        
        # Select all button
        col_sel_btn1, col_sel_btn2 = st.columns(2)
        with col_sel_btn1:
            if st.button("Select All", key="select_all"):
                st.session_state.export_selection = selectable_images
                st.rerun()
        with col_sel_btn2:
            if st.button("Clear All", key="clear_all"):
                st.session_state.export_selection = []
                st.rerun()
        
        # Get default selection
        default_sel = st.session_state.get('export_selection', selectable_images[:min(3, len(selectable_images))])
        # Filter to only valid images
        default_sel = [img for img in default_sel if img in selectable_images]
        
        selected = st.multiselect(
            "Select images", selectable_images, default=default_sel,
            label_visibility="collapsed"
        )
        st.session_state.export_selection = selected
        st.caption(f"{len(selected)} / {len(selectable_images)} images selected")
    
    with col_method:
        st.markdown("**Segmentation Method**")
        seg_method = st.radio(
            "Method",
            ["🔷 Superpixel (SLIC)", "🎯 Adaptive (Density-based)", "📊 Graph-based (Felzenszwalb)", "🔀 Hybrid (SLIC + Graph)", "🔍 Graph-First (Anchor + Fill)"],
            key="export_method", label_visibility="collapsed"
        )
        if seg_method == "🔷 Superpixel (SLIC)":
            st.caption("Superpixels: divides the image into compact, boundary-respecting regions (SLIC algorithm), then assigns each region the label of the nearest annotation point inside it. Best all-round default when objects have clear visual edges.")
        elif seg_method == "🎯 Adaptive (Density-based)":
            st.caption("Adaptive: grows regions outward from each annotation point like a watershed, filling gaps proportionally. Works well when annotation density varies across the image or when Superpixels leave uncolored gaps.")
        elif seg_method == "📊 Graph-based (Felzenszwalb)":
            st.caption("Graph-based: uses Felzenszwalb's algorithm to merge pixels by color/texture similarity into regions, then labels each region from your points. Follows natural boundaries well but produces many small fragments -- use Region Merging (below) to consolidate.")
        elif seg_method == "🔀 Hybrid (SLIC + Graph)":
            st.caption("Hybrid: lets you choose Superpixel (S) or Graph (G) independently for each round. Useful when you want SLIC's clean regions for an initial pass and Graph's texture-following for gap-filling, or vice versa.")
        else:
            st.caption("Graph-First: discovers the obvious coherent objects with a high-scale Felzenszwalb pass first, then fills remaining areas with progressive rounds. Anchored discovery labels are preserved during fill-in.")
        
        with st.expander("ℹ️ Method details (what it does / when to use)", expanded=False):
            if seg_method == "🔷 Superpixel (SLIC)":
                st.markdown(
                    """**SLIC Superpixels**

                    - **What it does:** breaks the image into many small regions (superpixels) that try to follow edges.
                    - **How labels spread:** each point label is assigned to its local region; multiple scales fill remaining gaps.
                    - **When to use:** best default when boundaries are visually clear.
                    """
                )
            elif seg_method == "🎯 Adaptive (Density-based)":
                st.markdown(
                    """**Adaptive (density-based / watershed-like growth)**

                    - **What it does:** grows regions outward from your sparse points.
                    - **Why it helps:** can fill larger gaps when points are sparse.
                    - **When to use:** when Superpixels leave holes or when point density varies a lot.
                    """
                )
            elif seg_method == "📊 Graph-based (Felzenszwalb)":
                st.markdown(
                    """**Graph-based (Felzenszwalb)**

                    - **What it does:** merges pixels/regions based on similarity (color/texture) to form natural segments.
                    - **When to use:** when you want segments that follow texture/color patterns rather than a fixed grid.
                    """
                )
            elif seg_method == "🔀 Hybrid (SLIC + Graph)":
                st.markdown(
                    """**Hybrid (SLIC + Graph combined)**

                    - **What it does:** each round can be either Superpixel or Graph-based.
                    - **S rounds:** number = superpixel count (higher = smaller regions)
                    - **G rounds:** number = merge threshold (higher = larger regions)
                    - **When to use:** when you want to combine the strengths of both methods.
                    """
                )
            else:
                st.markdown(
                    """**Graph-First (Anchor + Fill)**

                    - **What it does:** runs Felzenszwalb at a high scale first to discover the obvious coherent objects, then fills remaining areas with progressive rounds.
                    - **Discovery phase:** high scale → large regions → only the most prominent structures survive → anchor labels.
                    - **Fill-in phase:** Superpixel or Graph rounds progressively fill unlabeled areas without overwriting anchors.
                    - **When to use:** when distinct objects (corals, rocks) are surrounded by ambiguous background and you want the algorithm to identify the important structures first.
                    """
                )
    
    with col_params:
        st.markdown("**Parameters**")
        exp_scale_factor = st.slider("Scale", 0.1, 1.0, st.session_state.custom_defaults['scale_factor'], 0.1,
            key="export_scale", help="Controls the resolution at which segmentation is computed. "
                 "The image is resized to this fraction before processing (e.g. 0.4 = 40%% of original pixels). "
                 "Lower values are much faster because the algorithm works on fewer pixels, but fine details and small objects may be lost. "
                 "Higher values preserve detail but take significantly longer -- this matters a lot when batch-processing many images. "
                 "Recommended: 0.3-0.5 for most images. Use 0.2 for very large images or quick tests, 0.6+ when you need precise boundaries on small objects.")
        
        exp_num_rounds = st.slider("Rounds", 1, 4, st.session_state.custom_defaults['num_rounds'], 1, key="exp_rounds",
            help="How many segmentation passes to run per image. Each round creates segments at a different scale and labels unlabeled pixels. "
                 "Round 1 typically covers the majority of the image. Rounds 2-4 progressively fill in remaining gaps with coarser/finer segments. "
                 "More rounds = better pixel coverage (closer to 100%%) but slower processing. "
                 "3 rounds is a good default. Use 1-2 for quick tests, 4 if you see gaps in the final mask.")
        exp_use_smart = st.toggle("Adapt values", value=True, key="exp_smart",
            help="When ON, the round parameter values are automatically adjusted based on your Processing Scale. "
                 "For example, at 40%% scale the image has only 16%% of its original pixels, so superpixel counts are reduced proportionally "
                 "to keep region sizes visually similar. Without this, the same superpixel count on a smaller image would create tiny, useless regions. "
                 "When OFF, you control the exact raw values. Leave this ON unless you know exactly what values you need.")
        
        # Get method key
        if seg_method == "🔷 Superpixel (SLIC)":
            exp_method_key = "superpixel"
            exp_options = LOG_OPTIONS_10K
        elif seg_method == "🎯 Adaptive (Density-based)":
            exp_method_key = "adaptive"
            exp_options = AD_OPTIONS
        elif seg_method == "📊 Graph-based (Felzenszwalb)":
            exp_method_key = "graph"
            exp_options = LOG_OPTIONS_5K
        elif seg_method == "🔀 Hybrid (SLIC + Graph)":
            exp_method_key = "hybrid"
            exp_options = LOG_OPTIONS_10K
        else:  # Graph-First
            exp_method_key = "graph_first"
            exp_options = LOG_OPTIONS_5K
        
        # Initialize base values (persistent, at 100% scale)
        exp_base_values = init_base_values("exp", exp_method_key, exp_num_rounds)
        
        # Derive display values from base values
        exp_display_values = get_display_values(exp_base_values, exp_scale_factor, exp_method_key, exp_use_smart, exp_options)
        
        # Track scale changes - when adapt is ON and scale changes, update widget keys BEFORE widget creation
        exp_scale_track = "exp_last_scale_track"
        if exp_scale_track not in st.session_state:
            st.session_state[exp_scale_track] = exp_scale_factor
        
        if exp_use_smart and st.session_state[exp_scale_track] != exp_scale_factor and exp_method_key not in ("hybrid", "graph_first"):
            st.session_state[exp_scale_track] = exp_scale_factor
            # Pre-set widget keys to new display values (before widgets are created)
            prefixes = {"superpixel": "exp_sp_", "adaptive": "exp_ad_", "graph": "exp_gr_"}
            prefix = prefixes[exp_method_key]
            for i in range(exp_num_rounds):
                st.session_state[f"{prefix}{i}"] = exp_display_values[i]
        
        st.caption(f"{'Adapted' if exp_use_smart else 'Manual'} ({exp_scale_factor*100:.0f}%)")
        
        if seg_method == "🔷 Superpixel (SLIC)":
            exp_scale_values = []
            for i in range(exp_num_rounds):
                def make_exp_callback(idx, mkey, pscale, adapt):
                    def cb():
                        widget_key = f"exp_sp_{idx}"
                        if widget_key in st.session_state:
                            update_base_from_display("exp", mkey, idx, st.session_state[widget_key], pscale, adapt)
                    return cb
                
                val = st.select_slider(
                    f"Round {i+1}", 
                    options=LOG_OPTIONS_10K, 
                    value=exp_display_values[i], 
                    key=f"exp_sp_{i}",
                    on_change=make_exp_callback(i, exp_method_key, exp_scale_factor, exp_use_smart)
                )
                exp_scale_values.append(val)
            seg_params = {'scales': exp_scale_values}
            
        elif seg_method == "🎯 Adaptive (Density-based)":
            exp_scale_values = []
            for i in range(exp_num_rounds):
                def make_exp_callback(idx, mkey, pscale, adapt):
                    def cb():
                        widget_key = f"exp_ad_{idx}"
                        if widget_key in st.session_state:
                            update_base_from_display("exp", mkey, idx, st.session_state[widget_key], pscale, adapt)
                    return cb
                
                val = st.select_slider(
                    f"Round {i+1}", 
                    options=AD_OPTIONS, 
                    value=exp_display_values[i], 
                    key=f"exp_ad_{i}",
                    format_func=lambda x: f"{x:.2f}",
                    on_change=make_exp_callback(i, exp_method_key, exp_scale_factor, exp_use_smart)
                )
                exp_scale_values.append(val)
            with st.expander("⚙️ Advanced", expanded=False):
                exp_min_dist = st.slider("Min dist", 1, 50, st.session_state.custom_defaults['adaptive_min_dist'], 1, key="exp_min_dist",
                    help="Minimum distance (in pixels) between watershed seed points. Higher values = fewer seeds = larger, simpler regions. Default 10.")
                exp_density = st.slider("Density", 1, 20, st.session_state.custom_defaults['adaptive_density'], 1, key="exp_density",
                    help="How many annotation points must be nearby for a region to be 'densely annotated'. Areas below this get coarser regions. Default 5.")
                exp_allow_ow = st.checkbox("Overwrite", value=False, key="exp_allow_ow_ad",
                    help="When ON, later rounds can overwrite earlier labels if the new round has higher confidence. Leave OFF unless you see obvious mislabeling.")
            seg_params = {'scales': exp_scale_values, 'min_distance': exp_min_dist, 'density_threshold': exp_density, 'allow_overwrite': exp_allow_ow}
            
        elif seg_method == "📊 Graph-based (Felzenszwalb)":
            exp_scale_values = []
            for i in range(exp_num_rounds):
                def make_exp_callback(idx, mkey, pscale, adapt):
                    def cb():
                        widget_key = f"exp_gr_{idx}"
                        if widget_key in st.session_state:
                            update_base_from_display("exp", mkey, idx, st.session_state[widget_key], pscale, adapt)
                    return cb
                
                val = st.select_slider(
                    f"Round {i+1}", 
                    options=LOG_OPTIONS_5K, 
                    value=exp_display_values[i], 
                    key=f"exp_gr_{i}",
                    on_change=make_exp_callback(i, exp_method_key, exp_scale_factor, exp_use_smart)
                )
                exp_scale_values.append(val)
            with st.expander("⚙️ Advanced", expanded=False):
                exp_allow_ow_g = st.checkbox("Overwrite", value=False, key="exp_allow_ow_g",
                    help="When ON, later rounds can overwrite earlier labels if the new round has higher confidence. Leave OFF unless you see obvious mislabeling.")
            seg_params = {'scales': exp_scale_values, 'allow_overwrite': exp_allow_ow_g}
        
        elif seg_method == "🔀 Hybrid (SLIC + Graph)":
            st.caption("**S** = Superpixel (count), **G** = Graph (threshold)")
            exp_round_configs = []
            
            # Clear any cached values that might be invalid
            for i in range(4):
                for key in [f"exp_hybrid_s_{i}", f"exp_hybrid_g_{i}"]:
                    if key in st.session_state:
                        val = st.session_state[key]
                        if key.endswith(f"s_{i}") and val not in LOG_OPTIONS_10K:
                            del st.session_state[key]
                        elif key.endswith(f"g_{i}") and val not in LOG_OPTIONS_5K:
                            del st.session_state[key]
            
            for i in range(exp_num_rounds):
                col_type, col_val = st.columns([1, 2])
                
                with col_type:
                    exp_round_type = st.radio(
                        f"R{i+1}",
                        ["S", "G"],
                        horizontal=True,
                        key=f"exp_hybrid_type_{i}",
                        label_visibility="collapsed"
                    )
                
                with col_val:
                    if exp_round_type == "S":
                        # Defaults: 3000 → 1000 → 100
                        exp_val = st.select_slider(
                            f"Round {i+1}",
                            options=LOG_OPTIONS_10K,
                            value=3000 if i == 0 else (1000 if i == 1 else 100),
                            key=f"exp_hybrid_s_{i}",
                            label_visibility="collapsed"
                        )
                    else:
                        # Defaults: 100 → 300 → 1000
                        exp_val = st.select_slider(
                            f"Round {i+1}",
                            options=LOG_OPTIONS_5K,
                            value=100 if i == 0 else (300 if i == 1 else 1000),
                            key=f"exp_hybrid_g_{i}",
                            label_visibility="collapsed"
                        )
                
                exp_round_configs.append({
                    'type': 'superpixel' if exp_round_type == 'S' else 'graph',
                    'value': exp_val
                })
            
            with st.expander("⚙️ Advanced", expanded=False):
                exp_h_allow_ow = st.checkbox("Overwrite", value=False, key="exp_hybrid_ow",
                    help="When ON, later rounds can overwrite earlier labels if the new round has higher confidence. Leave OFF unless you see obvious mislabeling.")
            
            seg_params = {
                'round_configs': exp_round_configs,
                'allow_overwrite': exp_h_allow_ow
            }
        
        else:  # Graph-First (Anchor + Fill)
            st.markdown("**Phase 1: Discovery**")
            exp_gf_discovery_scale = st.select_slider(
                "Discovery scale",
                options=LOG_OPTIONS_5K,
                value=1000,
                key="exp_gf_discovery",
                help="Felzenszwalb scale for the discovery phase. Higher values create fewer, larger regions that represent "
                     "the most obvious objects. These become 'anchor' labels that fill-in rounds respect. "
                     "Try 500 for more detail, 2000+ for only the most prominent objects."
            )
            
            st.markdown("**Phase 2: Fill-in**")
            exp_gf_fill_method = st.radio(
                "Fill method",
                ["Superpixel", "Graph"],
                horizontal=True,
                key="exp_gf_fill_method",
                help="Superpixel (SLIC) for clean edges, Graph (Felzenszwalb) for natural texture boundaries."
            )
            exp_gf_fill_key = "graph_first_sp" if exp_gf_fill_method == "Superpixel" else "graph_first_gr"
            exp_gf_fill_options = LOG_OPTIONS_10K if exp_gf_fill_method == "Superpixel" else LOG_OPTIONS_5K
            
            # Initialize fill-round base values
            exp_gf_fill_base = init_base_values("exp", exp_gf_fill_key, exp_num_rounds)
            exp_gf_fill_display = get_display_values(exp_gf_fill_base, exp_scale_factor, exp_gf_fill_key, exp_use_smart, exp_gf_fill_options)
            
            exp_gf_fill_values = []
            for i in range(exp_num_rounds):
                def make_exp_gf_callback(idx, mkey, pscale, adapt):
                    def cb():
                        widget_key = f"exp_gf_fill_{idx}"
                        if widget_key in st.session_state:
                            update_base_from_display("exp", mkey, idx, st.session_state[widget_key], pscale, adapt)
                    return cb
                
                val = st.select_slider(
                    f"Fill {i+1}",
                    options=exp_gf_fill_options,
                    value=exp_gf_fill_display[i],
                    key=f"exp_gf_fill_{i}",
                    on_change=make_exp_gf_callback(i, exp_gf_fill_key, exp_scale_factor, exp_use_smart)
                )
                exp_gf_fill_values.append(val)
            
            with st.expander("⚙️ Advanced", expanded=False):
                exp_gf_allow_ow = st.checkbox("Allow fill to overwrite anchors", value=False, key="exp_gf_ow",
                    help="When OFF (default), fill-in rounds only label pixels the discovery round left unlabeled. "
                         "When ON, fill-in rounds can overwrite discovery labels.")
            
            seg_params = {
                'discovery_scale': exp_gf_discovery_scale,
                'fill_method': 'superpixel' if exp_gf_fill_method == 'Superpixel' else 'graph',
                'fill_values': exp_gf_fill_values,
                'allow_overwrite': exp_gf_allow_ow
            }
        
        # Region merging (graph/hybrid/graph_first only)
        exp_merge_params = None
        if exp_method_key in ("graph", "hybrid", "graph_first"):
            exp_merge_enabled = st.checkbox(
                "Enable region merging", value=True, key="exp_merge_enabled",
                help="Felzenszwalb graph segmentation often produces hundreds or thousands of tiny fragments per object instead of one clean region. "
                     "Region merging is a postprocessing step that consolidates those fragments into coherent, object-level masks. "
                     "This is strongly recommended when exporting to Roboflow, because without merging each tiny fragment becomes a separate annotation, "
                     "which is unusable for training. With merging enabled, you typically go from thousands of segments down to tens."
            )
            if exp_merge_enabled:
                with st.expander("🔗 Merge settings", expanded=False):
                    exp_merge_min_area = st.slider("Min area (speckle removal)", 10, 2000, 100, 10,
                        key="exp_merge_min_area",
                        help="Any isolated region (connected group of same-class pixels) smaller than this many pixels is removed entirely and set to background. "
                             "This cleans up tiny specks and noise that Felzenszwalb leaves behind -- single pixels, small dots, sensor noise artifacts. "
                             "These specks would each become a tiny, useless annotation in Roboflow. "
                             "Higher values remove more aggressively (but may delete legitimately small objects). "
                             "Default 100 is conservative. Increase to 300-500 if you still see many tiny speckles.")
                    exp_merge_small = st.slider("Small region merge", 50, 5000, 500, 50,
                        key="exp_merge_small",
                        help="After speckle removal, regions of the same class that are still smaller than this pixel count get merged into their nearest same-class neighbor. "
                             "This dramatically reduces the number of separate instances per class, turning many scattered fragments into a few coherent objects. "
                             "Higher values = more aggressive merging (larger fragments get absorbed). "
                             "Default 500 works well. Increase to 1000-2000 if you want fewer, larger objects. Decrease if small separate objects are important.")
                    exp_merge_color = st.slider("Color similarity threshold", 0.0, 100.0, 30.0, 5.0,
                        key="exp_merge_color",
                        help="After size-based merging, this step looks at adjacent regions of the same class and merges them if their average colors are similar enough. "
                             "The value is the maximum Euclidean distance in RGB color space (0-255 per channel) between two regions' mean colors. "
                             "Higher values = merge more aggressively (even regions with somewhat different colors). Set to 0 to disable. "
                             "Default 30 merges visually similar regions. Increase to 50-80 for more merging, decrease to 10-15 to only merge near-identical colors.")
                    exp_merge_morph = st.slider("Morph close kernel", 0, 51, 5, 1,
                        key="exp_merge_morph",
                        help="Morphological closing is an image processing operation that fills small holes and gaps inside regions. "
                             "Imagine each class mask as a shape with tiny pin-holes and ragged edges -- closing smooths those out by expanding the shape slightly, then shrinking it back. "
                             "The kernel size controls how large the 'fill brush' is (in pixels on the processed image): larger values fill bigger gaps but may also merge regions that should stay separate. "
                             "IMPORTANT: this value is in processed-image pixels, so it depends on your Processing Scale. "
                             "At 20%% scale, kernel 5 covers ~25 original pixels. At 100%% scale, kernel 5 only covers 5 original pixels -- "
                             "so at high scales you may need kernel 15-30+ to get the same physical gap-filling effect. "
                             "This runs BEFORE speckle removal and size-based merging. Set to 0 to skip. Default 5 is good at low scales (0.2-0.4). "
                             "At higher scales (0.6+), try 11-25. Values above 30 are aggressive and may over-smooth boundaries.")
                exp_merge_params = {
                    'min_area': exp_merge_min_area,
                    'small_region_merge': exp_merge_small,
                    'color_threshold': exp_merge_color,
                    'morph_close_ksize': exp_merge_morph
                }
        
        # Confidence filtering (optional for export)
        exp_conf_enabled = st.checkbox(
            "Enable confidence filtering (slower)", value=False, key="exp_conf_enabled",
            help="Calculates a confidence score (0-100) for each labeled region based on how well it matches the nearby annotation points. "
                 "Regions far from any annotation point, or where the label disagrees with nearby points, get low confidence. "
                 "This adds noticeable processing time per image because it must analyze every region's relationship to every annotation point. "
                 "Use this when you want to remove uncertain or likely-incorrect labels from your export before sending to Roboflow."
        )
        if exp_conf_enabled:
            exp_conf_threshold = st.slider(
                "Confidence threshold", 0, 100, 40, 5,
                key="exp_conf_threshold",
                help="Regions with a confidence score below this value are removed (set to background) in the exported COCO JSON. "
                     "Higher threshold = stricter filtering = more regions removed = less coverage but higher quality labels. "
                     "Lower threshold = keeps more regions including uncertain ones = better coverage but some labels may be wrong. "
                     "Default 40 is a moderate filter. Try 20 for lenient, 60-80 for strict."
            )
        else:
            exp_conf_threshold = 0
            st.caption("Confidence filtering is OFF (faster). Turn it on to filter uncertain regions; it will add noticeable processing time.")
        
        # Export settings button
        if seg_method == "🔀 Hybrid (SLIC + Graph)":
            exp_hybrid_scale_values = [f"{c['type'][0].upper()}:{c['value']}" for c in exp_round_configs]
            exp_settings_txt = format_settings_txt(seg_method, exp_scale_factor, exp_num_rounds, exp_hybrid_scale_values, seg_params, exp_use_smart, exp_conf_threshold, exp_conf_enabled, exp_merge_params)
        elif seg_method == "🔍 Graph-First (Anchor + Fill)":
            exp_gf_display_values = [f"D:{seg_params['discovery_scale']}"] + [f"F:{v}" for v in seg_params['fill_values']]
            exp_settings_txt = format_settings_txt(seg_method, exp_scale_factor, exp_num_rounds, exp_gf_display_values, seg_params, exp_use_smart, exp_conf_threshold, exp_conf_enabled, exp_merge_params)
        else:
            exp_settings_txt = format_settings_txt(seg_method, exp_scale_factor, exp_num_rounds, exp_scale_values, seg_params, exp_use_smart, exp_conf_threshold, exp_conf_enabled, exp_merge_params)
        st.download_button(
            "💾 Export Settings",
            exp_settings_txt,
            file_name="segmentation_settings.txt",
            mime="text/plain",
            use_container_width=True,
            key="exp_settings_download",
            help="Save current settings to a .txt file"
        )
    
    st.markdown("---")
    
    col_process, col_export = st.columns(2)
    
    with col_process:
        st.markdown(f"**{len(selected)} images** ready to process")
        
        if st.button("🎨 Process All Images", type="primary"):
            if not selected:
                st.warning("Select at least one image")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                st.session_state.processed = {}
                
                for i, image_name in enumerate(selected):
                    status_text.text(f"Processing {image_name}...")
                    
                    image = st.session_state.images[image_name]
                    # Use normalized lookup for annotations
                    norm_name = normalize_image_name(image_name)
                    ann_key = st.session_state.norm_to_ann.get(norm_name, image_name)
                    points_df = st.session_state.points_dict[ann_key]
                    
                    scaled_image, scaled_points = scale_image_and_points(image, points_df, exp_scale_factor)
                    
                    if seg_method == "🔷 Superpixel (SLIC)":
                        final_mask, _ = multi_scale_labeling(
                            scaled_image, scaled_points, st.session_state.labelset, seg_params['scales']
                        )
                    elif seg_method == "🎯 Adaptive (Density-based)":
                        final_mask, _ = multi_scale_adaptive_labeling(
                            scaled_image, scaled_points, st.session_state.labelset, seg_params['scales'],
                            min_distance=seg_params.get('min_distance', 10),
                            density_threshold=seg_params.get('density_threshold', 5),
                            allow_overwrite=seg_params.get('allow_overwrite', False)
                        )
                    elif seg_method == "📊 Graph-based (Felzenszwalb)":
                        final_mask, _ = multi_scale_graph_labeling(
                            scaled_image, scaled_points, st.session_state.labelset, seg_params['scales'],
                            allow_overwrite=seg_params.get('allow_overwrite', False)
                        )
                    elif seg_method == "🔀 Hybrid (SLIC + Graph)":
                        final_mask, _ = multi_scale_hybrid_labeling(
                            scaled_image, scaled_points, st.session_state.labelset,
                            seg_params['round_configs'],
                            allow_overwrite=seg_params.get('allow_overwrite', False)
                        )
                    else:  # Graph-First
                        final_mask, _ = multi_scale_graph_first_labeling(
                            scaled_image, scaled_points, st.session_state.labelset,
                            discovery_scale=seg_params['discovery_scale'],
                            fill_method=seg_params['fill_method'],
                            fill_values=seg_params['fill_values'],
                            allow_overwrite=seg_params.get('allow_overwrite', False)
                        )
                    
                    # Apply region merging for graph/hybrid/graph_first methods
                    if exp_merge_params is not None:
                        final_mask = merge_regions(
                            final_mask, image=scaled_image,
                            **exp_merge_params
                        )
                    
                    # Apply confidence filtering if enabled and threshold > 0
                    if exp_conf_enabled and exp_conf_threshold > 0:
                        confidence_map, _ = calculate_region_confidence(
                            final_mask, scaled_points, st.session_state.labelset
                        )
                        final_mask = apply_confidence_threshold(final_mask, confidence_map, exp_conf_threshold)
                    
                    final_mask = cv2.resize(final_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    st.session_state.processed[image_name] = {
                        'mask': final_mask, 'width': image.shape[1], 'height': image.shape[0]
                    }
                    
                    progress_bar.progress((i + 1) / len(selected))
                
                status_text.text(f"✓ Processed {len(selected)} images!")
                st.success(f"✓ All {len(selected)} images processed!")
    
    with col_export:
        if st.session_state.processed:
            st.markdown(f"**{len(st.session_state.processed)} images** processed")
            
            coco_dict = export_to_coco_dict(st.session_state.processed, st.session_state.labelset)
            coco_json = json.dumps(coco_dict, indent=2)
            
            st.download_button(
                "📥 Download COCO JSON", data=coco_json,
                file_name="coco_annotations.json", mime="application/json", type="primary"
            )
            
            st.caption(f"Images: {len(coco_dict['images'])} | Annotations: {len(coco_dict['annotations'])}")
        else:
            st.info("Process images first to enable export.")
    
    # Preview
    if st.session_state.processed:
        st.markdown("---")
        st.subheader("👁️ Preview Results")
        
        preview_image = st.selectbox("Select image to preview", list(st.session_state.processed.keys()))
        
        if preview_image:
            col_orig, col_result = st.columns(2)
            
            image = st.session_state.images[preview_image]
            mask = st.session_state.processed[preview_image]['mask']
            
            colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for entry in st.session_state.labelset:
                class_id = int(entry['Count'])
                color_code = entry['Color Code']
                if isinstance(color_code, list):
                    colored_mask[mask == class_id] = color_code
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(image_rgb, 0.4, colored_mask, 0.6, 0)
            
            with col_orig:
                st.markdown("**Original**")
                st.image(image_rgb, use_container_width=True)
            
            with col_result:
                st.markdown("**Segmentation**")
                st.image(overlay, use_container_width=True)
                coverage = (mask > 0).sum() / mask.size * 100
                st.caption(f"Coverage: {coverage:.1f}%")
            
            # Segment count panel for export preview
            with st.expander("📊 Segment Counts (for Roboflow)", expanded=False):
                seg_stats = count_segments(mask)
                st.markdown(f"**Total segments:** {seg_stats.get('total', 0)}")
                
                seg_rows = []
                for entry in st.session_state.labelset:
                    cid = int(entry.get('Count', 0))
                    n = seg_stats.get(cid, 0)
                    if n <= 0:
                        continue
                    seg_rows.append({
                        'Class': entry.get('Name', entry.get('Short Code', '')),
                        'Segments': n,
                    })
                
                if seg_rows:
                    seg_rows.sort(key=lambda x: x['Segments'], reverse=True)
                    import pandas as _pd
                    st.dataframe(_pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)
                
                st.caption("Each 'segment' is a separate connected-component instance that Roboflow will see as an individual annotation.")
