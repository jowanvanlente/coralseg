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
from coco_export import export_to_coco_dict

st.set_page_config(page_title="Annotation Segmentation", layout="wide")

st.title("Annotation-Based Segmentation")
st.caption("Test segmentation methods on sparse point annotations and export to COCO format.")

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
        'superpixel': [3000, 900, 30],
        'adaptive': [1.0, 0.5, 0.25],
        'adaptive_min_dist': 10,
        'adaptive_density': 5,
        'graph': [100, 300, 1000]
    }

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
        df = pd.read_csv(sample_csv_path)
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
    help="Upload your reef/coral images (.jpg, .jpeg, .png). These are the images you want to segment."
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
    help="CSV from CoralNet with columns: Name, Row, Column, Label. One point annotation per row."
)

if uploaded_csv:
    try:
        df = pd.read_csv(uploaded_csv)
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
st.sidebar.info(f"Annotated images: {len(st.session_state.points_dict)}")
st.sidebar.info(f"Classes: {len(st.session_state.labelset)}")

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
    tab1, tab2, tab3 = st.tabs(["✅ Matched", "⚠️ Images without annotations", "❌ Annotations without images"])
    
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
        st.caption(f"{len(images_without_annotations)} images have no matching annotations")
        items = images_without_annotations
        if search_query:
            items = [n for n in items if search_query.lower() in n.lower()]
        for name in sorted(items)[:100]:
            norm = normalize_image_name(name)
            st.markdown(f"⚠️ `{name}` (normalized: `{norm}`)")
    
    with tab3:
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

# Get images with annotations (using normalized name matching)
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

# Store mappings in session state for use elsewhere
st.session_state.norm_to_ann = norm_to_ann
st.session_state.image_norm_map = image_norm_map
st.session_state.ann_norm_map = ann_norm_map

if not annotated_images:
    st.warning("⚠️ No images have matching annotations. Make sure image filenames in your CSV match uploaded image names (check for double extensions like .JPG.JPG).")
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
    
    # General settings - Method selection vertical, info and scale on right
    st.subheader("🔧 General Settings")
    col_method, col_info_scale = st.columns([1.2, 2])
    
    with col_method:
        st.markdown("**Segmentation Method**")
        seg_method = st.radio(
            "Method",
            ["🔷 Superpixel (SLIC)", "🎯 Adaptive (Density-based)", "📊 Graph-based (Felzenszwalb)"],
            horizontal=False,
            label_visibility="collapsed"
        )
    
    with col_info_scale:
        col_info, col_scale_inner = st.columns([1.5, 1])
        
        with col_info:
            if seg_method == "🔷 Superpixel (SLIC)":
                with st.expander("ℹ️ How Superpixel works", expanded=False):
                    st.markdown("""**SLIC Superpixel Segmentation**

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
                    st.markdown("""**Adaptive Density-Based Segmentation**

**Scale progression: 1.0 → 0.5 → 0.25 (DECREASING resolution)**

The number represents **image resolution multiplier**:
- **1.0** = Full resolution = **Larger segments**
- **0.25** = Quarter resolution = **Smaller, finer segments**

**Why does lower resolution create finer segments?**
At lower resolution, watershed creates segments relative to the smaller image. When upscaled back, these become many small segments.

**Can overwrite:** If enabled and a finer scale has high confidence, it can refine boundaries.""")
            
            else:  # Graph-based
                with st.expander("ℹ️ How Graph-based works", expanded=False):
                    st.markdown("""**Felzenszwalb Graph-Based Segmentation**

**Scale progression: 100 → 300 → 1000 (INCREASING)**

The number is a **similarity threshold** for merging regions:
- **Lower number (100)** = Strict merging = **Many small segments**
- **Higher number (1000)** = Loose merging = **Few large segments**

**Can overwrite:** If enabled and a coarser scale has high confidence, it can correct over-segmentation.""")
        
        with col_scale_inner:
            scale_factor = st.slider("Processing Scale", 0.1, 1.0, st.session_state.custom_defaults['scale_factor'], 0.05,
                help="Lower = faster but less detail. 0.4 recommended for good balance.")
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
        test_image = st.selectbox("Image", annotated_images, label_visibility="collapsed")
        
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
            help="Toggle to show/hide the sparse point annotations on the image")
        
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
        
        if seg_method == "🔷 Superpixel (SLIC)":
            scale_1 = st.number_input("Scale 1 (many small)", 100, 10000, st.session_state.custom_defaults['superpixel'][0], 100,
                help="First pass: many small superpixels for fine detail. Higher = smaller regions.")
            scale_2 = st.number_input("Scale 2 (medium)", 10, 5000, st.session_state.custom_defaults['superpixel'][1], 10,
                help="Second pass: medium superpixels to fill gaps.")
            scale_3 = st.number_input("Scale 3 (few large)", 10, 1000, st.session_state.custom_defaults['superpixel'][2], 10,
                help="Final pass: few large superpixels for complete coverage.")
            
            if st.button("💾 Save", key="save_sp", use_container_width=True, help="Save these values as your defaults"):
                st.session_state.custom_defaults['superpixel'] = [scale_1, scale_2, scale_3]
                st.toast(f"✓ Saved [{scale_1}, {scale_2}, {scale_3}] as new defaults")
            
            seg_params = {'scales': [scale_1, scale_2, scale_3]}
            
        elif seg_method == "🎯 Adaptive (Density-based)":
            as1 = st.slider("Scale 1 (coarse)", 0.1, 1.0, st.session_state.custom_defaults['adaptive'][0], 0.1,
                help="Full resolution creates larger segments.")
            as2 = st.slider("Scale 2 (medium)", 0.1, 1.0, st.session_state.custom_defaults['adaptive'][1], 0.1,
                help="Half resolution for medium segments.")
            as3 = st.slider("Scale 3 (fine)", 0.1, 1.0, st.session_state.custom_defaults['adaptive'][2], 0.05,
                help="Quarter resolution for fine gap-filling.")
            
            with st.expander("⚙️ Advanced"):
                min_dist = st.slider("Min segment distance", 5, 50, st.session_state.custom_defaults['adaptive_min_dist'], 5,
                    help="Minimum pixels between watershed seeds.")
                density_thresh = st.slider("Density threshold", 1, 20, st.session_state.custom_defaults['adaptive_density'], 1,
                    help="Points needed for 'dense' area treatment.")
                allow_ow = st.checkbox("Allow overwriting", value=False,
                    help="Let later scales overwrite earlier labels if confident.")
            
            if st.button("💾 Save", key="save_ad", use_container_width=True, help="Save these values as your defaults"):
                st.session_state.custom_defaults['adaptive'] = [as1, as2, as3]
                st.session_state.custom_defaults['adaptive_min_dist'] = min_dist
                st.session_state.custom_defaults['adaptive_density'] = density_thresh
                st.toast(f"✓ Saved adaptive settings as new defaults")
            
            seg_params = {
                'scales': [as1, as2, as3],
                'min_distance': min_dist if 'min_dist' in dir() else 10,
                'density_threshold': density_thresh if 'density_thresh' in dir() else 5,
                'allow_overwrite': allow_ow if 'allow_ow' in dir() else False
            }
            
        else:  # Graph-based
            gs1 = st.number_input("Scale 1 (fine)", 10, 500, st.session_state.custom_defaults['graph'][0], 10,
                help="Low threshold = strict merging = many small segments.")
            gs2 = st.number_input("Scale 2 (medium)", 50, 1000, st.session_state.custom_defaults['graph'][1], 50,
                help="Medium threshold for gap filling.")
            gs3 = st.number_input("Scale 3 (coarse)", 100, 5000, st.session_state.custom_defaults['graph'][2], 100,
                help="High threshold = loose merging = complete coverage.")
            
            with st.expander("⚙️ Advanced"):
                g_allow_ow = st.checkbox("Allow overwriting", value=False, key="g_ow",
                    help="Let later scales overwrite earlier labels if confident.")
            
            if st.button("💾 Save", key="save_gr", use_container_width=True, help="Save these values as your defaults"):
                st.session_state.custom_defaults['graph'] = [gs1, gs2, gs3]
                st.toast(f"✓ Saved [{gs1}, {gs2}, {gs3}] as new defaults")
            
            seg_params = {
                'scales': [gs1, gs2, gs3],
                'allow_overwrite': g_allow_ow if 'g_allow_ow' in dir() else False
            }
        
        st.markdown("---")
        run_viz = st.button("🎨 Visualize", type="primary", use_container_width=True)
    
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
            
            # Opacity slider
            mask_opacity = st.slider("Overlay strength", 0, 100, 60, 5)
            mask_alpha = mask_opacity / 100.0
            
            # Compute overlay
            overlay = cv2.addWeighted(
                result['base_rgb'], 1.0 - mask_alpha,
                result['colored_mask'], mask_alpha, 0
            )
            
            # Resize and display
            result_h = 450
            result_w = int(overlay.shape[1] * (result_h / overlay.shape[0]))
            result_display = cv2.resize(overlay, (result_w, result_h))
            
            # Image + Legend side by side
            img_col, legend_col = st.columns([3, 1.2], gap="small")
            with img_col:
                st.image(result_display, caption=f"Coverage: {result['coverage']:.1f}%")
            with legend_col:
                final_mask = result['final_mask']
                labeled_pixels = (final_mask > 0).sum()
                if labeled_pixels > 0:
                    with st.expander("Legend", expanded=False):
                        unique_ids, counts = np.unique(final_mask, return_counts=True)
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
                else:
                    final_mask, intermediate = multi_scale_graph_labeling(
                        scaled_image, scaled_points, st.session_state.labelset, seg_params['scales'],
                        allow_overwrite=seg_params.get('allow_overwrite', False)
                    )
                
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
                    'intermediate': intermediate,
                    'scaled_image': scaled_image,
                    'coverage': coverage
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
    
    # Settings
    col_select, col_method, col_params = st.columns([1, 1.2, 1.2])
    
    with col_select:
        st.markdown("**Image Selection**")
        selected = st.multiselect(
            "Select images", annotated_images, default=annotated_images[:min(3, len(annotated_images))],
            label_visibility="collapsed"
        )
        st.caption(f"{len(selected)} images selected")
    
    with col_method:
        st.markdown("**Segmentation Method**")
        seg_method = st.radio(
            "Method",
            ["🔷 Superpixel (SLIC)", "🎯 Adaptive (Density-based)", "📊 Graph-based (Felzenszwalb)"],
            key="export_method", label_visibility="collapsed"
        )
    
    with col_params:
        st.markdown("**Parameters**")
        scale_factor = st.slider("Processing Scale", 0.1, 1.0, st.session_state.custom_defaults['scale_factor'], 0.1, key="export_scale")
        
        if seg_method == "🔷 Superpixel (SLIC)":
            with st.expander("⚙️ Scale Settings", expanded=False):
                exp_s1 = st.number_input("Scale 1 (many small)", 100, 10000, st.session_state.custom_defaults['superpixel'][0], 100, key="exp_sp1")
                exp_s2 = st.number_input("Scale 2 (medium)", 10, 5000, st.session_state.custom_defaults['superpixel'][1], 10, key="exp_sp2")
                exp_s3 = st.number_input("Scale 3 (few large)", 10, 1000, st.session_state.custom_defaults['superpixel'][2], 10, key="exp_sp3")
            seg_params = {'scales': [exp_s1, exp_s2, exp_s3]}
        elif seg_method == "🎯 Adaptive (Density-based)":
            with st.expander("⚙️ Scale Settings", expanded=False):
                exp_as1 = st.slider("Scale 1 (coarse)", 0.1, 1.0, st.session_state.custom_defaults['adaptive'][0], 0.1, key="exp_ad1")
                exp_as2 = st.slider("Scale 2 (medium)", 0.1, 1.0, st.session_state.custom_defaults['adaptive'][1], 0.1, key="exp_ad2")
                exp_as3 = st.slider("Scale 3 (fine)", 0.1, 1.0, st.session_state.custom_defaults['adaptive'][2], 0.05, key="exp_ad3")
                exp_min_dist = st.slider("Min segment distance", 5, 50, st.session_state.custom_defaults['adaptive_min_dist'], 5, key="exp_min_dist")
                exp_density = st.slider("Density threshold", 1, 20, st.session_state.custom_defaults['adaptive_density'], 1, key="exp_density")
                exp_allow_ow = st.checkbox("Allow overwriting", value=False, key="exp_allow_ow_ad")
            seg_params = {'scales': [exp_as1, exp_as2, exp_as3], 'min_distance': exp_min_dist, 'density_threshold': exp_density, 'allow_overwrite': exp_allow_ow}
        else:
            with st.expander("⚙️ Scale Settings", expanded=False):
                exp_g1 = st.number_input("Scale 1 (fine)", 10, 500, st.session_state.custom_defaults['graph'][0], 10, key="exp_g1")
                exp_g2 = st.number_input("Scale 2 (medium)", 50, 1000, st.session_state.custom_defaults['graph'][1], 50, key="exp_g2")
                exp_g3 = st.number_input("Scale 3 (coarse)", 100, 5000, st.session_state.custom_defaults['graph'][2], 100, key="exp_g3")
                exp_allow_ow_g = st.checkbox("Allow overwriting", value=False, key="exp_allow_ow_g")
            seg_params = {'scales': [exp_g1, exp_g2, exp_g3], 'allow_overwrite': exp_allow_ow_g}
    
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
                    
                    scaled_image, scaled_points = scale_image_and_points(image, points_df, scale_factor)
                    
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
                    else:
                        final_mask, _ = multi_scale_graph_labeling(
                            scaled_image, scaled_points, st.session_state.labelset, seg_params['scales'],
                            allow_overwrite=seg_params.get('allow_overwrite', False)
                        )
                    
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
