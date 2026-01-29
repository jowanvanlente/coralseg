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

from utils import load_labelset_from_json, load_annotations_from_df, scale_image_and_points
from superpixel_labeling import multi_scale_labeling
from adaptive_segmentation import multi_scale_adaptive_labeling
from graph_segmentation import multi_scale_graph_labeling
from coco_export import export_to_coco_dict

st.set_page_config(page_title="Sparse to Dense COCO", layout="wide")

st.title("Sparse Annotations → Dense Segmentation → COCO")
st.caption("Upload your images and CoralNet annotations to generate COCO format segmentation masks for Roboflow.")

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

# Image upload
st.sidebar.markdown("### 🖼️ Images")
uploaded_images = st.sidebar.file_uploader(
    "Upload images",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    help="Upload your reef/coral images (.jpg, .jpeg, .png). These are the images you want to segment."
)

if uploaded_images:
    for img_file in uploaded_images:
        if img_file.name not in st.session_state.images:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.session_state.images[img_file.name] = img
            img_file.seek(0)
    st.sidebar.success(f"✓ {len(st.session_state.images)} images loaded")

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
        required_cols = {'Name', 'Row', 'Column', 'Label'}
        if required_cols.issubset(df.columns):
            st.session_state.points_dict = load_annotations_from_df(df)
            total_points = sum(len(d) for d in st.session_state.points_dict.values())
            st.sidebar.success(f"✓ {total_points:,} annotations loaded")
        else:
            missing = required_cols - set(df.columns)
            st.sidebar.error(f"Missing columns: {missing}")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")

# Labelset editor
st.sidebar.markdown("### 🏷️ Labelset")
with st.sidebar.expander("View/Edit Labelset", expanded=False):
    st.caption("The labelset defines class colors. You can edit the JSON below.")
    labelset_json = st.text_area(
        "Labelset JSON",
        value=json.dumps(st.session_state.labelset, indent=2),
        height=300,
        label_visibility="collapsed"
    )
    if st.button("Update Labelset"):
        try:
            new_labelset = json.loads(labelset_json)
            st.session_state.labelset = load_labelset_from_json(new_labelset)
            st.success("Labelset updated!")
            st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

# Apply metadata to labelset
st.session_state.labelset = load_labelset_from_json(st.session_state.labelset)

# Status
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Status")
st.sidebar.info(f"Images: {len(st.session_state.images)}")
st.sidebar.info(f"Annotated images: {len(st.session_state.points_dict)}")
st.sidebar.info(f"Classes: {len(st.session_state.labelset)}")

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

# Get images with annotations
annotated_images = [name for name in st.session_state.images.keys() 
                   if name in st.session_state.points_dict]

if not annotated_images:
    st.warning("⚠️ No images have matching annotations. Make sure image filenames in your CSV match uploaded image names.")
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
    
    # General settings
    st.subheader("🔧 Settings")
    col_method, col_scale = st.columns([2, 1])
    
    with col_method:
        seg_method = st.radio(
            "Method",
            ["🔷 Superpixel (SLIC)", "🎯 Adaptive (Density-based)", "📊 Graph-based (Felzenszwalb)"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    with col_scale:
        scale_factor = st.slider("Processing Scale", 0.1, 1.0, 0.4, 0.05,
            help="Lower = faster but less detail. 0.4 recommended.")
    
    st.markdown("---")
    
    # Three columns layout
    col_left, col_mid, col_right = st.columns([1.6, 0.7, 1.6])
    
    # LEFT: Image selection and preview
    with col_left:
        st.markdown("**🖼️ Select Image**")
        test_image = st.selectbox("Image", annotated_images, label_visibility="collapsed")
        
        total_points_in_image = 0
        if test_image and test_image in st.session_state.points_dict:
            total_points_in_image = len(st.session_state.points_dict[test_image])
        
        show_points = st.toggle("📍 Show annotations", value=False)
        
        if test_image:
            image = st.session_state.images[test_image]
            preview_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preview_h = 450
            preview_w = int(image.shape[1] * (preview_h / image.shape[0]))
            preview_resized = cv2.resize(preview_rgb, (preview_w, preview_h))
            
            points_df = st.session_state.points_dict.get(test_image)
            
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
            
            st.image(preview_resized, caption=f"Original: {image.shape[1]}x{image.shape[0]} | {total_points_in_image} annotations")
    
    # MIDDLE: Parameters
    with col_mid:
        st.markdown("**⚙️ Parameters**")
        
        if seg_method == "🔷 Superpixel (SLIC)":
            scale_1 = st.number_input("Scale 1 (many)", 100, 10000, 3000, 100)
            scale_2 = st.number_input("Scale 2 (medium)", 10, 5000, 900, 10)
            scale_3 = st.number_input("Scale 3 (few)", 10, 1000, 30, 10)
            seg_params = {'scales': [scale_1, scale_2, scale_3]}
            
        elif seg_method == "🎯 Adaptive (Density-based)":
            as1 = st.slider("Scale 1", 0.1, 1.0, 1.0, 0.1)
            as2 = st.slider("Scale 2", 0.1, 1.0, 0.5, 0.1)
            as3 = st.slider("Scale 3", 0.1, 1.0, 0.25, 0.05)
            with st.expander("⚙️ Advanced"):
                min_dist = st.slider("Min distance", 5, 50, 10, 5)
                density_thresh = st.slider("Density thresh", 1, 20, 5, 1)
                allow_ow = st.checkbox("Allow overwrite", value=False)
            seg_params = {
                'scales': [as1, as2, as3],
                'min_distance': min_dist if 'min_dist' in dir() else 10,
                'density_threshold': density_thresh if 'density_thresh' in dir() else 5,
                'allow_overwrite': allow_ow if 'allow_ow' in dir() else False
            }
            
        else:  # Graph-based
            gs1 = st.number_input("Scale 1 (fine)", 10, 500, 100, 10)
            gs2 = st.number_input("Scale 2 (medium)", 50, 1000, 300, 50)
            gs3 = st.number_input("Scale 3 (coarse)", 100, 5000, 1000, 100)
            with st.expander("⚙️ Advanced"):
                g_allow_ow = st.checkbox("Allow overwrite", value=False, key="g_ow")
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
            
            st.image(result_display, caption=f"Coverage: {result['coverage']:.1f}%")
            
            # Legend
            with st.expander("Legend", expanded=False):
                final_mask = result['final_mask']
                labeled_pixels = (final_mask > 0).sum()
                if labeled_pixels > 0:
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
                        rows.append({'cnt': cnt, 'name': entry.get('Name', entry.get('Short Code', '')),
                                    'short': entry.get('Short Code', ''), 'color': color_str})
                    
                    for r in sorted(rows, key=lambda x: x['cnt'], reverse=True):
                        pct = (r['cnt'] / labeled_pixels) * 100
                        st.markdown(
                            f"<div style='display:flex;align-items:center;margin-bottom:4px;'>"
                            f"<span style='display:inline-block;width:12px;height:12px;background:{r['color']};border:1px solid #555;border-radius:2px;margin-right:6px;'></span>"
                            f"<span style='font-size:12px;'>{r['name']} ({r['short']}) - {pct:.1f}%</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
        else:
            st.info("👈 Click 'Visualize' to see result")
    
    # Process visualization
    if run_viz:
        if test_image not in st.session_state.points_dict:
            st.warning(f"No annotations for {test_image}")
        else:
            with st.spinner(f"Processing {test_image}..."):
                image = st.session_state.images[test_image]
                points_df = st.session_state.points_dict[test_image]
                
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
        scale_factor = st.slider("Processing Scale", 0.1, 1.0, 0.4, 0.1, key="export_scale")
        
        if seg_method == "🔷 Superpixel (SLIC)":
            seg_params = {'scales': [3000, 900, 30]}
        elif seg_method == "🎯 Adaptive (Density-based)":
            seg_params = {'scales': [1.0, 0.5, 0.25], 'min_distance': 10, 'density_threshold': 5, 'allow_overwrite': False}
        else:
            seg_params = {'scales': [100, 300, 1000], 'allow_overwrite': False}
    
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
                    points_df = st.session_state.points_dict[image_name]
                    
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
