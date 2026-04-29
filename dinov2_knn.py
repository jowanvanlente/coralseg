"""
DINOv2 + KNN segmentation using pretrained foundation model features.
"""

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import gc
from scipy.ndimage import generic_filter
from scipy.stats import mode as scipy_mode


# Cache the DINOv2 model to avoid reloading it multiple times
@st.cache_resource
def load_dinov2_model():
    """Load DINOv2 ViT-S/14 model (cached)."""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    return model


def preprocess_image_for_dinov2(image):
    """Preprocess image for DINOv2: resize to multiple of 14 and normalize."""
    h, w = image.shape[:2]
    
    # Make dimensions divisible by 14
    new_h = ((h + 13) // 14) * 14
    new_w = ((w + 13) // 14) * 14
    
    # Resize image
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # Normalize with ImageNet stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
    image_tensor = normalize(image_tensor)
    
    return image_tensor, (h, w), (new_h, new_w)


def denoise_patch_features(feature_grid):
    """Remove position-dependent artifact noise from DINOv2 patch tokens.
    
    Fits a low-rank linear model (2D polynomial basis of patch coordinates)
    to predict the position-dependent artifact component, then subtracts it.
    Based on the observation from "Denoising Vision Transformers" (DVT, ECCV 2024)
    that ViT outputs contain input-independent artifacts from position embeddings.
    
    Args:
        feature_grid: numpy array [patch_h, patch_w, D] of patch features
    
    Returns:
        Denoised feature grid of the same shape
    """
    patch_h, patch_w, D = feature_grid.shape
    N = patch_h * patch_w
    
    # Flatten to [N, D]
    T = feature_grid.reshape(N, D)
    
    # Build 2D position basis matrix [N, 6]: (1, y, x, y², x², yx)
    ys = np.linspace(-1, 1, patch_h)
    xs = np.linspace(-1, 1, patch_w)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')  # [patch_h, patch_w]
    yy = yy.reshape(N)
    xx = xx.reshape(N)
    
    P = np.stack([
        np.ones(N),   # bias
        yy,           # y
        xx,           # x
        yy ** 2,      # y²
        xx ** 2,      # x²
        yy * xx,      # yx
    ], axis=1).astype(np.float32)  # [N, 6]
    
    # Least-squares: W = (P^T P)^{-1} P^T T  ->  artifact = P @ W
    W, _, _, _ = np.linalg.lstsq(P, T, rcond=None)  # [6, D]
    artifact = P @ W  # [N, D]
    
    # Subtract the position-dependent artifact
    T_denoised = T - artifact
    
    return T_denoised.reshape(patch_h, patch_w, D)


def extract_dinov2_features(image, _model, denoise=True):
    """Extract DINOv2 patch features and keep them at patch resolution.
    
    Args:
        denoise: If True, apply low-rank denoising to remove position-dependent
                 artifacts from patch tokens (DVT method).
    """
    # Preprocess image
    image_tensor, original_size, resized_size = preprocess_image_for_dinov2(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    # Extract patch tokens via forward_features
    with torch.no_grad():
        patch_features_dict = _model.forward_features(image_tensor)
    
    # Get patch tokens from the dictionary
    patch_features = patch_features_dict['x_norm_patchtokens']  # [1, num_patches, 384]
    
    # Remove batch dimension
    patch_features = patch_features.squeeze(0)  # [num_patches, 384]
    
    # Calculate patch grid dimensions
    _, _, new_h, new_w = image_tensor.shape
    patch_h = new_h // 14
    patch_w = new_w // 14
    
    # Verify we have the right number of patches
    expected_patches = patch_h * patch_w
    if patch_features.shape[0] != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches, got {patch_features.shape[0]}")
    
    # Reshape to spatial grid
    feature_grid = patch_features.reshape(patch_h, patch_w, 384)  # [patch_h, patch_w, 384]
    feature_grid = feature_grid.numpy()
    
    # Apply denoising to remove position-dependent artifacts
    if denoise:
        feature_grid = denoise_patch_features(feature_grid)
    
    return feature_grid, original_size


def label_patches_from_knn(feature_map, points_df, labelset, original_size, k=5,
                           confidence_threshold=0.0):
    """Use KNN to propagate labels from sparse points to patch grid.
    
    Args:
        confidence_threshold: Minimum KNN probability for a patch to be labeled.
            Patches where the top-class probability is below this value are set
            to 0 (background/unlabeled).  Range 0.0-1.0.  0 = keep everything.
    """
    label_to_id = {entry['Short Code']: int(entry['Count']) for entry in labelset}
    patch_h, patch_w = feature_map.shape[:2]
    orig_h, orig_w = original_size
    
    # Extract features and labels for annotated points
    point_features = []
    point_labels = []
    
    for _, row in points_df.iterrows():
        col = int(row['Column'])
        row_y = int(row['Row'])
        label_name = row['Label']
        
        if label_name not in label_to_id:
            continue
            
        class_id = label_to_id[label_name]
        
        # Map full-resolution pixel coordinates to nearest patch position
        if 0 <= row_y < orig_h and 0 <= col < orig_w:
            if orig_h > 1:
                patch_y = int(round(row_y * (patch_h - 1) / (orig_h - 1)))
            else:
                patch_y = 0
            if orig_w > 1:
                patch_x = int(round(col * (patch_w - 1) / (orig_w - 1)))
            else:
                patch_x = 0
            patch_y = np.clip(patch_y, 0, patch_h - 1)
            patch_x = np.clip(patch_x, 0, patch_w - 1)
            feature_vector = feature_map[patch_y, patch_x]
            point_features.append(feature_vector)
            point_labels.append(class_id)
    
    if len(point_features) == 0:
        # Return empty patch mask if no valid points
        return np.zeros(feature_map.shape[:2], dtype=np.uint8)
    
    # Convert to numpy arrays
    point_features = np.array(point_features)
    point_labels = np.array(point_labels)
    
    # Clamp K to number of labeled points to avoid sklearn ValueError
    effective_k = min(k, len(point_features))
    
    # Fit KNN classifier
    knn = KNeighborsClassifier(n_neighbors=effective_k)
    knn.fit(point_features, point_labels)
    
    # Predict for all patches
    h, w = feature_map.shape[:2]
    all_features = feature_map.reshape(-1, feature_map.shape[-1])
    predicted_labels = knn.predict(all_features)
    
    # Apply probability-based confidence thresholding
    if confidence_threshold > 0:
        probabilities = knn.predict_proba(all_features)
        max_probs = probabilities.max(axis=1)
        predicted_labels[max_probs < confidence_threshold] = 0
    
    # Reshape back to patch-grid dimensions
    labeled_mask = predicted_labels.reshape(h, w).astype(np.uint8)
    
    return labeled_mask


def _remove_small_patch_regions(patch_mask, min_patch_region):
    """Remove connected components in patch grid smaller than min_patch_region patches."""
    if min_patch_region <= 1:
        return patch_mask
    out = patch_mask.copy()
    for cid in np.unique(out):
        if cid == 0:
            continue
        binary = (out == cid).astype(np.uint8)
        n_comp, labels = cv2.connectedComponents(binary)
        for comp in range(1, n_comp):
            if (labels == comp).sum() < min_patch_region:
                out[labels == comp] = 0
    return out


def _enforce_max_segments(mask, max_segments):
    """Remove smallest connected components until total count <= max_segments.
    
    Operates on the full-resolution labeled mask.  Iteratively removes the
    smallest component (by pixel area) across all classes until the total
    number of connected components is at or below max_segments.
    """
    if max_segments <= 0:
        return mask
    
    out = mask.copy()
    
    while True:
        # Count all connected components
        components = []  # (class_id, comp_label, area, labels_array)
        for cid in np.unique(out):
            if cid == 0:
                continue
            binary = (out == cid).astype(np.uint8)
            n_comp, labels = cv2.connectedComponents(binary)
            for comp in range(1, n_comp):
                area = (labels == comp).sum()
                components.append((cid, comp, area, labels))
        
        if len(components) <= max_segments:
            break
        
        # Remove the smallest component
        components.sort(key=lambda x: x[2])
        _, comp_id, _, comp_labels = components[0]
        out[comp_labels == comp_id] = 0
    
    return out


def multi_scale_dinov2_knn_labeling(image, points_df, labelset, k=5, denoise=True,
                                    mode_filter_size=3, confidence_threshold=0.0,
                                    min_patch_region=2, max_segments=50,
                                    _model=None, **kwargs):
    """Apply DINOv2 + KNN segmentation.
    
    Args:
        k: Number of neighbors for KNN (default: 5)
        denoise: If True, apply low-rank denoising to patch tokens (default: True)
        mode_filter_size: Kernel size for the majority-vote mode filter applied to
                          patch labels before upsampling. Must be odd. Set to 1 to
                          disable filtering. (default: 3)
        confidence_threshold: Minimum KNN probability for a patch to keep its label.
                              Patches below this are set to background.  Range 0.0-1.0.
                              0 = keep everything (default: 0.0)
        min_patch_region: Minimum number of contiguous patches for a region to survive.
                          Isolated clusters smaller than this are removed before
                          upsampling. (default: 2)
        max_segments: Maximum number of connected-component segments in the final mask.
                      After upsampling, the smallest segments are iteratively removed
                      until this cap is met.  0 = no cap. (default: 50)
        _model: Optional pre-loaded DINOv2 model (skips load_dinov2_model if provided).
                Useful for batch processing outside Streamlit where @st.cache_resource
                is not available.
    """
    # Load model (cached in Streamlit, or use pre-loaded model for batch)
    model = _model if _model is not None else load_dinov2_model()
    
    # Extract patch-resolution features (with optional denoising)
    feature_map, original_size = extract_dinov2_features(image, model, denoise=denoise)
    
    # Apply KNN labeling at patch resolution (with optional probability filtering)
    patch_mask = label_patches_from_knn(
        feature_map, points_df, labelset, original_size, k,
        confidence_threshold=confidence_threshold
    )
    
    # Majority-vote filter: collapse noisy per-patch labels into coherent regions.
    # A 5x5 mode filter replaces each patch label with the most common label in its
    # neighbourhood, eliminating isolated misclassified patches while preserving
    # real species boundaries.  Reduces ~1500 tiny segments to ~30-50 regions.
    def _mode_filter(values):
        return scipy_mode(values, keepdims=False).mode
    if mode_filter_size > 1:
        patch_mask = generic_filter(
            patch_mask.astype(np.float64), _mode_filter, size=mode_filter_size
        ).astype(np.uint8)
    
    # Remove small isolated patch-level regions before upsampling
    patch_mask = _remove_small_patch_regions(patch_mask, min_patch_region)
    
    # Upsample patch labels to full image resolution (nearest-neighbor to preserve class IDs)
    orig_h, orig_w = original_size
    labeled_mask = cv2.resize(patch_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    
    # Enforce maximum segment count by removing smallest components
    labeled_mask = _enforce_max_segments(labeled_mask, max_segments)
    
    # Explicitly free feature memory between images (important for large batch processing)
    del feature_map
    gc.collect()
    
    # Create intermediate masks list (for consistency with other methods)
    intermediate_masks = [{
        'k': k,
        'labeled_mask': labeled_mask.copy(),
        'cumulative_mask': labeled_mask.copy()
    }]
    
    return labeled_mask, intermediate_masks
