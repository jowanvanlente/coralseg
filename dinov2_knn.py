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


def extract_dinov2_features(image, _model):
    """Extract DINOv2 patch features and keep them at patch resolution."""
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
    
    return feature_grid.numpy(), original_size


def label_patches_from_knn(feature_map, points_df, labelset, original_size, k=5):
    """Use KNN to propagate labels from sparse points to patch grid."""
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
    
    # Reshape back to patch-grid dimensions
    labeled_mask = predicted_labels.reshape(h, w).astype(np.uint8)
    
    return labeled_mask


def multi_scale_dinov2_knn_labeling(image, points_df, labelset, k=5, **kwargs):
    """Apply DINOv2 + KNN segmentation.
    
    Args:
        k: Number of neighbors for KNN (default: 5)
    """
    # Load model (cached)
    model = load_dinov2_model()
    
    # Extract patch-resolution features
    feature_map, original_size = extract_dinov2_features(image, model)
    
    # Apply KNN labeling at patch resolution
    patch_mask = label_patches_from_knn(feature_map, points_df, labelset, original_size, k)
    
    # Upsample patch labels to full image resolution (nearest-neighbor to preserve class IDs)
    orig_h, orig_w = original_size
    labeled_mask = cv2.resize(patch_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    
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
