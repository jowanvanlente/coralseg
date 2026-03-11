"""
DINOv2 + KNN segmentation using pretrained foundation model features.
"""

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st


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


@st.cache_data(show_spinner=False)
def extract_dinov2_features(image, _model):
    """Extract DINOv2 patch features from image (cached per image)."""
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
    
    # Upsample to original image resolution
    orig_h, orig_w = original_size
    feature_map = torch.nn.functional.interpolate(
        feature_grid.permute(2, 0, 1).unsqueeze(0),  # [1, 384, patch_h, patch_w]
        size=(orig_h, orig_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).permute(1, 2, 0)  # [orig_h, orig_w, 384]
    
    return feature_map.numpy()


def label_pixels_from_knn(feature_map, points_df, labelset, k=5):
    """Use KNN to propagate labels from sparse points to all pixels."""
    label_to_id = {entry['Short Code']: int(entry['Count']) for entry in labelset}
    
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
        
        # Check bounds
        if 0 <= row_y < feature_map.shape[0] and 0 <= col < feature_map.shape[1]:
            feature_vector = feature_map[row_y, col]
            point_features.append(feature_vector)
            point_labels.append(class_id)
    
    if len(point_features) == 0:
        # Return empty mask if no valid points
        return np.zeros(feature_map.shape[:2], dtype=np.uint8)
    
    # Convert to numpy arrays
    point_features = np.array(point_features)
    point_labels = np.array(point_labels)
    
    # Clamp K to number of labeled points to avoid sklearn ValueError
    effective_k = min(k, len(point_features))
    
    # Fit KNN classifier
    knn = KNeighborsClassifier(n_neighbors=effective_k)
    knn.fit(point_features, point_labels)
    
    # Predict for all pixels
    h, w = feature_map.shape[:2]
    all_features = feature_map.reshape(-1, feature_map.shape[-1])
    predicted_labels = knn.predict(all_features)
    
    # Reshape back to image dimensions
    labeled_mask = predicted_labels.reshape(h, w).astype(np.uint8)
    
    return labeled_mask


def multi_scale_dinov2_knn_labeling(image, points_df, labelset, k=5, **kwargs):
    """Apply DINOv2 + KNN segmentation.
    
    Args:
        k: Number of neighbors for KNN (default: 5)
    """
    # Load model (cached)
    model = load_dinov2_model()
    
    # Extract features
    feature_map = extract_dinov2_features(image, model)  # cached per image
    
    # Apply KNN labeling
    labeled_mask = label_pixels_from_knn(feature_map, points_df, labelset, k)
    
    # Create intermediate masks list (for consistency with other methods)
    intermediate_masks = [{
        'k': k,
        'labeled_mask': labeled_mask.copy(),
        'cumulative_mask': labeled_mask.copy()
    }]
    
    return labeled_mask, intermediate_masks
