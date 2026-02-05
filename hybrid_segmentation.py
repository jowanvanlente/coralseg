"""
Hybrid segmentation combining SLIC superpixels and Felzenszwalb graph-based segmentation.
Each round can be configured as either type.
"""

import numpy as np
import cv2
from skimage.segmentation import slic, felzenszwalb


def create_superpixels(n_segments, image):
    """Create superpixels using SLIC algorithm."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segments = slic(image_rgb, n_segments=n_segments, compactness=10, sigma=1, start_label=1)
    return segments


def create_graph_segments(scale, image):
    """Create segments using Felzenszwalb graph-based algorithm."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segments = felzenszwalb(image_rgb, scale=scale, sigma=0.8, min_size=20)
    segments = segments + 1  # Start from 1 instead of 0
    return segments


def label_segments_from_points(segments, points_df, labelset):
    """Label segments based on sparse point annotations."""
    labeled_mask = np.zeros_like(segments, dtype=np.uint8)
    label_to_id = {entry['Short Code']: int(entry['Count']) for entry in labelset}
    
    for _, row in points_df.iterrows():
        col = int(row['Column'])
        row_y = int(row['Row'])
        label_name = row['Label']
        if label_name not in label_to_id:
            continue
        class_id = label_to_id[label_name]
        if 0 <= row_y < segments.shape[0] and 0 <= col < segments.shape[1]:
            segment_id = segments[row_y, col]
            labeled_mask[segments == segment_id] = class_id
    return labeled_mask


def join_masks(mask_old, mask_new):
    """Join two masks by filling unlabeled pixels in mask_old with mask_new."""
    return np.where(mask_old == 0, mask_new, mask_old)


def multi_scale_hybrid_labeling(image, points_df, labelset, round_configs, allow_overwrite=False):
    """
    Apply multi-scale hybrid labeling with configurable round types.
    
    Args:
        image: Input image
        points_df: DataFrame with point annotations
        labelset: List of label definitions
        round_configs: List of dicts with keys:
            - 'type': 'superpixel' or 'graph'
            - 'value': n_segments for superpixel, scale for graph
        allow_overwrite: Whether later rounds can overwrite earlier labels
    
    Returns:
        combined_mask: Final labeled mask
        intermediate_masks: List of intermediate results
    """
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    intermediate_masks = []
    
    for config in round_configs:
        round_type = config['type']
        value = config['value']
        
        if round_type == 'superpixel':
            segments = create_superpixels(int(value), image)
        else:  # graph
            segments = create_graph_segments(value, image)
        
        labeled_mask = label_segments_from_points(segments, points_df, labelset)
        
        if allow_overwrite:
            combined_mask = np.where(labeled_mask > 0, labeled_mask, combined_mask)
        else:
            combined_mask = join_masks(combined_mask, labeled_mask)
        
        intermediate_masks.append({
            'type': round_type,
            'value': value,
            'segments': segments,
            'labeled_mask': labeled_mask.copy(),
            'cumulative_mask': combined_mask.copy()
        })
    
    return combined_mask, intermediate_masks
