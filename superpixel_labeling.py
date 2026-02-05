"""
Superpixel-based dense labeling from sparse annotations.
"""

import numpy as np
import cv2
from skimage.segmentation import slic


def create_superpixels(n_segments, image, compactness=10, sigma=1):
    """Create superpixels using SLIC algorithm."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    superpixels = slic(image_rgb, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=1)
    return superpixels


def label_superpixels_from_points(superpixels, points_df, labelset):
    """Label superpixels based on sparse point annotations."""
    labeled_mask = np.zeros_like(superpixels, dtype=np.uint8)
    label_to_id = {entry['Short Code']: int(entry['Count']) for entry in labelset}
    
    for _, row in points_df.iterrows():
        col = int(row['Column'])
        row_y = int(row['Row'])
        label_name = row['Label']
        if label_name not in label_to_id:
            continue
        class_id = label_to_id[label_name]
        if 0 <= row_y < superpixels.shape[0] and 0 <= col < superpixels.shape[1]:
            superpixel_id = superpixels[row_y, col]
            labeled_mask[superpixels == superpixel_id] = class_id
    return labeled_mask


def join_masks(mask_old, mask_new):
    """Join two masks by filling unlabeled pixels in mask_old with mask_new."""
    return np.where(mask_old == 0, mask_new, mask_old)


def multi_scale_labeling(image, points_df, labelset, superpixel_counts, compactness=10, sigma=1, **kwargs):
    """Apply multi-scale superpixel labeling.
    
    Args:
        compactness: Higher = more regular/square superpixels, lower = follows edges more (1-50)
        sigma: Smoothing before segmentation (0.1-3)
    """
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    intermediate_masks = []
    
    for n_superpixels in superpixel_counts:
        superpixels = create_superpixels(n_superpixels, image, compactness=compactness, sigma=sigma)
        labeled_mask = label_superpixels_from_points(superpixels, points_df, labelset)
        combined_mask = join_masks(combined_mask, labeled_mask)
        intermediate_masks.append({
            'n_superpixels': n_superpixels,
            'superpixels': superpixels,
            'labeled_mask': labeled_mask.copy(),
            'cumulative_mask': combined_mask.copy()
        })
    return combined_mask, intermediate_masks
