"""
Graph-based segmentation method using Felzenszwalb's algorithm.
"""

import numpy as np
import cv2
from skimage.segmentation import felzenszwalb
from utils import label_segments_from_points


def create_graph_segments(image, scale=100, sigma=0.5, min_size=50):
    """Create segments using Felzenszwalb's graph-based segmentation."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segments = felzenszwalb(image_rgb, scale=scale, sigma=sigma, min_size=min_size)
    return segments


def multi_scale_graph_labeling(image, points_df, labelset, scales=[100, 300, 1000],
                               allow_overwrite=False, overwrite_threshold=2,
                               sigma=0.8, min_size=20):
    """Apply multi-scale graph-based segmentation.
    
    Args:
        sigma: Smoothing before segmentation (0.1-2), higher = smoother boundaries
        min_size: Minimum segment size in pixels (10-200), smaller = more detail
    """
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    intermediate_masks = []
    
    for scale in scales:
        segments = create_graph_segments(image, scale=scale, sigma=sigma, min_size=min_size)
        labeled_mask, confidence_map = label_segments_from_points(segments, points_df, labelset, return_confidence=True)
        
        if allow_overwrite:
            high_confidence = confidence_map >= overwrite_threshold
            combined_mask = np.where((combined_mask == 0) | (high_confidence & (labeled_mask > 0)), labeled_mask, combined_mask)
        else:
            combined_mask = np.where(combined_mask == 0, labeled_mask, combined_mask)
        
        intermediate_masks.append({'scale': scale, 'segments': segments, 'labeled_mask': labeled_mask.copy(), 'cumulative_mask': combined_mask.copy()})
    
    return combined_mask, intermediate_masks
