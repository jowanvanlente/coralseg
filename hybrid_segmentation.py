"""
Hybrid segmentation combining SLIC superpixels and Felzenszwalb graph-based segmentation.
Each round can be configured as either type.
"""

import numpy as np
from superpixel_labeling import create_superpixels
from graph_first_segmentation import create_graph_segments
from utils import label_segments_from_points


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
            segments = create_graph_segments(image, scale=value)

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
