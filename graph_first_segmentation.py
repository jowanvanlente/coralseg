"""
Graph-First segmentation method (Anchor + Fill).

Phase 1 - Discovery: Uses Felzenszwalb graph segmentation at a HIGH scale
to identify the obvious, coherent objects in the image. High scale values
produce large regions that follow natural color/texture boundaries, effectively
demarcating the most prominent structures (e.g. coral pieces, rocks).
These "anchor" labels are high-confidence and are preserved by default.

Phase 2 - Fill-in: The remaining unlabeled areas (typically large homogeneous
background regions or gaps between objects) are filled using progressive
multi-round segmentation. Each fill round uses either superpixels (SLIC) or
graph segments at increasing/decreasing granularity to progressively cover
all remaining pixels. Fill rounds respect the anchor labels and do not
overwrite them unless explicitly allowed.
"""

import numpy as np
import cv2
from skimage.segmentation import slic, felzenszwalb
from utils import label_segments_from_points


def create_graph_segments(image, scale=100, sigma=0.8, min_size=20):
    """Create segments using Felzenszwalb graph-based algorithm."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segments = felzenszwalb(image_rgb, scale=scale, sigma=sigma, min_size=min_size)
    segments = segments + 1  # Start from 1 instead of 0
    return segments


def create_superpixels(image, n_segments, compactness=10, sigma=1):
    """Create superpixels using SLIC algorithm."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segments = slic(image_rgb, n_segments=n_segments, compactness=compactness,
                    sigma=sigma, start_label=1)
    return segments


def multi_scale_graph_first_labeling(image, points_df, labelset,
                                      discovery_scale=1000,
                                      fill_method='superpixel',
                                      fill_values=None,
                                      allow_overwrite=False):
    """
    Apply Graph-First (Anchor + Fill) segmentation.
    
    Phase 1 - Discovery:
        Runs Felzenszwalb at a high scale to find large, coherent objects.
        These become "anchor" labels that are preserved through fill-in.
    
    Phase 2 - Fill-in:
        Runs multiple rounds of superpixel or graph segmentation to fill
        in the remaining unlabeled areas. Each round uses progressively
        different granularity.
    
    Args:
        image: Input image (BGR)
        points_df: DataFrame with 'Column', 'Row', 'Label'
        labelset: Label definitions
        discovery_scale: Felzenszwalb scale for discovery (higher = larger regions)
        fill_method: 'superpixel' or 'graph' for fill-in rounds
        fill_values: List of values for fill-in rounds.
                     For superpixel: n_segments counts (e.g. [3000, 900, 30])
                     For graph: scale values (e.g. [100, 300, 1000])
        allow_overwrite: If True, fill-in rounds can overwrite discovery labels
    
    Returns:
        combined_mask: Final dense labeled mask
        intermediate_masks: List of intermediate results (discovery + fill rounds)
    """
    if fill_values is None:
        fill_values = [3000, 900, 30] if fill_method == 'superpixel' else [100, 300, 1000]
    
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    intermediate_masks = []
    
    # ===== Phase 1: Discovery =====
    discovery_segments = create_graph_segments(image, scale=discovery_scale)
    discovery_mask, discovery_confidence = label_segments_from_points(
        discovery_segments, points_df, labelset, return_confidence=True
    )
    
    # The discovery mask becomes the anchor
    combined_mask = discovery_mask.copy()
    
    intermediate_masks.append({
        'phase': 'discovery',
        'type': 'graph',
        'value': discovery_scale,
        'segments': discovery_segments,
        'labeled_mask': discovery_mask.copy(),
        'cumulative_mask': combined_mask.copy(),
        'confidence_map': discovery_confidence
    })
    
    # ===== Phase 2: Fill-in =====
    for i, value in enumerate(fill_values):
        if fill_method == 'superpixel':
            segments = create_superpixels(image, n_segments=int(value))
        else:  # graph
            segments = create_graph_segments(image, scale=value)
        
        labeled_mask = label_segments_from_points(segments, points_df, labelset)
        
        if allow_overwrite:
            # Fill-in can overwrite (not recommended for anchor labels)
            combined_mask = np.where(labeled_mask > 0, labeled_mask, combined_mask)
        else:
            # Fill unlabeled pixels only (preserve anchor labels)
            combined_mask = np.where(combined_mask == 0, labeled_mask, combined_mask)
        
        intermediate_masks.append({
            'phase': 'fill',
            'type': fill_method,
            'value': value,
            'segments': segments,
            'labeled_mask': labeled_mask.copy(),
            'cumulative_mask': combined_mask.copy()
        })
    
    return combined_mask, intermediate_masks
