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


def label_segments_from_points(segments, points_df, labelset, return_confidence=False):
    """
    Label segments based on sparse point annotations.
    
    Each segment gets the label of the majority of points within it.
    If no points fall inside a segment, it remains unlabeled (0).
    
    Args:
        segments: Segment map
        points_df: DataFrame with 'Column', 'Row', 'Label'
        labelset: Label definitions
        return_confidence: If True, also return vote counts per pixel
    
    Returns:
        labeled_mask: Mask with class IDs
        confidence_map: (optional) Number of annotation votes per pixel
    """
    label_to_id = {entry['Short Code']: int(entry['Count']) for entry in labelset}
    labeled_mask = np.zeros(segments.shape, dtype=np.uint8)
    
    # Count votes for each segment
    segment_votes = {}
    
    for _, row in points_df.iterrows():
        col = int(row['Column'])
        row_y = int(row['Row'])
        label_name = row['Label']
        
        if label_name not in label_to_id:
            continue
        class_id = label_to_id[label_name]
        
        if 0 <= row_y < segments.shape[0] and 0 <= col < segments.shape[1]:
            segment_id = segments[row_y, col]
            if segment_id not in segment_votes:
                segment_votes[segment_id] = {}
            if class_id not in segment_votes[segment_id]:
                segment_votes[segment_id][class_id] = 0
            segment_votes[segment_id][class_id] += 1
    
    confidence_map = np.zeros(segments.shape, dtype=np.uint8) if return_confidence else None
    
    for segment_id, votes in segment_votes.items():
        if votes:
            winning_class = max(votes.items(), key=lambda x: x[1])[0]
            total_votes = sum(votes.values())
            mask = segments == segment_id
            labeled_mask[mask] = winning_class
            if return_confidence:
                confidence_map[mask] = min(total_votes, 255)
    
    if return_confidence:
        return labeled_mask, confidence_map
    return labeled_mask


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
