"""
Graph-based segmentation method using Felzenszwalb's algorithm.
"""

import numpy as np
import cv2
from skimage.segmentation import felzenszwalb


def create_graph_segments(image, scale=100, sigma=0.5, min_size=50):
    """Create segments using Felzenszwalb's graph-based segmentation."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segments = felzenszwalb(image_rgb, scale=scale, sigma=sigma, min_size=min_size)
    return segments


def label_segments_from_points(segments, points_df, labelset, return_confidence=False):
    """Label segments based on sparse point annotations."""
    label_to_id = {entry['Short Code']: int(entry['Count']) for entry in labelset}
    labeled_mask = np.zeros(segments.shape, dtype=np.uint8)
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
    
    confidence_map = np.zeros(segments.shape, dtype=np.uint8)
    for segment_id, votes in segment_votes.items():
        if votes:
            winning_class = max(votes.items(), key=lambda x: x[1])[0]
            total_votes = sum(votes.values())
            labeled_mask[segments == segment_id] = winning_class
            confidence_map[segments == segment_id] = min(total_votes, 255)
    
    if return_confidence:
        return labeled_mask, confidence_map
    return labeled_mask


def multi_scale_graph_labeling(image, points_df, labelset, scales=[100, 300, 1000],
                               allow_overwrite=False, overwrite_threshold=2):
    """Apply multi-scale graph-based segmentation."""
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    intermediate_masks = []
    
    for scale in scales:
        segments = create_graph_segments(image, scale=scale, sigma=0.8, min_size=20)
        labeled_mask, confidence_map = label_segments_from_points(segments, points_df, labelset, return_confidence=True)
        
        if allow_overwrite:
            high_confidence = confidence_map >= overwrite_threshold
            combined_mask = np.where((combined_mask == 0) | (high_confidence & (labeled_mask > 0)), labeled_mask, combined_mask)
        else:
            combined_mask = np.where(combined_mask == 0, labeled_mask, combined_mask)
        
        intermediate_masks.append({'scale': scale, 'segments': segments, 'labeled_mask': labeled_mask.copy(), 'cumulative_mask': combined_mask.copy()})
    
    return combined_mask, intermediate_masks
