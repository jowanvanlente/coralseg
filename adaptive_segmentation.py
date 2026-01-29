"""
Adaptive segmentation method - alternative to superpixels.
"""

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def create_adaptive_segments(image, points_df, min_distance=10, density_threshold=5):
    """Create adaptive segments based on annotation density."""
    h, w = image.shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)
    
    for _, row in points_df.iterrows():
        col = int(row['Column'])
        row_y = int(row['Row'])
        if 0 <= row_y < h and 0 <= col < w:
            density_map[row_y, col] = 1.0
    
    density_map = cv2.GaussianBlur(density_map, (51, 51), 0)
    if density_map.max() > 0:
        density_map = density_map / density_map.max()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    combined = edges.astype(np.float32) + (density_map * 255)
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    distance = ndi.distance_transform_edt(255 - combined)
    
    adaptive_min_dist = int(min_distance * (1.5 - density_map.mean()))
    adaptive_min_dist = max(5, min(adaptive_min_dist, 50))
    
    local_max = peak_local_max(distance, min_distance=adaptive_min_dist, labels=np.ones_like(distance, dtype=bool))
    markers = np.zeros(distance.shape, dtype=np.int32)
    markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
    segments = watershed(-distance, markers, mask=np.ones_like(distance, dtype=bool))
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


def multi_scale_adaptive_labeling(image, points_df, labelset, scales=[1.0, 0.5, 0.25], 
                                  min_distance=10, density_threshold=5, allow_overwrite=False,
                                  overwrite_threshold=3):
    """Apply multi-scale adaptive segmentation."""
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    intermediate_masks = []
    
    for scale in scales:
        if scale != 1.0:
            scaled_h = int(image.shape[0] * scale)
            scaled_w = int(image.shape[1] * scale)
            scaled_image = cv2.resize(image, (scaled_w, scaled_h))
            scaled_points = points_df.copy()
            scaled_points['Column'] = (scaled_points['Column'] * scale).astype(int)
            scaled_points['Row'] = (scaled_points['Row'] * scale).astype(int)
        else:
            scaled_image = image
            scaled_points = points_df
        
        min_dist = int(min_distance / scale)
        segments = create_adaptive_segments(scaled_image, scaled_points, min_distance=min_dist, density_threshold=density_threshold)
        labeled_mask, confidence_map = label_segments_from_points(segments, scaled_points, labelset, return_confidence=True)
        
        if scale != 1.0:
            labeled_mask = cv2.resize(labeled_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            segments = cv2.resize(segments.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            confidence_map = cv2.resize(confidence_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        if allow_overwrite:
            high_confidence = confidence_map >= overwrite_threshold
            combined_mask = np.where((combined_mask == 0) | (high_confidence & (labeled_mask > 0)), labeled_mask, combined_mask)
        else:
            combined_mask = np.where(combined_mask == 0, labeled_mask, combined_mask)
        
        intermediate_masks.append({'scale': scale, 'segments': segments, 'labeled_mask': labeled_mask.copy(), 'cumulative_mask': combined_mask.copy()})
    
    return combined_mask, intermediate_masks
