"""
Confidence scoring for segmented regions.
Calculates a 0-100 confidence score based on:
- Number of annotation points supporting the region
- Region size (very small or huge regions are less confident)
- Label consistency within the region
"""

import numpy as np
from scipy import ndimage


def calculate_region_confidence(mask, points_df, labelset):
    """
    Calculate confidence scores for each labeled region in the mask.
    
    Returns:
        confidence_map: Same shape as mask, with confidence values 0-100 per pixel
        region_stats: Dict with stats per (class_id, region_id)
    """
    label_to_id = {entry['Short Code']: int(entry['Count']) for entry in labelset}
    
    # Get unique class IDs (excluding 0 = background)
    class_ids = np.unique(mask)
    class_ids = class_ids[class_ids > 0]
    
    confidence_map = np.zeros_like(mask, dtype=np.float32)
    region_stats = {}
    
    total_pixels = mask.size
    
    for class_id in class_ids:
        # Get binary mask for this class
        class_mask = (mask == class_id)
        
        # Label connected components within this class
        labeled_regions, num_regions = ndimage.label(class_mask)
        
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            region_size = region_mask.sum()
            
            # Count points in this region
            points_in_region = 0
            matching_points = 0
            
            for _, row in points_df.iterrows():
                col = int(row['Column'])
                row_y = int(row['Row'])
                label_name = row['Label']
                
                if 0 <= row_y < mask.shape[0] and 0 <= col < mask.shape[1]:
                    if region_mask[row_y, col]:
                        points_in_region += 1
                        # Check if point label matches region class
                        if label_name in label_to_id and label_to_id[label_name] == class_id:
                            matching_points += 1
            
            # Calculate confidence components
            
            # 1. Point density score (0-40 points)
            # More points = higher confidence, but with diminishing returns
            if points_in_region == 0:
                point_score = 0
            elif points_in_region == 1:
                point_score = 20
            elif points_in_region == 2:
                point_score = 30
            else:
                point_score = min(40, 30 + points_in_region * 2)
            
            # 2. Label consistency score (0-30 points)
            # All points match = full score
            if points_in_region > 0:
                consistency_ratio = matching_points / points_in_region
                consistency_score = consistency_ratio * 30
            else:
                consistency_score = 0
            
            # 3. Size score (0-30 points)
            # Penalize very small (<0.1% of image) or very large (>30% of image) regions
            size_ratio = region_size / total_pixels
            if size_ratio < 0.001:  # Very tiny
                size_score = 5
            elif size_ratio < 0.005:  # Small
                size_score = 15
            elif size_ratio < 0.01:  # Reasonable small
                size_score = 25
            elif size_ratio < 0.30:  # Good range
                size_score = 30
            elif size_ratio < 0.50:  # Getting large
                size_score = 20
            else:  # Very large
                size_score = 10
            
            # Total confidence (0-100)
            confidence = point_score + consistency_score + size_score
            
            # Store stats
            region_stats[(class_id, region_id)] = {
                'size': region_size,
                'size_ratio': size_ratio,
                'points_in_region': points_in_region,
                'matching_points': matching_points,
                'point_score': point_score,
                'consistency_score': consistency_score,
                'size_score': size_score,
                'confidence': confidence
            }
            
            # Apply confidence to map
            confidence_map[region_mask] = confidence
    
    return confidence_map, region_stats


def apply_confidence_threshold(mask, confidence_map, threshold):
    """
    Zero out regions in mask where confidence is below threshold.
    
    Args:
        mask: Original segmentation mask
        confidence_map: Confidence values per pixel (0-100)
        threshold: Minimum confidence to keep (0-100)
    
    Returns:
        filtered_mask: Mask with low-confidence regions removed
    """
    filtered_mask = mask.copy()
    filtered_mask[confidence_map < threshold] = 0
    return filtered_mask


def get_confidence_summary(region_stats):
    """Get summary statistics about confidence scores."""
    if not region_stats:
        return {'min': 0, 'max': 0, 'mean': 0, 'count': 0}
    
    confidences = [s['confidence'] for s in region_stats.values()]
    return {
        'min': min(confidences),
        'max': max(confidences),
        'mean': sum(confidences) / len(confidences),
        'count': len(confidences)
    }
