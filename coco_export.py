"""
COCO JSON export functions.

Exports dense segmentation masks to COCO format with polygon encoding.
One annotation per connected component (object instance).
"""

import numpy as np
import cv2
import json


def process_single_image_to_coco(args):
    """
    Process one image's mask into COCO annotations with polygon contours.
    
    Args:
        args: Tuple of (image_id, image_name, mask, orig_width, orig_height, scale_factor)
              Mask is at scaled resolution. Coordinates are rescaled back to original dims.
    """
    image_id, image_name, mask, orig_width, orig_height, scale_factor = args
    
    H, W = mask.shape
    inv_scale = 1.0 / scale_factor
    
    image_entry = {
        'id': image_id,
        'file_name': image_name,
        'width': orig_width,
        'height': orig_height
    }
    
    annotations = []
    
    for class_id in np.unique(mask):
        if class_id == 0:
            continue
        
        class_mask = (mask == class_id).astype(np.uint8)
        num_components, component_labels = cv2.connectedComponents(class_mask)
        
        for component_id in range(1, num_components):
            component_mask = (component_labels == component_id).astype(np.uint8)
            
            if component_mask.sum() < 10:
                continue
            
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            polygons = []
            for contour in contours:
                if len(contour) < 3:
                    continue
                poly = [round(v * inv_scale, 1) for v in contour.flatten().tolist()]
                if len(poly) >= 6:
                    polygons.append(poly)
            
            if not polygons:
                continue
            
            rows, cols = np.where(component_mask)
            if len(rows) == 0:
                continue
            
            bbox = [
                round(float(cols.min()) * inv_scale, 1),
                round(float(rows.min()) * inv_scale, 1),
                round(float(cols.max() - cols.min() + 1) * inv_scale, 1),
                round(float(rows.max() - rows.min() + 1) * inv_scale, 1)
            ]
            
            area = round(float(component_mask.sum()) * inv_scale * inv_scale, 1)
            
            annotations.append({
                'image_id': image_id,
                'category_id': int(class_id),
                'segmentation': polygons,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0
            })
    
    return image_entry, annotations


def export_to_coco_dict(processed_images, labelset):
    """Export processed images to COCO dictionary format.
    
    Used by the webapp where masks are already at original resolution.
    """
    coco = {'images': [], 'annotations': [], 'categories': []}
    
    for entry in labelset:
        coco['categories'].append({'id': int(entry['Count']), 'name': entry['Short Code']})
    
    all_annotations = []
    for image_id, (image_name, data) in enumerate(processed_images.items(), start=1):
        args = (image_id, image_name, data['mask'], data['width'], data['height'], 1.0)
        image_entry, annotations = process_single_image_to_coco(args)
        coco['images'].append(image_entry)
        all_annotations.extend(annotations)
    
    for annotation_id, annotation in enumerate(all_annotations, start=1):
        annotation['id'] = annotation_id
        coco['annotations'].append(annotation)
    
    return coco
