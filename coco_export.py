"""
COCO JSON export functions.
"""

import numpy as np
import cv2
import json
from pycocotools import mask as mask_utils


def process_single_image_to_coco(args):
    """Process one image's mask into COCO annotations."""
    image_id, image_name, mask, width, height = args
    H, W = mask.shape
    
    image_entry = {'id': image_id, 'file_name': image_name, 'width': width, 'height': height}
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
            
            rle = mask_utils.encode(np.asfortranarray(component_mask))
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')
            
            rows, cols = np.where(component_mask)
            if len(rows) == 0:
                continue
            
            bbox = [int(cols.min()), int(rows.min()), int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)]
            annotations.append({
                'image_id': image_id, 'category_id': int(class_id),
                'segmentation': {'size': [H, W], 'counts': rle['counts']},
                'bbox': bbox, 'area': int(component_mask.sum()), 'iscrowd': 0
            })
    return image_entry, annotations


def export_to_coco_dict(processed_images, labelset):
    """Export processed images to COCO dictionary format."""
    coco = {'images': [], 'annotations': [], 'categories': []}
    
    for entry in labelset:
        coco['categories'].append({'id': int(entry['Count']), 'name': entry['Short Code']})
    
    all_annotations = []
    for image_id, (image_name, data) in enumerate(processed_images.items(), start=1):
        args = (image_id, image_name, data['mask'], data['width'], data['height'])
        image_entry, annotations = process_single_image_to_coco(args)
        coco['images'].append(image_entry)
        all_annotations.extend(annotations)
    
    for annotation_id, annotation in enumerate(all_annotations, start=1):
        annotation['id'] = annotation_id
        coco['annotations'].append(annotation)
    
    return coco
