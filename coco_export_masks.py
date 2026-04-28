"""
COCO JSON export functions — RLE mask encoding.

Exports dense segmentation masks to COCO format with RLE (Run-Length Encoding)
instead of polygon encoding. One annotation per connected component (object instance).
"""

import numpy as np
import cv2
from pycocotools import mask as mask_utils


def process_single_image_to_coco_masks(args):
    """
    Process one image's mask into COCO annotations with RLE mask encoding.
    
    Args:
        args: Tuple of (image_id, image_name, mask, orig_width, orig_height, scale_factor)
              Mask is at scaled resolution. Binary masks are rescaled to original dims
              and encoded as RLE.
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
            
            # Resize binary mask to original image dimensions
            full_res_mask = cv2.resize(
                component_mask, (orig_width, orig_height),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Encode as RLE (pycocotools expects Fortran-order uint8 array)
            rle = mask_utils.encode(np.asfortranarray(full_res_mask))
            # counts is bytes — decode to str for JSON serialization
            rle['counts'] = rle['counts'].decode('utf-8')
            
            # Compute bbox and area from the full-res mask
            rows, cols = np.where(full_res_mask)
            if len(rows) == 0:
                continue
            
            bbox = [
                int(cols.min()),
                int(rows.min()),
                int(cols.max() - cols.min() + 1),
                int(rows.max() - rows.min() + 1)
            ]
            
            area = int(full_res_mask.sum())
            
            annotations.append({
                'image_id': image_id,
                'category_id': int(class_id),
                'segmentation': rle,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0
            })
    
    return image_entry, annotations


def export_to_coco_dict_masks(processed_images, labelset):
    """Export processed images to COCO dictionary format with RLE mask encoding.
    
    Used by the webapp where masks are already at original resolution.
    """
    coco = {'images': [], 'annotations': [], 'categories': []}
    
    for entry in labelset:
        coco['categories'].append({'id': int(entry['Count']), 'name': entry['Short Code']})
    
    all_annotations = []
    for image_id, (image_name, data) in enumerate(processed_images.items(), start=1):
        args = (image_id, image_name, data['mask'], data['width'], data['height'], 1.0)
        image_entry, annotations = process_single_image_to_coco_masks(args)
        coco['images'].append(image_entry)
        all_annotations.extend(annotations)
    
    for annotation_id, annotation in enumerate(all_annotations, start=1):
        annotation['id'] = annotation_id
        coco['annotations'].append(annotation)
    
    return coco
