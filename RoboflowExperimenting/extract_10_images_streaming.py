import json
import ijson

# Read the large JSON file using ijson for streaming
def extract_10_images():
    selected_image_ids = []
    first_10_images = []
    selected_annotations = []
    categories = []
    
    print("Processing large JSON file...")
    
    # First pass: get first 10 images and their IDs
    with open('C:/Coding/coralseg/output/coco_run_graph200.json', 'rb') as f:
        parser = ijson.parse(f)
        current_image = {}
        in_images = False
        image_count = 0
        
        for prefix, event, value in parser:
            if prefix == 'images' and event == 'start_array':
                in_images = True
            elif prefix == 'images' and event == 'end_array':
                in_images = False
                break
            elif in_images and prefix.startswith('images.item'):
                if event == 'map_key' and value in ['id', 'file_name', 'width', 'height']:
                    current_key = value
                elif prefix.endswith('.id') and event == 'number':
                    current_image['id'] = int(value)
                elif prefix.endswith('.file_name') and event == 'string':
                    current_image['file_name'] = value
                elif prefix.endswith('.width') and event == 'number':
                    current_image['width'] = int(value)
                elif prefix.endswith('.height') and event == 'number':
                    current_image['height'] = int(value)
                elif event == 'end_map' and current_image:
                    first_10_images.append(current_image.copy())
                    selected_image_ids.append(current_image['id'])
                    image_count += 1
                    current_image = {}
                    if image_count >= 10:
                        break
    
    print(f"Selected {len(first_10_images)} images with IDs: {selected_image_ids}")
    
    # Second pass: get annotations for selected images
    with open('C:/Coding/coralseg/output/coco_run_graph200.json', 'rb') as f:
        parser = ijson.parse(f)
        current_annotation = {}
        in_annotations = False
        annotation_count = 0
        
        for prefix, event, value in parser:
            if prefix == 'annotations' and event == 'start_array':
                in_annotations = True
            elif prefix == 'annotations' and event == 'end_array':
                in_annotations = False
                break
            elif in_annotations and prefix.startswith('annotations.item'):
                if event == 'map_key' and value in ['image_id', 'category_id', 'segmentation', 'area', 'iscrowd', 'id']:
                    current_key = value
                elif prefix.endswith('.image_id') and event == 'number':
                    current_annotation['image_id'] = int(value)
                elif prefix.endswith('.category_id') and event == 'number':
                    current_annotation['category_id'] = int(value)
                elif prefix.endswith('.segmentation') and event == 'start_array':
                    # We'll collect the segmentation data
                    pass
                elif prefix.endswith('.area') and event == 'number':
                    current_annotation['area'] = float(value)
                elif prefix.endswith('.iscrowd') and event == 'number':
                    current_annotation['iscrowd'] = int(value)
                elif prefix.endswith('.id') and event == 'number':
                    current_annotation['id'] = int(value)
                elif event == 'end_map' and current_annotation:
                    if current_annotation.get('image_id') in selected_image_ids:
                        selected_annotations.append(current_annotation.copy())
                        annotation_count += 1
                    current_annotation = {}
    
    print(f"Found {annotation_count} annotations for selected images")
    
    # Third pass: get all categories
    with open('C:/Coding/coralseg/output/coco_run_graph200.json', 'rb') as f:
        parser = ijson.parse(f)
        current_category = {}
        in_categories = False
        
        for prefix, event, value in parser:
            if prefix == 'categories' and event == 'start_array':
                in_categories = True
            elif prefix == 'categories' and event == 'end_array':
                in_categories = False
                break
            elif in_categories and prefix.startswith('categories.item'):
                if event == 'map_key' and value in ['id', 'name']:
                    current_key = value
                elif prefix.endswith('.id') and event == 'number':
                    current_category['id'] = int(value)
                elif prefix.endswith('.name') and event == 'string':
                    current_category['name'] = value
                elif event == 'end_map' and current_category:
                    categories.append(current_category.copy())
                    current_category = {}
    
    print(f"Collected {len(categories)} categories")
    
    # Create the new JSON structure
    new_coco_data = {
        "images": first_10_images,
        "annotations": selected_annotations,
        "categories": categories
    }
    
    # Save to new file
    with open('C:/Coding/coralseg/output/coco_run_graph200_10images.json', 'w') as f:
        json.dump(new_coco_data, f, indent=4)
    
    print("Saved to: C:/Coding/coralseg/output/coco_run_graph200_10images.json")
    
    # Print summary
    print("\nSelected images:")
    for img in first_10_images:
        annotation_count = sum(1 for ann in selected_annotations if ann['image_id'] == img['id'])
        print(f"  Image ID {img['id']}: {img['file_name']} ({annotation_count} annotations)")

if __name__ == "__main__":
    extract_10_images()
