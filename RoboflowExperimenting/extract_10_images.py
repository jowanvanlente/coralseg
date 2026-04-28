import json

# Load the original COCO JSON file
with open('C:/Coding/coralseg/output/coco_run_graph200.json', 'r') as f:
    coco_data = json.load(f)

# Get the first 10 images
first_10_images = coco_data['images'][:10]
selected_image_ids = [img['id'] for img in first_10_images]

# Filter annotations for only the selected image IDs
selected_annotations = []
for annotation in coco_data['annotations']:
    if annotation['image_id'] in selected_image_ids:
        selected_annotations.append(annotation)

# Get all categories (we keep all categories)
categories = coco_data['categories']

# Create the new COCO format JSON with only 10 images
new_coco_data = {
    "images": first_10_images,
    "annotations": selected_annotations,
    "categories": categories
}

# Save to new file
with open('C:/Coding/coralseg/output/coco_run_graph200_10images.json', 'w') as f:
    json.dump(new_coco_data, f, indent=4)

print(f"Extracted {len(first_10_images)} images")
print(f"Found {len(selected_annotations)} annotations for these images")
print(f"Kept {len(categories)} categories")
print("Saved to: C:/Coding/coralseg/output/coco_run_graph200_10images.json")

# Print summary of selected images
print("\nSelected images:")
for img in first_10_images:
    annotation_count = sum(1 for ann in selected_annotations if ann['image_id'] == img['id'])
    print(f"  Image ID {img['id']}: {img['file_name']} ({annotation_count} annotations)")
