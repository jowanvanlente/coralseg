import json
from pathlib import Path
from typing import Any, Dict, List


# --- CONFIGURATION ---
# Set your input JSON file path here
INPUT_JSON_PATH = Path(r"C:\Coding\coralseg\output\coco_run_graphAll.json")

# Set output directory (will be created if it doesn't exist)
OUTPUT_DIR = Path(r"C:\Coding\coralseg\output\coco_run_graphAll_split")

# Split by number of images per file (fast and predictable)
IMAGES_PER_CHUNK = 200

# Set output filename prefix (None = use input filename stem)
OUTPUT_PREFIX = None  # Will default to "coco_run_graphAll"
# -------------------


def split_coco(input_json: Path, output_dir: Path, images_per_chunk: int, prefix: str) -> None:
    if images_per_chunk <= 0:
        raise ValueError("IMAGES_PER_CHUNK must be greater than 0")

    with input_json.open("r", encoding="utf-8-sig") as f:
        coco = json.load(f)

    if not isinstance(coco, dict):
        raise ValueError("COCO JSON root must be an object")

    images = coco.get("images")
    annotations = coco.get("annotations")
    categories = coco.get("categories")

    if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        raise ValueError("COCO JSON must contain list fields: images, annotations, categories")

    annotations_by_image: Dict[Any, List[Dict[str, Any]]] = {}
    for ann in annotations:
        image_id = ann.get("image_id")
        annotations_by_image.setdefault(image_id, []).append(ann)

    template = {k: v for k, v in coco.items() if k not in {"images", "annotations", "categories"}}

    output_dir.mkdir(parents=True, exist_ok=True)

    total_images = len(images)
    total_annotations = len(annotations)
    chunks_written = 0

    for start in range(0, total_images, images_per_chunk):
        chunk_images = images[start : start + images_per_chunk]

        chunk_annotations: List[Dict[str, Any]] = []
        for image in chunk_images:
            chunk_annotations.extend(annotations_by_image.get(image.get("id"), []))

        chunk_obj = dict(template)
        chunk_obj["images"] = chunk_images
        chunk_obj["annotations"] = chunk_annotations
        chunk_obj["categories"] = categories

        part_number = chunks_written + 1
        out_path = output_dir / f"{prefix}_part_{part_number:03d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(chunk_obj, f, separators=(",", ":"), ensure_ascii=False)

        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(
            f"Wrote {out_path.name}: images={len(chunk_images)}, "
            f"annotations={len(chunk_annotations)}, size={size_mb:.2f} MB"
        )
        chunks_written += 1

    print(f"Input file: {input_json}")
    print(f"Output dir: {output_dir}")
    print(f"Images per chunk: {images_per_chunk}")
    print(f"Chunks written: {chunks_written}")
    print(f"Total images processed: {total_images}")
    print(f"Total annotations processed: {total_annotations}")


def main() -> None:
    input_json = INPUT_JSON_PATH
    output_dir = OUTPUT_DIR
    images_per_chunk = IMAGES_PER_CHUNK
    prefix = OUTPUT_PREFIX or input_json.stem

    if not input_json.exists():
        raise FileNotFoundError(f"Input file not found: {input_json}")

    split_coco(
        input_json=input_json,
        output_dir=output_dir,
        images_per_chunk=images_per_chunk,
        prefix=prefix,
    )


if __name__ == "__main__":
    main()
