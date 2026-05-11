"""Test de-mangling Roboflow filenames to recover original CoralNet names."""
import json, os, re
import pandas as pd
from utils import normalize_image_name

def demangle_roboflow_name(extra_name):
    """Recover original filename from Roboflow's extra.name.
    
    Roboflow replaces the original extension '.' with '-', then appends a new ext.
    e.g. 'image.JPG' -> 'image-JPG.JPG', 'image.jpeg' -> 'image-jpeg.jpeg'
    """
    base, ext = os.path.splitext(extra_name)
    ext_noDot = ext.lstrip(".")
    
    # Common image extensions
    img_exts = {"jpg", "jpeg", "png", "tif", "tiff", "bmp", "JPG", "JPEG", "PNG"}
    
    # Check if base ends with -<ext_variant>
    pattern = re.compile(r'^(.*)-(' + '|'.join(img_exts) + r')$', re.IGNORECASE)
    m = pattern.match(base)
    if m:
        original_base = m.group(1)
        original_ext = m.group(2)
        return original_base + "." + original_ext
    
    return extra_name

# Load CoralNet CSV names
df = pd.read_csv(r"input/annotations/annotations_confirmed_merged.csv", low_memory=False)
csv_names_raw = set(df["Name"].astype(str).unique())
csv_names_norm = {normalize_image_name(n): n for n in csv_names_raw}

# Load Roboflow COCO
with open(r"input/annotations/Roboflow/20260508_annotations.coco.json") as f:
    d = json.load(f)

matched, unmatched = 0, 0
unmatched_examples = []
for im in d["images"]:
    extra_name = im.get("extra", {}).get("name", im["file_name"])
    original = demangle_roboflow_name(extra_name)
    norm = normalize_image_name(original)
    
    found = (original in csv_names_raw or norm in csv_names_norm)
    if found:
        matched += 1
    else:
        unmatched += 1
        if len(unmatched_examples) < 10:
            unmatched_examples.append((extra_name, original, norm))

print(f"Matched: {matched}, Unmatched: {unmatched}")
if unmatched_examples:
    print("\nUnmatched examples:")
    for extra, orig, norm in unmatched_examples:
        print(f"  extra={extra} -> demangled={orig} -> norm={norm}")
