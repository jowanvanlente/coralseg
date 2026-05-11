"""Check how Roboflow image names map to CoralNet CSV image names."""
import json
import pandas as pd
from utils import normalize_image_name

# Load CoralNet CSV names
df = pd.read_csv(r"input/annotations/annotations_confirmed_merged.csv", low_memory=False)
col = "Label code" if "Label code" in df.columns else "Label"
csv_names_raw = set(df["Name"].astype(str).unique())
csv_names_norm = {normalize_image_name(n): n for n in csv_names_raw}

# Load Roboflow COCO
with open(r"input/annotations/Roboflow/20260508_annotations.coco.json") as f:
    d = json.load(f)

# Try to match using extra.name
matched, unmatched = 0, 0
unmatched_examples = []
for im in d["images"][:]:
    extra_name = im.get("extra", {}).get("name", "")
    fn = im["file_name"]
    
    # Try matching extra.name directly
    norm_extra = normalize_image_name(extra_name)
    norm_fn = normalize_image_name(fn)
    
    found = (extra_name in csv_names_raw or 
             norm_extra in csv_names_norm or
             fn in csv_names_raw or 
             norm_fn in csv_names_norm)
    
    if found:
        matched += 1
    else:
        unmatched += 1
        if len(unmatched_examples) < 10:
            unmatched_examples.append((fn, extra_name))

print(f"Matched: {matched}, Unmatched: {unmatched}")
print("\nUnmatched examples:")
for fn, extra in unmatched_examples:
    print(f"  file_name: {fn}")
    print(f"  extra.name: {extra}")
    # Try to find close matches
    extra_lower = extra.lower().replace("-jpg", ".jpg").replace("-jpeg", ".jpeg").replace("_jpg", ".jpg")
    for cn in list(csv_names_norm.keys())[:]:
        if cn.lower().startswith(extra_lower[:20].lower()):
            print(f"    possible match: {cn}")
            break
    print()

# Show some CSV name examples for comparison
print("Sample CSV names:")
for n in sorted(csv_names_raw)[:10]:
    print(f"  {n}")
