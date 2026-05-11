"""Quick inspection of the COCO JSON."""
import json
from collections import Counter

with open(r"input/annotations/Roboflow/20260508_annotations.coco.json") as f:
    d = json.load(f)

cats = {c["id"]: c["name"] for c in d["categories"]}
print("CATEGORIES:")
for k, v in cats.items():
    print(f"  {k}: {v}")

ann = d["annotations"][0]
seg = ann.get("segmentation", {})
print(f"\nSeg type: {type(seg)}")
if isinstance(seg, dict):
    print(f"Seg keys: {list(seg.keys())}")
elif isinstance(seg, list):
    print(f"Seg is list, len={len(seg)}, first polygon len={len(seg[0]) if seg else 0}")

# Check a few annotations for seg type
types = Counter()
for a in d["annotations"][:100]:
    s = a.get("segmentation")
    types[type(s).__name__] += 1
print(f"\nSeg types in first 100: {dict(types)}")

cc = Counter(a["category_id"] for a in d["annotations"])
print("\nPER-CAT COUNTS:")
for k, v in sorted(cc.items(), key=lambda x: -x[1]):
    print(f"  {cats[k]:25s}: {v}")

img_ids = set(a["image_id"] for a in d["annotations"])
print(f"\nImages with anns: {len(img_ids)} / {len(d['images'])}")

print("\nSAMPLE IMAGE NAMES:")
for im in d["images"][:5]:
    extra_name = im.get("extra", {}).get("name", "")
    print(f"  file_name={im['file_name']}")
    print(f"    extra.name={extra_name}")

# Check area distribution
areas = [a["area"] for a in d["annotations"]]
print(f"\nArea stats: min={min(areas)}, max={max(areas)}, mean={sum(areas)/len(areas):.0f}")

# Show one RLE annotation if exists
for a in d["annotations"][:5]:
    s = a.get("segmentation")
    if isinstance(s, dict):
        print(f"\nRLE example: counts_len={len(s.get('counts',''))}, size={s.get('size')}")
        break
