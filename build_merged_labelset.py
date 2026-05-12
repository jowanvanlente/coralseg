"""
Build a merged labelset and merged annotation CSVs based on the marine
biologist's class merge structure.

Edit `input/labelset/class_merge_config.py` to change the merge structure,
then run this script:

    python build_merged_labelset.py

Outputs:
    input/labelset/labelset_merged.json
    input/annotations/annotations_confirmed_merged.csv
    input/annotations/Roboflow/<name>_merged.coco.json

Prints a before/after summary so you can sanity check the result.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Allow importing the config from input/labelset/
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "input" / "labelset"))
import class_merge_config as cfg  # noqa: E402

LABELSET_IN = ROOT / "input" / "labelset" / "labelset.json"
LABELSET_OUT = ROOT / "input" / "labelset" / "labelset_merged.json"
CONFIRMED_IN = ROOT / "input" / "annotations" / "annotations_confirmed.csv"
CONFIRMED_OUT = ROOT / "input" / "annotations" / "annotations_confirmed_merged.csv"

COCO_DIR = ROOT / "input" / "annotations" / "Roboflow"


def resolve_target(code: str, merge_map: dict, drop_codes: set,
                   _seen: set | None = None) -> str | None:
    """Follow the merge chain. Returns final target short code, or None if dropped."""
    if code in drop_codes:
        return None
    if code not in merge_map:
        return code
    _seen = _seen or set()
    if code in _seen:
        raise ValueError(f"Cycle detected in MERGE_MAP at '{code}'")
    _seen.add(code)
    return resolve_target(merge_map[code], merge_map, drop_codes, _seen)


def build_resolution(all_codes, merge_map, drop_codes):
    """code -> final target (or None if dropped)."""
    return {c: resolve_target(c, merge_map, drop_codes) for c in all_codes}


def build_merged_labelset(original: list[dict], resolution: dict,
                          new_classes: dict) -> list[dict]:
    """Keep entries that resolve to themselves, add new classes."""
    by_code = {e["Short Code"]: e for e in original}

    # Codes that survive (resolve to themselves).
    keep_codes = sorted({tgt for tgt in resolution.values() if tgt is not None})

    merged = []
    for code in keep_codes:
        if code in by_code:
            entry = dict(by_code[code])
        elif code in new_classes:
            nc = new_classes[code]
            entry = {
                "Label ID": nc["Label ID"],
                "Short Code": code,
                "Color Code": list(nc["Color Code"]),
                "Color Name": "",
            }
        else:
            raise KeyError(
                f"Target '{code}' is not in the original labelset and not "
                f"declared in NEW_CLASSES."
            )
        merged.append(entry)

    # Renumber Count and Color Name for consistency.
    for i, e in enumerate(merged, start=1):
        e["Count"] = i
        e["Color Name"] = f"Color {i}"
    return merged


def remap_label_codes(df: pd.DataFrame, resolution: dict) -> pd.DataFrame:
    """Drop rows whose Label code resolves to None; remap the rest."""
    codes = df["Label code"].astype(str)
    targets = codes.map(resolution)

    unknown_mask = targets.isna() & ~codes.isin([k for k, v in resolution.items() if v is None])
    if unknown_mask.any():
        unseen = sorted(codes[unknown_mask].unique())
        print(f"  WARNING: {unknown_mask.sum()} rows have Label codes not in "
              f"the original labelset and not in MERGE_MAP/DROP_CODES: {unseen}")

    keep_mask = targets.notna()
    out = df.loc[keep_mask].copy()
    out["Label code"] = targets.loc[keep_mask].values
    return out


def update_label_id_column(df: pd.DataFrame, code_to_id: dict) -> pd.DataFrame:
    """For confirmed.csv: also update Label ID to match the new code."""
    if "Label ID" in df.columns:
        df = df.copy()
        df["Label ID"] = df["Label code"].map(code_to_id)
    return df


def print_summary(name: str, before: pd.Series, after: pd.Series) -> None:
    print(f"\n=== {name} ===")
    print(f"  rows before: {len(before):>8}")
    print(f"  rows after : {len(after):>8}")
    print(f"  classes before: {before.nunique()}")
    print(f"  classes after : {after.nunique()}")
    print("  per-class counts (after):")
    for code, n in after.value_counts().items():
        print(f"    {code:<14} {n}")


def remap_coco(coco_path: Path, output_path: Path, resolution: dict,
               roboflow_to_short: dict) -> None:
    """Remap a Roboflow COCO JSON to the merged class structure.

    - Categories are replaced with one entry per surviving merged short code.
    - Annotations whose class is dropped are removed.
    - Annotations whose class is merged get their category_id updated.
    """
    raw = json.loads(coco_path.read_text(encoding="utf-8"))

    # old category id → merged short code (None = drop)
    old_to_merged: dict[int, str | None] = {}
    for cat in raw["categories"]:
        short = roboflow_to_short.get(cat["name"])
        if short is None:
            continue  # root / unmapped (e.g. "REEFo")
        old_to_merged[cat["id"]] = resolution.get(short)

    # Build new category list (sorted unique merged targets)
    unique_targets = sorted({t for t in old_to_merged.values() if t is not None})
    new_cat_by_code = {
        code: {"id": i, "name": code, "supercategory": "none"}
        for i, code in enumerate(unique_targets, start=1)
    }

    # Remap annotations
    new_annotations = []
    dropped = 0
    for ann in raw["annotations"]:
        merged = old_to_merged.get(ann["category_id"])
        if merged is None:
            dropped += 1
            continue
        new_ann = dict(ann)
        new_ann["category_id"] = new_cat_by_code[merged]["id"]
        new_annotations.append(new_ann)

    out = {
        "info": raw.get("info", {}),
        "licenses": raw.get("licenses", []),
        "categories": list(new_cat_by_code.values()),
        "images": raw["images"],
        "annotations": new_annotations,
    }
    output_path.write_text(json.dumps(out), encoding="utf-8")

    print(f"\nWrote {output_path}")
    print(f"  Categories: {len(raw['categories'])} -> {len(new_cat_by_code)}")
    print(f"  Annotations: {len(raw['annotations'])} -> {len(new_annotations)} "
          f"(dropped {dropped})")
    # Show which old categories merged
    merge_report: dict[str, list[str]] = {}
    for cat in raw["categories"]:
        short = roboflow_to_short.get(cat["name"])
        if short is None:
            continue
        target = resolution.get(short)
        if target and target != short:
            merge_report.setdefault(target, []).append(f"{cat['name']}({short})")
    if merge_report:
        print("  Merged categories:")
        for tgt, srcs in sorted(merge_report.items()):
            print(f"    {tgt} <- {', '.join(srcs)}")


def main() -> None:
    print(f"Reading labelset: {LABELSET_IN}")
    original = json.loads(LABELSET_IN.read_text(encoding="utf-8"))
    original_codes = {e["Short Code"] for e in original}

    # Sanity: every source/target in the config must be known.
    all_referenced = (set(cfg.MERGE_MAP) | set(cfg.MERGE_MAP.values())
                      | set(cfg.DROP_CODES))
    missing = all_referenced - original_codes - set(cfg.NEW_CLASSES)
    if missing:
        raise ValueError(
            f"These short codes appear in the merge config but are unknown: "
            f"{sorted(missing)}"
        )

    # Resolution covers every original code + every new class.
    resolution = build_resolution(
        original_codes | set(cfg.NEW_CLASSES),
        cfg.MERGE_MAP,
        cfg.DROP_CODES,
    )

    merged = build_merged_labelset(original, resolution, cfg.NEW_CLASSES)
    code_to_id = {e["Short Code"]: e["Label ID"] for e in merged}

    LABELSET_OUT.write_text(json.dumps(merged, indent=4), encoding="utf-8")
    print(f"Wrote {LABELSET_OUT} ({len(merged)} classes)")

    # ---- annotations_confirmed ----
    print(f"\nReading {CONFIRMED_IN}")
    confirmed = pd.read_csv(CONFIRMED_IN, low_memory=False)
    confirmed_out = remap_label_codes(confirmed, resolution)
    confirmed_out = update_label_id_column(confirmed_out, code_to_id)
    confirmed_out.to_csv(CONFIRMED_OUT, index=False)
    print(f"Wrote {CONFIRMED_OUT}")
    print_summary("annotations_confirmed", confirmed["Label code"], confirmed_out["Label code"])

    # Show the merge actions taken (sources that changed or were dropped).
    print("\n=== Merge / drop actions ===")
    changed = {c: t for c, t in resolution.items() if t != c}
    for src in sorted(changed):
        tgt = changed[src]
        arrow = "DROP" if tgt is None else f"-> {tgt}"
        print(f"  {src:<14} {arrow}")

    # ---- Roboflow COCO JSON(s) ----
    from utils import _ROBOFLOW_NAME_TO_SHORT_CODE

    if COCO_DIR.is_dir():
        coco_files = sorted(COCO_DIR.glob("*.coco.json"))
        # Skip already-merged files
        coco_files = [f for f in coco_files if "_merged" not in f.stem]
        if not coco_files:
            print(f"\nNo *.coco.json files found in {COCO_DIR} — skipping COCO merge.")
        for coco_in in coco_files:
            coco_out = coco_in.with_name(
                coco_in.name.replace(".coco.json", "_merged.coco.json")
            )
            remap_coco(coco_in, coco_out, resolution, _ROBOFLOW_NAME_TO_SHORT_CODE)
    else:
        print(f"\n{COCO_DIR} does not exist — skipping COCO merge.")


if __name__ == "__main__":
    main()
