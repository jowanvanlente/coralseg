"""
Build a merged labelset and merged annotation CSVs based on the marine
biologist's class merge structure.

Edit `input/labelset/class_merge_config.py` to change the merge structure,
then run this script:

    python build_merged_labelset.py

Outputs:
    input/labelset/labelset_merged.json
    input/annotations/annotations_cleaned_merged.csv
    input/annotations/annotations_confirmed_merged.csv

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
CLEANED_IN = ROOT / "input" / "annotations" / "annotations_cleaned.csv"
CLEANED_OUT = ROOT / "input" / "annotations" / "annotations_cleaned_merged.csv"
CONFIRMED_IN = ROOT / "input" / "annotations" / "annotations_confirmed.csv"
CONFIRMED_OUT = ROOT / "input" / "annotations" / "annotations_confirmed_merged.csv"


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

    # ---- annotations_cleaned ----
    print(f"\nReading {CLEANED_IN}")
    cleaned = pd.read_csv(CLEANED_IN)
    cleaned_out = remap_label_codes(cleaned, resolution)
    cleaned_out.to_csv(CLEANED_OUT, index=False)
    print(f"Wrote {CLEANED_OUT}")
    print_summary("annotations_cleaned", cleaned["Label code"], cleaned_out["Label code"])

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


if __name__ == "__main__":
    main()
