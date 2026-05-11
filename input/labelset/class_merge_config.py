"""
Single source of truth for the marine-biologist class merge.

Edit this file to change the merge structure. Then run:
    python build_merged_labelset.py

That regenerates:
    input/labelset/labelset_merged.json
    input/annotations/annotations_cleaned_merged.csv
    input/annotations/annotations_confirmed_merged.csv

Conventions
-----------
- All keys/values are Short Codes (the same strings used in the
  `Label code` column of annotations_*.csv and `Short Code` field of
  labelset.json).
- MERGE_MAP entries cascade: if A -> B and B -> C, A effectively becomes C.
- If the final target after cascading is in DROP_CODES, all sources that
  resolve to it are dropped from the merged annotations as well.
- Short codes that appear in the original labelset but are NOT mentioned
  here are kept as-is (their own class). They simply have no annotations
  yet, so they will not appear in the merged CSVs.
"""

# Codes whose annotations are dropped entirely (rows removed from CSVs,
# class removed from the merged labelset).
DROP_CODES = {
    "Unknown",
    "MA",
}

# source short code -> target short code
MERGE_MAP = {
    # -> Rock
    "TA": "Rock",
    "HS_AR": "Rock",
    "CCA": "Rock",
    "Dead (ex)": "Rock",
    "Biofilm ex": "Rock",
    # -> SP
    "Didae": "SP",
    "Tun": "SP",
    # -> GA
    "Hal": "GA",
    "Cau": "GA",
    "Fila (ex)": "GA",
    "Val": "GA",
    # -> HC
    "Turb-HC": "HC",
    "Myc": "HC",
    "Echphy": "HC",
    "Cyph": "HC",
    "Diplo": "HC",
    "Ser": "HC",
    "Lepta": "HC",
    "Mer": "HC",
    "Oul": "HC",
    "Leptos": "HC",
    "Leptor": "HC",
    "Dun": "HC",
    "Plero": "HC",
    "Bla": "HC",
    "Oxy": "HC",
    "Para": "HC",
    "Pod": "HC",
    "Pachy": "HC",
    "Alv": "HC",
    "Sym": "HC",
    "Psa": "HC",
    # -> Xe
    "Tubmus": "Xe",
    # -> Frame
    "Urchins ex": "Frame",
    # -> Other (new class, see NEW_CLASSES below)
    "Zoan": "Other",
    "Bivalve": "Other",
    "Bryo": "Other",
    # -> MA (which is itself dropped -> cascades to drop)
    "BA": "MA",
}

# New classes that don't exist in the original labelset.
# Short Code -> {"Label ID": str, "Color Code": [R, G, B]}
# Pick Label IDs that don't collide with any existing ones in labelset.json
# (existing IDs are all <= 7251; we use 9000+ to stay safe).
NEW_CLASSES = {
    "Other": {
        "Label ID": "9001",
        "Color Code": [150, 150, 170],
    },
}
