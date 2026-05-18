"""
Single source of truth for the marine-biologist class merge.

This file matches the inline definitions from segformer_train.ipynb Cell 3
to ensure training and evaluation use the same ground truth taxonomy.

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
- Short codes that appear in the original labelset but are NOT mentioned
  here are kept as-is (their own class). They simply have no annotations
  yet, so they will not appear in the merged CSVs.
"""

# Labels that are excluded entirely (rows removed from CSVs).
_EXCLUDED_LABELS = {
    'Unknown', 'Unkn', 'Unk', 'MA', 'BA',
    '?', 'NA', 'nan', 'None', '', 'Off',
}

# source short code -> target short code (direct mapping, no cascading)
_LABEL_MERGE_MAP = {
    # → Rock
    'TA': 'Rock', 'HS_AR': 'Rock', 'CCA': 'Rock',
    'Dead (ex)': 'Rock', 'Biofilm ex': 'Rock',
    # → SP
    'Didae': 'SP', 'Tun': 'SP',
    # → GA
    'Hal': 'GA', 'Fila (ex)': 'GA', 'Val': 'GA', 'Cau': 'GA',
    # → HC  (rare genera, each < 400 annotations)
    'Turb-HC': 'HC', 'Myc': 'HC', 'Echphy': 'HC', 'Cyph': 'HC',
    'Diplo': 'HC', 'Ser': 'HC', 'Lepta': 'HC', 'Mer': 'HC',
    'Oul': 'HC', 'Leptos': 'HC', 'Leptor': 'HC', 'Dun': 'HC',
    'Plero': 'HC', 'Bla': 'HC', 'Oxy': 'HC', 'Para': 'HC',
    'Pod': 'HC', 'Pachy': 'HC', 'Alv': 'HC', 'Sym': 'HC', 'Psa': 'HC',
    # → Frame / Xe / Other
    'Urchins ex': 'Frame', 'Tubmus': 'Xe',
    'Zoan': 'Other', 'Bivalve': 'Other', 'Bryo': 'Other',
}
