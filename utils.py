"""
Utility functions for loading data and image operations.
"""

import numpy as np
import pandas as pd
import cv2
import json
import os
import re


def normalize_image_name(name):
    """
    Normalize image filename by removing duplicate extensions and copy suffixes.
    Examples:
        'image.jpeg.jpeg' -> 'image.jpeg'
        'image.JPG.JPG' -> 'image.JPG'
        'image (1).jpg' -> 'image.jpg'
        'image_copy.jpg' -> 'image.jpg'
    """
    # Remove common copy suffixes like (1), (2), _copy, -copy
    name = re.sub(r'\s*\(\d+\)\s*(?=\.)', '', name)
    name = re.sub(r'[-_]?copy\d*(?=\.)', '', name, flags=re.IGNORECASE)
    
    # Handle duplicate extensions (e.g., .jpeg.jpeg, .JPG.JPG)
    base, ext = os.path.splitext(name)
    if ext:
        # Check if base also ends with same extension (case-insensitive)
        base_lower = base.lower()
        ext_lower = ext.lower()
        if base_lower.endswith(ext_lower):
            # Remove the duplicate extension from base
            name = base[:-len(ext)] + ext
    
    return name


LABEL_META_BY_SHORT_CODE = {
    'Aca': {'Name': 'Acanthastrea', 'Functional Group': 'Hard coral'},
    'Acr': {'Name': 'Acropora', 'Functional Group': 'Hard coral'},
    'Alv': {'Name': 'Alveopora', 'Functional Group': 'Hard coral'},
    'Ano': {'Name': 'Anomastraea', 'Functional Group': 'Hard coral'},
    'Astrea': {'Name': 'Astrea', 'Functional Group': 'Hard coral'},
    'Astreo': {'Name': 'Astreopora', 'Functional Group': 'Hard coral'},
    'Bla': {'Name': 'Blastomussa', 'Functional Group': 'Hard coral'},
    'Caula': {'Name': 'Caulastrea', 'Functional Group': 'Hard coral'},
    'Cos': {'Name': 'Coscinaraea', 'Functional Group': 'Hard coral'},
    'Cyph': {'Name': 'Cyphastrea', 'Functional Group': 'Hard coral'},
    'Diplo': {'Name': 'Diploastrea', 'Functional Group': 'Hard coral'},
    'Dipsa': {'Name': 'Dipsastraea', 'Functional Group': 'Hard coral'},
    'Dun': {'Name': 'Duncanopsammia', 'Functional Group': 'Hard coral'},
    'Echphy': {'Name': 'Echinophyllia', 'Functional Group': 'Hard coral'},
    'Echpo': {'Name': 'Echinopora', 'Functional Group': 'Hard coral'},
    'Favit': {'Name': 'Favites', 'Functional Group': 'Hard coral'},
    'Fungii': {'Name': 'Fungiidae', 'Functional Group': 'Hard coral'},
    'Gal': {'Name': 'Galaxea', 'Functional Group': 'Hard coral'},
    'Gar': {'Name': 'Gardineroseris', 'Functional Group': 'Hard coral'},
    'Gonia': {'Name': 'Goniastrea', 'Functional Group': 'Hard coral'},
    'Gonio': {'Name': 'Goniopora', 'Functional Group': 'Hard coral'},
    'HC': {'Name': 'Hard coral', 'Functional Group': 'Hard coral'},
    'Hydno': {'Name': 'Hydnophora', 'Functional Group': 'Hard coral'},
    'Iso': {'Name': 'Isopora', 'Functional Group': 'Hard coral'},
    'Lepta': {'Name': 'Leptastrea', 'Functional Group': 'Hard coral'},
    'Leptor': {'Name': 'Leptoria', 'Functional Group': 'Hard coral'},
    'Leptos': {'Name': 'Leptoseris', 'Functional Group': 'Hard coral'},
    'Lobophyl': {'Name': 'Lobophyllia', 'Functional Group': 'Hard coral'},
    'Mer': {'Name': 'Merulina', 'Functional Group': 'Hard coral'},
    'Micro': {'Name': 'Micromussa', 'Functional Group': 'Hard coral'},
    'Monti': {'Name': 'Montipora', 'Functional Group': 'Hard coral'},
    'Myc': {'Name': 'Mycedium', 'Functional Group': 'Hard coral'},
    'Oul': {'Name': 'Oulophyllia', 'Functional Group': 'Hard coral'},
    'Oxy': {'Name': 'Oxypora', 'Functional Group': 'Hard coral'},
    'Pachy': {'Name': 'Pachyseris', 'Functional Group': 'Hard coral'},
    'Para': {'Name': 'Paramontastraea', 'Functional Group': 'Hard coral'},
    'Pav': {'Name': 'Pavona', 'Functional Group': 'Hard coral'},
    'Pec': {'Name': 'Pectinia', 'Functional Group': 'Hard coral'},
    'Phy': {'Name': 'Physogyra', 'Functional Group': 'Hard coral'},
    'Platy': {'Name': 'Platygyra', 'Functional Group': 'Hard coral'},
    'Plero': {'Name': 'Plerogyra', 'Functional Group': 'Hard coral'},
    'Plesia': {'Name': 'Plesiastrea', 'Functional Group': 'Hard coral'},
    'Poc': {'Name': 'Pocillopora', 'Functional Group': 'Hard coral'},
    'Pod': {'Name': 'Podabacia', 'Functional Group': 'Hard coral'},
    'PorB': {'Name': 'Porites (branching)', 'Functional Group': 'Hard coral'},
    'PorM': {'Name': 'Porites (massive)', 'Functional Group': 'Hard coral'},
    'Psa': {'Name': 'Psammocora', 'Functional Group': 'Hard coral'},
    'Ser': {'Name': 'Seriatopora', 'Functional Group': 'Hard coral'},
    'Styl': {'Name': 'Stylophora', 'Functional Group': 'Hard coral'},
    'Sym': {'Name': 'Symphyllia', 'Functional Group': 'Hard coral'},
    'Turb-HC': {'Name': 'Turbinaria (coral)', 'Functional Group': 'Hard coral'},
    'Agl': {'Name': 'Aglaophenia spp.', 'Functional Group': 'Other Invertebrates'},
    'Anemone': {'Name': 'Anemone', 'Functional Group': 'Other Invertebrates'},
    'Bivalve': {'Name': 'Bivalve', 'Functional Group': 'Other Invertebrates'},
    'Bryo': {'Name': 'Bryozoan', 'Functional Group': 'Other Invertebrates'},
    'Cormor': {'Name': 'Corallimorpharia', 'Functional Group': 'Other Invertebrates'},
    'Didae': {'Name': 'Didemnidae', 'Functional Group': 'Other Invertebrates'},
    'Hydro': {'Name': 'Hydroid', 'Functional Group': 'Other Invertebrates'},
    'Lobphyt': {'Name': 'Lobophytum', 'Functional Group': 'Other Invertebrates'},
    'Mil': {'Name': 'Millepora', 'Functional Group': 'Other Invertebrates'},
    'Sarcopydae': {'Name': 'Sarcophyton', 'Functional Group': 'Other Invertebrates'},
    'SC': {'Name': 'Soft Coral', 'Functional Group': 'Other Invertebrates'},
    'SP': {'Name': 'Sponge', 'Functional Group': 'Other Invertebrates'},
    'Tubmus': {'Name': 'Tubipora musica', 'Functional Group': 'Other Invertebrates'},
    'Tun': {'Name': 'Tunicate', 'Functional Group': 'Other Invertebrates'},
    'Urchins ex': {'Name': 'Echinoderms: sea urchin', 'Functional Group': 'Other Invertebrates'},
    'Xe': {'Name': 'XENIIDAE', 'Functional Group': 'Other Invertebrates'},
    'Zoan': {'Name': 'Zoanthid', 'Functional Group': 'Other Invertebrates'},
    'Biofilm ex': {'Name': 'Biofilm', 'Functional Group': 'Soft Substrate'},
    'Rhy': {'Name': 'Rhytisma', 'Functional Group': 'Soft Substrate'},
    'S': {'Name': 'Sand', 'Functional Group': 'Soft Substrate'},
    'Dead (ex)': {'Name': 'Dead coral', 'Functional Group': 'Hard Substrate'},
    'HS_AR': {'Name': 'Hard Substrate', 'Functional Group': 'Hard Substrate'},
    'Rock': {'Name': 'Rock', 'Functional Group': 'Hard Substrate'},
    'Cya': {'Name': 'Cyanobacteria', 'Functional Group': 'Other'},
    'Frame': {'Name': 'Framer', 'Functional Group': 'Other'},
    'R': {'Name': 'Rubble', 'Functional Group': 'Other'},
    'Unknown': {'Name': 'Unknown', 'Functional Group': 'Other'},
    'BA': {'Name': 'Ochrophyta', 'Functional Group': 'Algae'},
    'Cau': {'Name': 'Caulerpa', 'Functional Group': 'Algae'},
    'CCA': {'Name': 'CCA (crustose coralline algae)', 'Functional Group': 'Algae'},
    'Dic': {'Name': 'Dictyota', 'Functional Group': 'Algae'},
    'Fila (ex)': {'Name': 'Algae (filamentous)', 'Functional Group': 'Algae'},
    'GA': {'Name': 'Chlorophyta', 'Functional Group': 'Algae'},
    'Hal': {'Name': 'Halimeda', 'Functional Group': 'Algae'},
    'Lobvar': {'Name': 'Lobophora variegata', 'Functional Group': 'Algae'},
    'MA': {'Name': 'Macroalgae', 'Functional Group': 'Algae'},
    'Pad': {'Name': 'Padina', 'Functional Group': 'Algae'},
    'RA': {'Name': 'Rhodophyta', 'Functional Group': 'Algae'},
    'Sar': {'Name': 'Sargassum', 'Functional Group': 'Algae'},
    'TA': {'Name': 'Turf algae', 'Functional Group': 'Algae'},
    'Turb-BA': {'Name': 'Turbinaria (algae)', 'Functional Group': 'Algae'},
    'Val': {'Name': 'Valonia spp.', 'Functional Group': 'Algae'},
    'SG': {'Name': 'Seagrass', 'Functional Group': 'Seagrass'},
}


def load_labelset_from_json(labelset_data):
    """Load labelset from JSON data (already parsed)."""
    for entry in labelset_data:
        short_code = entry.get('Short Code')
        meta = LABEL_META_BY_SHORT_CODE.get(short_code)
        if meta:
            entry['Name'] = meta['Name']
            entry['Functional Group'] = meta['Functional Group']
        else:
            entry.setdefault('Name', short_code)
            entry.setdefault('Functional Group', 'Unknown')
    return labelset_data


def load_annotations_from_df(df):
    """
    Load sparse point annotations from DataFrame.
    
    Handles CSVs with extra columns - only uses Name, Row, Column, and Label.
    Label column can be named 'Label', 'Label code', or 'label'.
    """
    df = df.copy()
    
    # Find the label column (could be 'Label', 'Label code', etc.)
    label_col = None
    for col in ['Label', 'Label code', 'label', 'label code']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f"CSV must have a label column (Label, Label code). Found: {list(df.columns)}")
    
    # Rename to standard 'Label' if needed
    if label_col != 'Label':
        df = df.rename(columns={label_col: 'Label'})
    
    # Check for required columns
    required = ['Name', 'Row', 'Column', 'Label']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")
    
    # Keep only required columns
    df = df[required].copy()
    
    return {name: group for name, group in df.groupby('Name')}


def load_labelset(path):
    """Load labelset JSON file."""
    with open(path, 'r') as f:
        labelset = json.load(f)

    for entry in labelset:
        short_code = entry.get('Short Code')
        meta = LABEL_META_BY_SHORT_CODE.get(short_code)
        if meta:
            entry['Name'] = meta['Name']
            entry['Functional Group'] = meta['Functional Group']
        else:
            entry.setdefault('Name', short_code)
            entry.setdefault('Functional Group', 'Unknown')

    return labelset


def load_annotations(path):
    """
    Load sparse point annotations CSV.
    
    Handles CSVs with extra columns - only uses Name, Row, Column, and Label.
    Label column can be named 'Label', 'Label code', or 'label'.
    """
    df = pd.read_csv(path, low_memory=False)
    
    label_col = None
    for col in ['Label', 'Label code', 'label', 'label code']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f"CSV must have a label column (Label, Label code). Found columns: {list(df.columns)}")
    
    if label_col != 'Label':
        df = df.rename(columns={label_col: 'Label'})
    
    required = ['Name', 'Row', 'Column', 'Label']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")
    
    df = df[required].copy()
    
    annotations = {}
    for name, group in df.groupby('Name'):
        annotations[name] = group
        normalized = normalize_image_name(name)
        if normalized != name and normalized not in annotations:
            annotations[normalized] = group
    
    return annotations


def load_image_files(path, recursive=False):
    """Get list of image files from directory."""
    extensions = ('.jpg', '.jpeg', '.png')
    
    if recursive:
        files = []
        for root, dirs, filenames in os.walk(path):
            for f in filenames:
                if f.lower().endswith(extensions):
                    files.append(f)
        return files
    else:
        return [f for f in os.listdir(path) if f.lower().endswith(extensions)]


def find_image_path(base_path, image_name):
    """
    Find the full path to an image, searching recursively in subfolders.
    Also tries normalized name variants to handle double extensions.
    """
    names_to_try = [image_name]
    normalized = normalize_image_name(image_name)
    if normalized != image_name:
        names_to_try.append(normalized)
    
    for name in names_to_try:
        direct_path = os.path.join(base_path, name)
        if os.path.exists(direct_path):
            return direct_path
    
    for root, dirs, files in os.walk(base_path):
        for name in names_to_try:
            if name in files:
                return os.path.join(root, name)
        for f in files:
            if normalize_image_name(f) == normalized:
                return os.path.join(root, f)
    
    return None


def scale_image_and_points(image, points_df, scale_factor):
    """Scale image and annotation points by a factor."""
    orig_h, orig_w = image.shape[:2]
    scaled_w = int(orig_w * scale_factor)
    scaled_h = int(orig_h * scale_factor)
    scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    scaled_points = points_df.copy()
    scaled_points['Column'] = (scaled_points['Column'] * scale_factor).round().astype(int)
    scaled_points['Row'] = (scaled_points['Row'] * scale_factor).round().astype(int)
    return scaled_image, scaled_points


def rescale_mask(mask, original_image):
    """Rescale mask back to original image size."""
    orig_h, orig_w = original_image.shape[:2]
    return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
