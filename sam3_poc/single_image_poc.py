"""
SAM3 single-image POC: cluster annotations → SAM3 segmentation → majority-vote labeling.

Picks the 5 largest spatial clusters of same-class annotation points from the
dipsa training image, sends each as a separate SAM3 visual_segment call (reusing
the cached image embedding via image_id), and produces a 3-panel figure.
"""

import os
import json
import base64
import uuid
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path as MplPath
from sklearn.cluster import DBSCAN
import requests
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY or API_KEY == "your_api_key_here":
    raise RuntimeError("Set ROBOFLOW_API_KEY in sam3_poc/.env")

IMAGE_PATH = SCRIPT_DIR.parent / "input" / "images" / "Training Data" / "dipsa.jpeg.jpeg"
ANNOTATIONS_PATH = SCRIPT_DIR.parent / "input" / "annotations" / "annotations_confirmed.csv"
IMAGE_NAME = "dipsa.jpeg"  # name in annotations CSV (no double extension)
IMAGE_ID = "dipsa_poc"     # consistent ID for embedding cache
SAM3_URL = "https://serverless.roboflow.com/sam3/visual_segment"

N_CLUSTERS = 5
DBSCAN_EPS = 80       # max pixel distance between points in a cluster
DBSCAN_MIN_SAMPLES = 3

# ── Load image ──────────────────────────────────────────────────────────────
image_bgr = cv2.imread(str(IMAGE_PATH))
if image_bgr is None:
    raise FileNotFoundError(f"Cannot read: {IMAGE_PATH}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
h, w = image_bgr.shape[:2]
print(f"Image: {IMAGE_PATH.name}  ({w}x{h})")

_, buffer = cv2.imencode(".jpg", image_bgr)
image_b64 = base64.b64encode(buffer).decode("utf-8")

# ── Load annotations ────────────────────────────────────────────────────────
df = pd.read_csv(str(ANNOTATIONS_PATH))
ann = df.loc[df["Name"] == IMAGE_NAME, ["Row", "Column", "Label code"]].copy()
ann.columns = ["y", "x", "label"]
ann = ann.reset_index(drop=True)
print(f"Annotations: {len(ann)} points, {ann['label'].nunique()} classes")
print(f"Classes: {dict(ann['label'].value_counts())}")

# ── Cluster same-class nearby points with DBSCAN ───────────────────────────
clusters = []
for label, group in ann.groupby("label"):
    coords = group[["x", "y"]].values
    if len(coords) < DBSCAN_MIN_SAMPLES:
        continue  # skip classes with too few points for a meaningful cluster
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(coords)
    for cid in set(db.labels_):
        if cid == -1:
            continue
        mask = db.labels_ == cid
        clusters.append({
            "label": label,
            "points": coords[mask],
            "size": int(mask.sum()),
        })

clusters.sort(key=lambda c: c["size"], reverse=True)
top_clusters = clusters[:N_CLUSTERS]
print(f"\nTop {N_CLUSTERS} clusters:")
for i, c in enumerate(top_clusters):
    centroid = c["points"].mean(axis=0).astype(int)
    print(f"  {i+1}. class={c['label']:<8s}  size={c['size']}  centroid=({centroid[0]}, {centroid[1]})")

# ── Color map for classes ───────────────────────────────────────────────────
all_labels = sorted(ann["label"].unique())
tab_colors = list(mcolors.TABLEAU_COLORS.values())
class_colors = {label: tab_colors[i % len(tab_colors)] for i, label in enumerate(all_labels)}


def hex_to_rgb(hex_color):
    """Convert matplotlib hex color to (R, G, B) 0-255 tuple."""
    rgb = mcolors.to_rgb(hex_color)
    return tuple(int(c * 255) for c in rgb)


# ── Send SAM3 calls (one per cluster, reusing embedding) ───────────────────
session = requests.Session()
sam3_results = []

for i, cluster in enumerate(top_clusters):
    # Use cluster centroid as a single positive point for a focused segmentation
    cx, cy = cluster["points"].mean(axis=0).astype(int)
    points_payload = [{"positive": True, "x": int(cx), "y": int(cy)}]
    # Unique image_id per call to prevent cache from returning stale masks
    call_image_id = f"{IMAGE_ID}_cluster{i}"
    payload = {
        "image": {"type": "base64", "value": image_b64},
        "image_id": call_image_id,
        "prompts": [{"prompts": [{"points": points_payload}]}],
        "format": "json",
        "sam2_version_id": "hiera_large",
        "multimask_output": True,
        "save_logits_to_cache": False,
        "load_logits_from_cache": False,
    }

    print(f"\nCluster {i+1}/{N_CLUSTERS}: centroid=({cx},{cy}) class={cluster['label']} ...")
    resp = session.post(f"{SAM3_URL}?api_key={API_KEY}", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print(f"  API time: {data.get('time', '?')}s")

    # Extract predictions (handle both response formats)
    predictions = []
    if "prompt_results" in data:
        for pr in data["prompt_results"]:
            predictions.extend(pr.get("predictions", []))
    elif "predictions" in data:
        predictions = data["predictions"]

    if not predictions:
        print("  No predictions returned.")
        sam3_results.append({"cluster": cluster, "polygons": [], "majority_class": None})
        continue

    # Log all predictions, pick highest confidence
    for pi, pred in enumerate(predictions):
        print(f"  prediction[{pi}]: conf={pred.get('confidence', 0):.4f}, polygons={len(pred.get('masks', []))}")
    best = max(predictions, key=lambda p: p.get("confidence", 0))
    polygons = best.get("masks", [])
    conf = best.get("confidence", 0)
    print(f"  -> Selected: confidence={conf:.4f}, polygons={len(polygons)}")

    # Majority vote: count ALL annotation points inside ANY of these polygons
    votes = Counter()
    for _, pt in ann.iterrows():
        px, py = int(pt["x"]), int(pt["y"])
        for poly_coords in polygons:
            contour = np.array(poly_coords, dtype=np.float32).reshape(-1, 1, 2)
            if cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0:
                votes[pt["label"]] += 1
                break  # count each point only once across all polygon parts

    majority_class = votes.most_common(1)[0][0] if votes else cluster["label"]
    print(f"  Majority vote: {majority_class} (votes: {dict(votes)})")

    sam3_results.append({
        "cluster": cluster,
        "polygons": polygons,
        "majority_class": majority_class,
        "confidence": conf,
    })

# ── Build 3-panel figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=100)

# Panel 1: Original image
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image", fontsize=14)
axes[0].axis("off")

# Panel 2: Annotation points colored by class
axes[1].imshow(image_rgb)
for label in all_labels:
    pts = ann[ann["label"] == label]
    axes[1].scatter(pts["x"], pts["y"], c=class_colors[label], label=label,
                    s=30, edgecolors="white", linewidths=0.5, zorder=5)
axes[1].legend(loc="upper right", fontsize=7, framealpha=0.8, ncol=2)
axes[1].set_title("Annotation Points by Class", fontsize=14)
axes[1].axis("off")

# Panel 3: SAM3 polygons colored by majority class
axes[2].imshow(image_rgb)
for res in sam3_results:
    mc = res["majority_class"]
    if mc is None or not res["polygons"]:
        continue
    color = class_colors.get(mc, "gray")
    for poly_coords in res["polygons"]:
        poly_np = np.array(poly_coords)
        patch = MplPolygon(poly_np, closed=True, facecolor=color, edgecolor=color,
                           alpha=0.35, linewidth=1.5)
        axes[2].add_patch(patch)

    # Label at centroid of the cluster
    cx, cy = res["cluster"]["points"].mean(axis=0)
    axes[2].annotate(mc, (cx, cy), color="white", fontsize=8, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.8))

# Add annotation points on panel 3 too for reference
axes[2].scatter(ann["x"], ann["y"], c="white", s=8, zorder=10, alpha=0.6)
axes[2].set_title("SAM3 Polygons (majority-vote class)", fontsize=14)
axes[2].axis("off")

plt.tight_layout()
output_path = SCRIPT_DIR / "single_image_poc_result.png"
fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
print(f"\nFigure saved to: {output_path}")
plt.close()
