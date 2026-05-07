import os
import json
import base64
import requests
import cv2
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

# Load API key from .env in the same directory as this script
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY or API_KEY == "your_api_key_here":
    raise RuntimeError("Set ROBOFLOW_API_KEY in sam3_poc/.env")

# --- Config ---
IMAGE_PATH = SCRIPT_DIR.parent / "input" / "images" / "Training Data" / "dipsa.jpeg.jpeg"
# Hardcoded annotation point from annotations_confirmed.csv:
#   Name=dipsa.jpeg, Row(y)=101, Column(x)=384, Label=Dipsa
POINT_X = 384
POINT_Y = 101

SAM3_URL = "https://serverless.roboflow.com/sam3/visual_segment"

# --- Load and encode image ---
image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

_, buffer = cv2.imencode(".jpg", image)
image_b64 = base64.b64encode(buffer).decode("utf-8")

print(f"Image loaded: {IMAGE_PATH}  ({image.shape[1]}x{image.shape[0]})")
print(f"Point prompt: x={POINT_X}, y={POINT_Y} (label=TA)")

# --- Build payload (SAM3 PVS) ---
payload = {
    "image": {
        "type": "base64",
        "value": image_b64,
    },
    "image_id": "DSCN2136",
    "prompts": [
        {
            "prompts": [
                {
                    "points": [
                        {"positive": True, "x": POINT_X, "y": POINT_Y}
                    ]
                }
            ]
        }
    ],
    "format": "json",
    "sam2_version_id": "hiera_large",
    "multimask_output": True,
    "save_logits_to_cache": True,
    "load_logits_from_cache": False,
}

# --- Call SAM3 API ---
print("\nSending request to SAM3 visual_segment ...")
response = requests.post(
    f"{SAM3_URL}?api_key={API_KEY}",
    json=payload,
    timeout=120,
)
response.raise_for_status()
data = response.json()

# --- Print response summary (full JSON is huge due to polygon coords) ---
print(f"\n=== Response top-level keys: {list(data.keys())} ===")
print(f"API time: {data.get('time', 'N/A')}s")

# Save full response to file for inspection
resp_path = SCRIPT_DIR / "test_connection_response.json"
with open(resp_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"Full JSON response saved to: {resp_path}")

# --- Extract predictions (handle both response formats) ---
predictions = []
if "prompt_results" in data:
    for pr in data["prompt_results"]:
        predictions.extend(pr.get("predictions", []))
elif "predictions" in data:
    predictions = data["predictions"]
else:
    print(f"\nUnexpected response structure. Keys: {list(data.keys())}")

# --- Overlay mask / polygon on image ---
output = image.copy()
overlay = image.copy()

if not predictions:
    print("\nNo predictions found in response.")
else:
    # Pick the highest-confidence prediction
    best = max(predictions, key=lambda p: p.get("confidence", 0))
    conf = best.get("confidence", 0)
    masks = best.get("masks", [])
    fmt = best.get("format", "unknown")
    print(f"\nBest prediction — confidence: {conf:.4f}, format: {fmt}, "
          f"num polygons: {len(masks)}")

    color = (0, 255, 0)  # green
    for polygon in masks:
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=2)

    # Blend the filled overlay with the original
    alpha = 0.35
    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

# Draw the prompt point
cv2.circle(output, (POINT_X, POINT_Y), radius=10, color=(0, 0, 255), thickness=-1)
cv2.putText(output, "TA", (POINT_X + 15, POINT_Y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

# Save result
OUTPUT_PATH = SCRIPT_DIR / "test_connection_result.jpg"
cv2.imwrite(str(OUTPUT_PATH), output)
print(f"\nResult saved to: {OUTPUT_PATH}")
