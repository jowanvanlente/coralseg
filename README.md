# Sparse to Dense COCO - Web App

Convert sparse point annotations (from CoralNet) into dense segmentation masks in COCO format for Roboflow.

## Azure Deployment

### Runtime Stack
**Python 3.11** (recommended)

The app requires Python 3.10+ due to these dependencies:
- `numpy>=2.0.0` requires Python 3.10+
- `scipy>=1.14.0` requires Python 3.10+
- `scikit-image>=0.24.0` requires Python 3.10+

### Azure Web App Settings
1. **Runtime stack:** Python 3.11
2. **Startup command:** `python -m streamlit run webapp.py --server.port 8000 --server.address 0.0.0.0`

### Deployment Steps
1. Create Azure Web App with Python 3.11 runtime
2. Deploy the `webapp/` folder contents
3. Set the startup command in Configuration > General settings

## Local Development

```bash
cd webapp
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run webapp.py
```

## Features

- **Test Mode:** Visualize segmentation on single images before batch processing
- **Export Mode:** Process multiple images and download COCO JSON
- **Sample Data:** Built-in sample image with 900 annotations for testing
- **Three segmentation methods:** Superpixel (SLIC), Adaptive (Density-based), Graph-based (Felzenszwalb)

## Input Format

### Images
- JPG, JPEG, or PNG format
- Any resolution (automatically scaled for processing)

### Annotations CSV (CoralNet format)
```csv
Name,Row,Column,Label
image001.jpg,120,340,Acr
image001.jpg,245,512,TA
image002.jpg,100,200,S
```

## Output
- COCO JSON file ready for Roboflow import
- One annotation per connected region (instance segmentation)
