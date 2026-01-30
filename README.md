# Sparse to Dense COCO - Web App

Convert sparse point annotations (from CoralNet) into dense segmentation masks in COCO format for Roboflow.

## Azure Deployment

### Runtime Stack
**Python 3.10 or 3.11** (recommended: 3.11)

### Azure Web App Settings
1. **Runtime stack:** Python 3.11
2. **Startup command:** `python -m streamlit run webapp.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`

> **Note:** Azure sets the `PORT` environment variable automatically. The startup command uses `$PORT` to bind to the correct port.

### Deployment Steps (GitHub Integration)
1. Create Azure Web App with Python 3.11 runtime
2. Connect to your GitHub repository (select the `webapp/` folder as source if using monorepo)
3. Set the **Startup command** in Configuration > General settings
4. The `.streamlit/config.toml` handles headless mode and CORS settings

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
