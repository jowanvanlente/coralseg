#!/bin/bash

# Install system dependencies for OpenCV
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libxcb1 libx11-6

# Run Streamlit
python -m streamlit run webapp.py --server.port 8000 --server.address 0.0.0.0
