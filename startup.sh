#!/bin/bash

# Azure App Service startup script
# Uses PORT env var set by Azure, defaults to 8000 for local testing

python -m streamlit run webapp.py --server.port ${PORT:-8000} --server.address 0.0.0.0 --server.headless true
