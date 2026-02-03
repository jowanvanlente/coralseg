FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV headless and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Environment variables for Azure Container Apps
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500
ENV STREAMLIT_SERVER_MAX_MESSAGE_SIZE=500
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check for Container Apps (longer timeout for cold starts)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/_stcore/health || exit 1

# Run Streamlit
CMD ["python", "-m", "streamlit", "run", "webapp.py", "--server.port=8000", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableWebsocketCompression=false"]
