FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for both projects
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    sqlite3 \
    curl \
    libx11-6 \
    libxext6 \
    libxrender1 \
    x11-xserver-utils \
    xauth \
    xvfb \
    libcublas-11-8 \
    gcc \
    libc6-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
COPY docker/requirements.txt docker_requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r docker_requirements.txt

# Copy project files
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/
COPY footage/ ./footage/
COPY detected_faces/ ./detected_faces/
COPY logs/ ./logs/
COPY init_db.sql ./init_db.sql

# Create directories and set permissions
RUN mkdir -p /app/data && \
    chmod -R u+rw /app && \
    chown -R 1000:1000 /app/data && \
    chmod -R 777 /app/detected_faces && \
    chmod -R 777 /app/logs

# Verify critical files
RUN if [ ! -f /app/src/kinwatch_agent.py ]; then echo "Error: kinwatch_agent.py not found"; exit 1; fi && \
    if [ ! -f /app/models/yolov8n.pt ]; then echo "Error: yolov8n.pt not found"; exit 1; fi && \
    if [ ! -d /app/models/buffalo_l ]; then echo "Error: buffalo_l directory not found"; exit 1; fi

# Create non-root user
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g 1000 -ms /bin/bash appuser && \
    chown -R appuser:appgroup /app

USER appuser

# Generic CMD to allow docker-compose overrides
CMD ["bash"]