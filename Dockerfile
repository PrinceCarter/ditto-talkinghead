# Use Ubuntu base with Python (matching our working local environment)
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace/ditto-talkinghead

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies (using what worked locally on A100)
RUN python3 -m pip install --no-cache-dir \
    runpod \
    librosa \
    tqdm \
    filetype \
    imageio \
    opencv_python_headless \
    scikit-image \
    imageio-ffmpeg \
    colored \
    onnxruntime-gpu \
    mediapipe \
    einops

# Initialize git lfs (for model downloads if needed)
RUN git lfs install

# Download model checkpoints from HuggingFace
RUN python3 -m pip install huggingface_hub && \
    python3 -c "from huggingface_hub import snapshot_download; import os; os.makedirs('./checkpoints', exist_ok=True); snapshot_download(repo_id='DITTO-TTS/ditto-talkinghead', local_dir='./checkpoints', allow_patterns=['*.pth', '*.pkl', '*.onnx', '*.engine', '*.bin']); print('Model checkpoints downloaded successfully')"

# Set working directory
WORKDIR /workspace/ditto-talkinghead

# Start the handler
CMD ["python3", "-u", "rp_handler.py"]