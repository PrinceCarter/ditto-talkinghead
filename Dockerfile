# Use NVIDIA CUDA base image with conda
FROM continuumio/miniconda3:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/miniconda3/bin:$PATH"

WORKDIR /workspace/ditto-talkinghead

# Install system dependencies
RUN apt-get update && apt-get install -y \
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

# Make sure conda environment can be activated first
RUN conda init bash

# Copy project files
COPY . .

# Install Python dependencies using conda and pip
RUN /bin/bash -c "source ~/.bashrc && conda activate base && pip install --no-cache-dir runpod numpy==1.26.4"

RUN /bin/bash -c "source ~/.bashrc && conda activate base && pip install --no-cache-dir tensorrt librosa tqdm filetype"

RUN /bin/bash -c "source ~/.bashrc && conda activate base && pip install --no-cache-dir imageio opencv_python_headless scikit-image"

RUN /bin/bash -c "source ~/.bashrc && conda activate base && pip install --no-cache-dir cython cuda-python imageio-ffmpeg colored"

RUN /bin/bash -c "source ~/.bashrc && conda activate base && pip install --no-cache-dir polygraphy onnxruntime-gpu mediapipe einops"

# Initialize git lfs (for model downloads if needed)
RUN git lfs install

# Download model checkpoints from HuggingFace
RUN /bin/bash -c "source ~/.bashrc && conda activate base && pip install huggingface_hub && python -c \"from huggingface_hub import snapshot_download; import os; os.makedirs('./checkpoints', exist_ok=True); snapshot_download(repo_id='DITTO-TTS/ditto-talkinghead', local_dir='./checkpoints', allow_patterns=['*.pth', '*.pkl', '*.onnx', '*.engine', '*.bin']); print('Model checkpoints downloaded successfully')\""

# Set working directory
WORKDIR /workspace/ditto-talkinghead

# Start the handler
CMD ["python3", "-u", "rp_handler.py"]