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

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    tensorrt \
    librosa \
    tqdm \
    filetype \
    imageio \
    opencv_python_headless \
    scikit-image \
    cython \
    cuda-python \
    imageio-ffmpeg \
    colored \
    polygraphy \
    numpy==1.26.4 \
    onnxruntime-gpu \
    mediapipe \
    einops

# Initialize git lfs (for model downloads if needed)
RUN git lfs install

# Make sure conda environment can be activated
RUN conda init bash

# Set working directory
WORKDIR /workspace/ditto-talkinghead

# Start the handler
CMD ["python3", "-u", "rp_handler.py"]