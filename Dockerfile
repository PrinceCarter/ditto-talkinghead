# Use miniconda base that matches the working environment
FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs \
    ffmpeg libsndfile1 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ditto-talkinghead

# Create conda environment (simplified, no complex environment.yaml)
RUN conda create -n ditto python=3.10 -y && \
    conda run -n ditto pip install --no-cache-dir \
    runpod \
    huggingface_hub hf_transfer \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    librosa tqdm filetype imageio opencv-python-headless \
    scikit-image imageio-ffmpeg colored onnxruntime einops \
    requests numpy scipy

# Set conda environment path
ENV PATH="/opt/conda/envs/ditto/bin:$PATH"

# Copy code
COPY rp_handler.py inference.py stream_pipeline_*.py ./
COPY core ./core
COPY scripts ./scripts

# Create model cache directory
RUN mkdir -p /workspace/models/ditto
ENV DITTO_CKPT_DIR=/workspace/models/ditto

# Simple startup script
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

CMD ["/startup.sh"]