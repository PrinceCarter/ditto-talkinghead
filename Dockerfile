# Use the official miniconda image that works
FROM condaforge/mambaforge:latest

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

# Accept conda Terms of Service first, then create environment
RUN conda config --set channel_priority flexible && \
    echo "yes" | conda tos accept --all && \
    conda create -n ditto python=3.10 -y

# Install packages the simple way
RUN conda run -n ditto pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN conda run -n ditto pip install --no-cache-dir \
    runpod huggingface_hub hf_transfer requests numpy scipy

RUN conda run -n ditto pip install --no-cache-dir \
    librosa tqdm filetype imageio opencv-python-headless \
    scikit-image imageio-ffmpeg colored onnxruntime einops

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