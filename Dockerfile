# Use the exact base that matches our working RunPod environment
FROM runpod/base:0.6.2-cuda12.2.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install miniconda (to match your exact setup)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh && \
    /opt/miniconda/bin/conda init bash

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs \
    ffmpeg libsndfile1 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ditto-talkinghead

# Create conda environment exactly like we did locally
RUN /opt/miniconda/bin/conda create -n ditto python=3.10 -y

# Install PyTorch first using miniconda path
RUN /opt/miniconda/bin/conda run -n ditto pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install basic dependencies
RUN /opt/miniconda/bin/conda run -n ditto pip install --no-cache-dir \
    runpod huggingface_hub hf_transfer requests numpy scipy

# Install remaining packages
RUN /opt/miniconda/bin/conda run -n ditto pip install --no-cache-dir \
    librosa tqdm filetype imageio opencv-python-headless \
    scikit-image imageio-ffmpeg colored onnxruntime einops

# Set conda environment path (using miniconda path)
ENV PATH="/opt/miniconda/envs/ditto/bin:$PATH"

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