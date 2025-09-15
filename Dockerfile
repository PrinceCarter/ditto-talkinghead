# Use the exact base that matches our working RunPod environment
FROM runpod/base:0.6.2-cuda12.2.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system deps first (including wget)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl \
    git git-lfs \
    ffmpeg libsndfile1 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install miniconda (to match your exact setup)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh

# Initialize conda and add to PATH
ENV PATH="/opt/miniconda/bin:$PATH"
RUN conda init bash && \
    echo ". /opt/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc

WORKDIR /workspace/ditto-talkinghead

# Create conda environment (now that conda is properly set up)
RUN /bin/bash -c "source /opt/miniconda/etc/profile.d/conda.sh && conda create -n ditto python=3.10 -y"

# Install packages in conda environment using bash with proper sourcing
RUN /bin/bash -c "source /opt/miniconda/etc/profile.d/conda.sh && conda activate ditto && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"

RUN /bin/bash -c "source /opt/miniconda/etc/profile.d/conda.sh && conda activate ditto && \
    pip install --no-cache-dir runpod huggingface_hub hf_transfer requests numpy scipy"

RUN /bin/bash -c "source /opt/miniconda/etc/profile.d/conda.sh && conda activate ditto && \
    pip install --no-cache-dir librosa tqdm filetype imageio opencv-python-headless \
    scikit-image imageio-ffmpeg colored onnxruntime einops"

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