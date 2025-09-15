# CUDA runtime base (matches RunPod GPU runtime)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    git git-lfs \
    ffmpeg libsndfile1 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ditto-talkinghead

# Install PyTorch (CUDA build that matches base)
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# App deps (pin what matters)
RUN python3 -m pip install --no-cache-dir \
    runpod \
    huggingface_hub hf_transfer \
    librosa tqdm filetype imageio opencv_python_headless \
    scikit-image imageio-ffmpeg colored onnxruntime-gpu mediapipe einops \
    requests

# Copy only the minimal code to avoid cache-busting
COPY rp_handler.py inference.py stream_pipeline_*.py ./
COPY core ./core
COPY scripts ./scripts

# Create checkpoints cache path (no models baked!)
RUN mkdir -p /workspace/models/ditto
ENV DITTO_CKPT_DIR=/workspace/models/ditto

# On cold start: fetch missing models (PyTorch version - tight allow-list), then run handler
CMD python3 -c "import os; os.makedirs(os.environ.get('DITTO_CKPT_DIR','/workspace/models/ditto'), exist_ok=True); \
from huggingface_hub import snapshot_download; d=os.environ['DITTO_CKPT_DIR']; \
print('Model dir:', d); \
snapshot_download(repo_id='digital-avatar/ditto-talkinghead', local_dir=d, local_dir_use_symlinks=False, \
allow_patterns=['ditto_cfg/v0.4_hubert_cfg_pytorch.pkl', \
                'ditto_pytorch/models/appearance_extractor.pth', \
                'ditto_pytorch/models/decoder.pth', \
                'ditto_pytorch/models/lmdm_v0.4_hubert.pth', \
                'ditto_pytorch/models/motion_extractor.pth', \
                'ditto_pytorch/models/stitch_network.pth', \
                'ditto_pytorch/models/warp_network.pth', \
                'ditto_pytorch/aux_models/2d106det.onnx', \
                'ditto_pytorch/aux_models/det_10g.onnx', \
                'ditto_pytorch/aux_models/face_landmarker.task', \
                'ditto_pytorch/aux_models/hubert_streaming_fix_kv.onnx', \
                'ditto_pytorch/aux_models/landmark203.onnx']);" \
 && ln -sf \"$DITTO_CKPT_DIR\" /workspace/ditto-talkinghead/checkpoints \
 && python3 -u rp_handler.py