#!/bin/bash
set -e

echo "Starting Ditto TalkingHead container..."

# Activate conda environment (using miniconda path)
source /opt/miniconda/etc/profile.d/conda.sh
conda activate ditto

# Create model directory
mkdir -p $DITTO_CKPT_DIR

# Download models on first run
echo "Downloading models from HuggingFace..."
python3 -c "
import os
from huggingface_hub import snapshot_download

d = os.environ.get('DITTO_CKPT_DIR', '/workspace/models/ditto')
print(f'Model dir: {d}')

snapshot_download(
    repo_id='digital-avatar/ditto-talkinghead',
    local_dir=d,
    local_dir_use_symlinks=False,
    allow_patterns=[
        'ditto_cfg/v0.4_hubert_cfg_pytorch.pkl',
        'ditto_pytorch/models/appearance_extractor.pth',
        'ditto_pytorch/models/decoder.pth',
        'ditto_pytorch/models/lmdm_v0.4_hubert.pth',
        'ditto_pytorch/models/motion_extractor.pth',
        'ditto_pytorch/models/stitch_network.pth',
        'ditto_pytorch/models/warp_network.pth',
        'ditto_pytorch/aux_models/2d106det.onnx',
        'ditto_pytorch/aux_models/det_10g.onnx',
        'ditto_pytorch/aux_models/face_landmarker.task',
        'ditto_pytorch/aux_models/hubert_streaming_fix_kv.onnx',
        'ditto_pytorch/aux_models/landmark203.onnx'
    ]
)
"

# Create symlink to checkpoints
ln -sf "$DITTO_CKPT_DIR" /workspace/ditto-talkinghead/checkpoints

# Start the handler
echo "Starting RunPod handler..."
exec python3 -u rp_handler.py