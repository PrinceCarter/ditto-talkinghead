# Ditto LiveKit Avatar Integration - RunPod Setup Guide

This project integrates the Ditto talking head avatar system with LiveKit agents for real-time avatar conversations.

## Overview

The integration allows users to have real-time conversations with a high-quality Ditto avatar that:
- Responds to voice input with AI-generated speech
- Displays synchronized lip movements and facial expressions
- Streams video/audio through LiveKit for web/mobile clients

## Prerequisites

### Environment Variables
Create a `.env` file with:
```bash
LIVEKIT_URL=wss://your-livekit-server.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
OPENAI_API_KEY=your_openai_api_key
```

### Required Files
- Avatar image: `./example/avatar1.jpeg` (or update path in config)
- Ditto checkpoints and models (see Installation section)

## RunPod Setup

### 1. Launch RunPod Instance
- **Template**: PyTorch 2.1.0 (or similar with CUDA support)
- **GPU**: A100 recommended (RTX 4090/3090 minimum)
- **Storage**: 50GB+ recommended
- **Python**: 3.10 or 3.11

### 2. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/antgroup/ditto-talkinghead
cd ditto-talkinghead

# Install conda environment
conda env create -f environment.yaml
conda activate ditto

# Install additional LiveKit dependencies
pip install livekit-agents livekit-plugins-openai python-dotenv

# Install CUDA Python if missing
pip install cuda-python
```

### 3. Download Ditto Models

```bash
# Download checkpoints from HuggingFace
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```

### 4. Verify Model Structure
Ensure your checkpoints directory looks like:
```
./checkpoints/
├── ditto_cfg/
│   ├── v0.4_hubert_cfg_trt.pkl
│   └── v0.4_hubert_cfg_trt_online.pkl
├── ditto_onnx/
│   └── [various .onnx files]
└── ditto_trt_Ampere_Plus/
    └── [various .engine files]
```

### 5. Test Ditto Standalone (Optional)
```bash
# Test basic Ditto functionality
python inference.py \
    --data_root "./checkpoints/ditto_trt_Ampere_Plus" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/avatar1.jpeg" \
    --output_path "./tmp/result.mp4"
```

### 6. Setup LiveKit Integration Files

Copy the integration files to your RunPod instance:

**Core Integration Files:**
- `livekit_integration/__init__.py`
- `livekit_integration/ditto_session.py`
- `livekit_integration/stream_sdk_livekit.py`
- `livekit_integration/video_publisher.py`
- `livekit_integration/audio_handler.py`
- `livekit_integration/log.py`

**Test Script:**
- `test_simli_pattern.py`

### 7. Create Environment File
```bash
# Create .env file with your credentials
cat > .env << EOF
LIVEKIT_URL=wss://your-livekit-server.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
OPENAI_API_KEY=your_openai_api_key
EOF
```

### 8. Run the LiveKit Avatar Agent
```bash
# Activate conda environment
conda activate ditto

# Run the avatar agent
python test_simli_pattern.py dev
```

## Project Structure

```
ditto-talkinghead/
├── livekit_integration/          # LiveKit integration code
│   ├── __init__.py              # Package exports
│   ├── ditto_session.py         # Main avatar session management
│   ├── stream_sdk_livekit.py    # Modified StreamSDK for LiveKit
│   ├── video_publisher.py       # Video streaming to LiveKit
│   ├── audio_handler.py         # Audio processing and publishing
│   └── log.py                   # Logging configuration
├── test_simli_pattern.py        # Test script using AgentSession pattern
├── checkpoints/                 # Ditto model files
├── example/                     # Sample avatar images and audio
└── .env                        # Environment variables
```

## Key Configuration

### Avatar Configuration (in test_simli_pattern.py)
```python
ditto_config = DittoConfig(
    data_root="./checkpoints/ditto_trt_Ampere_Plus",
    cfg_pkl="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl", 
    avatar_image_path="./example/avatar1.jpeg",
    online_mode=True,
    max_session_length=600,
    max_idle_time=30,
)
```

### Performance Settings
- **FPS**: 30 (configured in stream_sdk_livekit.py)
- **Chunk size**: (1,2,1) for fastest inference
- **Frame buffer**: 10 frames for smooth playback
- **Audio sample rate**: 16kHz

## Troubleshooting

### Common Issues

1. **CUDA Compatibility**: If you get TensorRT errors, convert ONNX to TRT for your GPU:
   ```bash
   python scripts/cvt_onnx_to_trt.py --onnx_dir "./checkpoints/ditto_onnx" --trt_dir "./checkpoints/ditto_trt_custom"
   ```

2. **Memory Issues**: 
   - Use A100 or similar high-memory GPU
   - Monitor GPU memory usage with `nvidia-smi`

3. **Audio Sync Issues**:
   - Ensure audio sample rate is 16kHz
   - Check chunk size matches HuBERT requirements (6480 samples)

4. **Video Choppy/Glitchy**:
   - Reduce inference chunk size in stream_sdk_livekit.py
   - Increase frame buffer size
   - Check GPU utilization

### Logs and Monitoring
- Check logs for "=== TTS AUDIO RECEIVED ===" and "DittoVideoWriter: Frame" messages
- Monitor frame generation rate: should see ~30 FPS for smooth video
- Watch for timeout errors in audio processing

## Architecture Notes

The integration follows the LiveKit avatar pattern similar to Simli:
1. **AgentSession** handles LLM, TTS, and STT
2. **DittoAvatarSession** connects as separate participant
3. **DataStreamAudioOutput** streams TTS audio to avatar
4. **Avatar processes audio** through Ditto for lip sync
5. **Video frames published** to LiveKit room for users

## Performance Optimization

For best results:
- Use TensorRT models (not PyTorch) for inference speed
- Keep chunk sizes small for low latency
- Use frame repetition instead of idle frames during gaps
- Monitor and tune based on GPU capabilities

## Support

Common commands for debugging:
```bash
# Check GPU utilization
nvidia-smi

# Monitor logs in real-time
tail -f /var/log/livekit-agent.log

# Test individual components
python inference.py [args]  # Test Ditto standalone
```

Remember to update avatar image paths and model configurations based on your specific setup and requirements.