# Ditto Talking Head API Server

Self-hosted API server for generating talking head videos using the Ditto model.

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate ditto

# Install API dependencies
pip install -r requirements_api.txt

# Download Ditto models
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```

### 2. Start API Server

```bash
python api_server.py
```

Server runs on `http://localhost:5000` by default.

### 3. Generate Videos

#### Using curl:

```bash
curl -X POST \
  -F "audio=@./example/audio.wav" \
  -F "image=@./example/image.png" \
  http://localhost:5000/generate
```

#### Using Python client:

```python
from client_example import DittoClient

client = DittoClient()
result = client.generate_video('./example/audio.wav', './example/image.png')
print(result)
```

## API Endpoints

### Health Check
- **GET** `/`
- Returns server status and SDK initialization status

### Generate Video
- **POST** `/generate`
- **Form Data:**
  - `audio`: Audio file (wav, mp3, m4a)
  - `image`: Source image (png, jpg, jpeg)
  - `config`: Optional JSON configuration
- **Returns:** Session ID and download URL

### Check Status
- **GET** `/status/<session_id>`
- Returns generation status for a session

### Download Video
- **GET** `/download/<filename>`
- Downloads the generated video file

### List Models
- **GET** `/models`
- Returns available models and configurations

## Configuration

### Server Configuration

Environment variables:
- `PORT`: Server port (default: 5000)
- `DEBUG`: Enable debug mode (default: False)

### Video Generation Configuration

Optional config parameter for `/generate`:

```json
{
  "setup_kwargs": {},
  "run_kwargs": {
    "chunksize": [3, 5, 2],
    "fade_in": 5,
    "fade_out": 5,
    "ctrl_info": {}
  }
}
```

## Docker Setup

Create `Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY . .

RUN conda env create -f environment.yaml
RUN /opt/conda/envs/ditto/bin/pip install -r requirements_api.txt

EXPOSE 5000

CMD ["/opt/conda/envs/ditto/bin/python", "api_server.py"]
```

Build and run:

```bash
docker build -t ditto-api .
docker run -p 5000:5000 --gpus all ditto-api
```

## RunPod Deployment

1. Use PyTorch template with GPU support
2. Clone repository and setup environment
3. Download models from HuggingFace
4. Run `python api_server.py`
5. Access via RunPod's public URL

## Performance Optimization

- Use TensorRT models for faster inference
- Pre-warm the SDK on startup
- Implement request queuing for multiple concurrent requests
- Use Redis for session storage in production

## Troubleshooting

### Common Issues

1. **SDK Initialization Failed**
   - Check if model files exist in `./checkpoints/`
   - Verify GPU compatibility with TensorRT models

2. **CUDA/TensorRT Errors**
   - Convert ONNX models to TensorRT for your GPU:
   ```bash
   python scripts/cvt_onnx_to_trt.py --onnx_dir "./checkpoints/ditto_onnx" --trt_dir "./checkpoints/ditto_trt_custom"
   ```

3. **Memory Issues**
   - Use A100 or similar high-memory GPU
   - Monitor GPU memory with `nvidia-smi`

4. **File Upload Errors**
   - Check file format (only wav, mp3, m4a for audio; png, jpg, jpeg for images)
   - Ensure files are under 50MB

## Security Considerations

- Add authentication for production use
- Implement rate limiting
- Validate file types and sizes
- Clean up temporary files regularly
- Use HTTPS in production