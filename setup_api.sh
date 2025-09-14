#!/bin/bash

# Ditto Talking Head API Setup Script

echo "🚀 Setting up Ditto Talking Head API Server..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command_exists conda; then
    print_error "Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

if ! command_exists git; then
    print_error "Git not found. Please install git first."
    exit 1
fi

if ! command_exists ffmpeg; then
    print_warning "FFmpeg not found. Installing via conda..."
fi

# Setup conda environment
print_status "Setting up conda environment..."
if conda env list | grep -q "ditto"; then
    print_warning "Conda environment 'ditto' already exists. Updating..."
    conda env update -f environment.yaml
else
    conda env create -f environment.yaml
fi

# Activate environment
print_status "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ditto

# Install API dependencies
print_status "Installing API dependencies..."
pip install -r requirements_api.txt

# Create necessary directories
print_status "Creating directories..."
mkdir -p tmp/uploads
mkdir -p tmp/outputs
mkdir -p downloads
mkdir -p checkpoints

# Download models if not exists
if [ ! -d "checkpoints/ditto_trt_Ampere_Plus" ]; then
    print_status "Downloading Ditto models from HuggingFace..."

    if ! command_exists git-lfs; then
        print_status "Installing git-lfs..."
        git lfs install
    fi

    git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints

    if [ $? -eq 0 ]; then
        print_status "✅ Models downloaded successfully"
    else
        print_error "Failed to download models"
        exit 1
    fi
else
    print_status "✅ Models already exist"
fi

# Check GPU compatibility
print_status "Checking GPU compatibility..."
if command_exists nvidia-smi; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    print_status "Detected GPU: $GPU_INFO"

    # Check if TensorRT models are compatible
    if [[ $GPU_INFO == *"RTX"* ]] && [[ ! $GPU_INFO == *"40"* ]]; then
        print_warning "Your GPU might not be compatible with pre-built TensorRT models."
        print_warning "You may need to convert ONNX models to TensorRT:"
        print_warning "python scripts/cvt_onnx_to_trt.py --onnx_dir \"./checkpoints/ditto_onnx\" --trt_dir \"./checkpoints/ditto_trt_custom\""
    fi
else
    print_error "No GPU detected or nvidia-smi not available"
    exit 1
fi

# Test basic setup
print_status "Testing basic setup..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.get_device_name()}')
" 2>/dev/null

if [ $? -eq 0 ]; then
    print_status "✅ PyTorch and CUDA setup verified"
else
    print_error "PyTorch setup verification failed"
fi

# Create example .env file
if [ ! -f ".env" ]; then
    print_status "Creating example .env file..."
    cat > .env << EOF
# API Server Configuration
PORT=5000
DEBUG=false

# Optional: Add other configurations here
EOF
fi

# Make scripts executable
chmod +x api_server.py
chmod +x client_example.py

print_status "🎉 Setup completed successfully!"
echo
echo "Next steps:"
echo "1. Activate the conda environment: conda activate ditto"
echo "2. Start the API server: python api_server.py"
echo "3. Test with the client: python client_example.py"
echo "4. Or visit http://localhost:5000 in your browser"
echo
echo "For more information, see API_README.md"