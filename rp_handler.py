import runpod
import os
import sys
import subprocess
import base64
import tempfile
import requests
from pathlib import Path

# Add the current directory to Python path
sys.path.append('/workspace/ditto-talkinghead')

def download_file(url, suffix):
    """Download file from URL to temporary file"""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        return tmp_file.name

def handler(event):
    """
    RunPod handler for ditto-talkinghead video inference

    Expected input format (2 options):

    Option 1 - URLs (for large files):
    {
        "input": {
            "audio_url": "https://example.com/audio.wav",
            "image_url": "https://example.com/image.png"
        }
    }

    Option 2 - Base64 (for files <8MB total):
    {
        "input": {
            "audio_base64": "base64_encoded_audio_data",
            "image_base64": "base64_encoded_image_data",
            "audio_format": "wav",  # optional, defaults to wav
            "image_format": "png"   # optional, defaults to png
        }
    }

    Use client_upload.py script to easily upload local files via base64.
    """
    try:
        print("Ditto TalkingHead Worker Started")

        # Extract input data
        input_data = event.get('input', {})

        # Check input method
        audio_url = input_data.get('audio_url')
        image_url = input_data.get('image_url')
        audio_base64 = input_data.get('audio_base64')
        image_base64 = input_data.get('image_base64')

        audio_path = None
        image_path = None

        # Method 1: URLs (for large files)
        if audio_url and image_url:
            print("Using URL input method")
            audio_path = download_file(audio_url, '.wav')
            image_path = download_file(image_url, '.png')

        # Method 2: Base64 (for files <8MB total)
        elif audio_base64 and image_base64:
            print("Using base64 input method")
            audio_format = input_data.get('audio_format', 'wav')
            image_format = input_data.get('image_format', 'png')

            with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as audio_file:
                audio_data = base64.b64decode(audio_base64)
                audio_file.write(audio_data)
                audio_path = audio_file.name

            with tempfile.NamedTemporaryFile(suffix=f'.{image_format}', delete=False) as image_file:
                image_data = base64.b64decode(image_base64)
                image_file.write(image_data)
                image_path = image_file.name
        else:
            return {
                "error": "Please provide files via: 1) URLs (audio_url + image_url) or 2) Base64 (audio_base64 + image_base64). Use client_upload.py for local files."
            }

        # Create output file path
        output_path = tempfile.mktemp(suffix='.mp4')

        print(f"Processing audio: {audio_path}")
        print(f"Processing image: {image_path}")
        print(f"Output will be: {output_path}")

        # Change to ditto directory
        os.chdir('/workspace/ditto-talkinghead')

        # Run inference with conda environment activated
        cmd = [
            '/bin/bash', '-c',
            f'''
            source /opt/conda/etc/profile.d/conda.sh && \
            conda activate ditto && \
            python3 inference.py \
                --data_root "./checkpoints/ditto_pytorch" \
                --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" \
                --audio_path "{audio_path}" \
                --source_path "{image_path}" \
                --output_path "{output_path}"
            '''
        ]

        print("Running ditto inference...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"Error running inference: {result.stderr}")
            return {
                "error": f"Inference failed: {result.stderr}"
            }

        print("Inference completed successfully!")

        # Read the output video and encode as base64
        if os.path.exists(output_path):
            with open(output_path, 'rb') as video_file:
                video_data = video_file.read()
                video_base64 = base64.b64encode(video_data).decode('utf-8')

            # Clean up temporary files
            try:
                os.unlink(audio_path)
                os.unlink(image_path)
                os.unlink(output_path)
            except:
                pass

            return {
                "video_base64": video_base64,
                "message": "Video generated successfully"
            }
        else:
            return {
                "error": "Output video file was not created"
            }

    except Exception as e:
        print(f"Handler error: {str(e)}")
        return {
            "error": f"Handler error: {str(e)}"
        }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})