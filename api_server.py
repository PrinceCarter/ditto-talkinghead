#!/usr/bin/env python3
"""
Ditto Talking Head API Server
Self-hosted video generation API
"""

import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import logging

from stream_pipeline_offline import StreamSDK
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = './tmp/uploads'
OUTPUT_FOLDER = './tmp/outputs'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Global SDK instance
SDK = None


def load_pkl(pkl_path):
    """Load pickle configuration file"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def initialize_sdk():
    """Initialize the Ditto SDK"""
    global SDK
    try:
        # Default configuration
        data_root = "./checkpoints/ditto_trt_Ampere_Plus"
        cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"

        # Check if files exist
        if not Path(data_root).exists():
            logger.error(f"Model directory not found: {data_root}")
            return False

        if not Path(cfg_pkl).exists():
            logger.error(f"Config file not found: {cfg_pkl}")
            return False

        SDK = StreamSDK(cfg_pkl, data_root)
        logger.info("✅ Ditto SDK initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize SDK: {str(e)}")
        return False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_video(audio_path: str, source_path: str, output_path: str, config_overrides: dict = None) -> bool:
    """Generate talking head video"""
    global SDK

    try:
        # Load configuration overrides
        more_kwargs = {}
        if config_overrides:
            more_kwargs = config_overrides

        setup_kwargs = more_kwargs.get("setup_kwargs", {})
        run_kwargs = more_kwargs.get("run_kwargs", {})

        # Setup SDK
        SDK.setup(source_path, output_path, **setup_kwargs)

        # Load and process audio
        audio, sr = librosa.core.load(audio_path, sr=16000)
        num_f = int(len(audio) / 16000 * 25)

        # Configure run parameters
        fade_in = run_kwargs.get("fade_in", -1)
        fade_out = run_kwargs.get("fade_out", -1)
        ctrl_info = run_kwargs.get("ctrl_info", {})
        SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

        # Process audio through SDK
        if SDK.online_mode:
            chunksize = run_kwargs.get("chunksize", (3, 5, 2))
            audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
            split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
            for i in range(0, len(audio), chunksize[1] * 640):
                audio_chunk = audio[i:i + split_len]
                if len(audio_chunk) < split_len:
                    audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
                SDK.run_chunk(audio_chunk, chunksize)
        else:
            aud_feat = SDK.wav2feat.wav2feat(audio)
            SDK.audio2motion_queue.put(aud_feat)

        SDK.close()

        # Combine video with audio using ffmpeg
        final_output = output_path.replace('.mp4', '_final.mp4')
        cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{final_output}"'
        result = os.system(cmd)

        if result == 0:
            # Replace the original output with the final one
            os.rename(final_output, output_path)
            logger.info(f"✅ Video generated successfully: {output_path}")
            return True
        else:
            logger.error(f"FFmpeg failed with code: {result}")
            return False

    except Exception as e:
        logger.error(f"Video generation failed: {str(e)}")
        return False


@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Ditto Talking Head API Server',
        'version': '1.0.0',
        'sdk_initialized': SDK is not None
    })


@app.route('/generate', methods=['POST'])
def generate_talking_head():
    """Generate talking head video from audio and source image"""

    # Check if SDK is initialized
    if SDK is None:
        return jsonify({'error': 'SDK not initialized'}), 500

    # Check if the post request has the files
    if 'audio' not in request.files or 'image' not in request.files:
        return jsonify({'error': 'Missing audio or image file'}), 400

    audio_file = request.files['audio']
    image_file = request.files['image']

    # Check if files were selected
    if audio_file.filename == '' or image_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Validate file extensions
    if not (allowed_file(audio_file.filename) and allowed_file(image_file.filename)):
        return jsonify({'error': 'Invalid file format'}), 400

    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save uploaded files
        audio_filename = f"{session_id}_{timestamp}_audio.wav"
        image_filename = f"{session_id}_{timestamp}_image.{image_file.filename.rsplit('.', 1)[1].lower()}"
        output_filename = f"{session_id}_{timestamp}_output.mp4"

        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        audio_file.save(audio_path)
        image_file.save(image_path)

        # Get optional configuration parameters
        config_overrides = {}
        if request.form.get('config'):
            try:
                import json
                config_overrides = json.loads(request.form.get('config'))
            except json.JSONDecodeError:
                logger.warning("Invalid config JSON provided, using defaults")

        # Generate video
        logger.info(f"Starting video generation for session: {session_id}")
        success = generate_video(audio_path, image_path, output_path, config_overrides)

        if success and os.path.exists(output_path):
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Video generated successfully',
                'output_filename': output_filename,
                'download_url': f'/download/{output_filename}'
            })
        else:
            return jsonify({'error': 'Video generation failed'}), 500

    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

    finally:
        # Clean up uploaded files (keep output for download)
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
            if 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
        except:
            pass


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download generated video file"""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)

    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(file_path, as_attachment=True)


@app.route('/status/<session_id>', methods=['GET'])
def get_status(session_id):
    """Get status of a video generation session"""
    # Look for output file with this session_id
    output_files = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) if f.startswith(session_id)]

    if output_files:
        return jsonify({
            'session_id': session_id,
            'status': 'completed',
            'output_filename': output_files[0],
            'download_url': f'/download/{output_files[0]}'
        })
    else:
        return jsonify({
            'session_id': session_id,
            'status': 'processing'
        })


@app.route('/models', methods=['GET'])
def list_models():
    """List available models and configurations"""
    try:
        models = {}
        checkpoints_dir = Path('./checkpoints')

        if checkpoints_dir.exists():
            # List available model directories
            for model_dir in checkpoints_dir.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith('ditto_'):
                    models[model_dir.name] = {
                        'path': str(model_dir),
                        'type': 'tensorrt' if 'trt' in model_dir.name else 'onnx' if 'onnx' in model_dir.name else 'pytorch'
                    }

        return jsonify({
            'available_models': models,
            'current_model': './checkpoints/ditto_trt_Ampere_Plus',
            'current_config': './checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl'
        })

    except Exception as e:
        return jsonify({'error': f'Failed to list models: {str(e)}'}), 500


if __name__ == '__main__':
    logger.info("🚀 Starting Ditto Talking Head API Server...")

    # Initialize SDK
    if not initialize_sdk():
        logger.error("❌ Failed to initialize SDK. Server will not start.")
        exit(1)

    # Start Flask server
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
    )