#!/usr/bin/env python3
"""
Test script to verify Ditto setup and functionality
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required packages can be imported"""
    logger.info("🔍 Testing imports...")

    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__}")

        import librosa
        logger.info(f"✅ librosa {librosa.__version__}")

        import numpy as np
        logger.info(f"✅ NumPy {np.__version__}")

        import pickle
        logger.info(f"✅ pickle")

        # Test Flask imports
        import flask
        logger.info(f"✅ Flask {flask.__version__}")

        return True

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    logger.info("🔍 Testing CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.version.cuda}")
            logger.info(f"✅ GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  - {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            logger.warning("⚠️  CUDA not available")
            return False

    except Exception as e:
        logger.error(f"❌ CUDA test failed: {e}")
        return False

def test_file_structure():
    """Test if required files and directories exist"""
    logger.info("🔍 Testing file structure...")

    required_files = [
        './example/audio.wav',
        './example/image.png',
        './stream_pipeline_offline.py',
        './inference.py',
    ]

    required_dirs = [
        './example',
        './core',
        './scripts',
    ]

    all_good = True

    for file_path in required_files:
        if Path(file_path).exists():
            logger.info(f"✅ {file_path}")
        else:
            logger.error(f"❌ Missing: {file_path}")
            all_good = False

    for dir_path in required_dirs:
        if Path(dir_path).exists():
            logger.info(f"✅ {dir_path}/")
        else:
            logger.error(f"❌ Missing: {dir_path}/")
            all_good = False

    return all_good

def test_models():
    """Test if model files exist"""
    logger.info("🔍 Testing model files...")

    model_paths = [
        './checkpoints/ditto_trt_Ampere_Plus',
        './checkpoints/ditto_cfg',
        './checkpoints/ditto_onnx',
    ]

    all_good = True

    for model_path in model_paths:
        if Path(model_path).exists():
            logger.info(f"✅ {model_path}")
            # Count files in directory
            if Path(model_path).is_dir():
                file_count = len(list(Path(model_path).iterdir()))
                logger.info(f"   Contains {file_count} files")
        else:
            logger.warning(f"⚠️  Missing: {model_path}")
            all_good = False

    return all_good

def test_basic_inference():
    """Test basic inference functionality"""
    logger.info("🔍 Testing basic inference import...")

    try:
        from stream_pipeline_offline import StreamSDK
        logger.info("✅ StreamSDK import successful")

        # Try to initialize SDK (this will fail if models are missing, but that's ok)
        cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
        data_root = "./checkpoints/ditto_trt_Ampere_Plus"

        if Path(cfg_pkl).exists() and Path(data_root).exists():
            try:
                SDK = StreamSDK(cfg_pkl, data_root)
                logger.info("✅ SDK initialization successful")
                return True
            except Exception as e:
                logger.warning(f"⚠️  SDK initialization failed: {e}")
                logger.warning("   This is expected if models haven't been downloaded yet")
                return False
        else:
            logger.warning("⚠️  Model files not found - run setup_api.sh first")
            return False

    except ImportError as e:
        logger.error(f"❌ StreamSDK import failed: {e}")
        return False

def test_api_server():
    """Test API server components"""
    logger.info("🔍 Testing API server components...")

    try:
        # Test Flask imports
        from flask import Flask, request, jsonify, send_file
        logger.info("✅ Flask components")

        # Check if API server file exists and has basic structure
        api_server_path = Path('./api_server.py')
        if api_server_path.exists():
            logger.info("✅ api_server.py exists")

            # Check file size (should be substantial)
            file_size = api_server_path.stat().st_size
            if file_size > 1000:
                logger.info(f"✅ API server file size: {file_size} bytes")
            else:
                logger.warning(f"⚠️  API server file seems small: {file_size} bytes")
        else:
            logger.error("❌ api_server.py not found")
            return False

        # Test client example
        client_path = Path('./client_example.py')
        if client_path.exists():
            logger.info("✅ client_example.py exists")
        else:
            logger.warning("⚠️  client_example.py not found")

        return True

    except ImportError as e:
        logger.error(f"❌ API server test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Ditto setup verification...")

    tests = [
        ("Package Imports", test_imports),
        ("CUDA Support", test_cuda),
        ("File Structure", test_file_structure),
        ("Model Files", test_models),
        ("Basic Inference", test_basic_inference),
        ("API Server", test_api_server),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")

        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY")
    logger.info(f"{'='*50}")

    passed = 0
    total = len(tests)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        if result:
            passed += 1

    logger.info(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Your setup is ready.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. See details above.")

        # Provide guidance
        if not results.get("Model Files", True):
            logger.info("💡 To download models: ./setup_api.sh")

        if not results.get("CUDA Support", True):
            logger.info("💡 Ensure you're running on a GPU-enabled environment")

        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)