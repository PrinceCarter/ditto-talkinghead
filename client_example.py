#!/usr/bin/env python3
"""
Example client for Ditto Talking Head API Server
"""

import requests
import json
import time
import os
from pathlib import Path


class DittoClient:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url

    def health_check(self):
        """Check if the API server is healthy"""
        try:
            response = requests.get(f'{self.base_url}/')
            return response.json()
        except requests.RequestException as e:
            return {'error': str(e)}

    def generate_video(self, audio_path, image_path, config=None):
        """
        Generate a talking head video

        Args:
            audio_path (str): Path to audio file (wav, mp3, m4a)
            image_path (str): Path to source image (png, jpg, jpeg)
            config (dict): Optional configuration overrides

        Returns:
            dict: Response from server
        """
        files = {
            'audio': open(audio_path, 'rb'),
            'image': open(image_path, 'rb')
        }

        data = {}
        if config:
            data['config'] = json.dumps(config)

        try:
            response = requests.post(
                f'{self.base_url}/generate',
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout
            )
            return response.json()

        except requests.RequestException as e:
            return {'error': str(e)}

        finally:
            # Close file handles
            for file in files.values():
                file.close()

    def get_status(self, session_id):
        """Get status of video generation"""
        try:
            response = requests.get(f'{self.base_url}/status/{session_id}')
            return response.json()
        except requests.RequestException as e:
            return {'error': str(e)}

    def download_video(self, filename, save_path=None):
        """Download generated video"""
        try:
            response = requests.get(f'{self.base_url}/download/{filename}', stream=True)

            if response.status_code == 200:
                # Determine save path
                if save_path is None:
                    save_path = f'./downloads/{filename}'

                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Save file
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                return {'success': True, 'saved_to': save_path}
            else:
                return {'error': f'Download failed: {response.status_code}'}

        except requests.RequestException as e:
            return {'error': str(e)}

    def list_models(self):
        """List available models"""
        try:
            response = requests.get(f'{self.base_url}/models')
            return response.json()
        except requests.RequestException as e:
            return {'error': str(e)}


def main():
    """Example usage"""
    # Initialize client
    client = DittoClient()

    # Check health
    print("🔍 Checking server health...")
    health = client.health_check()
    print(f"Health: {health}")

    if not health.get('sdk_initialized'):
        print("❌ SDK not initialized on server")
        return

    # List available models
    print("\n📋 Available models:")
    models = client.list_models()
    print(json.dumps(models, indent=2))

    # Check if example files exist
    audio_path = './example/audio.wav'
    image_path = './example/image.png'

    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return

    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return

    # Generate video
    print(f"\n🎬 Generating video...")
    print(f"Audio: {audio_path}")
    print(f"Image: {image_path}")

    # Optional configuration
    config = {
        "run_kwargs": {
            "chunksize": (3, 5, 2),
            "fade_in": 5,
            "fade_out": 5
        }
    }

    result = client.generate_video(audio_path, image_path, config)
    print(f"Generation result: {result}")

    if result.get('success'):
        session_id = result['session_id']
        output_filename = result['output_filename']

        print(f"\n✅ Video generated successfully!")
        print(f"Session ID: {session_id}")
        print(f"Output filename: {output_filename}")

        # Download the video
        print(f"\n⬇️  Downloading video...")
        download_result = client.download_video(output_filename)
        print(f"Download result: {download_result}")

        if download_result.get('success'):
            print(f"✅ Video saved to: {download_result['saved_to']}")
        else:
            print(f"❌ Download failed: {download_result}")

    else:
        print(f"❌ Video generation failed: {result}")


if __name__ == '__main__':
    main()