#!/usr/bin/env python3
"""
Ditto + LiveKit Integration Example
Based on Simli avatar example pattern
"""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, WorkerType, cli
from livekit.plugins import openai
# from livekit.plugins import deepgram, elevenlabs

from livekit_integration import DittoAvatarSession, DittoConfig

logger = logging.getLogger("ditto-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    """Main entrypoint following LiveKit avatar pattern"""
    
    # IMPORTANT: Connect to room first before accessing local_participant
    await ctx.connect()
    
    # Create AgentSession with your preferred STT/LLM/TTS
    session = AgentSession(
        # Option 1: Use OpenAI Realtime (like Simli example)
        llm=openai.realtime.RealtimeModel(voice="alloy"),
        
        # Option 2: Use separate STT/LLM/TTS (uncomment to use)
        # stt=deepgram.STT(),
        # llm=openai.LLM(),  
        # tts=elevenlabs.TTS(),
    )
    
    # Configure Ditto avatar
    ditto_config = DittoConfig(
        data_root="./checkpoints/ditto_trt_Ampere_Plus",
        cfg_pkl="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
        avatar_image_path="./example/avatar.png",
        online_mode=True,
        max_session_length=600,
        max_idle_time=30,
    )
    
    # Create and start avatar session
    ditto_avatar = DittoAvatarSession(ditto_config=ditto_config)
    await ditto_avatar.start(session, room=ctx.room)
    
    # Start the agent, it will join the room and wait for the avatar to join
    await session.start(
        agent=Agent(instructions="You are a helpful AI assistant with a Ditto avatar. Keep responses conversational and engaging!"),
        room=ctx.room,
    )


def check_setup():
    """Check if all required files and environment variables are present"""
    logger.info("Checking setup...")
    
    # Check if required files exist
    required_files = [
        "./checkpoints/ditto_trt_Ampere_Plus",
        "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl", 
        "./example/avatar.png",
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Required file not found: {file_path}")
            logger.info("Please run setup_runpod.sh first")
            return False
    
    logger.info("✅ All required files found")
    
    # Check environment variables
    required_env = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "OPENAI_API_KEY"]
    missing_env = []
    for env_var in required_env:
        if not os.getenv(env_var):
            missing_env.append(env_var)
    
    if missing_env:
        logger.error(f"Missing environment variables: {', '.join(missing_env)}")
        return False
    
    logger.info("✅ Environment variables configured")
    logger.info("Ready to run Ditto LiveKit agent!")
    return True


if __name__ == "__main__":
    if not check_setup():
        exit(1)
        
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint, 
            worker_type=WorkerType.ROOM
        )
    )