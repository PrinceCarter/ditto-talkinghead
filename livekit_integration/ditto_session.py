from __future__ import annotations

import os
import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional

from livekit import api, rtc
from livekit.agents import (
    NOT_GIVEN,
    AgentSession,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .log import logger
from .stream_sdk_livekit import StreamSDKLiveKit

SAMPLE_RATE = 16000
_AVATAR_AGENT_IDENTITY = "ditto-avatar-agent"
_AVATAR_AGENT_NAME = "ditto-avatar-agent"


@dataclass
class DittoConfig:
    """
    Args:
        data_root (str): Path to TensorRT models directory
        cfg_pkl (str): Path to configuration pickle file
        avatar_image_path (str): Path to source avatar image
        max_session_length (int): Maximum session duration in seconds
        max_idle_time (int): Maximum idle time before disconnection
        online_mode (bool): Enable streaming mode for real-time processing
        crop_scale (float): Crop scale for face detection
        emotion (int): Emotion ID for avatar expression
    """
    data_root: str
    cfg_pkl: str
    avatar_image_path: str
    max_session_length: int = 600
    max_idle_time: int = 30
    online_mode: bool = True
    crop_scale: float = 2.3
    emotion: int = 4


class DittoAvatarSession:
    """A Ditto avatar session for LiveKit"""

    def __init__(
        self,
        *,
        ditto_config: DittoConfig,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        self._ditto_config = ditto_config
        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._stream_sdk: Optional[StreamSDKLiveKit] = None
        self._room: Optional[rtc.Room] = None

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise Exception(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        try:
            job_ctx = get_job_context()
            local_participant_identity = job_ctx.token_claims().identity
        except RuntimeError as e:
            if not room.isconnected():
                raise Exception("failed to get local participant identity") from e
            local_participant_identity = room.local_participant.identity

        # Create LiveKit token for avatar participant
        livekit_token = (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})
            .to_jwt()
        )

        logger.debug("starting ditto avatar session")
        
        # Initialize Ditto StreamSDK with LiveKit integration
        self._stream_sdk = StreamSDKLiveKit(
            cfg_pkl=self._ditto_config.cfg_pkl,
            data_root=self._ditto_config.data_root,
            room=room,
            avatar_participant_identity=self._avatar_participant_identity,
            livekit_token=livekit_token,
            livekit_url=livekit_url,
        )
        
        self._room = room
        
        # Setup avatar with source image
        await self._stream_sdk.setup_avatar(
            source_path=self._ditto_config.avatar_image_path,
            online_mode=self._ditto_config.online_mode,
            crop_scale=self._ditto_config.crop_scale,
            emotion=self._ditto_config.emotion,
        )
        
        # Set up audio streaming from AgentSession TTS to avatar (Simli pattern)
        from livekit.agents.voice.avatar import DataStreamAudioOutput
        
        logger.info("Setting up DataStreamAudioOutput to stream TTS to avatar participant")
        
        # Stream TTS audio to the avatar participant (like Simli does)
        agent_session.output.audio = DataStreamAudioOutput(
            room=room,
            destination_identity=self._avatar_participant_identity,
            sample_rate=SAMPLE_RATE,
        )
        
        logger.info(f"Audio output configured to stream to avatar participant: {self._avatar_participant_identity}")
        
        logger.info("Ditto avatar session started successfully")

    async def stop(self) -> None:
        """Stop the avatar session"""
        if self._stream_sdk:
            await self._stream_sdk.stop()
            self._stream_sdk = None
        logger.debug("Ditto avatar session stopped")