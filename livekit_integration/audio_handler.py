from __future__ import annotations

import asyncio
import numpy as np
from typing import TYPE_CHECKING, Optional

from livekit.agents.voice.io import AudioOutput
from livekit.agents import tts
from livekit import rtc

from .log import logger

if TYPE_CHECKING:
    from .stream_sdk_livekit import StreamSDKLiveKit


class DittoAudioPublisher:
    """Publishes audio to LiveKit room while processing through Ditto"""
    
    def __init__(self, room: rtc.Room, participant_identity: str, sample_rate: int = 16000):
        self._room = room  
        self._participant_identity = participant_identity
        self._sample_rate = sample_rate
        self._audio_source: Optional[rtc.AudioSource] = None
        self._audio_track: Optional[rtc.LocalAudioTrack] = None
        self._running = False
        
    async def start(self):
        """Start audio publishing"""
        try:
            # Create audio source and track
            self._audio_source = rtc.AudioSource(self._sample_rate, 1)  # 16kHz mono
            self._audio_track = rtc.LocalAudioTrack.create_audio_track(
                "ditto-voice", self._audio_source
            )
            
            # Publish the audio track
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            
            publication = await self._room.local_participant.publish_track(
                self._audio_track, options
            )
            
            self._running = True
            logger.info(f"Published audio track: {publication.sid}")
            
        except Exception as e:
            logger.error(f"Failed to start audio publisher: {e}")
            raise
    
    async def stop(self):
        """Stop audio publishing"""
        self._running = False
        
        if self._audio_track:
            await self._room.local_participant.unpublish_track(self._audio_track.sid)
            self._audio_track = None
            
        if self._audio_source:
            self._audio_source = None
            
        logger.info("Audio publisher stopped")
    
    async def publish_audio(self, audio_data: np.ndarray):
        """Publish audio frame to LiveKit"""
        if not self._running or not self._audio_source:
            return
            
        try:
            # Convert float32 to int16
            if audio_data.dtype == np.float32:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)
            
            # Create AudioFrame
            audio_frame = rtc.AudioFrame(
                data=audio_int16.tobytes(),
                sample_rate=self._sample_rate,
                num_channels=1,
                samples_per_channel=len(audio_int16)
            )
            
            # Publish to room
            await self._audio_source.capture_frame(audio_frame)
            
        except Exception as e:
            logger.error(f"Error publishing audio frame: {e}")


class DittoDataStreamAudioOutput(AudioOutput):
    """Audio output handler that streams audio to Ditto for avatar generation AND publishes to LiveKit"""
    
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        stream_sdk: StreamSDKLiveKit,
        audio_publisher: DittoAudioPublisher,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            label="ditto-audio-output",
        )
        self._stream_sdk = stream_sdk
        self._audio_publisher = audio_publisher
        self._audio_buffer = np.array([], dtype=np.float32)
        # Ditto expects: int(sum((3,5,2)) * 0.04 * 16000) + 80 = 6480 samples
        self._chunk_size = 6480  # Correct chunk size for Ditto TensorRT model
        
    async def aplay(
        self,
        data: rtc.AudioFrame,
        *,
        volume: float = 1.0,
    ) -> None:
        """Process audio frame and stream to Ditto for avatar generation AND publish to LiveKit"""
        try:
            # Process the audio with Ditto directly
            # Convert memoryview to numpy array first
            audio_np = np.frombuffer(data.data, dtype=np.int16)
            audio_data = audio_np.astype(np.float32) / 32768.0  # Convert int16 to float32
            
            # Ensure correct sample rate
            if data.sample_rate != self._sample_rate:
                logger.warning(f"Audio sample rate mismatch: expected {self._sample_rate}, got {data.sample_rate}")
                return
            
            # Apply volume
            if volume != 1.0:
                audio_data = audio_data * volume
            
            logger.info(f"=== TTS AUDIO RECEIVED: {len(audio_data)} samples ===")
            
            # FIRST: Publish the original audio to LiveKit room so users hear the voice
            await self._audio_publisher.publish_audio(audio_data)
            logger.info(f"Published {len(audio_data)} audio samples to LiveKit room")
            
            # SECOND: Add to buffer for Ditto processing (lip sync)
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_data])
            logger.info(f"Audio buffer now: {len(self._audio_buffer)} samples")
            
            # Process chunks for Ditto lip sync
            while len(self._audio_buffer) >= self._chunk_size:
                chunk = self._audio_buffer[:self._chunk_size]
                self._audio_buffer = self._audio_buffer[self._chunk_size:]
                
                # Send chunk to Ditto for lip sync processing
                logger.info(f"Processing audio chunk of size {len(chunk)} for lip sync")
                await self._stream_sdk.process_audio_chunk(chunk)
                
        except Exception as e:
            logger.error(f"CRITICAL ERROR in audio processing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Don't re-raise - let the speech generation complete normally
    
    async def aflush(self) -> None:
        """Flush remaining audio data"""
        try:
            logger.info("=== TTS SEGMENT COMPLETE - FLUSHING BUFFER ===")
            # Flush our buffer
            if len(self._audio_buffer) > 0:
                logger.info(f"Flushing remaining audio buffer of size {len(self._audio_buffer)}")
                await self._stream_sdk.process_audio_chunk(self._audio_buffer)
                self._audio_buffer = np.array([], dtype=np.float32)
            logger.info("=== AUDIO FLUSH COMPLETE ===")
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR flushing audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Don't re-raise - let the speech generation complete normally
    
    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Required abstract method - must call super() to track segments"""
        # IMPORTANT: Call super() to properly track playback segments
        await super().capture_frame(frame)
        
        # Now process our audio
        await self.aplay(frame)
    
    def clear_buffer(self) -> None:
        """Clear the audio buffer"""
        try:
            logger.info("=== CLEARING AUDIO BUFFER ===")
            self._audio_buffer = np.array([], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error clearing buffer: {e}")
            # Don't raise exceptions in sync methods
    
    def flush(self) -> None:
        """Mark end of current audio segment - sync version"""
        try:
            logger.info("=== SYNC FLUSH CALLED ===")
            
            # Call super() to properly track segment completion
            super().flush()
            
            # Clear the buffer to prepare for next audio segment
            self._audio_buffer = np.array([], dtype=np.float32)
            
            # CRITICAL: Notify that playback is finished for this segment
            # This is what fixes the speech generation lifecycle
            self.on_playback_finished(
                playback_position=0.0,  # We don't track position
                interrupted=False,
            )
            logger.info("=== NOTIFIED PLAYBACK FINISHED ===")
            
        except Exception as e:
            logger.error(f"Error in sync flush: {e}")
            # Don't raise exceptions in sync methods


class AudioChunkProcessor:
    """Helper class to handle audio chunk processing for Ditto"""
    
    def __init__(self, stream_sdk: StreamSDKLiveKit):
        self._stream_sdk = stream_sdk
        self._chunk_queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the audio processing task"""
        self._running = True
        self._processing_task = asyncio.create_task(self._process_audio_chunks())
    
    async def stop(self):
        """Stop the audio processing"""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
    
    async def add_chunk(self, audio_chunk: np.ndarray):
        """Add audio chunk to processing queue"""
        if self._running:
            await self._chunk_queue.put(audio_chunk)
    
    async def _process_audio_chunks(self):
        """Process audio chunks from the queue"""
        while self._running:
            try:
                # Wait for audio chunk with timeout
                audio_chunk = await asyncio.wait_for(
                    self._chunk_queue.get(), 
                    timeout=1.0
                )
                
                # Process the chunk with Ditto
                await self._stream_sdk.process_audio_chunk(audio_chunk)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in audio chunk processor: {e}")
                if self._running:
                    await asyncio.sleep(0.1)


class DittoTTSInterceptor(tts.TTS):
    """TTS wrapper that intercepts audio and sends it to Ditto for avatar generation"""
    
    def __init__(
        self,
        original_tts: tts.TTS,
        stream_sdk: "StreamSDKLiveKit", 
        audio_publisher: DittoAudioPublisher,
    ):
        super().__init__(
            capabilities=original_tts.capabilities,
            sample_rate=original_tts.sample_rate,
            num_channels=original_tts.num_channels,
        )
        self._original_tts = original_tts
        self._stream_sdk = stream_sdk
        self._audio_publisher = audio_publisher
        self._chunk_size = 6480  # Ditto chunk size
        
    def synthesize(self, text: str) -> tts.ChunkedStream:
        """Synthesize speech and intercept the audio for Ditto processing"""
        logger.info(f"=== TTS SYNTHESIZING: {text} ===")
        
        # Get the original TTS stream
        original_stream = self._original_tts.synthesize(text)
        
        # Return our intercepting stream
        return DittoChunkedStream(
            original_stream=original_stream,
            stream_sdk=self._stream_sdk,
            audio_publisher=self._audio_publisher,
            chunk_size=self._chunk_size,
        )


class DittoChunkedStream(tts.ChunkedStream):
    """ChunkedStream that intercepts audio frames for Ditto processing"""
    
    def __init__(
        self,
        original_stream: tts.ChunkedStream,
        stream_sdk: "StreamSDKLiveKit",
        audio_publisher: DittoAudioPublisher, 
        chunk_size: int,
    ):
        self._original_stream = original_stream
        self._stream_sdk = stream_sdk
        self._audio_publisher = audio_publisher
        self._chunk_size = chunk_size
        self._audio_buffer = np.array([], dtype=np.float32)
        
    async def collect(self) -> rtc.AudioFrame:
        """Collect the complete audio and process it through Ditto"""
        logger.info("=== COLLECTING TTS AUDIO ===")
        
        # Get the original audio frame
        audio_frame = await self._original_stream.collect()
        
        # Process the audio through our pipeline
        await self._process_audio_frame(audio_frame)
        
        return audio_frame
    
    async def _process_audio_frame(self, audio_frame: rtc.AudioFrame):
        """Process audio frame through Ditto and publish to LiveKit"""
        try:
            # Convert audio frame to numpy array
            audio_np = np.frombuffer(audio_frame.data, dtype=np.int16)
            audio_data = audio_np.astype(np.float32) / 32768.0  # Convert to float32
            
            logger.info(f"=== TTS AUDIO RECEIVED: {len(audio_data)} samples ===")
            
            # FIRST: Publish the original audio to LiveKit room so users hear the voice
            await self._audio_publisher.publish_audio(audio_data)
            logger.info(f"Published {len(audio_data)} audio samples to LiveKit room")
            
            # SECOND: Add to buffer for Ditto processing (lip sync)
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_data])
            logger.info(f"Audio buffer now: {len(self._audio_buffer)} samples")
            
            # Process chunks for Ditto lip sync
            while len(self._audio_buffer) >= self._chunk_size:
                chunk = self._audio_buffer[:self._chunk_size]
                self._audio_buffer = self._audio_buffer[self._chunk_size:]
                
                # Send chunk to Ditto for lip sync processing
                logger.info(f"Processing audio chunk of size {len(chunk)} for lip sync")
                await self._stream_sdk.process_audio_chunk(chunk)
            
            # Process any remaining buffer at the end
            if len(self._audio_buffer) > 0:
                logger.info(f"Processing final audio chunk of size {len(self._audio_buffer)}")
                await self._stream_sdk.process_audio_chunk(self._audio_buffer)
                self._audio_buffer = np.array([], dtype=np.float32)
                
            logger.info("=== TTS AUDIO PROCESSING COMPLETE ===")
                
        except Exception as e:
            logger.error(f"CRITICAL ERROR processing TTS audio: {e}")
            import traceback
            logger.error(traceback.format_exc())