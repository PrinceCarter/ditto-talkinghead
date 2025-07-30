from __future__ import annotations

import asyncio
import threading
import queue
import numpy as np
import traceback
from typing import Optional

from livekit import rtc

from stream_pipeline_online import StreamSDK
from .video_publisher import LiveKitVideoPublisher, DittoVideoWriter
from .audio_handler import DittoAudioPublisher
from .log import logger


class StreamSDKLiveKit(StreamSDK):
    """Modified StreamSDK for LiveKit integration"""
    
    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        room: rtc.Room,
        avatar_participant_identity: str,
        livekit_token: str,
        livekit_url: str,
        **kwargs
    ):
        # Initialize base StreamSDK
        super().__init__(cfg_pkl, data_root, **kwargs)
        
        self._main_room = room  # The main room
        self._avatar_participant_identity = avatar_participant_identity
        self._livekit_token = livekit_token
        self._livekit_url = livekit_url
        
        # Avatar will connect as separate participant
        self._avatar_room: Optional[rtc.Room] = None
        self._video_publisher: Optional[LiveKitVideoPublisher] = None
        self._audio_receiver = None
        self._is_setup = False
        
        # Frame buffering for smoother playback
        self._frame_buffer_queue = asyncio.Queue(maxsize=10)
        self._last_generated_frame = None
        
    async def setup_avatar(
        self,
        source_path: str,
        online_mode: bool = True,
        crop_scale: float = 2.3,
        emotion: int = 4,
        **kwargs
    ):
        """Setup avatar for LiveKit streaming"""
        try:
            # Setup parameters
            setup_kwargs = {
                'online_mode': online_mode,
                'crop_scale': crop_scale,
                'emo': emotion,
                **kwargs
            }
            
            # Use a temporary output path (won't actually be used)
            temp_output_path = "/tmp/ditto_livekit_temp.mp4"
            
            # Setup the base StreamSDK
            self.setup(source_path, temp_output_path, **setup_kwargs)
            
            # Connect avatar participant to room
            logger.info("Connecting avatar participant to room...")
            self._avatar_room = rtc.Room()
            await self._avatar_room.connect(self._livekit_url, self._livekit_token)
            logger.info("Avatar participant connected to room")
            
            # Set up audio receiver to get TTS audio from main agent
            from livekit.agents.voice.avatar import DataStreamAudioReceiver
            self._audio_receiver = DataStreamAudioReceiver(
                room=self._avatar_room,
                # No sender_identity specified = receive from any agent
            )
            await self._audio_receiver.start()
            logger.info("Audio receiver started")
            
            # Set up event handler for received audio
            @self._audio_receiver.on("audio_frame_received")
            def on_audio_frame(frame: rtc.AudioFrame):
                # Process audio through Ditto
                asyncio.create_task(self._process_received_audio(frame))
            
            # Create video publisher using avatar room
            self._video_publisher = LiveKitVideoPublisher(
                room=self._avatar_room,
                participant_identity=self._avatar_participant_identity,
                width=1432,  # Actual Ditto output size
                height=1432,
                fps=30  # Higher FPS for smoother video
            )
            
            # Start video publisher
            await self._video_publisher.start()
            
            # Replace the writer with our LiveKit video writer
            self.writer = DittoVideoWriter(self._video_publisher)
            
            self._is_setup = True
            logger.info("Avatar setup completed for LiveKit streaming")
            
        except Exception as e:
            logger.error(f"Failed to setup avatar: {e}")
            raise
    
    async def _process_received_audio(self, audio_frame: rtc.AudioFrame):
        """Process audio frame received from DataStreamAudioReceiver"""
        try:
            # Convert audio frame to numpy array
            audio_np = np.frombuffer(audio_frame.data, dtype=np.int16) 
            audio_data = audio_np.astype(np.float32) / 32768.0  # Convert to float32
            
            logger.info(f"=== RECEIVED TTS AUDIO: {len(audio_data)} samples ===")
            
            # Process through Ditto in chunks
            chunk_size = 6480  # Ditto chunk size
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                
                # Pad if needed
                if len(chunk) < chunk_size:
                    padding = np.zeros(chunk_size - len(chunk), dtype=np.float32)
                    chunk = np.concatenate([chunk, padding])
                
                logger.info(f"Processing audio chunk of size {len(chunk)} for lip sync")
                await self.process_audio_chunk(chunk)
                
        except Exception as e:
            logger.error(f"Error processing received audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _check_system_health(self):
        """Check if all Ditto worker threads are alive"""
        if not hasattr(self, 'thread_list'):
            return False
            
        dead_threads = []
        for i, thread in enumerate(self.thread_list):
            if not thread.is_alive():
                dead_threads.append(i)
        
        if dead_threads:
            logger.error(f"Dead worker threads detected: {dead_threads}")
            return False
            
        if hasattr(self, 'stop_event') and self.stop_event.is_set():
            logger.error("Stop event is set - system is stopped")
            return False
            
        if hasattr(self, 'worker_exception') and self.worker_exception is not None:
            logger.error(f"Worker exception present: {self.worker_exception}")
            return False
            
        return True

    async def process_audio_chunk(self, audio_chunk: np.ndarray):
        """Process audio chunk for avatar generation with optimizations for smooth video"""
        if not self._is_setup:
            logger.warning("Avatar not setup, ignoring audio chunk")
            return
            
        try:
            # Skip health checks for faster processing - only check every 10th chunk
            if not hasattr(self, '_chunk_counter'):
                self._chunk_counter = 0
            self._chunk_counter += 1
            
            if self._chunk_counter % 10 == 0:  # Only check health every 10 chunks
                if not self._check_system_health():
                    await self._restart_ditto_system()
                    if not self._check_system_health():
                        return
            
            # Ensure audio is in the right format
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Ditto expects 16kHz audio with specific chunk size
            expected_chunk_size = 6480  # int(sum((3,5,2)) * 0.04 * 16000) + 80
            if len(audio_chunk) < expected_chunk_size:
                padding = np.zeros(expected_chunk_size - len(audio_chunk), dtype=np.float32)
                audio_chunk = np.concatenate([audio_chunk, padding])
            elif len(audio_chunk) > expected_chunk_size:
                # Truncate if too large
                audio_chunk = audio_chunk[:expected_chunk_size]
            
            # Use optimal chunksize for fastest inference while maintaining quality
            chunksize = (1, 2, 1)  # Much smaller chunks = much faster processing
            
            # Run chunk processing in thread with reduced timeout for responsiveness
            if self._chunk_counter % 20 == 0:  # Log less frequently
                logger.info(f"Processing audio chunk {self._chunk_counter} with shape: {audio_chunk.shape}")
            
            loop = asyncio.get_event_loop()
            
            # Reduced timeout for more responsive processing
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self.run_chunk, audio_chunk, chunksize),
                    timeout=5.0  # Reduced from 10s to 5s for faster response
                )
                if self._chunk_counter % 20 == 0:  # Log less frequently
                    logger.info(f"Audio chunk {self._chunk_counter} processing completed")
            except asyncio.TimeoutError:
                logger.error(f"Audio chunk {self._chunk_counter} processing timed out!")
                # Don't restart system on timeout - just skip this chunk for smoother video
                return
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Try to restart the system if it failed
            try:
                await self._restart_ditto_system()
            except Exception as restart_error:
                logger.error(f"Failed to restart Ditto system: {restart_error}")
    
    def get_audio_publisher(self):
        """No longer needed - audio comes via DataStreamAudioReceiver"""
        return None
    
    async def _restart_ditto_system(self):
        """Restart the Ditto system workers and state"""
        try:
            logger.info("Restarting Ditto system...")
            
            # Clear all error states
            if hasattr(self, 'stop_event'):
                self.stop_event.clear()
                
            if hasattr(self, 'worker_exception'):
                self.worker_exception = None
            
            # Clear all queues to start fresh
            import queue
            queue_names = ['audio2motion_queue', 'motion_stitch_queue', 'warp_f3d_queue', 
                          'decode_f3d_queue', 'putback_queue', 'writer_queue']
            
            for queue_name in queue_names:
                if hasattr(self, queue_name):
                    q = getattr(self, queue_name)
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except:
                            break
            
            # Restart worker threads
            import threading
            if hasattr(self, 'thread_list'):
                worker_methods = [
                    self.audio2motion_worker,
                    self.motion_stitch_worker, 
                    self.warp_f3d_worker,
                    self.decode_f3d_worker,
                    self.putback_worker,
                    self.writer_worker
                ]
                
                for i, thread in enumerate(self.thread_list):
                    if not thread.is_alive() and i < len(worker_methods):
                        new_thread = threading.Thread(target=worker_methods[i])
                        new_thread.daemon = True
                        new_thread.start()
                        self.thread_list[i] = new_thread
            
            await asyncio.sleep(0.2)  # Brief pause for stability
            logger.info("Ditto system restarted")
            
        except Exception as e:
            logger.error(f"Error restarting Ditto system: {e}")
            raise
    
    async def reset_session(self):
        """Reset the avatar session for a new conversation"""
        try:
            logger.info("Resetting avatar session...")
            
            # Reset the base StreamSDK state
            if hasattr(self, 'stop_event'):
                self.stop_event.clear()
            
            # Don't recreate video publisher - just clear frame queue
            if self._video_publisher and self._video_publisher._frame_queue:
                # Clear any pending frames
                while not self._video_publisher._frame_queue.empty():
                    try:
                        self._video_publisher._frame_queue.get_nowait()
                    except:
                        break
                logger.info("Cleared video frame queue")
            
            logger.info("Avatar session reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting session: {e}")
            
    async def stop(self):
        """Stop the LiveKit streaming"""
        try:
            # Stop audio receiver
            if self._audio_receiver:
                await self._audio_receiver.aclose()
                self._audio_receiver = None
            
            # Stop video publishing
            if self._video_publisher:
                await self._video_publisher.stop()
                self._video_publisher = None
            
            # Disconnect avatar room
            if self._avatar_room:
                await self._avatar_room.disconnect()
                self._avatar_room = None
            
            # Stop worker threads if needed
            if hasattr(self, 'stop_event'):
                self.stop_event.set()
            
            logger.info("StreamSDK LiveKit stopped")
            
        except Exception as e:
            logger.error(f"Error stopping StreamSDK: {e}")
    
    def _writer_worker(self):
        """Override writer worker to publish frames to LiveKit with buffering"""
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.writer_queue.get(timeout=1)
                except queue.Empty:
                    # If no new frames, repeat last frame for smoothness
                    if self._last_generated_frame is not None:
                        self.writer(self._last_generated_frame, fmt="rgb")
                    continue

                if item is None:
                    break
                    
                res_frame_rgb = item
                self._last_generated_frame = res_frame_rgb.copy()  # Save for repetition
                
                # Add to buffer queue (non-blocking)
                try:
                    self._frame_buffer_queue.put_nowait(res_frame_rgb)
                except asyncio.QueueFull:
                    # If buffer full, remove oldest and add new
                    try:
                        self._frame_buffer_queue.get_nowait()
                        self._frame_buffer_queue.put_nowait(res_frame_rgb)
                    except:
                        pass
                
                # Use our custom LiveKit writer
                self.writer(res_frame_rgb, fmt="rgb")
                self.writer_pbar.update()
                
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()
    
    def setup_Nd(self, N_d, fade_in=-1, fade_out=-1, ctrl_info=None):
        """Override to handle infinite streaming"""
        # For LiveKit streaming, we don't know the final length
        # Set a large number for continuous streaming
        if N_d <= 0:
            N_d = 999999  # Very large number for continuous streaming
        
        super().setup_Nd(N_d, fade_in, fade_out, ctrl_info)


class AsyncStreamWrapper:
    """Wrapper to make StreamSDK work with async/await"""
    
    def __init__(self, stream_sdk: StreamSDKLiveKit):
        self._stream_sdk = stream_sdk
        self._executor = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stream_sdk.stop()
    
    async def process_audio(self, audio_data: np.ndarray):
        """Process audio data asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._stream_sdk.process_audio_chunk,
            audio_data
        )