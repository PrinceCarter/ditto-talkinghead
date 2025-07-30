from __future__ import annotations

import asyncio
import numpy as np
from typing import Optional
from PIL import Image

from livekit import rtc

from .log import logger


class LiveKitVideoPublisher:
    """Publishes video frames from Ditto to LiveKit room"""
    
    def __init__(
        self,
        room: rtc.Room,
        participant_identity: str,
        width: int = 512,
        height: int = 512,
        fps: int = 25,
    ):
        self._room = room
        self._participant_identity = participant_identity
        self._width = width
        self._height = height
        self._fps = fps
        self._video_track: Optional[rtc.LocalVideoTrack] = None
        self._video_source: Optional[rtc.VideoSource] = None
        self._frame_queue = asyncio.Queue(maxsize=2)  # Smaller queue for lower latency
        self._publishing_task: Optional[asyncio.Task] = None
        self._running = False
        self._idle_frame: Optional[np.ndarray] = None
        self._has_active_frames = False
        self._last_frame_time = 0.0
        self._last_publish_time = 0.0
        
        # Frame smoothing for choppy video
        self._frame_buffer = []  # Buffer of recent frames for interpolation
        self._max_buffer_size = 5
        self._adaptive_fps = fps  # Can be adjusted based on performance
        
    async def start(self):
        """Start video publishing"""
        try:
            # Create idle frame - a simple static image
            self._create_idle_frame()
            
            # Create video source and track (ONCE - never recreate during operation)
            self._video_source = rtc.VideoSource(self._width, self._height)
            self._video_track = rtc.LocalVideoTrack.create_video_track(
                "ditto-avatar", self._video_source
            )
            
            # Publish the video track
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_CAMERA
            
            publication = await self._room.local_participant.publish_track(
                self._video_track, options
            )
            
            logger.info(f"Video track published: {publication.sid}")
            
            # Start continuous frame publishing task
            self._running = True
            self._publishing_task = asyncio.create_task(self._publish_frames())
            
        except Exception as e:
            logger.error(f"Failed to start video publisher: {e}")
            raise
    
    async def stop(self):
        """Stop video publishing"""
        self._running = False
        
        if self._publishing_task:
            self._publishing_task.cancel()
            try:
                await self._publishing_task
            except asyncio.CancelledError:
                pass
        
        if self._video_track:
            try:
                await self._room.local_participant.unpublish_track(self._video_track.sid)
            except:
                pass
            self._video_track = None
        
        self._video_source = None
            
        logger.info("Video publisher stopped")
    
    def _create_idle_frame(self):
        """Create a simple idle frame to show when no avatar video is active"""
        try:
            # Create a simple gradient or solid color frame
            idle_frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            
            # Create a nice gradient or pattern
            # You can replace this with loading an actual image file
            for y in range(self._height):
                for x in range(self._width):
                    # Simple radial gradient from center
                    center_x, center_y = self._width // 2, self._height // 2
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    max_distance = np.sqrt(center_x**2 + center_y**2)
                    intensity = int(50 + (205 * (1 - distance / max_distance)))
                    idle_frame[y, x] = [intensity//3, intensity//2, intensity]  # Blue-ish gradient
            
            self._idle_frame = idle_frame
            logger.info(f"Created idle frame: {self._idle_frame.shape}")
            
        except Exception as e:
            # Fallback to solid black frame
            logger.warning(f"Failed to create gradient idle frame: {e}, using solid black")
            self._idle_frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
    
    def _add_to_buffer(self, frame: np.ndarray):
        """Add frame to buffer for smoothing"""
        self._frame_buffer.append(frame.copy())
        if len(self._frame_buffer) > self._max_buffer_size:
            self._frame_buffer.pop(0)  # Remove oldest frame
    
    def _get_smoothed_frame(self) -> Optional[np.ndarray]:
        """Get a smoothed frame using simple averaging"""
        if not self._frame_buffer:
            return None
        
        if len(self._frame_buffer) == 1:
            return self._frame_buffer[0].copy()
        
        # Simple frame blending for smoother video
        # Use the two most recent frames
        frame1 = self._frame_buffer[-1].astype(np.float32)
        frame2 = self._frame_buffer[-2].astype(np.float32) if len(self._frame_buffer) > 1 else frame1
        
        # Blend 70% current frame, 30% previous frame for smoothness
        blended = (0.7 * frame1 + 0.3 * frame2).astype(np.uint8)
        return blended

    async def publish_frame(self, frame_rgb: np.ndarray):
        """Add active avatar frame to queue with smoothing"""
        if not self._running:
            logger.warning("Video publisher not running, dropping frame")
            return
            
        try:
            # Convert frame format if needed
            if frame_rgb.dtype != np.uint8:
                frame_rgb = (frame_rgb * 255).astype(np.uint8)
            
            # Ensure correct dimensions
            if frame_rgb.shape[:2] != (self._height, self._width):
                logger.warning(f"Frame size mismatch: expected ({self._height}, {self._width}), got {frame_rgb.shape[:2]}")
                return
            
            # Add to frame buffer for smoothing
            self._add_to_buffer(frame_rgb)
            
            # Get smoothed frame
            smoothed_frame = self._get_smoothed_frame()
            if smoothed_frame is None:
                smoothed_frame = frame_rgb
            
            # Mark that we have active frames
            if not self._has_active_frames:
                logger.info("=== SWITCHING TO ACTIVE MODE - AVATAR FRAMES AVAILABLE ===")
            self._has_active_frames = True
            import time
            self._last_frame_time = time.time()
            
            # Add to queue (non-blocking, drop if full for low latency)
            try:
                self._frame_queue.put_nowait(smoothed_frame)
                logger.debug(f"Smoothed frame queued, queue size: {self._frame_queue.qsize()}")
            except asyncio.QueueFull:
                # Drop ALL old frames and add new one for lowest latency
                while not self._frame_queue.empty():
                    try:
                        self._frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                self._frame_queue.put_nowait(smoothed_frame)
                logger.debug("Dropped old frames for low latency")
                    
        except Exception as e:
            logger.error(f"Error publishing frame: {e}")
    
    async def _publish_frames(self):
        """Continuously publish frames - either active avatar frames or idle frames"""
        target_fps = self._fps
        frame_interval = 1.0 / target_fps
        idle_timeout = 1.0  # Shorter timeout to switch to idle faster
        
        # Frame interpolation variables for smoother playback
        last_active_frame = None
        frame_hold_count = 0
        max_frame_hold = 3  # Hold frames longer for smoother appearance
        
        logger.info(f"Starting continuous video stream at {target_fps} FPS...")
        
        while self._running:
            try:
                import time
                current_time = time.time()
                
                # More flexible frame timing - allow slight variations
                if self._last_publish_time > 0:
                    time_since_last = current_time - self._last_publish_time
                    target_sleep = frame_interval - time_since_last
                    if target_sleep > 0.001:  # Only sleep if meaningful time left
                        await asyncio.sleep(min(target_sleep, frame_interval * 0.8))
                        current_time = time.time()
                
                frame_to_send = None
                
                # Try to get the LATEST active frame from queue (skip old ones)
                latest_frame = None
                frames_skipped = 0
                while True:
                    try:
                        latest_frame = self._frame_queue.get_nowait()
                        if frame_to_send is not None:
                            frames_skipped += 1
                        frame_to_send = latest_frame
                    except asyncio.QueueEmpty:
                        break
                
                if frames_skipped > 0:
                    logger.debug(f"Skipped {frames_skipped} old frames for smooth playback")
                
                if frame_to_send is not None:
                    # New active frame available
                    last_active_frame = frame_to_send.copy()
                    frame_hold_count = 0
                    logger.debug("Using latest active avatar frame")
                elif last_active_frame is not None and frame_hold_count < max_frame_hold:
                    # Hold the last active frame for smoother transitions
                    frame_to_send = last_active_frame
                    frame_hold_count += 1
                    logger.debug(f"Holding last active frame ({frame_hold_count}/{max_frame_hold})")
                else:
                    # Switch to idle mode
                    if (not self._has_active_frames or 
                        (current_time - self._last_frame_time) > idle_timeout):
                        if self._has_active_frames:
                            logger.info("=== SWITCHING TO IDLE MODE - NO ACTIVE FRAMES ===")
                            self._has_active_frames = False
                        frame_to_send = self._idle_frame.copy()
                        logger.debug("Using idle frame")
                
                # Send the frame (either active or idle) to video source
                if frame_to_send is not None and self._video_source is not None:
                    try:
                        # Create video frame using correct API
                        frame = rtc.VideoFrame(
                            self._width,
                            self._height,
                            rtc.VideoBufferType.RGB24,
                            frame_to_send.tobytes()
                        )
                        
                        # Capture frame to video source with safety check
                        capture_result = self._video_source.capture_frame(frame)
                        if capture_result is not None:
                            await capture_result
                        self._last_publish_time = current_time
                        
                    except Exception as e:
                        logger.error(f"Failed to capture frame: {e}")
                        # Don't continue, just skip this iteration
                        await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in continuous video streaming: {e}")
                if self._running:
                    await asyncio.sleep(0.01)  # Shorter sleep on error


class DittoVideoWriter:
    """Custom video writer that publishes frames to LiveKit instead of saving to file"""
    
    def __init__(self, video_publisher: LiveKitVideoPublisher):
        self._video_publisher = video_publisher
        self._frame_count = 0
        
    def __call__(self, frame_rgb: np.ndarray, fmt: str = "rgb"):
        """Write frame (publish to LiveKit) - robust and smooth"""
        # Log every 10th frame for monitoring
        if self._frame_count % 10 == 0:
            logger.info(f"DittoVideoWriter: Frame {self._frame_count}, shape: {frame_rgb.shape}")
        
        # Robust frame publishing with proper async handling
        try:
            loop = asyncio.get_running_loop()
            # Use create_task for proper async handling
            loop.create_task(self._publish_frame_safe(frame_rgb, fmt))
        except RuntimeError:
            # No event loop - create one temporarily
            logger.warning("No event loop running, creating temporary loop")
            try:
                # Create a new event loop for this frame
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                new_loop.run_until_complete(self._publish_frame_safe(frame_rgb, fmt))
                new_loop.close()
            except Exception as e:
                logger.error(f"Failed to publish frame with new loop: {e}")
        
        self._frame_count += 1
    
    async def _publish_frame_safe(self, frame_rgb: np.ndarray, fmt: str):
        """Safe async frame publishing with error handling"""
        try:
            if fmt == "rgb" and self._video_publisher is not None:
                await self._video_publisher.publish_frame(frame_rgb)
        except Exception as e:
            # Log error but don't crash
            if self._frame_count % 10 == 0:  # Only log errors occasionally
                logger.error(f"Error publishing frame {self._frame_count}: {e}")
    
    async def _publish_frame_ultra_fast(self, frame_rgb: np.ndarray, fmt: str):
        """Ultra-fast async frame publishing with minimal overhead"""
        if fmt == "rgb":
            # Direct call without error handling for maximum speed
            await self._video_publisher.publish_frame(frame_rgb)
    
    async def _publish_frame_fast(self, frame_rgb: np.ndarray, fmt: str):
        """Fast async frame publishing with minimal logging"""
        try:
            if fmt == "rgb":
                await self._video_publisher.publish_frame(frame_rgb)
            # Don't log every frame - too much overhead
        except Exception as e:
            if self._frame_count % 10 == 0:  # Only log errors occasionally
                logger.error(f"Error publishing frame {self._frame_count-1}: {e}")
    
    async def _publish_frame(self, frame_rgb: np.ndarray, fmt: str):
        """Async frame publishing (legacy method)"""
        await self._publish_frame_fast(frame_rgb, fmt)
    
    def close(self):
        """Close the writer"""
        logger.info(f"DittoVideoWriter closed. Total frames: {self._frame_count}")
        # Note: actual cleanup is handled by the video publisher