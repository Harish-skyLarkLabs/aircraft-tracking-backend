"""
LiveKit Publisher for backend-rendered video with AI overlays.

This module provides an alternative approach where the backend draws
detection bounding boxes on video frames and publishes the processed
video to LiveKit. This ensures perfect synchronization between video
and detection overlays at the cost of higher backend CPU usage.

Use this approach when:
- Perfect detection-video sync is critical
- Frontend overlay rendering is insufficient
- You need consistent overlay rendering across all clients

Note: This is optional. The default approach uses frontend overlay
rendering which is more scalable.

Usage:
    publisher = LiveKitPublisher(camera_id, rtsp_url, camera_name)
    await publisher.start()
    # ... process frames and publish ...
    await publisher.stop()
"""
import os
import asyncio
import logging
import threading
import time
import cv2
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check if LiveKit SDK is available
try:
    from livekit import rtc
    from livekit.rtc import Room, VideoSource, VideoFrame, LocalVideoTrack
    LIVEKIT_RTC_AVAILABLE = True
except ImportError:
    LIVEKIT_RTC_AVAILABLE = False
    logger.warning("LiveKit RTC SDK not installed. Run: pip install livekit")


@dataclass
class PublisherConfig:
    """Configuration for LiveKit publisher"""
    livekit_url: str = "ws://localhost:7880"
    api_key: str = "devkey"
    api_secret: str = "secret"
    video_width: int = 1280
    video_height: int = 720
    fps: int = 15
    
    @classmethod
    def from_env(cls) -> 'PublisherConfig':
        """Load configuration from environment variables"""
        return cls(
            livekit_url=os.getenv("LIVEKIT_URL", "ws://localhost:7880"),
            api_key=os.getenv("LIVEKIT_API_KEY", "devkey"),
            api_secret=os.getenv("LIVEKIT_API_SECRET", "secret"),
            video_width=int(os.getenv("LIVEKIT_VIDEO_WIDTH", "1280")),
            video_height=int(os.getenv("LIVEKIT_VIDEO_HEIGHT", "720")),
            fps=int(os.getenv("LIVEKIT_FPS", "15")),
        )


class LiveKitPublisher:
    """
    Publishes processed video frames with AI overlays to LiveKit.
    
    This class handles:
    - Connecting to LiveKit as a publisher
    - Converting OpenCV frames to LiveKit video frames
    - Publishing frames at a consistent rate
    - Drawing detection overlays on frames before publishing
    """
    
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        camera_name: str = "",
        config: Optional[PublisherConfig] = None,
    ):
        """
        Initialize LiveKit publisher.
        
        Args:
            camera_id: Unique camera identifier
            rtsp_url: RTSP stream URL (for reference, actual frames come from processor)
            camera_name: Display name for the camera
            config: Publisher configuration
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name or f"Camera {camera_id}"
        self.config = config or PublisherConfig.from_env()
        
        self.room: Optional[Room] = None
        self.video_source: Optional[VideoSource] = None
        self.video_track: Optional[LocalVideoTrack] = None
        
        self.is_running = False
        self.is_connected = False
        self._lock = threading.Lock()
        
        self._enabled = LIVEKIT_RTC_AVAILABLE and os.getenv("USE_LIVEKIT", "false").lower() == "true"
    
    @property
    def is_enabled(self) -> bool:
        """Check if LiveKit publishing is enabled"""
        return self._enabled
    
    async def start(self) -> bool:
        """
        Start the LiveKit publisher.
        
        Returns:
            True if started successfully, False otherwise
        """
        if not self.is_enabled:
            logger.warning("LiveKit publishing is not enabled")
            return False
        
        if self.is_running:
            logger.warning(f"Publisher for camera {self.camera_id} is already running")
            return True
        
        try:
            # Generate publisher token
            from .livekit_client import livekit_client
            
            room_name = f"camera-{self.camera_id}"
            
            # Create access token for publisher (using new chained API)
            from livekit import api
            
            token = (
                api.AccessToken(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                )
                .with_identity(f"publisher-{self.camera_id}")
                .with_name(self.camera_name)
                .with_grants(api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=False,
                ))
            )
            
            jwt_token = token.to_jwt()
            
            # Create room and connect
            self.room = Room()
            
            # Connect to room
            await self.room.connect(self.config.livekit_url, jwt_token)
            
            # Create video source and track
            self.video_source = VideoSource(
                width=self.config.video_width,
                height=self.config.video_height,
            )
            
            self.video_track = LocalVideoTrack.create_video_track(
                "camera",
                self.video_source,
            )
            
            # Publish track
            await self.room.local_participant.publish_track(self.video_track)
            
            self.is_running = True
            self.is_connected = True
            
            logger.info(f"LiveKit publisher started for camera {self.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start LiveKit publisher: {e}")
            await self.stop()
            return False
    
    async def stop(self):
        """Stop the LiveKit publisher"""
        self.is_running = False
        
        try:
            if self.video_track:
                await self.room.local_participant.unpublish_track(self.video_track)
                self.video_track = None
            
            if self.room:
                await self.room.disconnect()
                self.room = None
            
            self.video_source = None
            self.is_connected = False
            
            logger.info(f"LiveKit publisher stopped for camera {self.camera_id}")
            
        except Exception as e:
            logger.error(f"Error stopping LiveKit publisher: {e}")
    
    def publish_frame(
        self,
        frame: np.ndarray,
        detections: list = None,
        draw_overlay: bool = True,
    ):
        """
        Publish a video frame to LiveKit.
        
        Args:
            frame: OpenCV frame (BGR format)
            detections: List of detection dictionaries
            draw_overlay: Whether to draw detection overlays
        """
        if not self.is_running or not self.video_source:
            return
        
        try:
            # Draw detection overlays if requested
            if draw_overlay and detections:
                frame = self._draw_detections(frame, detections)
            
            # Resize frame if needed
            if frame.shape[1] != self.config.video_width or frame.shape[0] != self.config.video_height:
                frame = cv2.resize(frame, (self.config.video_width, self.config.video_height))
            
            # Convert BGR to RGBA
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            
            # Create LiveKit video frame
            video_frame = VideoFrame(
                width=self.config.video_width,
                height=self.config.video_height,
                type=rtc.VideoBufferType.RGBA,
                data=frame_rgba.tobytes(),
            )
            
            # Capture frame to video source
            self.video_source.capture_frame(video_frame)
            
        except Exception as e:
            logger.error(f"Error publishing frame: {e}")
    
    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list,
    ) -> np.ndarray:
        """
        Draw detection bounding boxes and labels on frame.
        
        Args:
            frame: OpenCV frame
            detections: List of detection dictionaries
            
        Returns:
            Frame with overlays drawn
        """
        frame = frame.copy()
        
        for det in detections:
            # Extract bounding box
            bbox = det.get('bbox', det.get('box', []))
            if len(bbox) < 4:
                continue
            
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Get detection info
            track_id = det.get('track_id', det.get('id', -1))
            confidence = det.get('confidence', det.get('conf', 0))
            class_name = det.get('class_name', det.get('label', 'Object'))
            action = det.get('action', '')
            
            # Choose color based on action
            if action == 'landing':
                color = (0, 255, 0)  # Green
            elif action == 'taking_off':
                color = (0, 165, 255)  # Orange
            elif action == 'hovering':
                color = (255, 255, 0)  # Cyan
            else:
                color = (0, 255, 0)  # Default green
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Build label
            label = f"#{track_id} {class_name}"
            if confidence > 0:
                label += f" {confidence*100:.0f}%"
            if action:
                label += f" [{action}]"
            
            # Draw label background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width + 10, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
        
        return frame
    
    def get_status(self) -> Dict[str, Any]:
        """Get publisher status"""
        return {
            "camera_id": self.camera_id,
            "is_running": self.is_running,
            "is_connected": self.is_connected,
            "config": {
                "video_width": self.config.video_width,
                "video_height": self.config.video_height,
                "fps": self.config.fps,
            },
        }


class LiveKitPublisherManager:
    """
    Manager for multiple LiveKit publishers.
    
    Provides a centralized way to manage publishers for multiple cameras.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.publishers: Dict[str, LiveKitPublisher] = {}
        self._lock = threading.Lock()
    
    async def start_publisher(
        self,
        camera_id: str,
        rtsp_url: str,
        camera_name: str = "",
    ) -> Optional[LiveKitPublisher]:
        """
        Start a publisher for a camera.
        
        Args:
            camera_id: Camera identifier
            rtsp_url: RTSP stream URL
            camera_name: Camera display name
            
        Returns:
            LiveKitPublisher instance if successful, None otherwise
        """
        with self._lock:
            if camera_id in self.publishers:
                return self.publishers[camera_id]
            
            publisher = LiveKitPublisher(camera_id, rtsp_url, camera_name)
            
            if await publisher.start():
                self.publishers[camera_id] = publisher
                return publisher
            
            return None
    
    async def stop_publisher(self, camera_id: str) -> bool:
        """
        Stop a publisher for a camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            True if stopped successfully
        """
        with self._lock:
            publisher = self.publishers.pop(camera_id, None)
            
            if publisher:
                await publisher.stop()
                return True
            
            return False
    
    def get_publisher(self, camera_id: str) -> Optional[LiveKitPublisher]:
        """Get a publisher by camera ID"""
        return self.publishers.get(camera_id)
    
    def publish_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        detections: list = None,
    ):
        """
        Publish a frame for a camera.
        
        Args:
            camera_id: Camera identifier
            frame: OpenCV frame
            detections: Detection list
        """
        publisher = self.publishers.get(camera_id)
        if publisher:
            publisher.publish_frame(frame, detections)
    
    async def stop_all(self):
        """Stop all publishers"""
        camera_ids = list(self.publishers.keys())
        for camera_id in camera_ids:
            await self.stop_publisher(camera_id)
        
        logger.info("Stopped all LiveKit publishers")


# Singleton instance
publisher_manager = LiveKitPublisherManager()

