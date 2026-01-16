"""
Enhanced RTSP Stream Handler with GStreamer support
Based on the ML team's aircraft_det.py demo script
Provides:
- GStreamer backend for better performance
- FFmpeg fallback
- Improved error recovery and reconnection
- Thread-safe frame buffering
"""
import cv2
import threading
import time
import logging
import numpy as np
from typing import Optional, Tuple
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class RTSPStreamHandler:
    """
    Enhanced RTSP stream handler with GStreamer support and improved error recovery
    
    Features:
    - GStreamer backend for better RTSP performance
    - FFmpeg fallback
    - Default OpenCV backend as last resort
    - Thread-safe frame buffering
    - Automatic reconnection
    """
    
    def __init__(
        self,
        rtsp_url: str,
        buffer_size: int = 10,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 10,
        use_gstreamer: bool = True,
    ):
        """
        Initialize stream handler
        
        Args:
            rtsp_url: RTSP stream URL
            buffer_size: Maximum frames to buffer
            reconnect_delay: Delay between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts
            use_gstreamer: Try GStreamer backend first
        """
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.use_gstreamer = use_gstreamer
        
        self.frame_queue: Queue = Queue(maxsize=buffer_size)
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.last_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.capture_thread: Optional[threading.Thread] = None
        
        # Error tracking
        self.error_count = 0
        self.max_errors = 50
        self.consecutive_failures = 0
        self.reconnect_count = 0
        
        # Stream properties
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.last_frame_time = 0
        self.backend_used = "none"
        
        logger.info(f"RTSPStreamHandler initialized for {rtsp_url}")
    
    def start(self) -> bool:
        """Start the stream capture thread"""
        if self.is_running:
            return True
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        logger.info(f"Stream handler started for {self.rtsp_url}")
        return True
    
    def stop(self):
        """Stop the stream capture"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        logger.info(f"Stream handler stopped for {self.rtsp_url}")
    
    def _create_capture_with_gstreamer(self) -> Optional[cv2.VideoCapture]:
        """Create capture using GStreamer pipeline for better performance"""
        if not self.use_gstreamer:
            return None
            
        try:
            gst_pipeline = (
                f'rtspsrc location={self.rtsp_url} latency=0 buffer-mode=0 ! '
                'rtph264depay ! h264parse ! avdec_h264 ! '
                'videoconvert ! appsink drop=true sync=false'
            )
            
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                logger.info("Successfully connected using GStreamer")
                self.backend_used = "gstreamer"
                return cap
            else:
                logger.warning("GStreamer failed, falling back to FFmpeg")
                return None
        except Exception as e:
            logger.debug(f"GStreamer not available: {e}")
            return None
    
    def _create_capture_with_ffmpeg(self) -> Optional[cv2.VideoCapture]:
        """Create capture using FFmpeg backend with optimized settings"""
        try:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 25)
                
                logger.info("Successfully connected using FFmpeg backend")
                self.backend_used = "ffmpeg"
                return cap
            else:
                return None
        except Exception as e:
            logger.error(f"FFmpeg backend failed: {e}")
            return None
    
    def _create_capture_default(self) -> Optional[cv2.VideoCapture]:
        """Create capture using default OpenCV backend"""
        try:
            cap = cv2.VideoCapture(self.rtsp_url)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 25)
                logger.info("Connected using default OpenCV backend")
                self.backend_used = "opencv"
                return cap
        except Exception as e:
            logger.error(f"Default backend failed: {e}")
        return None
    
    def _connect(self) -> bool:
        """Connect to RTSP stream with multiple fallback options"""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Try GStreamer first
        self.cap = self._create_capture_with_gstreamer()
        if self.cap and self.cap.isOpened():
            self._update_stream_properties()
            return True
        
        # Try FFmpeg backend
        self.cap = self._create_capture_with_ffmpeg()
        if self.cap and self.cap.isOpened():
            self._update_stream_properties()
            return True
        
        # Try default OpenCV backend
        self.cap = self._create_capture_default()
        if self.cap and self.cap.isOpened():
            self._update_stream_properties()
            return True
        
        return False
    
    def _update_stream_properties(self):
        """Update stream properties from capture"""
        if self.cap and self.cap.isOpened():
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            logger.info(f"Stream properties: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
    
    def _capture_frames(self):
        """Continuously capture frames from RTSP stream with improved error handling"""
        frame_skip_counter = 0
        skip_frames = 1  # Process every frame by default
        
        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.info("Attempting to connect to RTSP stream...")
                    if self._connect():
                        logger.info("Successfully connected to RTSP stream")
                        self.consecutive_failures = 0
                        self.error_count = 0
                        self.reconnect_count = 0
                    else:
                        self.reconnect_count += 1
                        if self.reconnect_count > self.max_reconnect_attempts:
                            logger.error(f"Max reconnection attempts reached for {self.rtsp_url}")
                            break
                        logger.warning(f"Failed to connect. Retrying in {self.reconnect_delay}s...")
                        time.sleep(self.reconnect_delay)
                        continue
                
                # Grab frame
                ret = self.cap.grab()
                
                if not ret:
                    self.consecutive_failures += 1
                    if self.consecutive_failures > 30:
                        logger.warning("Too many grab failures. Reconnecting...")
                        self._connect()
                        self.consecutive_failures = 0
                    continue
                
                # Frame skipping for performance
                frame_skip_counter += 1
                if frame_skip_counter % skip_frames != 0:
                    continue
                
                # Retrieve frame
                ret, frame = self.cap.retrieve()
                
                if ret and frame is not None:
                    self.consecutive_failures = 0
                    
                    if frame.shape[0] > 0 and frame.shape[1] > 0:
                        with self.frame_lock:
                            self.last_frame = frame.copy()
                        
                        # Clear old frames if queue is full
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except Empty:
                                pass
                        
                        self.frame_queue.put(frame)
                        self.last_frame_time = time.time()
                    else:
                        logger.warning("Invalid frame dimensions")
                
                else:
                    self.consecutive_failures += 1
                    self.error_count += 1
                    
                    if self.error_count > self.max_errors:
                        logger.warning("Max errors reached. Reconnecting...")
                        self._connect()
                        self.error_count = 0
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                self.consecutive_failures += 1
                if self.consecutive_failures > 10:
                    time.sleep(1)
        
        logger.info(f"Capture loop ended for {self.rtsp_url}")
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the latest frame from the buffer
        
        Returns:
            Tuple of (success, frame)
        """
        try:
            frame = self.frame_queue.get(timeout=0.05)
            return True, frame
        except Empty:
            with self.frame_lock:
                if self.last_frame is not None:
                    return True, self.last_frame.copy()
            return False, None
    
    def get_frame_nowait(self) -> Optional[np.ndarray]:
        """Get the latest frame without waiting"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            with self.frame_lock:
                if self.last_frame is not None:
                    return self.last_frame.copy()
            return None
    
    def is_connected(self) -> bool:
        """Check if the stream is connected"""
        return self.is_running and self.cap is not None and self.cap.isOpened()
    
    def get_info(self) -> dict:
        """Get stream information"""
        return {
            'source': self.rtsp_url,
            'running': self.is_running,
            'connected': self.is_connected(),
            'width': self.frame_width,
            'height': self.frame_height,
            'fps': self.fps,
            'backend': self.backend_used,
            'reconnect_count': self.reconnect_count,
            'error_count': self.error_count,
            'last_frame_time': self.last_frame_time,
            'buffer_size': self.frame_queue.qsize(),
        }


# Alias for backwards compatibility
class StreamHandler(RTSPStreamHandler):
    """
    Backwards compatible alias for RTSPStreamHandler
    """
    
    def __init__(
        self,
        source: str,
        buffer_size: int = 2,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 10,
    ):
        super().__init__(
            rtsp_url=source,
            buffer_size=buffer_size,
            reconnect_delay=reconnect_delay,
            max_reconnect_attempts=max_reconnect_attempts,
        )
    
    def check_running(self) -> bool:
        """Check if the stream handler is running (backwards compatible)"""
        return self.is_connected()
