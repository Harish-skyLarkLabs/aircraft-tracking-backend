"""
Stream handler for RTSP/video streams
"""
import cv2
import threading
import time
import logging
import numpy as np
from typing import Optional
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class StreamHandler:
    """Handle video stream capture in a separate thread"""
    
    def __init__(
        self,
        source: str,
        buffer_size: int = 2,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 10,
    ):
        """
        Initialize stream handler
        
        Args:
            source: Video source (RTSP URL, file path, or camera index)
            buffer_size: Maximum frames to buffer
            reconnect_delay: Delay between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts before giving up
        """
        self.source = source
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue: Queue = Queue(maxsize=buffer_size)
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.reconnect_count = 0
        self.last_frame_time = 0
    
    def start(self) -> bool:
        """Start the stream capture"""
        if self.running:
            return True
        
        # Try to open the stream
        if not self._open_stream():
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info(f"Stream handler started for {self.source}")
        return True
    
    def stop(self):
        """Stop the stream capture"""
        self.running = False
        
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
        
        logger.info(f"Stream handler stopped for {self.source}")
    
    def _open_stream(self) -> bool:
        """Open the video stream"""
        try:
            # Release existing capture if any
            if self.cap:
                self.cap.release()
            
            # Open new capture
            self.cap = cv2.VideoCapture(self.source)
            
            # Set buffer size to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open stream: {self.source}")
                return False
            
            # Get stream properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            
            logger.info(f"Opened stream: {self.source} ({self.frame_width}x{self.frame_height} @ {self.fps}fps)")
            return True
            
        except Exception as e:
            logger.error(f"Error opening stream {self.source}: {e}")
            return False
    
    def _capture_loop(self):
        """Main capture loop running in a separate thread"""
        consecutive_failures = 0
        
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if not self._reconnect():
                        break
                    continue
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame from {self.source} ({consecutive_failures} failures)")
                    
                    if consecutive_failures >= 10:
                        if not self._reconnect():
                            break
                        consecutive_failures = 0
                    
                    time.sleep(0.01)
                    continue
                
                consecutive_failures = 0
                self.last_frame_time = time.time()
                
                # Add frame to queue, dropping old frames if full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                
                self.frame_queue.put(frame)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
        
        logger.info(f"Capture loop ended for {self.source}")
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to the stream"""
        self.reconnect_count += 1
        
        if self.reconnect_count > self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts reached for {self.source}")
            return False
        
        logger.info(f"Attempting to reconnect to {self.source} (attempt {self.reconnect_count})")
        
        time.sleep(self.reconnect_delay)
        
        if self._open_stream():
            self.reconnect_count = 0
            return True
        
        return True  # Keep trying
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the buffer
        
        Returns:
            numpy.ndarray: The latest frame, or None if no frame available
        """
        try:
            return self.frame_queue.get(timeout=0.1)
        except Empty:
            return None
    
    def get_frame_nowait(self) -> Optional[np.ndarray]:
        """
        Get the latest frame without waiting
        
        Returns:
            numpy.ndarray: The latest frame, or None if no frame available
        """
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def is_running(self) -> bool:
        """Check if the stream handler is running"""
        return self.running and self.cap is not None and self.cap.isOpened()
    
    def get_info(self) -> dict:
        """Get stream information"""
        return {
            'source': self.source,
            'running': self.running,
            'width': self.frame_width,
            'height': self.frame_height,
            'fps': self.fps,
            'reconnect_count': self.reconnect_count,
            'last_frame_time': self.last_frame_time,
            'buffer_size': self.frame_queue.qsize(),
        }


