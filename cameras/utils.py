"""
Utility functions for cameras app
"""
import cv2
import logging
import threading
import time
from typing import Generator, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class MJPEGStreamManager:
    """
    Manages MJPEG streams for multiple cameras.
    Uses a singleton pattern to share streams across requests.
    """
    _instance: Optional['MJPEGStreamManager'] = None
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
        
        self._streams: dict = {}  # camera_id -> stream info
        self._stream_locks: dict = defaultdict(threading.Lock)
        self._client_counts: dict = defaultdict(int)
        self._initialized = True
        logger.info("MJPEGStreamManager initialized")
    
    def _create_stream(self, camera_id: str, rtsp_link: str) -> bool:
        """Create a new stream for a camera."""
        try:
            cap = cv2.VideoCapture(rtsp_link)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                logger.error(f"Failed to open stream for camera {camera_id}")
                return False
            
            self._streams[camera_id] = {
                'cap': cap,
                'rtsp_link': rtsp_link,
                'last_frame': None,
                'last_frame_time': 0,
                'running': True,
            }
            
            # Start frame capture thread
            thread = threading.Thread(
                target=self._capture_frames,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            self._streams[camera_id]['thread'] = thread
            
            logger.info(f"Started stream for camera {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating stream for camera {camera_id}: {e}")
            return False
    
    def _capture_frames(self, camera_id: str):
        """Background thread to capture frames from RTSP stream."""
        stream_info = self._streams.get(camera_id)
        if not stream_info:
            return
        
        cap = stream_info['cap']
        target_fps = 15  # Limit to 15 FPS for MJPEG
        frame_interval = 1.0 / target_fps
        
        while stream_info.get('running', False):
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Resize frame if too large (max 720p for streaming)
                    h, w = frame.shape[:2]
                    if w > 1280:
                        scale = 1280 / w
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    
                    stream_info['last_frame'] = frame
                    stream_info['last_frame_time'] = time.time()
                else:
                    # Try to reconnect
                    logger.warning(f"Lost connection to camera {camera_id}, reconnecting...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(stream_info['rtsp_link'])
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    stream_info['cap'] = cap
                
                time.sleep(frame_interval)
                
            except Exception as e:
                logger.error(f"Error capturing frame for camera {camera_id}: {e}")
                time.sleep(1)
        
        cap.release()
        logger.info(f"Stopped capture thread for camera {camera_id}")
    
    def get_stream(self, camera_id: str, rtsp_link: str) -> Generator[bytes, None, None]:
        """
        Get MJPEG stream generator for a camera.
        Creates stream if not exists.
        """
        with self._stream_locks[camera_id]:
            if camera_id not in self._streams:
                if not self._create_stream(camera_id, rtsp_link):
                    return
            self._client_counts[camera_id] += 1
        
        try:
            stream_info = self._streams.get(camera_id)
            if not stream_info:
                return
            
            while stream_info.get('running', False):
                frame = stream_info.get('last_frame')
                if frame is not None:
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_bytes = buffer.tobytes()
                    
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                    )
                
                time.sleep(0.066)  # ~15 FPS
                
        finally:
            with self._stream_locks[camera_id]:
                self._client_counts[camera_id] -= 1
                
                # Stop stream if no more clients
                if self._client_counts[camera_id] <= 0:
                    self._stop_stream(camera_id)
    
    def _stop_stream(self, camera_id: str):
        """Stop a stream for a camera."""
        stream_info = self._streams.pop(camera_id, None)
        if stream_info:
            stream_info['running'] = False
            if stream_info.get('cap'):
                stream_info['cap'].release()
            logger.info(f"Stopped stream for camera {camera_id}")
        
        self._client_counts.pop(camera_id, None)
    
    def stop_all_streams(self):
        """Stop all active streams."""
        for camera_id in list(self._streams.keys()):
            self._stop_stream(camera_id)
    
    def is_streaming(self, camera_id: str) -> bool:
        """Check if a camera is currently streaming."""
        return camera_id in self._streams and self._streams[camera_id].get('running', False)
    
    def get_client_count(self, camera_id: str) -> int:
        """Get the number of clients watching a stream."""
        return self._client_counts.get(camera_id, 0)


# Singleton instance
mjpeg_manager = MJPEGStreamManager()


def check_rtsp(rtsp_link: str, timeout: int = 10):
    """
    Check if an RTSP stream is accessible and return a thumbnail frame.
    
    Args:
        rtsp_link: The RTSP URL to check
        timeout: Connection timeout in seconds
        
    Returns:
        numpy.ndarray: The captured frame if successful, None otherwise
    """
    if not rtsp_link:
        return None

    try:
        cap = cv2.VideoCapture(rtsp_link)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            logger.warning(f"Failed to open RTSP stream: {rtsp_link}")
            return None

        # Try to read a frame
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            return frame
        else:
            logger.warning(f"Failed to read frame from RTSP stream: {rtsp_link}")
            return None

    except Exception as e:
        logger.error(f"Error checking RTSP stream {rtsp_link}: {str(e)}")
        return None


def get_stream_info(rtsp_link: str):
    """
    Get information about an RTSP stream.
    
    Args:
        rtsp_link: The RTSP URL to check
        
    Returns:
        dict: Stream information including resolution and FPS
    """
    if not rtsp_link:
        return None

    try:
        cap = cv2.VideoCapture(rtsp_link)
        
        if not cap.isOpened():
            return None

        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        
        cap.release()
        return info

    except Exception as e:
        logger.error(f"Error getting stream info for {rtsp_link}: {str(e)}")
        return None

