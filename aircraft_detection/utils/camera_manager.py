"""
Camera Manager for handling multiple camera streams and processing
"""
import cv2
import threading
import time
import logging
import numpy as np
from typing import Dict, Optional, Set
from uuid import UUID
from collections import defaultdict

from .frame_processor import FrameProcessor
from .stream_handler import StreamHandler

logger = logging.getLogger(__name__)


class CameraManager:
    """Singleton manager for handling multiple camera streams"""
    
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
        self.cameras: Dict[str, Dict] = {}
        self.processors: Dict[str, FrameProcessor] = {}
        self.stream_handlers: Dict[str, StreamHandler] = {}
        self.clients: Dict[str, Set[str]] = defaultdict(set)
        self.latest_frames: Dict[str, Dict] = {}
        self.frame_ids: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        
        logger.info("CameraManager initialized")
    
    def initialize(self):
        """Initialize the camera manager (called on first use)"""
        pass  # Already initialized in __init__
    
    def start_camera_processing(
        self,
        camera_id: UUID,
        rtsp_link: str,
        camera_name: str = "",
        roi_points: Optional[list] = None,
    ) -> bool:
        """
        Start processing for a camera
        
        Args:
            camera_id: Camera UUID
            rtsp_link: RTSP stream URL
            camera_name: Optional camera name
            roi_points: Optional ROI polygon points
            
        Returns:
            bool: True if started successfully
        """
        camera_id_str = str(camera_id)
        
        with self._lock:
            # Check if already processing
            if camera_id_str in self.stream_handlers:
                logger.warning(f"Camera {camera_id_str} is already being processed")
                return True
            
            try:
                # Create stream handler
                stream_handler = StreamHandler(rtsp_link)
                if not stream_handler.start():
                    logger.error(f"Failed to start stream handler for camera {camera_id_str}")
                    return False
                
                # Create frame processor
                processor = FrameProcessor(
                    camera_id=camera_id_str,
                    camera_name=camera_name,
                    roi_points=roi_points,
                )
                
                # Store references
                self.stream_handlers[camera_id_str] = stream_handler
                self.processors[camera_id_str] = processor
                self.cameras[camera_id_str] = {
                    'rtsp_link': rtsp_link,
                    'camera_name': camera_name,
                    'started_at': time.time(),
                }
                
                # Start frame processing thread
                processing_thread = threading.Thread(
                    target=self._process_camera_frames,
                    args=(camera_id_str,),
                    daemon=True
                )
                processing_thread.start()
                
                logger.info(f"Started processing for camera {camera_id_str}")
                return True
                
            except Exception as e:
                logger.error(f"Error starting camera processing: {e}")
                return False
    
    def stop_camera_processing(self, camera_id: UUID) -> bool:
        """
        Stop processing for a camera
        
        Args:
            camera_id: Camera UUID
            
        Returns:
            bool: True if stopped successfully
        """
        camera_id_str = str(camera_id)
        
        with self._lock:
            if camera_id_str not in self.stream_handlers:
                logger.warning(f"Camera {camera_id_str} is not being processed")
                return True
            
            try:
                # Stop stream handler
                if camera_id_str in self.stream_handlers:
                    self.stream_handlers[camera_id_str].stop()
                    del self.stream_handlers[camera_id_str]
                
                # Clean up processor
                if camera_id_str in self.processors:
                    self.processors[camera_id_str].cleanup()
                    del self.processors[camera_id_str]
                
                # Clean up other data
                if camera_id_str in self.cameras:
                    del self.cameras[camera_id_str]
                if camera_id_str in self.latest_frames:
                    del self.latest_frames[camera_id_str]
                if camera_id_str in self.frame_ids:
                    del self.frame_ids[camera_id_str]
                
                logger.info(f"Stopped processing for camera {camera_id_str}")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping camera processing: {e}")
                return False
    
    def _process_camera_frames(self, camera_id: str):
        """Process frames for a camera in a background thread"""
        logger.info(f"Started frame processing thread for camera {camera_id}")
        
        while camera_id in self.stream_handlers:
            try:
                stream_handler = self.stream_handlers.get(camera_id)
                processor = self.processors.get(camera_id)
                
                if not stream_handler or not processor:
                    break
                
                # Get frame from stream
                frame = stream_handler.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                start_time = time.time()
                processed_frame, detections = processor._process_frame_sync(frame)
                process_time = time.time() - start_time
                
                # Update latest frame
                self.frame_ids[camera_id] += 1
                self.latest_frames[camera_id] = {
                    'frame': processed_frame,
                    'detections': detections,
                    'frame_id': self.frame_ids[camera_id],
                    'process_time': process_time,
                    'timestamp': time.time(),
                }
                
                # Rate limiting
                target_interval = 1.0 / 15  # 15 FPS
                if process_time < target_interval:
                    time.sleep(target_interval - process_time)
                    
            except Exception as e:
                logger.error(f"Error processing frame for camera {camera_id}: {e}")
                time.sleep(0.1)
        
        logger.info(f"Stopped frame processing thread for camera {camera_id}")
    
    def get_latest_frame(self, camera_id: UUID) -> Dict:
        """
        Get the latest processed frame for a camera
        
        Args:
            camera_id: Camera UUID
            
        Returns:
            Dict with frame data
        """
        camera_id_str = str(camera_id)
        
        if camera_id_str in self.latest_frames:
            return self.latest_frames[camera_id_str]
        
        # Return empty frame if no data
        return {
            'frame': np.zeros((480, 640, 3), dtype=np.uint8),
            'detections': [],
            'frame_id': 0,
            'process_time': 0,
            'timestamp': time.time(),
        }
    
    def get_camera_status(self, camera_id: UUID) -> Dict:
        """
        Get status of a camera
        
        Args:
            camera_id: Camera UUID
            
        Returns:
            Dict with camera status
        """
        camera_id_str = str(camera_id)
        
        is_active = camera_id_str in self.stream_handlers
        
        status = {
            'active': is_active,
            'camera_id': camera_id_str,
        }
        
        if is_active:
            processor = self.processors.get(camera_id_str)
            if processor:
                status.update(processor.get_stats())
            
            camera_info = self.cameras.get(camera_id_str, {})
            status['started_at'] = camera_info.get('started_at')
            status['clients_connected'] = len(self.clients.get(camera_id_str, set()))
        
        return status
    
    def register_client(self, camera_id: UUID, client_id: str):
        """Register a client for a camera"""
        camera_id_str = str(camera_id)
        self.clients[camera_id_str].add(client_id)
        logger.debug(f"Client {client_id} registered for camera {camera_id_str}")
    
    def unregister_client(self, camera_id: UUID, client_id: str):
        """Unregister a client from a camera"""
        camera_id_str = str(camera_id)
        self.clients[camera_id_str].discard(client_id)
        logger.debug(f"Client {client_id} unregistered from camera {camera_id_str}")
    
    def get_all_cameras_status(self) -> Dict:
        """Get status of all cameras"""
        return {
            camera_id: self.get_camera_status(UUID(camera_id))
            for camera_id in self.cameras.keys()
        }


# Global camera manager instance
camera_manager = CameraManager()

