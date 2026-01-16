"""
Enhanced Camera Manager for handling multiple camera streams with PTZ support
Based on the ML team's aircraft_det.py demo script
Provides:
- Multiple camera stream management
- PTZ controller integration
- Enhanced RTSP streaming with GStreamer support
- Frame processing with detection and tracking
"""
import cv2
import threading
import time
import logging
import numpy as np
from typing import Dict, Optional, Set, Tuple
from uuid import UUID
from collections import defaultdict

from .frame_processor import FrameProcessor
from .stream_handler import RTSPStreamHandler, StreamHandler
from .ptz_controller import PTZController, get_ptz_controller, remove_ptz_controller

logger = logging.getLogger(__name__)


class CameraSystem:
    """
    Complete camera system with stream handler, frame processor, and optional PTZ
    Mirrors the CameraSystem class from the demo script
    """
    
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        camera_name: str = "",
        roi_points: Optional[list] = None,
        # PTZ configuration
        ptz_enabled: bool = False,
        ptz_ip: Optional[str] = None,
        ptz_username: Optional[str] = None,
        ptz_password: Optional[str] = None,
        ptz_channel: int = 1,
        ptz_preset_number: int = 20,
        enable_ptz_tracking: bool = True,
        enable_zoom_control: bool = True,
        # Processing settings
        enable_recording: bool = True,
        debug_visualization: bool = True,
    ):
        """
        Initialize complete camera system
        
        Args:
            camera_id: Unique camera identifier
            rtsp_url: RTSP stream URL
            camera_name: Display name for camera
            roi_points: Optional ROI polygon points
            ptz_enabled: Whether PTZ control is enabled
            ptz_ip: PTZ camera IP (defaults to extracted from RTSP URL)
            ptz_username: PTZ camera username
            ptz_password: PTZ camera password
            ptz_channel: PTZ channel number
            ptz_preset_number: Default preset position
            enable_ptz_tracking: Enable PTZ tracking
            enable_zoom_control: Enable zoom control
            enable_recording: Enable video recording
            debug_visualization: Enable debug overlays
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name
        self.roi_points = roi_points
        
        # Initialize stream handler with GStreamer support
        self.stream_handler = RTSPStreamHandler(rtsp_url, buffer_size=5)
        
        # Initialize PTZ controller if enabled
        self.ptz_controller: Optional[PTZController] = None
        if ptz_enabled and ptz_ip and ptz_username and ptz_password:
            self.ptz_controller = PTZController(
                ip=ptz_ip,
                username=ptz_username,
                password=ptz_password,
                channel=ptz_channel,
                enable_ptz_tracking=enable_ptz_tracking,
                enable_zoom_control=enable_zoom_control,
                preset_number=ptz_preset_number,
            )
            if not self.ptz_controller.connect():
                logger.warning(f"Failed to connect PTZ controller for camera {camera_id}")
                self.ptz_controller = None
        
        # Initialize frame processor
        self.processor = FrameProcessor(
            camera_id=camera_id,
            camera_name=camera_name,
            roi_points=roi_points,
            ptz_controller=self.ptz_controller,
            enable_ptz_tracking=enable_ptz_tracking,
            enable_zoom_control=enable_zoom_control,
            enable_recording=enable_recording,
            debug_visualization=debug_visualization,
        )
        
        # State
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.started_at: Optional[float] = None
        
        logger.info(f"CameraSystem initialized for {camera_id} (PTZ: {ptz_enabled})")
    
    def start(self) -> bool:
        """Start the camera system"""
        if self.is_running:
            return True
        
        # Start stream handler
        if not self.stream_handler.start():
            logger.error(f"Failed to start stream handler for camera {self.camera_id}")
            return False
        
        self.is_running = True
        self.started_at = time.time()
        
        logger.info(f"CameraSystem started for {self.camera_id}")
        return True
    
    def stop(self):
        """Stop the camera system"""
        self.is_running = False
        
        # Stop stream handler
        self.stream_handler.stop()
        
        # Stop PTZ controller
        if self.ptz_controller:
            self.ptz_controller.stop()
        
        # Cleanup processor
        self.processor.cleanup()
        
        logger.info(f"CameraSystem stopped for {self.camera_id}")
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get frame from camera"""
        return self.stream_handler.get_frame()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """Process a frame through the detector and tracker"""
        return self.processor._process_frame_sync(frame)
    
    def get_status(self) -> dict:
        """Get camera system status"""
        status = {
            'camera_id': self.camera_id,
            'camera_name': self.camera_name,
            'is_running': self.is_running,
            'started_at': self.started_at,
            'stream': self.stream_handler.get_info(),
            'processor': self.processor.get_stats(),
        }
        
        if self.ptz_controller:
            status['ptz'] = self.ptz_controller.get_status()
        
        return status


class CameraManager:
    """
    Singleton manager for handling multiple camera streams with PTZ support
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
        self.camera_systems: Dict[str, CameraSystem] = {}
        self.cameras: Dict[str, Dict] = {}
        self.processors: Dict[str, FrameProcessor] = {}
        self.stream_handlers: Dict[str, RTSPStreamHandler] = {}
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
        # PTZ configuration
        ptz_enabled: bool = False,
        ptz_ip: Optional[str] = None,
        ptz_username: Optional[str] = None,
        ptz_password: Optional[str] = None,
        ptz_channel: int = 1,
        ptz_preset_number: int = 20,
        enable_ptz_tracking: bool = True,
        enable_zoom_control: bool = True,
    ) -> bool:
        """
        Start processing for a camera with optional PTZ support
        
        Args:
            camera_id: Camera UUID
            rtsp_link: RTSP stream URL
            camera_name: Optional camera name
            roi_points: Optional ROI polygon points
            ptz_enabled: Whether PTZ control is enabled
            ptz_ip: PTZ camera IP address
            ptz_username: PTZ camera username
            ptz_password: PTZ camera password
            ptz_channel: PTZ channel number
            ptz_preset_number: Default preset position
            enable_ptz_tracking: Enable PTZ tracking
            enable_zoom_control: Enable zoom control
            
        Returns:
            bool: True if started successfully
        """
        camera_id_str = str(camera_id)
        
        with self._lock:
            # Check if already processing
            if camera_id_str in self.camera_systems:
                logger.warning(f"Camera {camera_id_str} is already being processed")
                return True
            
            try:
                # Create camera system
                camera_system = CameraSystem(
                    camera_id=camera_id_str,
                    rtsp_url=rtsp_link,
                    camera_name=camera_name,
                    roi_points=roi_points,
                    ptz_enabled=ptz_enabled,
                    ptz_ip=ptz_ip,
                    ptz_username=ptz_username,
                    ptz_password=ptz_password,
                    ptz_channel=ptz_channel,
                    ptz_preset_number=ptz_preset_number,
                    enable_ptz_tracking=enable_ptz_tracking,
                    enable_zoom_control=enable_zoom_control,
                )
                
                if not camera_system.start():
                    logger.error(f"Failed to start camera system for {camera_id_str}")
                    return False
                
                # Store references
                self.camera_systems[camera_id_str] = camera_system
                self.stream_handlers[camera_id_str] = camera_system.stream_handler
                self.processors[camera_id_str] = camera_system.processor
                self.cameras[camera_id_str] = {
                    'rtsp_link': rtsp_link,
                    'camera_name': camera_name,
                    'started_at': time.time(),
                    'ptz_enabled': ptz_enabled,
                }
                
                # Start frame processing thread
                processing_thread = threading.Thread(
                    target=self._process_camera_frames,
                    args=(camera_id_str,),
                    daemon=True
                )
                processing_thread.start()
                
                logger.info(f"Started processing for camera {camera_id_str} (PTZ: {ptz_enabled})")
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
            if camera_id_str not in self.camera_systems:
                logger.warning(f"Camera {camera_id_str} is not being processed")
                return True
            
            try:
                # Stop camera system
                camera_system = self.camera_systems.pop(camera_id_str, None)
                if camera_system:
                    camera_system.stop()
                
                # Clean up references
                self.stream_handlers.pop(camera_id_str, None)
                self.processors.pop(camera_id_str, None)
                self.cameras.pop(camera_id_str, None)
                self.latest_frames.pop(camera_id_str, None)
                self.frame_ids.pop(camera_id_str, None)
                
                logger.info(f"Stopped processing for camera {camera_id_str}")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping camera processing: {e}")
                return False
    
    def _process_camera_frames(self, camera_id: str):
        """Process frames for a camera in a background thread"""
        logger.info(f"Started frame processing thread for camera {camera_id}")
        
        while camera_id in self.camera_systems:
            try:
                camera_system = self.camera_systems.get(camera_id)
                
                if not camera_system or not camera_system.is_running:
                    break
                
                # Get frame from stream
                ret, frame = camera_system.get_frame()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                start_time = time.time()
                processed_frame, detections = camera_system.process_frame(frame)
                process_time = time.time() - start_time
                
                # Validate processed frame before storing
                if processed_frame is None or not isinstance(processed_frame, np.ndarray):
                    logger.warning(f"Invalid processed frame for camera {camera_id}")
                    continue
                
                if processed_frame.size == 0:
                    logger.warning(f"Empty processed frame for camera {camera_id}")
                    continue
                
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
        
        is_active = camera_id_str in self.camera_systems
        
        status = {
            'active': is_active,
            'camera_id': camera_id_str,
        }
        
        if is_active:
            camera_system = self.camera_systems.get(camera_id_str)
            if camera_system:
                status.update(camera_system.get_status())
            
            camera_info = self.cameras.get(camera_id_str, {})
            status['started_at'] = camera_info.get('started_at')
            status['ptz_enabled'] = camera_info.get('ptz_enabled', False)
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
    
    # PTZ Control Methods
    
    def ptz_go_to_preset(self, camera_id: UUID, preset_number: Optional[int] = None) -> bool:
        """Move PTZ camera to preset position"""
        camera_id_str = str(camera_id)
        camera_system = self.camera_systems.get(camera_id_str)
        
        if camera_system and camera_system.ptz_controller:
            return camera_system.ptz_controller.go_to_preset(preset_number)
        return False
    
    def ptz_emergency_stop(self, camera_id: UUID) -> bool:
        """Emergency stop PTZ movement"""
        camera_id_str = str(camera_id)
        camera_system = self.camera_systems.get(camera_id_str)
        
        if camera_system and camera_system.ptz_controller:
            camera_system.ptz_controller.emergency_stop()
            return True
        return False
    
    def ptz_clear_tracking_lock(self, camera_id: UUID) -> bool:
        """Clear PTZ tracking lock"""
        camera_id_str = str(camera_id)
        camera_system = self.camera_systems.get(camera_id_str)
        
        if camera_system:
            camera_system.processor.clear_tracking_lock()
            return True
        return False
    
    def ptz_set_tracking_enabled(self, camera_id: UUID, enabled: bool) -> bool:
        """Enable/disable PTZ tracking"""
        camera_id_str = str(camera_id)
        camera_system = self.camera_systems.get(camera_id_str)
        
        if camera_system:
            camera_system.processor.set_ptz_tracking_enabled(enabled)
            if camera_system.ptz_controller:
                camera_system.ptz_controller.enable_ptz_tracking = enabled
            return True
        return False
    
    def ptz_set_zoom_enabled(self, camera_id: UUID, enabled: bool) -> bool:
        """Enable/disable zoom control"""
        camera_id_str = str(camera_id)
        camera_system = self.camera_systems.get(camera_id_str)
        
        if camera_system:
            camera_system.processor.set_zoom_control_enabled(enabled)
            return True
        return False
    
    def get_ptz_status(self, camera_id: UUID) -> Optional[Dict]:
        """Get PTZ controller status"""
        camera_id_str = str(camera_id)
        camera_system = self.camera_systems.get(camera_id_str)
        
        if camera_system and camera_system.ptz_controller:
            return camera_system.ptz_controller.get_status()
        return None


# Global camera manager instance
camera_manager = CameraManager()
