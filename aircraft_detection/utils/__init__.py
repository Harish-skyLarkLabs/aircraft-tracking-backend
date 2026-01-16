"""
Aircraft Detection Utilities with PTZ Support

Includes:
- Aircraft detector (YOLO-based)
- Enhanced Aircraft Tracker with PTZ support
- IOU Tracker (legacy compatibility)
- PTZ Controller for Hikvision cameras
- Alert service (with cooldown logic)
- Video recorder (pre-roll/post-roll buffer)
- Frame processor (integrates all components with PTZ)
- Enhanced RTSP stream handler with GStreamer support
- Drawing utilities
"""

from .aircraft_detector import AircraftDetector, get_detector
from .tracker import AircraftTracker, IOUTracker, TrackedAircraft, calculate_iou, calculate_distance
from .ptz_controller import PTZController, get_ptz_controller, remove_ptz_controller, get_all_ptz_status
from .alert_service import AlertService, get_alert_service
from .video_recorder import VideoRecorder, get_video_recorder
from .frame_processor import FrameProcessor
from .stream_handler import RTSPStreamHandler, StreamHandler
from .camera_manager import CameraManager, CameraSystem, camera_manager

__all__ = [
    # Detector
    'AircraftDetector',
    'get_detector',
    # Tracker
    'AircraftTracker',
    'IOUTracker',
    'TrackedAircraft',
    'calculate_iou',
    'calculate_distance',
    # PTZ Controller
    'PTZController',
    'get_ptz_controller',
    'remove_ptz_controller',
    'get_all_ptz_status',
    # Alert Service
    'AlertService',
    'get_alert_service',
    # Video Recorder
    'VideoRecorder',
    'get_video_recorder',
    # Frame Processor
    'FrameProcessor',
    # Stream Handler
    'RTSPStreamHandler',
    'StreamHandler',
    # Camera Manager
    'CameraManager',
    'CameraSystem',
    'camera_manager',
]
