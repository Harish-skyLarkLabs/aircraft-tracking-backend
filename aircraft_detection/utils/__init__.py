"""
Aircraft Detection Utilities

Includes:
- Aircraft detector (YOLO-based)
- IOU Tracker
- Alert service (with cooldown logic)
- Video recorder (pre-roll/post-roll buffer)
- Frame processor (integrates all components)
- Drawing utilities
"""

from .aircraft_detector import AircraftDetector, get_detector
from .tracker import IOUTracker
from .alert_service import AlertService, get_alert_service
from .video_recorder import VideoRecorder, get_video_recorder
from .frame_processor import FrameProcessor
from .camera_manager import camera_manager

__all__ = [
    'AircraftDetector',
    'get_detector',
    'IOUTracker',
    'AlertService',
    'get_alert_service',
    'VideoRecorder',
    'get_video_recorder',
    'FrameProcessor',
    'camera_manager',
]

