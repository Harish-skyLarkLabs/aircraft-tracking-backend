"""
Alert Service for creating and managing aircraft detection alerts
Handles:
- Session-based alerting (one alert per track session)
- Image snapshots (initial and final)
- Video recording
- WebSocket broadcast to frontend
"""
import cv2
import time
import logging
import threading
from typing import Dict, Optional, List
from datetime import datetime
from django.utils import timezone
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

logger = logging.getLogger(__name__)

# Session timeout settings
TRACK_TIMEOUT_SECONDS = 10  # Time after which a track is considered "gone"


class TrackSession:
    """Represents an active tracking session for one aircraft"""
    
    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        track_id: int,
        detection_type: str,
        confidence: float,
        bbox: List[int],
        initial_frame,
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.track_id = track_id
        self.detection_type = detection_type
        self.confidence = confidence
        self.bbox = bbox
        self.action = 'unknown'
        
        # Timestamps
        self.start_time = time.time()
        self.last_seen_time = time.time()
        
        # Images
        self.initial_frame = initial_frame.copy() if initial_frame is not None else None
        self.final_frame = None
        self.final_bbox = bbox
        
        # Track confidence updates (keep max)
        self.max_confidence = confidence
        
        # Alert ID (set when alert is created)
        self.alert_id: Optional[str] = None
        
        # Session state
        self.is_completed = False


class AlertService:
    """
    Service for managing aircraft detection alerts.
    
    Session-based approach:
    - One alert per track session
    - Alert created when track session ends (timeout or explicit end)
    - Includes initial image, final image, and video
    """
    
    def __init__(self):
        self.channel_layer = None
        self.lock = threading.Lock()
        
        # Active track sessions: {camera_id: {track_id: TrackSession}}
        self.active_sessions: Dict[str, Dict[int, TrackSession]] = {}
        
        logger.info("AlertService initialized (session-based)")
    
    def _get_channel_layer(self):
        """Lazy load channel layer"""
        if self.channel_layer is None:
            self.channel_layer = get_channel_layer()
        return self.channel_layer
    
    def start_or_update_session(
        self,
        camera_id: str,
        camera_name: str,
        track_id: int,
        detection_type: str,
        confidence: float,
        bbox: List[int],
        frame,
        action: str = 'unknown',
    ) -> TrackSession:
        """
        Start a new tracking session or update an existing one.
        
        Args:
            camera_id: Camera identifier
            camera_name: Camera display name
            track_id: Tracking ID
            detection_type: Type of detection (Aircraft, etc.)
            confidence: Detection confidence
            bbox: Bounding box [x1, y1, x2, y2]
            frame: Current video frame
            action: Aircraft action (landing, takeoff, etc.)
            
        Returns:
            TrackSession object
        """
        with self.lock:
            # Initialize camera sessions if needed
            if camera_id not in self.active_sessions:
                self.active_sessions[camera_id] = {}
            
            if track_id in self.active_sessions[camera_id]:
                # Update existing session
                session = self.active_sessions[camera_id][track_id]
                session.last_seen_time = time.time()
                session.final_frame = frame.copy() if frame is not None else None
                session.final_bbox = bbox
                session.action = action
                
                # Update max confidence
                if confidence > session.max_confidence:
                    session.max_confidence = confidence
                    session.confidence = confidence
                
                return session
            else:
                # Create new session
                session = TrackSession(
                    camera_id=camera_id,
                    camera_name=camera_name,
                    track_id=track_id,
                    detection_type=detection_type,
                    confidence=confidence,
                    bbox=bbox,
                    initial_frame=frame,
                )
                session.action = action
                self.active_sessions[camera_id][track_id] = session
                
                logger.info(f"Started new tracking session for track {track_id} on camera {camera_id}")
                
                return session
    
    def update_track_seen(self, camera_id: str, track_id: int):
        """Update the last seen time for a track (called every frame)"""
        with self.lock:
            if camera_id in self.active_sessions:
                if track_id in self.active_sessions[camera_id]:
                    self.active_sessions[camera_id][track_id].last_seen_time = time.time()
    
    def check_and_complete_sessions(self, camera_id: str, current_track_ids: set) -> List[TrackSession]:
        """
        Check for sessions that should be completed.
        
        A session is completed when:
        1. Track is no longer detected (timeout)
        2. Track ID changes
        
        Args:
            camera_id: Camera identifier
            current_track_ids: Set of currently detected track IDs
            
        Returns:
            List of completed sessions
        """
        current_time = time.time()
        completed_sessions = []
        
        with self.lock:
            if camera_id not in self.active_sessions:
                return []
            
            for track_id, session in list(self.active_sessions[camera_id].items()):
                # Check if track is no longer present or timed out
                time_since_seen = current_time - session.last_seen_time
                
                if track_id not in current_track_ids or time_since_seen > TRACK_TIMEOUT_SECONDS:
                    # Session is complete
                    session.is_completed = True
                    completed_sessions.append(session)
                    del self.active_sessions[camera_id][track_id]
                    
                    logger.info(f"Track session {track_id} completed on camera {camera_id} "
                               f"(duration: {current_time - session.start_time:.1f}s)")
        
        return completed_sessions
    
    def create_alert_from_session(
        self,
        session: TrackSession,
        video_path: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Create an alert from a completed tracking session.
        
        Args:
            session: Completed TrackSession
            video_path: Path to video in MinIO (optional)
            
        Returns:
            Alert data dict if created, None otherwise
        """
        from aircraft_detection.models import AircraftDetection
        from backend.storage import minio_storage
        
        try:
            # Calculate session duration
            duration = time.time() - session.start_time
            
            # Create detection record
            detection = AircraftDetection(
                camera_id=session.camera_id,
                camera_name=session.camera_name,
                track_id=session.track_id,
                detection_type=session.detection_type.lower(),
                action=session.action,
                confidence=session.max_confidence,
                bbox_x1=session.bbox[0],
                bbox_y1=session.bbox[1],
                bbox_x2=session.bbox[2],
                bbox_y2=session.bbox[3],
                detection_time=timezone.now(),
                severity='medium',
                title=f"{session.detection_type} Detected",
                description=f"{session.detection_type} detected on {session.camera_name} "
                           f"for {duration:.1f}s with {session.max_confidence:.0%} confidence",
            )
            
            # Save initial frame to MinIO
            if session.initial_frame is not None:
                try:
                    _, buffer = cv2.imencode('.jpg', session.initial_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    image_bytes = buffer.tobytes()
                    
                    image_path = minio_storage.upload_alert_image(
                        camera_id=session.camera_id,
                        alert_id=str(detection.detection_id),
                        image_bytes=image_bytes,
                        suffix="_initial",
                    )
                    
                    if image_path:
                        detection.image_path = image_path
                        logger.info(f"Saved initial frame to {image_path}")
                except Exception as e:
                    logger.error(f"Failed to save initial frame: {e}")
            
            # Set video path if provided
            if video_path:
                detection.video_path = video_path
            
            # Save to database
            detection.save()
            session.alert_id = str(detection.detection_id)
            
            # Prepare alert data for WebSocket
            alert_data = {
                'detection_id': str(detection.detection_id),
                'id': str(detection.detection_id),
                'camera_id': session.camera_id,
                'camera_name': session.camera_name,
                'track_id': session.track_id,
                'detection_type': session.detection_type,
                'action': session.action,
                'confidence': session.max_confidence,
                'bbox': session.bbox,
                'image_url': detection.image_url,
                'video_url': detection.video_url,
                'title': detection.title,
                'description': detection.description,
                'severity': detection.severity,
                'status': detection.status,
                'is_read': detection.is_read,
                'detection_time': detection.detection_time.isoformat(),
                'timestamp': detection.detection_time.isoformat(),
                'duration': duration,
            }
            
            # Broadcast via WebSocket
            self._broadcast_alert(alert_data)
            
            logger.info(f"Created alert {detection.detection_id} for track {session.track_id} "
                       f"on camera {session.camera_id}")
            
            return alert_data
            
        except Exception as e:
            logger.error(f"Failed to create alert from session: {e}")
            return None
    
    def update_alert_video(self, alert_id: str, video_path: str):
        """Update an alert with the video path after recording completes"""
        from aircraft_detection.models import AircraftDetection
        
        try:
            detection = AircraftDetection.objects.get(detection_id=alert_id)
            detection.video_path = video_path
            detection.save(update_fields=['video_path', 'updated_at'])
            
            # Broadcast update
            alert_data = {
                'type': 'alert_updated',
                'detection_id': alert_id,
                'id': alert_id,
                'video_url': detection.video_url,
            }
            self._broadcast_alert(alert_data, event_type='alert_updated')
            
            logger.info(f"Updated alert {alert_id} with video path: {video_path}")
            
        except AircraftDetection.DoesNotExist:
            logger.error(f"Alert {alert_id} not found for video update")
        except Exception as e:
            logger.error(f"Failed to update alert video: {e}")
    
    def update_alert_image(self, alert_id: str, image_path: str):
        """Update an alert with the image path"""
        from aircraft_detection.models import AircraftDetection
        
        try:
            detection = AircraftDetection.objects.get(detection_id=alert_id)
            detection.image_path = image_path
            detection.save(update_fields=['image_path', 'updated_at'])
            
            # Broadcast update
            alert_data = {
                'type': 'alert_updated',
                'detection_id': alert_id,
                'id': alert_id,
                'image_url': detection.image_url,
            }
            self._broadcast_alert(alert_data, event_type='alert_updated')
            
            logger.info(f"Updated alert {alert_id} with image path: {image_path}")
            
        except AircraftDetection.DoesNotExist:
            logger.error(f"Alert {alert_id} not found for image update")
        except Exception as e:
            logger.error(f"Failed to update alert image: {e}")
    
    def _broadcast_alert(self, alert_data: Dict, event_type: str = 'new_alert'):
        """Broadcast alert to all connected WebSocket clients"""
        try:
            channel_layer = self._get_channel_layer()
            if channel_layer:
                async_to_sync(channel_layer.group_send)(
                    'alerts',
                    {
                        'type': 'alert_message',
                        'event': event_type,
                        'data': alert_data,
                    }
                )
                logger.debug(f"Broadcasted {event_type} to alerts channel")
        except Exception as e:
            logger.error(f"Failed to broadcast alert: {e}")
    
    def cleanup_camera(self, camera_id: str):
        """Clean up tracking data for a camera"""
        with self.lock:
            if camera_id in self.active_sessions:
                del self.active_sessions[camera_id]
        logger.info(f"Cleaned up alert tracking for camera {camera_id}")
    
    def get_stats(self) -> Dict:
        """Get alert service statistics"""
        with self.lock:
            total_cameras = len(self.active_sessions)
            total_sessions = sum(len(s) for s in self.active_sessions.values())
            return {
                'cameras_tracked': total_cameras,
                'active_sessions': total_sessions,
                'track_timeout_seconds': TRACK_TIMEOUT_SECONDS,
            }


# Global alert service instance
_alert_service: Optional[AlertService] = None


def get_alert_service() -> AlertService:
    """Get or create the global alert service instance"""
    global _alert_service
    if _alert_service is None:
        _alert_service = AlertService()
    return _alert_service
