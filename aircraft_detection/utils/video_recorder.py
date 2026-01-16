"""
Video Recorder for aircraft detection events
Session-based recording with memory-efficient frame storage:
- Pre-roll buffer (capture frames before detection)
- Recording while track is active
- Post-roll recording (continue after track disappears)
- Save to MinIO storage with alert
"""
import cv2
import time
import threading
import tempfile
import os
import logging
import subprocess
from typing import Dict, Optional, List, Tuple, Set
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Recording settings
PRE_ROLL_SECONDS = 3  # Reduced from 5 to save memory
POST_ROLL_SECONDS = 5  # Reduced from 8 to save memory
MAX_RECORDING_SECONDS = 60  # Reduced from 120 to prevent memory issues
DEFAULT_FPS = 10  # Reduced from 15 to save memory
MAX_FRAMES_PER_SESSION = 600  # Maximum frames to store (60s × 10fps)
JPEG_QUALITY = 70  # Quality for compressed frame storage


@dataclass
class RecordingSession:
    """Represents an active recording session for a track"""
    camera_id: str
    track_id: int
    start_time: float
    last_detection_time: float
    # Store compressed JPEG bytes instead of raw frames to save memory
    frames: List[Tuple[float, bytes]] = field(default_factory=list)
    is_complete: bool = False
    frame_width: int = 1920
    frame_height: int = 1080
    fps: int = DEFAULT_FPS
    
    # Initial and final frames for snapshots (compressed)
    initial_frame: Optional[bytes] = None
    final_frame: Optional[bytes] = None
    initial_bbox: List[int] = field(default_factory=list)
    final_bbox: List[int] = field(default_factory=list)
    
    # Detection info
    detection_type: str = 'Aircraft'
    max_confidence: float = 0.0
    action: str = 'unknown'


def compress_frame(frame, quality: int = JPEG_QUALITY) -> Optional[bytes]:
    """Compress a frame to JPEG bytes to save memory"""
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes()
    except Exception:
        return None


def decompress_frame(data: bytes) -> Optional[any]:
    """Decompress JPEG bytes back to a frame"""
    try:
        import numpy as np
        nparr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


class VideoRecorder:
    """
    Video recorder with pre-roll buffer for aircraft detection events.
    
    Memory-efficient approach:
    - Stores compressed JPEG frames instead of raw numpy arrays
    - Limited buffer sizes
    - Automatic cleanup
    """
    
    def __init__(self, fps: int = DEFAULT_FPS):
        self.fps = fps
        self.lock = threading.Lock()
        
        # Pre-roll buffer per camera: {camera_id: deque of (timestamp, compressed_frame_bytes)}
        self.pre_roll_buffers: Dict[str, deque] = {}
        
        # Active recording sessions: {camera_id: {track_id: RecordingSession}}
        self.active_sessions: Dict[str, Dict[int, RecordingSession]] = {}
        
        # Background thread for processing completed recordings
        self.processing_queue: List[RecordingSession] = []
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        
        # Frame skip counter per camera to reduce FPS
        self.frame_counters: Dict[str, int] = {}
        self.frame_skip = 2  # Process every Nth frame
        
        logger.info(f"VideoRecorder initialized with {PRE_ROLL_SECONDS}s pre-roll, {POST_ROLL_SECONDS}s post-roll")
    
    def add_frame_to_buffer(self, camera_id: str, frame: any):
        """
        Add a frame to the pre-roll buffer for a camera.
        Uses compressed JPEG to save memory.
        """
        # Frame skipping to reduce memory usage
        if camera_id not in self.frame_counters:
            self.frame_counters[camera_id] = 0
        
        self.frame_counters[camera_id] += 1
        if self.frame_counters[camera_id] % self.frame_skip != 0:
            return
        
        current_time = time.time()
        
        # Compress frame
        compressed = compress_frame(frame)
        if compressed is None:
            return
        
        with self.lock:
            # Initialize buffer if needed
            if camera_id not in self.pre_roll_buffers:
                # Buffer size = pre_roll_seconds * fps / frame_skip
                max_frames = int(PRE_ROLL_SECONDS * self.fps / self.frame_skip)
                self.pre_roll_buffers[camera_id] = deque(maxlen=max_frames)
            
            # Add compressed frame to buffer
            self.pre_roll_buffers[camera_id].append((current_time, compressed))
            
            # Also add to any active recording sessions for this camera
            if camera_id in self.active_sessions:
                for session in self.active_sessions[camera_id].values():
                    if not session.is_complete and len(session.frames) < MAX_FRAMES_PER_SESSION:
                        session.frames.append((current_time, compressed))
                        
                        # Update frame dimensions from first frame
                        if len(session.frames) == 1:
                            session.frame_height, session.frame_width = frame.shape[:2]
    
    def start_or_update_recording(
        self,
        camera_id: str,
        track_id: int,
        detection_type: str,
        confidence: float,
        bbox: List[int],
        frame: any,
        action: str = 'unknown',
    ) -> RecordingSession:
        """
        Start a new recording session or update an existing one.
        """
        current_time = time.time()
        
        # Compress the current frame for storage
        compressed_frame = compress_frame(frame, quality=85)  # Higher quality for snapshots
        
        with self.lock:
            # Initialize camera sessions if needed
            if camera_id not in self.active_sessions:
                self.active_sessions[camera_id] = {}
            
            if track_id in self.active_sessions[camera_id]:
                # Update existing session
                session = self.active_sessions[camera_id][track_id]
                session.last_detection_time = current_time
                session.final_frame = compressed_frame
                session.final_bbox = list(bbox)
                session.action = action
                
                # Update max confidence
                if confidence > session.max_confidence:
                    session.max_confidence = confidence
                
                return session
            else:
                # Get pre-roll frames (already compressed)
                pre_roll_frames = []
                if camera_id in self.pre_roll_buffers:
                    pre_roll_frames = list(self.pre_roll_buffers[camera_id])
                
                # Create new session
                session = RecordingSession(
                    camera_id=camera_id,
                    track_id=track_id,
                    start_time=current_time - PRE_ROLL_SECONDS,
                    last_detection_time=current_time,
                    frames=pre_roll_frames,
                    fps=self.fps // self.frame_skip,  # Adjusted FPS
                    initial_frame=compressed_frame,
                    final_frame=compressed_frame,
                    initial_bbox=list(bbox),
                    final_bbox=list(bbox),
                    detection_type=detection_type,
                    max_confidence=confidence,
                    action=action,
                )
                
                # Get frame dimensions
                if frame is not None:
                    session.frame_height, session.frame_width = frame.shape[:2]
                
                self.active_sessions[camera_id][track_id] = session
                
                logger.info(f"Started recording for track {track_id} on camera {camera_id} "
                           f"with {len(pre_roll_frames)} pre-roll frames")
                
                return session
    
    def check_and_complete_recordings(
        self,
        camera_id: str,
        current_track_ids: Optional[Set[int]] = None,
    ) -> List[RecordingSession]:
        """
        Check for recordings that should be completed.
        """
        current_time = time.time()
        completed_sessions = []
        
        with self.lock:
            if camera_id not in self.active_sessions:
                return []
            
            for track_id, session in list(self.active_sessions[camera_id].items()):
                # Check if track is still present
                track_present = current_track_ids is None or track_id in current_track_ids
                
                # Update last detection time if track is present
                if track_present:
                    session.last_detection_time = current_time
                    continue
                
                # Check post-roll timeout
                time_since_last_detection = current_time - session.last_detection_time
                recording_duration = current_time - session.start_time
                
                if (time_since_last_detection > POST_ROLL_SECONDS or 
                    recording_duration > MAX_RECORDING_SECONDS or
                    len(session.frames) >= MAX_FRAMES_PER_SESSION):
                    
                    session.is_complete = True
                    completed_sessions.append(session)
                    del self.active_sessions[camera_id][track_id]
                    
                    logger.info(f"Recording complete for track {track_id}: "
                               f"{len(session.frames)} frames, "
                               f"{recording_duration:.1f}s duration")
        
        # Queue completed sessions for processing
        for session in completed_sessions:
            self.processing_queue.append(session)
        
        return completed_sessions
    
    def _processing_worker(self):
        """Background worker to save completed recordings"""
        while self.processing_active:
            try:
                session = None
                if self.processing_queue:
                    session = self.processing_queue.pop(0)
                
                if session:
                    self._save_recording_and_create_alert(session)
                    # Clear session frames after processing to free memory
                    session.frames = []
                    session.initial_frame = None
                    session.final_frame = None
                else:
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error in recording processor: {e}")
                time.sleep(1)
    
    def _save_recording_and_create_alert(self, session: RecordingSession):
        """Save a completed recording and create alert with all media"""
        from backend.storage import minio_storage
        from aircraft_detection.models import AircraftDetection
        from cameras.models import Camera
        from django.utils import timezone
        
        if not session.frames:
            logger.warning(f"No frames to save for track {session.track_id}")
            return
        
        # Verify camera still has AI enabled before creating alert
        try:
            camera = Camera.objects.get(camera_id=session.camera_id)
            if not camera.ai_enabled:
                logger.info(f"Skipping alert creation for camera {session.camera_id} - AI is now disabled")
                return
        except Camera.DoesNotExist:
            logger.warning(f"Camera {session.camera_id} no longer exists, skipping alert creation")
            return
        
        video_path = None
        initial_image_path = None
        
        try:
            import uuid
            alert_id = str(uuid.uuid4())
            
            # 1. Save initial frame
            if session.initial_frame is not None:
                try:
                    initial_image_path = minio_storage.upload_alert_image(
                        camera_id=session.camera_id,
                        alert_id=alert_id,
                        image_bytes=session.initial_frame,  # Already compressed
                        suffix="_initial",
                    )
                    logger.info(f"Saved initial frame to {initial_image_path}")
                except Exception as e:
                    logger.error(f"Failed to save initial frame: {e}")
            
            # 2. Save video (decompress frames and encode with H.264)
            try:
                video_path = self._create_video(session, alert_id, minio_storage)
            except Exception as e:
                logger.error(f"Failed to save video: {e}")
            
            # 3. Create alert in database
            duration = time.time() - session.start_time
            
            detection = AircraftDetection(
                detection_id=alert_id,
                camera_id=session.camera_id,
                camera_name="",
                track_id=session.track_id,
                detection_type=session.detection_type.lower(),
                action=session.action,
                confidence=session.max_confidence,
                bbox_x1=session.initial_bbox[0] if session.initial_bbox else 0,
                bbox_y1=session.initial_bbox[1] if session.initial_bbox else 0,
                bbox_x2=session.initial_bbox[2] if session.initial_bbox else 0,
                bbox_y2=session.initial_bbox[3] if session.initial_bbox else 0,
                detection_time=timezone.now(),
                severity='medium',
                title=f"{session.detection_type} Detected",
                description=f"{session.detection_type} tracked for {duration:.1f}s "
                           f"with {session.max_confidence:.0%} confidence",
                image_path=initial_image_path,
                video_path=video_path,
            )
            detection.save()
            
            # 4. Broadcast alert via WebSocket
            self._broadcast_alert({
                'detection_id': alert_id,
                'id': alert_id,
                'camera_id': session.camera_id,
                'camera_name': detection.camera_name,
                'track_id': session.track_id,
                'detection_type': session.detection_type,
                'action': session.action,
                'confidence': session.max_confidence,
                'bbox': session.initial_bbox,
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
            })
            
            logger.info(f"Created alert {alert_id} for track {session.track_id}")
            
        except Exception as e:
            logger.error(f"Failed to save recording and create alert: {e}")
    
    def _create_video(self, session: RecordingSession, alert_id: str, minio_storage) -> Optional[str]:
        """Create video from compressed frames using ffmpeg for H.264 encoding"""
        tmp_raw_path = None
        tmp_h264_path = None
        
        try:
            # Create temporary file for raw video
            with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp_file:
                tmp_raw_path = tmp_file.name
            
            # Write frames using OpenCV (raw format)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fps = max(session.fps, 5)  # Minimum 5 FPS
            writer = cv2.VideoWriter(
                tmp_raw_path,
                fourcc,
                fps,
                (session.frame_width, session.frame_height)
            )
            
            if not writer.isOpened():
                logger.error("Failed to open video writer")
                return None
            
            # Decompress and write frames
            for timestamp, compressed_frame in session.frames:
                frame = decompress_frame(compressed_frame)
                if frame is not None:
                    if frame.shape[:2] != (session.frame_height, session.frame_width):
                        frame = cv2.resize(frame, (session.frame_width, session.frame_height))
                    writer.write(frame)
            
            writer.release()
            
            # Convert to H.264 using ffmpeg for browser compatibility
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_h264_path = tmp_file.name
            
            # Use ffmpeg to convert to H.264 (browser compatible)
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', tmp_raw_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                tmp_h264_path
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.warning(f"ffmpeg conversion failed: {result.stderr.decode()}")
                # Fall back to raw video
                tmp_h264_path = tmp_raw_path
                tmp_raw_path = None
            
            # Upload to MinIO
            with open(tmp_h264_path, 'rb') as f:
                video_bytes = f.read()
            
            video_path = minio_storage.upload_alert_video(
                camera_id=session.camera_id,
                alert_id=alert_id,
                video_bytes=video_bytes,
            )
            
            logger.info(f"Saved video to {video_path}: "
                       f"{len(session.frames)} frames, "
                       f"{len(video_bytes) / 1024 / 1024:.1f}MB")
            
            return video_path
            
        finally:
            # Clean up temp files
            if tmp_raw_path and os.path.exists(tmp_raw_path):
                os.unlink(tmp_raw_path)
            if tmp_h264_path and os.path.exists(tmp_h264_path):
                os.unlink(tmp_h264_path)
    
    def _broadcast_alert(self, alert_data: Dict):
        """Broadcast alert to WebSocket clients"""
        try:
            from asgiref.sync import async_to_sync
            from channels.layers import get_channel_layer
            
            channel_layer = get_channel_layer()
            if channel_layer:
                async_to_sync(channel_layer.group_send)(
                    'alerts',
                    {
                        'type': 'alert_message',
                        'event': 'new_alert',
                        'data': alert_data,
                    }
                )
        except Exception as e:
            logger.error(f"Failed to broadcast alert: {e}")
    
    def cleanup_camera(self, camera_id: str):
        """Clean up all data for a camera and cancel pending recordings"""
        with self.lock:
            # 1. Clear pre-roll buffers
            if camera_id in self.pre_roll_buffers:
                self.pre_roll_buffers[camera_id].clear()
                del self.pre_roll_buffers[camera_id]
            
            # 2. Cancel active recording sessions (don't save them)
            session_count = 0
            if camera_id in self.active_sessions:
                session_count = len(self.active_sessions[camera_id])
                # Clear frame data before deleting to free memory
                for session in self.active_sessions[camera_id].values():
                    session.frames = []
                    session.initial_frame = None
                    session.final_frame = None
                del self.active_sessions[camera_id]
                logger.info(f"Cancelled {session_count} active recording sessions for camera {camera_id}")
            
            # 3. Remove from frame counters
            if camera_id in self.frame_counters:
                del self.frame_counters[camera_id]
            
            # 4. Remove any pending sessions from processing queue for this camera
            original_queue_size = len(self.processing_queue)
            new_queue = []
            removed_count = 0
            
            for session in self.processing_queue:
                if session.camera_id == camera_id:
                    # Clear frame data to free memory
                    session.frames = []
                    session.initial_frame = None
                    session.final_frame = None
                    removed_count += 1
                else:
                    new_queue.append(session)
            
            self.processing_queue = new_queue
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} pending recordings from queue for camera {camera_id}")
        
        logger.info(f"✓ Fully cleaned up video recorder for camera {camera_id} "
                   f"({session_count} active + {removed_count} queued sessions cancelled)")
    
    def shutdown(self):
        """Shutdown the video recorder"""
        self.processing_active = False
        
        # Clear all buffers
        with self.lock:
            for camera_id in list(self.pre_roll_buffers.keys()):
                self.pre_roll_buffers[camera_id].clear()
            self.pre_roll_buffers.clear()
            
            for camera_id in list(self.active_sessions.keys()):
                for session in self.active_sessions[camera_id].values():
                    session.frames = []
                self.active_sessions[camera_id].clear()
            self.active_sessions.clear()
        
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        logger.info("VideoRecorder shutdown complete")
    
    def get_stats(self) -> Dict:
        """Get recorder statistics"""
        with self.lock:
            total_buffers = len(self.pre_roll_buffers)
            total_buffer_frames = sum(len(b) for b in self.pre_roll_buffers.values())
            total_sessions = sum(len(s) for s in self.active_sessions.values())
            total_session_frames = sum(
                len(session.frames) 
                for sessions in self.active_sessions.values() 
                for session in sessions.values()
            )
            pending_saves = len(self.processing_queue)
            
            return {
                'cameras_buffering': total_buffers,
                'buffer_frames': total_buffer_frames,
                'active_recordings': total_sessions,
                'recording_frames': total_session_frames,
                'pending_saves': pending_saves,
                'pre_roll_seconds': PRE_ROLL_SECONDS,
                'post_roll_seconds': POST_ROLL_SECONDS,
            }


# Global video recorder instance
_video_recorder: Optional[VideoRecorder] = None


def get_video_recorder() -> VideoRecorder:
    """Get or create the global video recorder instance"""
    global _video_recorder
    if _video_recorder is None:
        _video_recorder = VideoRecorder()
    return _video_recorder
