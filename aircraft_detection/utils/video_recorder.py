"""
Video Recorder for aircraft detection events
Time-based clip recording (5-second clips):
- When aircraft is detected, start recording
- Save 5-second clips every 5 seconds while aircraft is tracked
- Save both full video and cropped/zoomed aircraft video
- Create alert for each clip saved to MinIO
- Memory-efficient: stores compressed JPEG frames
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
CLIP_DURATION_SECONDS = 5  # Save 5-second clips when aircraft detected
PRE_ROLL_SECONDS = 2  # Small pre-roll for context
DEFAULT_FPS = 10  # Reduced from 15 to save memory
MAX_FRAMES_PER_CLIP = 50  # Maximum frames per clip (5s × 10fps)
JPEG_QUALITY = 70  # Quality for compressed frame storage


@dataclass
class RecordingSession:
    """Represents an active recording session for a track"""
    camera_id: str
    track_id: int
    start_time: float
    last_detection_time: float
    last_clip_save_time: float = 0  # When we last saved a clip
    clip_number: int = 0  # Current clip number for this session
    # Store compressed JPEG bytes with bbox: (timestamp, compressed_frame, bbox)
    frames: List[Tuple[float, bytes, List[int]]] = field(default_factory=list)
    # Store cropped/zoomed aircraft frames separately: (timestamp, compressed_crop)
    crop_frames: List[Tuple[float, Optional[bytes]]] = field(default_factory=list)
    is_complete: bool = False
    frame_width: int = 1920
    frame_height: int = 1080
    crop_width: int = 320  # Standard crop size
    crop_height: int = 240
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
        
        logger.info(f"VideoRecorder initialized with {CLIP_DURATION_SECONDS}s clip duration")
    
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
            
            # Note: Frames are now added directly in start_or_update_recording
            # This buffer is mainly for pre-roll when starting new sessions
    
    def start_or_update_recording(
        self,
        camera_id: str,
        track_id: int,
        detection_type: str,
        confidence: float,
        bbox: List[int],
        frame: any,
        action: str = 'unknown',
        is_ptz_target: bool = False,
    ) -> RecordingSession:
        """
        Start a new recording session or update an existing one.
        Only captures cropped/zoomed aircraft frames for PTZ-locked aircraft.
        """
        current_time = time.time()
        
        # Compress the current frame for storage
        compressed_frame = compress_frame(frame, quality=85)  # Higher quality for snapshots
        
        # Only create cropped frame for PTZ-locked aircraft (will be created when adding frames)
        compressed_crop = None
        
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
                
                # Add frames to current clip - ensure full and crop frames are always in sync
                if len(session.frames) < MAX_FRAMES_PER_CLIP:
                    # Store frame with bbox: (timestamp, compressed_frame, bbox)
                    session.frames.append((current_time, compressed_frame, list(bbox)))
                    
                    # Add corresponding crop frame (only for PTZ-locked aircraft)
                    # Crop frame must be created from the SAME frame that was just added
                    if is_ptz_target:
                        # Create crop from the same frame we just added
                        cropped_frame = self._create_crop_frame(frame, bbox)
                        if cropped_frame is not None:
                            compressed_crop = compress_frame(cropped_frame, quality=85)
                            session.crop_frames.append((current_time, compressed_crop))
                        else:
                            # If crop creation fails, add None to maintain sync
                            session.crop_frames.append((current_time, None))
                    else:
                        # Not PTZ target - add None to maintain frame count sync
                        session.crop_frames.append((current_time, None))
                
                # Update max confidence
                if confidence > session.max_confidence:
                    session.max_confidence = confidence
                
                # Check if 5 seconds have passed since last clip save
                time_since_last_save = current_time - session.last_clip_save_time
                if session.last_clip_save_time == 0:
                    # First detection, initialize
                    session.last_clip_save_time = current_time
                elif time_since_last_save >= CLIP_DURATION_SECONDS and len(session.frames) > 0:
                    # Create a copy of the session for this clip to save
                    clip_session = RecordingSession(
                        camera_id=session.camera_id,
                        track_id=session.track_id,
                        start_time=session.last_clip_save_time,
                        last_detection_time=current_time,
                        last_clip_save_time=session.last_clip_save_time,
                        clip_number=session.clip_number,
                        frames=session.frames.copy(),  # Already includes bbox
                        crop_frames=session.crop_frames.copy(),  # Same length as frames
                        frame_width=session.frame_width,
                        frame_height=session.frame_height,
                        crop_width=session.crop_width,
                        crop_height=session.crop_height,
                        fps=session.fps,
                        initial_frame=session.initial_frame,
                        final_frame=compressed_frame,
                        initial_bbox=session.initial_bbox.copy(),
                        final_bbox=list(bbox),
                        detection_type=session.detection_type,
                        max_confidence=session.max_confidence,
                        action=session.action,
                    )
                    # Queue clip for saving (outside lock)
                    self.processing_queue.append(clip_session)
                    # Reset for next clip
                    session.clip_number += 1
                    session.last_clip_save_time = current_time
                    session.frames = []
                    session.crop_frames = []
                    session.initial_frame = compressed_frame
                    session.initial_bbox = list(bbox)
                    logger.info(f"Queued clip {clip_session.clip_number + 1} for track {track_id}, starting clip {session.clip_number + 1}")
                
                return session
            else:
                # Get pre-roll frames (already compressed) - convert to new format with bbox
                pre_roll_frames = []
                pre_roll_crop_frames = []
                if camera_id in self.pre_roll_buffers:
                    # Pre-roll frames don't have bbox, use current bbox as fallback
                    for ts, compressed in self.pre_roll_buffers[camera_id]:
                        pre_roll_frames.append((ts, compressed, list(bbox)))
                        # Pre-roll doesn't have crops, add None to maintain sync
                        pre_roll_crop_frames.append((ts, None))
                
                # Create crop frame for current detection (only for PTZ targets)
                current_crop_frame = None
                if is_ptz_target:
                    cropped_frame = self._create_crop_frame(frame, bbox)
                    if cropped_frame is not None:
                        compressed_crop = compress_frame(cropped_frame, quality=85)
                        if compressed_crop is not None:
                            current_crop_frame = (current_time, compressed_crop)
                    else:
                        current_crop_frame = (current_time, None)
                else:
                    # Not PTZ target - add None to maintain sync
                    current_crop_frame = (current_time, None)
                
                # Create new session - ensure frames and crop_frames are synchronized
                session = RecordingSession(
                    camera_id=camera_id,
                    track_id=track_id,
                    start_time=current_time,
                    last_detection_time=current_time,
                    last_clip_save_time=current_time,  # Initialize clip timer
                    clip_number=0,
                    frames=pre_roll_frames + [(current_time, compressed_frame, list(bbox))],  # Include current frame with bbox
                    crop_frames=pre_roll_crop_frames + [current_crop_frame],  # Maintain same length
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
                
                # Set crop dimensions (standard size for PTZ crops)
                if is_ptz_target:
                    session.crop_width = 320
                    session.crop_height = 240
                
                self.active_sessions[camera_id][track_id] = session
                
                logger.info(f"Started recording for track {track_id} on camera {camera_id}")
                
                return session
    
    def _create_crop_frame(self, frame: any, bbox: List[int], padding: int = 50, target_size: Tuple[int, int] = (320, 240)) -> Optional[any]:
        """
        Create a cropped and resized frame of the aircraft.
        
        Args:
            frame: Full frame
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding around the bounding box
            target_size: Target size (width, height) for the crop
            
        Returns:
            Cropped and resized frame
        """
        if frame is None or bbox is None or len(bbox) < 4:
            return None
        
        try:
            import numpy as np
            
            x1, y1, x2, y2 = map(int, bbox)
            frame_height, frame_width = frame.shape[:2]
            
            # Add padding
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(frame_width, x2 + padding)
            crop_y2 = min(frame_height, y2 + padding)
            
            # Extract crop
            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            
            if crop.size == 0:
                return None
            
            # Resize to target size while maintaining aspect ratio
            crop_h, crop_w = crop.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scale to fit in target size
            scale = min(target_w / crop_w, target_h / crop_h)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            
            if new_w <= 0 or new_h <= 0:
                return None
            
            # Resize crop
            resized_crop = cv2.resize(crop, (new_w, new_h))
            
            # Create canvas with target size and place resized crop in center
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # Calculate position to center the crop
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_crop
            
            return canvas
            
        except Exception as e:
            logger.debug(f"Error creating crop frame: {e}")
            return None
    
    def check_and_complete_recordings(
        self,
        camera_id: str,
        current_track_ids: Optional[Set[int]] = None,
    ) -> List[RecordingSession]:
        """
        Check for recordings that should be completed or clips that should be saved.
        Now saves 5-second clips while track is active, and final clip when track disappears.
        """
        current_time = time.time()
        completed_sessions = []
        
        with self.lock:
            if camera_id not in self.active_sessions:
                return []
            
            for track_id, session in list(self.active_sessions[camera_id].items()):
                # Check if track is still present
                track_present = current_track_ids is None or track_id in current_track_ids
                
                if track_present:
                    # Track is still active - check if we need to save a 5-second clip
                    time_since_last_save = current_time - session.last_clip_save_time
                    if time_since_last_save >= CLIP_DURATION_SECONDS and len(session.frames) > 0:
                        # Create a copy of the session for this clip to save
                        clip_session = RecordingSession(
                            camera_id=session.camera_id,
                            track_id=session.track_id,
                            start_time=session.last_clip_save_time,
                            last_detection_time=current_time,
                            last_clip_save_time=session.last_clip_save_time,
                            clip_number=session.clip_number,
                            frames=session.frames.copy(),
                            crop_frames=session.crop_frames.copy(),
                            frame_width=session.frame_width,
                            frame_height=session.frame_height,
                            crop_width=session.crop_width,
                            crop_height=session.crop_height,
                            fps=session.fps,
                            initial_frame=session.initial_frame,
                            final_frame=session.final_frame,
                            initial_bbox=session.initial_bbox.copy(),
                            final_bbox=session.final_bbox.copy(),
                            detection_type=session.detection_type,
                            max_confidence=session.max_confidence,
                            action=session.action,
                        )
                        # Queue clip for saving
                        self.processing_queue.append(clip_session)
                        # Reset for next clip
                        session.clip_number += 1
                        session.last_clip_save_time = current_time
                        session.frames = []
                        session.crop_frames = []
                        session.initial_frame = None  # Will be set on next detection
                        logger.info(f"Queued clip {clip_session.clip_number + 1} for track {track_id}, starting clip {session.clip_number + 1}")
                else:
                    # Track disappeared - save final clip if there are frames
                    if len(session.frames) > 0:
                        # Create a copy for final clip
                        final_clip_session = RecordingSession(
                            camera_id=session.camera_id,
                            track_id=session.track_id,
                            start_time=session.last_clip_save_time if session.last_clip_save_time > 0 else session.start_time,
                            last_detection_time=session.last_detection_time,
                            last_clip_save_time=session.last_clip_save_time,
                            clip_number=session.clip_number,
                            frames=session.frames.copy(),
                            crop_frames=session.crop_frames.copy(),
                            frame_width=session.frame_width,
                            frame_height=session.frame_height,
                            crop_width=session.crop_width,
                            crop_height=session.crop_height,
                            fps=session.fps,
                            initial_frame=session.initial_frame,
                            final_frame=session.final_frame,
                            initial_bbox=session.initial_bbox.copy(),
                            final_bbox=session.final_bbox.copy(),
                            detection_type=session.detection_type,
                            max_confidence=session.max_confidence,
                            action=session.action,
                        )
                        # Queue final clip for saving
                        self.processing_queue.append(final_clip_session)
                        logger.info(f"Queued final clip for track {track_id}")
                    
                    # Mark session as complete and remove
                    session.is_complete = True
                    completed_sessions.append(session)
                    del self.active_sessions[camera_id][track_id]
                    
                    logger.info(f"Track {track_id} disappeared")
        
        return completed_sessions
    
    def _processing_worker(self):
        """Background worker to save clips and recordings"""
        while self.processing_active:
            try:
                session = None
                if self.processing_queue:
                    session = self.processing_queue.pop(0)
                
                if session:
                    # Use clip save method for all (handles both clips and final recordings)
                    self._save_clip_and_create_alert(session)
                    # Clear session frames after processing to free memory
                    session.frames = []
                    session.crop_frames = []
                    session.initial_frame = None
                    session.final_frame = None
                else:
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error in recording processor: {e}")
                time.sleep(1)
    
    def _save_clip_and_create_alert(self, session: RecordingSession):
        """Save a 5-second clip and create alert immediately"""
        from backend.storage import minio_storage
        from aircraft_detection.models import AircraftDetection
        from cameras.models import Camera
        from django.utils import timezone
        
        if not session.frames:
            logger.warning(f"No frames to save for clip {session.clip_number} of track {session.track_id}")
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
        crop_video_path = None
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
                        image_bytes=session.initial_frame,
                        suffix="_initial",
                    )
                    logger.info(f"Saved initial frame to {initial_image_path}")
                except Exception as e:
                    logger.error(f"Failed to save initial frame: {e}")
            
            # 2. Save full video
            try:
                video_path = self._create_video(session, alert_id, minio_storage)
            except Exception as e:
                logger.error(f"Failed to save video: {e}")
            
            # 3. Save cropped/zoomed aircraft video
            if session.crop_frames:
                try:
                    crop_video_path = self._create_crop_video(session, alert_id, minio_storage)
                    logger.info(f"Saved crop video to {crop_video_path}")
                except Exception as e:
                    logger.error(f"Failed to save crop video: {e}")
            
            # 4. Create alert in database
            clip_duration = len(session.frames) / session.fps if session.fps > 0 else CLIP_DURATION_SECONDS
            
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
                title=f"{session.detection_type} Detected (Clip {session.clip_number + 1})",
                description=f"{session.detection_type} detected - {clip_duration:.1f}s clip "
                           f"with {session.max_confidence:.0%} confidence",
                image_path=initial_image_path,
                video_path=video_path,
                crop_video_path=crop_video_path,
            )
            detection.save()
            
            # 5. Broadcast alert via WebSocket
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
                'crop_video_url': detection.crop_video_url,
                'title': detection.title,
                'description': detection.description,
                'severity': detection.severity,
                'status': detection.status,
                'is_read': detection.is_read,
                'detection_time': detection.detection_time.isoformat(),
                'timestamp': detection.detection_time.isoformat(),
                'duration': clip_duration,
            })
            
            logger.info(f"Created alert {alert_id} for clip {session.clip_number + 1} of track {session.track_id}")
            
        except Exception as e:
            logger.error(f"Failed to save clip and create alert: {e}")
    
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
        crop_video_path = None
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
            
            # 2. Save full video (decompress frames and encode with H.264)
            try:
                video_path = self._create_video(session, alert_id, minio_storage)
            except Exception as e:
                logger.error(f"Failed to save video: {e}")
            
            # 3. Save cropped/zoomed aircraft video
            if session.crop_frames:
                try:
                    crop_video_path = self._create_crop_video(session, alert_id, minio_storage)
                    logger.info(f"Saved crop video to {crop_video_path}")
                except Exception as e:
                    logger.error(f"Failed to save crop video: {e}")
            
            # 4. Create alert in database
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
                crop_video_path=crop_video_path,
            )
            detection.save()
            
            # 5. Broadcast alert via WebSocket
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
                'crop_video_url': detection.crop_video_url,
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
            
            # Decompress and write frames (format: timestamp, compressed_frame, bbox)
            # Draw red bbox on each frame for full video
            for frame_data in session.frames:
                if len(frame_data) == 3:
                    timestamp, compressed_frame, bbox = frame_data
                else:
                    # Backward compatibility with old format
                    timestamp, compressed_frame = frame_data[:2]
                    bbox = session.initial_bbox if hasattr(session, 'initial_bbox') else None
                
                frame = decompress_frame(compressed_frame)
                if frame is not None:
                    if frame.shape[:2] != (session.frame_height, session.frame_width):
                        frame = cv2.resize(frame, (session.frame_width, session.frame_height))
                    
                    # Draw red bbox on frame for full video (annotated)
                    if bbox and len(bbox) >= 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        color = (0, 0, 255)  # Red in BGR
                        thickness = 3
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
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
    
    def _create_crop_video(self, session: RecordingSession, alert_id: str, minio_storage) -> Optional[str]:
        """
        Create cropped/zoomed aircraft video from crop frames.
        Ensures crop video has the same length as full video by creating crops from full frames if needed.
        """
        if not session.frames:
            return None
        
        # If no crop frames, we can't create crop video (shouldn't happen for PTZ targets)
        if not session.crop_frames:
            logger.warning(f"No crop frames available for track {session.track_id}, skipping crop video")
            return None
        
        tmp_raw_path = None
        tmp_h264_path = None
        
        try:
            # Create temporary file for raw video
            with tempfile.NamedTemporaryFile(suffix='_crop.avi', delete=False) as tmp_file:
                tmp_raw_path = tmp_file.name
            
            # Write frames using OpenCV (raw format)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fps = max(session.fps, 5)  # Minimum 5 FPS
            
            # Use crop dimensions
            writer = cv2.VideoWriter(
                tmp_raw_path,
                fourcc,
                fps,
                (session.crop_width, session.crop_height)
            )
            
            if not writer.isOpened():
                logger.error("Failed to open crop video writer")
                return None
            
            # Create a map of crop frames by timestamp for quick lookup
            crop_frame_map = {ts: cf for ts, cf in session.crop_frames if cf is not None}
            
            # Process full frames and create corresponding crop frames
            # This ensures crop video has exactly the same frames as full video
            frames_written = 0
            for frame_idx, frame_data in enumerate(session.frames):
                # Extract data from frame (format: timestamp, compressed_frame, bbox)
                if len(frame_data) == 3:
                    timestamp, compressed_full_frame, bbox = frame_data
                else:
                    # Backward compatibility
                    timestamp, compressed_full_frame = frame_data[:2]
                    bbox = session.initial_bbox if frame_idx < len(session.frames) / 2 else session.final_bbox
                
                # Try to get existing crop frame from crop_frames list (by index for sync)
                compressed_crop = None
                if frame_idx < len(session.crop_frames):
                    _, compressed_crop = session.crop_frames[frame_idx]
                else:
                    # Fallback to timestamp lookup
                    compressed_crop = crop_frame_map.get(timestamp)
                
                if compressed_crop is None:
                    # Crop frame missing - create it from full frame using bbox from this frame
                    full_frame = decompress_frame(compressed_full_frame)
                    if full_frame is not None and bbox and len(bbox) >= 4:
                        cropped_frame = self._create_crop_frame(full_frame, bbox)
                        if cropped_frame is not None:
                            compressed_crop_bytes = compress_frame(cropped_frame, quality=85)
                            if compressed_crop_bytes:
                                compressed_crop = compressed_crop_bytes
                
                # Decompress and write crop frame
                if compressed_crop:
                    crop_frame = decompress_frame(compressed_crop)
                    if crop_frame is not None:
                        # Ensure frame matches expected dimensions
                        if crop_frame.shape[:2] != (session.crop_height, session.crop_width):
                            crop_frame = cv2.resize(crop_frame, (session.crop_width, session.crop_height))
                        writer.write(crop_frame)
                        frames_written += 1
            
            if frames_written == 0:
                logger.warning(f"No crop frames written for track {session.track_id}")
                writer.release()
                return None
            
            writer.release()
            
            # Convert to H.264 using ffmpeg for browser compatibility
            with tempfile.NamedTemporaryFile(suffix='_crop.mp4', delete=False) as tmp_file:
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
                logger.warning(f"ffmpeg crop conversion failed: {result.stderr.decode()}")
                # Fall back to raw video
                tmp_h264_path = tmp_raw_path
                tmp_raw_path = None
            
            # Upload to MinIO with _crop suffix
            with open(tmp_h264_path, 'rb') as f:
                video_bytes = f.read()
            
            # Upload crop video
            crop_video_path = minio_storage.upload_alert_video(
                camera_id=session.camera_id,
                alert_id=alert_id,
                video_bytes=video_bytes,
                suffix="_crop",
            )
            
            logger.info(f"Saved crop video to {crop_video_path}: "
                       f"{frames_written} frames (matching full video: {len(session.frames)} frames), "
                       f"{len(video_bytes) / 1024 / 1024:.1f}MB")
            
            return crop_video_path
            
        except Exception as e:
            logger.error(f"Error creating crop video: {e}")
            return None
            
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
                    session.crop_frames = []
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
                    session.crop_frames = []
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
                    session.crop_frames = []
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
                'clip_duration_seconds': CLIP_DURATION_SECONDS,
                'pre_roll_seconds': PRE_ROLL_SECONDS,
            }


# Global video recorder instance
_video_recorder: Optional[VideoRecorder] = None


def get_video_recorder() -> VideoRecorder:
    """Get or create the global video recorder instance"""
    global _video_recorder
    if _video_recorder is None:
        _video_recorder = VideoRecorder()
    return _video_recorder
