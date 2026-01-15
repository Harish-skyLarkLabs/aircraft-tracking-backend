"""
Frame processor for aircraft detection with tracking and session-based recording
"""
import numpy as np
import threading
import queue
import time
import logging
from typing import Dict, List, Optional, Tuple, Callable, Set
from datetime import datetime

from .aircraft_detector import get_detector
from .tracker import IOUTracker
from .video_recorder import get_video_recorder
from .drawing_utils import draw_polygon_roi

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Thread-safe frame buffer"""
    
    def __init__(self, size: int = 5):
        self.frames = []
        self.detections = []
        self.timestamps = []
        self.max_size = size
        self.lock = threading.Lock()
    
    def add(self, frame: np.ndarray, detections: List, timestamp: float = None):
        with self.lock:
            if timestamp is None:
                timestamp = time.time()
            
            self.frames.append(frame)
            self.detections.append(detections)
            self.timestamps.append(timestamp)
            
            if len(self.frames) > self.max_size:
                self.frames.pop(0)
                self.detections.pop(0)
                self.timestamps.pop(0)
    
    def get_latest(self) -> Tuple[Optional[np.ndarray], List]:
        with self.lock:
            if not self.frames:
                return None, []
            return self.frames[-1], self.detections[-1]


class FrameProcessor:
    """
    Process frames for aircraft detection with:
    - Object tracking (IOU-based)
    - Session-based recording (one alert per track with image + video)
    """
    
    def __init__(
        self,
        camera_id: str,
        camera_name: str = "",
        roi_points: Optional[List[List[int]]] = None,
        confidence_threshold: float = 0.25,
        max_fps: int = 15,
        debug_visualization: bool = True,
        enable_recording: bool = True,
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.roi_points = roi_points
        self.confidence_threshold = confidence_threshold
        self.max_fps = max_fps
        self.debug_visualization = debug_visualization
        self.enable_recording = enable_recording
        
        # Initialize detector
        self.detector = get_detector(confidence_threshold=confidence_threshold)
        
        # Initialize tracker
        self.tracker = IOUTracker(iou_threshold=0.3, max_age=30, min_hits=2)
        
        # Video recorder (handles recording + alert creation)
        self.video_recorder = get_video_recorder() if enable_recording else None
        
        # Frame buffer
        self.frame_buffer = FrameBuffer(size=5)
        
        # Processing state
        self.processing_enabled = True
        self.frame_counter = 0
        self.last_process_time = 0
        self.fps = 0
        
        # Track current active tracks
        self.current_track_ids: Set[int] = set()
        
        # Callbacks
        self.detection_callback: Optional[Callable] = None
        
        # Processing queue and thread
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_active = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        
        logger.info(f"FrameProcessor initialized for camera {camera_id} (recording={enable_recording})")
    
    def _processing_worker(self):
        """Background worker thread for frame processing"""
        while self.processing_active:
            try:
                try:
                    frame = self.frame_queue.get(timeout=0.2)
                except queue.Empty:
                    # Check for completed recordings periodically
                    if self.video_recorder:
                        self.video_recorder.check_and_complete_recordings(
                            self.camera_id, 
                            self.current_track_ids
                        )
                    continue
                
                if self.processing_enabled:
                    start_time = time.time()
                    processed_frame, detections = self._process_frame_sync(frame)
                    process_time = time.time() - start_time
                    
                    if process_time > 0:
                        self.fps = 1.0 / process_time
                    
                    self.last_process_time = process_time
                    self.result_queue.put((processed_frame, detections))
                else:
                    self.result_queue.put((frame.copy(), []))
                
                self.frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
    
    def _process_frame_sync(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame synchronously"""
        self.frame_counter += 1
        original_frame = frame.copy()
        
        try:
            # Add frame to video recorder buffer (for pre-roll)
            if self.video_recorder:
                self.video_recorder.add_frame_to_buffer(self.camera_id, original_frame)
            
            # Run detection and get annotated frame
            annotated_frame, raw_detections = self.detector.detect_and_plot(frame, self.roi_points)
            processed_frame = annotated_frame
            
            # Prepare detections for tracker
            bboxes = [d['bbox'] for d in raw_detections]
            class_ids = [d['class_id'] for d in raw_detections]
            
            # Update tracker
            tracked_objects = self.tracker.update(bboxes, class_ids)
            
            # Build detection results and update current track IDs
            detections = []
            new_track_ids: Set[int] = set()
            
            for track_id, bbox in tracked_objects:
                new_track_ids.add(track_id)
                
                # Find matching raw detection for confidence
                confidence = 0.0
                class_name = 'Aircraft'
                class_id = 0
                
                for raw_det in raw_detections:
                    if self._iou(bbox, raw_det['bbox']) > 0.5:
                        confidence = raw_det['confidence']
                        class_name = raw_det['class_name']
                        class_id = raw_det['class_id']
                        break
                
                detection = {
                    'track_id': track_id,
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'timestamp': datetime.now().isoformat(),
                }
                detections.append(detection)
                
                # Start or update recording session for this track
                if self.video_recorder:
                    self.video_recorder.start_or_update_recording(
                        camera_id=self.camera_id,
                        track_id=track_id,
                        detection_type=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        frame=original_frame,
                        action='unknown',  # Could be determined from tracker direction
                    )
            
            # Update current track IDs
            self.current_track_ids = new_track_ids
            
            # Check for completed recordings (tracks that disappeared)
            if self.video_recorder:
                self.video_recorder.check_and_complete_recordings(
                    self.camera_id,
                    self.current_track_ids
                )
            
            # Draw ROI if defined
            if self.debug_visualization and self.roi_points:
                processed_frame = draw_polygon_roi(processed_frame, self.roi_points)
            
            # Store in buffer
            self.frame_buffer.add(processed_frame, detections)
            
            return processed_frame, detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, []
    
    def _iou(self, box1: List, box2: List) -> float:
        """Calculate IOU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    async def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """Process a frame asynchronously"""
        import asyncio
        
        if not self.processing_enabled:
            self.frame_buffer.add(frame.copy(), [])
            return frame.copy(), []
        
        if self.frame_queue.qsize() >= 2:
            return self.frame_buffer.get_latest()
        
        try:
            self.frame_queue.put(frame, timeout=0.1)
            
            try:
                processed_frame, detections = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: self.result_queue.get(timeout=0.1)
                    ),
                    timeout=0.1
                )
                self.result_queue.task_done()
                return processed_frame, detections
            except (queue.Empty, asyncio.TimeoutError):
                return self.frame_buffer.get_latest()
                
        except queue.Full:
            return self.frame_buffer.get_latest()
    
    def set_detection_callback(self, callback: Callable):
        """Set callback function for detections"""
        self.detection_callback = callback
    
    def set_processing_enabled(self, enabled: bool):
        """Enable or disable processing"""
        self.processing_enabled = enabled
        logger.info(f"Processing {'enabled' if enabled else 'disabled'} for camera {self.camera_id}")
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        stats = {
            'camera_id': self.camera_id,
            'frame_counter': self.frame_counter,
            'fps': round(self.fps, 1),
            'processing_enabled': self.processing_enabled,
            'queue_size': self.frame_queue.qsize(),
            'active_tracks': len(self.current_track_ids),
        }
        
        if self.video_recorder:
            stats['video_recorder'] = self.video_recorder.get_stats()
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        self.processing_active = False
        
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Clean up tracking data
        self.tracker.clear()
        
        # Clean up video recorder data for this camera
        if self.video_recorder:
            self.video_recorder.cleanup_camera(self.camera_id)
        
        logger.info(f"FrameProcessor cleaned up for camera {self.camera_id}")
    
    def __del__(self):
        self.cleanup()
