"""
Frame processor for aircraft detection and tracking
"""
import cv2
import numpy as np
import threading
import queue
import time
import logging
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime

from .aircraft_detector import get_detector
from .tracker import IOUTracker
from .drawing_utils import (
    draw_aircraft_detection,
    draw_polygon_roi,
    draw_info_overlay,
)

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
    """Process frames for aircraft detection and tracking"""
    
    def __init__(
        self,
        camera_id: str,
        camera_name: str = "",
        roi_points: Optional[List[List[int]]] = None,
        confidence_threshold: float = 0.5,
        max_fps: int = 15,
        debug_visualization: bool = True,
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.roi_points = roi_points
        self.confidence_threshold = confidence_threshold
        self.max_fps = max_fps
        self.debug_visualization = debug_visualization
        
        # Initialize detector and tracker
        self.detector = get_detector(confidence_threshold=confidence_threshold)
        self.tracker = IOUTracker(iou_threshold=0.3, max_age=30, min_hits=3)
        
        # Frame buffer
        self.frame_buffer = FrameBuffer(size=5)
        
        # Processing state
        self.processing_enabled = True
        self.frame_counter = 0
        self.last_process_time = 0
        self.fps = 0
        
        # Detection history for action determination
        self.detection_history: Dict[int, List[Dict]] = {}
        
        # Callbacks
        self.detection_callback: Optional[Callable] = None
        
        # Processing queue and thread
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_active = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        
        logger.info(f"FrameProcessor initialized for camera {camera_id}")
    
    def _processing_worker(self):
        """Background worker thread for frame processing"""
        while self.processing_active:
            try:
                # Get frame from queue
                try:
                    frame = self.frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                
                if self.processing_enabled:
                    # Process the frame
                    start_time = time.time()
                    processed_frame, detections = self._process_frame_sync(frame)
                    process_time = time.time() - start_time
                    
                    # Calculate FPS
                    if process_time > 0:
                        self.fps = 1.0 / process_time
                    
                    self.last_process_time = process_time
                    self.result_queue.put((processed_frame, detections))
                else:
                    # Return original frame
                    self.result_queue.put((frame.copy(), []))
                
                self.frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
    
    def _process_frame_sync(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame synchronously"""
        self.frame_counter += 1
        processed_frame = frame.copy()
        
        try:
            # Run detection
            raw_detections = self.detector.detect(frame, self.roi_points)
            
            # Prepare detections for tracker
            bboxes = [d['bbox'] for d in raw_detections]
            class_ids = [d['class_id'] for d in raw_detections]
            
            # Update tracker
            tracked_objects = self.tracker.update(bboxes, class_ids)
            
            # Build detection results with tracking info
            detections = []
            
            for track_id, bbox in tracked_objects:
                # Find matching raw detection for confidence
                confidence = 0.0
                class_name = 'aircraft'
                class_id = 0
                
                for raw_det in raw_detections:
                    if self._iou(bbox, raw_det['bbox']) > 0.5:
                        confidence = raw_det['confidence']
                        class_name = raw_det['class_name']
                        class_id = raw_det['class_id']
                        break
                
                # Get track direction and determine action
                direction = self.tracker.get_track_direction(track_id)
                track = self.tracker.get_track(track_id)
                velocity = track.get_velocity() if track else None
                
                # Determine position in frame
                frame_height = frame.shape[0]
                bbox_center_y = (bbox[1] + bbox[3]) / 2
                if bbox_center_y < frame_height * 0.33:
                    position = 'top'
                elif bbox_center_y > frame_height * 0.66:
                    position = 'bottom'
                else:
                    position = 'middle'
                
                action = self.detector.determine_action(direction, position, velocity)
                
                detection = {
                    'track_id': track_id,
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'action': action,
                    'direction': direction,
                    'timestamp': datetime.now().isoformat(),
                }
                
                detections.append(detection)
                
                # Draw on frame if debug visualization is enabled
                if self.debug_visualization:
                    processed_frame = draw_aircraft_detection(
                        processed_frame,
                        bbox,
                        detection_type=class_name,
                        action=action,
                        track_id=track_id,
                        confidence=confidence,
                    )
                
                # Call detection callback if set
                if self.detection_callback and action in ['landing', 'takeoff']:
                    try:
                        self.detection_callback(detection)
                    except Exception as e:
                        logger.error(f"Error in detection callback: {e}")
            
            # Draw ROI if defined
            if self.debug_visualization and self.roi_points:
                processed_frame = draw_polygon_roi(processed_frame, self.roi_points)
            
            # Draw info overlay
            if self.debug_visualization:
                processed_frame = draw_info_overlay(
                    processed_frame,
                    fps=self.fps,
                    detection_count=len(detections),
                    camera_name=self.camera_name,
                )
            
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
        
        # Check if we're falling behind
        if self.frame_queue.qsize() >= 2:
            return self.frame_buffer.get_latest()
        
        try:
            # Queue the frame
            self.frame_queue.put(frame, timeout=0.1)
            
            # Wait for result
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
        return {
            'camera_id': self.camera_id,
            'frame_counter': self.frame_counter,
            'fps': round(self.fps, 1),
            'processing_enabled': self.processing_enabled,
            'queue_size': self.frame_queue.qsize(),
            'active_tracks': len(self.tracker.tracks),
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.processing_active = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        self.tracker.clear()
        logger.info(f"FrameProcessor cleaned up for camera {self.camera_id}")
    
    def __del__(self):
        self.cleanup()



