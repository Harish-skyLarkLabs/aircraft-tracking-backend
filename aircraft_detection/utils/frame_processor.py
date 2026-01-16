"""
Enhanced Frame Processor for aircraft detection with PTZ tracking support
Based on the ML team's aircraft_det.py demo script
Includes:
- Object tracking with PTZ target selection
- Session-based recording
- Edge filtering and size filtering
- Drawing utilities for visualization
"""
import cv2
import numpy as np
import threading
import queue
import time
import logging
from typing import Dict, List, Optional, Tuple, Callable, Set
from datetime import datetime

from .aircraft_detector import get_detector
from .tracker import AircraftTracker, IOUTracker
from .video_recorder import get_video_recorder
from .drawing_utils import draw_polygon_roi
from .ptz_controller import PTZController, get_ptz_controller

logger = logging.getLogger(__name__)


# Processing configuration
FRAMES_BEFORE_PRESET = 120  # Frames without detection before returning to preset


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
    Enhanced Frame Processor for aircraft detection with PTZ tracking
    
    Features:
    - Object tracking with enhanced AircraftTracker
    - PTZ target selection and control
    - Session-based recording (one alert per track)
    - Edge filtering and size filtering
    - Real-time visualization with aircraft crop overlay
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
        # PTZ settings
        ptz_controller: Optional[PTZController] = None,
        enable_ptz_tracking: bool = True,
        enable_zoom_control: bool = True,
        # Tracker settings
        enable_size_filter: bool = True,
        min_aircraft_width: int = 10,
        enable_edge_filtering: bool = True,
        edge_margin_percent: float = 7,
        lock_only_mode: bool = True,
        min_consecutive_detections: int = 5,
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.roi_points = roi_points
        self.confidence_threshold = confidence_threshold
        self.max_fps = max_fps
        self.debug_visualization = debug_visualization
        self.enable_recording = enable_recording
        
        # PTZ settings
        self.ptz_controller = ptz_controller
        self.enable_ptz_tracking = enable_ptz_tracking
        self.enable_zoom_control = enable_zoom_control
        
        # Initialize detector
        self.detector = get_detector(confidence_threshold=confidence_threshold)
        
        # Initialize enhanced tracker
        self.tracker = AircraftTracker(
            iou_threshold=0.3,
            max_age=30,
            min_hits=2,
            enable_size_filter=enable_size_filter,
            min_aircraft_width=min_aircraft_width,
            enable_edge_filtering=enable_edge_filtering,
            edge_margin_percent=edge_margin_percent,
            lock_only_mode=lock_only_mode,
            min_consecutive_detections=min_consecutive_detections,
        )
        
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
        
        # PTZ tracking state
        self.current_ptz_target: Optional[Dict] = None
        self.preset_triggered = False
        self.zoom_stabilization_counter = 0
        self.last_ptz_time = 0
        self.ptz_cooldown = 0.1
        
        # Callbacks
        self.detection_callback: Optional[Callable] = None
        
        # Processing queue and thread
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_active = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        
        logger.info(f"FrameProcessor initialized for camera {camera_id} "
                   f"(ptz={ptz_controller is not None}, recording={enable_recording})")
    
    def set_ptz_controller(self, ptz_controller: PTZController):
        """Set the PTZ controller for this processor"""
        self.ptz_controller = ptz_controller
        logger.info(f"PTZ controller set for camera {self.camera_id}")
    
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
        """Process a single frame synchronously with PTZ tracking"""
        self.frame_counter += 1
        original_frame = frame.copy()
        
        try:
            # Add frame to video recorder buffer (for pre-roll)
            if self.video_recorder:
                self.video_recorder.add_frame_to_buffer(self.camera_id, original_frame)
            
            # Run detection
            annotated_frame, raw_detections = self.detector.detect_and_plot(frame, self.roi_points)
            
            # Convert raw detections to tracker format
            detection_dicts = []
            for d in raw_detections:
                detection_dicts.append({
                    'box': d['bbox'],
                    'confidence': d['confidence'],
                    'class_id': d['class_id'],
                    'class_name': d['class_name'],
                })
            
            # Update tracker and get PTZ target
            detections_with_tracks, ptz_target = self.tracker.update(
                detection_dicts,
                frame.shape
            )
            self.current_ptz_target = ptz_target
            
            # Build detection results and update current track IDs
            detections = []
            new_track_ids: Set[int] = set()
            
            for detection in detections_with_tracks:
                track_id = detection.get('track_id')
                if track_id:
                    new_track_ids.add(track_id)
                
                is_ptz_target = ptz_target and track_id == ptz_target.get('track_id')
                
                det_result = {
                    'track_id': track_id,
                    'bbox': detection['box'],
                    'confidence': detection.get('confidence', 0),
                    'class_id': detection.get('class_id', 0),
                    'class_name': detection.get('class_name', 'Aircraft'),
                    'width': detection.get('width', 0),
                    'height': detection.get('height', 0),
                    'size_ratio': detection.get('size_ratio', 0),
                    'timestamp': datetime.now().isoformat(),
                    'is_ptz_target': is_ptz_target,
                }
                detections.append(det_result)
            
            # Update current track IDs
            self.current_track_ids = new_track_ids
            
            # Handle PTZ tracking
            if ptz_target and self.ptz_controller and self.enable_ptz_tracking:
                self._handle_ptz_tracking(ptz_target, frame.shape)
            elif self.ptz_controller:
                self._handle_no_detection()
            
            # Draw visualization - only PTZ-locked aircraft bbox on clean frame
            processed_frame = self._draw_visualization(
                original_frame,  # Use original clean frame
                original_frame,
                detections,
                ptz_target
            )
            
            # Only record PTZ-locked aircraft (create crop videos only for locked aircraft)
            # Store original frame - bbox will be drawn when creating full video
            if self.video_recorder and ptz_target:
                track_id = ptz_target.get('track_id')
                if track_id:
                    track = self.tracker.get_track(track_id)
                    action = track.get_direction() if track else 'unknown'
                    
                    # Find the detection for this PTZ target
                    ptz_detection = next(
                        (d for d in detections if d.get('track_id') == track_id),
                        None
                    )
                    
                    if ptz_detection:
                        self.video_recorder.start_or_update_recording(
                            camera_id=self.camera_id,
                            track_id=track_id,
                            detection_type=ptz_detection.get('class_name', 'Aircraft'),
                            confidence=ptz_detection.get('confidence', 0),
                            bbox=ptz_target['box'],
                            frame=original_frame,  # Store original frame, bbox drawn in video creation
                            action=action,
                            is_ptz_target=True,  # Indicate this is PTZ-locked
                        )
            
            # Check for completed recordings (tracks that disappeared)
            if self.video_recorder:
                self.video_recorder.check_and_complete_recordings(
                    self.camera_id,
                    self.current_track_ids
                )
            
            # Store in buffer
            self.frame_buffer.add(processed_frame, detections)
            
            return processed_frame, detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, []
    
    def _handle_ptz_tracking(self, ptz_target: Dict, frame_shape: Tuple):
        """Handle PTZ tracking for the target aircraft"""
        if not self.ptz_controller:
            return
        
        # Reset preset trigger
        self.preset_triggered = False
        
        # Get aircraft center
        box = ptz_target['box']
        aircraft_center_x = (box[0] + box[2]) / 2
        aircraft_center_y = (box[1] + box[3]) / 2
        
        # Calculate distance from frame center
        frame_center_x = frame_shape[1] // 2
        frame_center_y = frame_shape[0] // 2
        
        # Calculate normalized distance from center (-1 to 1)
        norm_x = (aircraft_center_x - frame_center_x) / frame_center_x if frame_center_x > 0 else 0
        norm_y = (aircraft_center_y - frame_center_y) / frame_center_y if frame_center_y > 0 else 0
        
        # Aggressive centering - smaller tolerance
        if abs(norm_x) > 0.05 or abs(norm_y) > 0.05:
            self.ptz_controller.track_object_smooth(
                frame_shape[1], frame_shape[0],
                aircraft_center_x, aircraft_center_y,
                ptz_target.get('width'), ptz_target.get('height')
            )
            logger.debug(f"[CENTERING] Object at ({norm_x:.2f}, {norm_y:.2f})")
        
        # Handle zoom control if enabled
        if self.enable_zoom_control and self.zoom_stabilization_counter <= 0:
            width = ptz_target.get('width', 0)
            height = ptz_target.get('height', 0)
            if width and height:
                zoomed = self.ptz_controller.control_zoom_for_aircraft_by_area(
                    width, height, self.ptz_cooldown
                )
                if zoomed:
                    self.zoom_stabilization_counter = 25
        
        # Decrease stabilization counter
        if self.zoom_stabilization_counter > 0:
            self.zoom_stabilization_counter -= 1
    
    def _handle_no_detection(self):
        """Handle case when no aircraft is detected"""
        if not self.ptz_controller:
            return
        
        # Check if we should go to preset
        if self.tracker.should_go_to_preset(FRAMES_BEFORE_PRESET) and not self.preset_triggered:
            logger.info(f"No detection for {FRAMES_BEFORE_PRESET} frames, going to preset")
            self.ptz_controller.go_to_preset()
            self.preset_triggered = True
    
    def _draw_visualization(
        self,
        annotated_frame: np.ndarray,
        original_frame: np.ndarray,
        detections: List[Dict],
        ptz_target: Optional[Dict]
    ) -> np.ndarray:
        """Draw visualization overlays on the frame - only PTZ-locked aircraft"""
        if not self.debug_visualization:
            return annotated_frame
        
        display_frame = original_frame.copy()  # Use original frame, not annotated
        
        # Only draw bounding box for PTZ-locked aircraft
        if ptz_target:
            x1, y1, x2, y2 = map(int, ptz_target['box'])
            
            # Draw red bounding box for PTZ-locked aircraft
            color = (0, 0, 255)  # Red in BGR
            thickness = 3
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
        
        return display_frame
    
    def _draw_edge_margins(self, frame: np.ndarray):
        """Draw edge margins on frame for visualization"""
        if not self.tracker.enable_edge_filtering:
            return
        
        frame_height, frame_width = frame.shape[:2]
        edge_margin_x = int((frame_width * self.tracker.edge_margin_percent) / 100)
        edge_margin_y = int((frame_height * self.tracker.edge_margin_percent) / 100)
        
        color = (100, 100, 100)
        thickness = 1
        
        cv2.line(frame, (edge_margin_x, 0), (edge_margin_x, frame_height), color, thickness)
        cv2.line(frame, (frame_width - edge_margin_x, 0), (frame_width - edge_margin_x, frame_height), color, thickness)
        cv2.line(frame, (0, edge_margin_y), (frame_width, edge_margin_y), color, thickness)
        cv2.line(frame, (0, frame_height - edge_margin_y), (frame_width, frame_height - edge_margin_y), color, thickness)
    
    def _draw_detection(self, frame: np.ndarray, detection: Dict, ptz_target: Optional[Dict]):
        """Draw detection box with info"""
        x1, y1, x2, y2 = map(int, detection['bbox'])
        track_id = detection.get('track_id')
        is_ptz_target = detection.get('is_ptz_target', False)
        
        # Get track info
        track = self.tracker.get_track(track_id) if track_id else None
        has_enough_detections = track and track.detection_count >= self.tracker.min_consecutive_detections
        
        # Choose color based on status
        width = detection.get('width', x2 - x1)
        if self.tracker.enable_size_filter and width < self.tracker.min_aircraft_width:
            color = (255, 255, 255)  # White for small aircraft
            thickness = 1
        elif is_ptz_target:
            color = (0, 0, 255)  # Red for PTZ tracked
            thickness = 3
        elif has_enough_detections:
            color = (0, 255, 0)  # Green for confirmed
            thickness = 2
        else:
            color = (0, 255, 255)  # Yellow for confirming
            thickness = 2
        
        # Add padding
        x1 -= 10
        y1 -= 10
        x2 += 10
        y2 += 10
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if track and not has_enough_detections:
            label = f"Confirming {track.detection_count}/{self.tracker.min_consecutive_detections}"
        else:
            confidence = detection.get('confidence', 0)
            ptz_status = " [PTZ-LOCKED]" if is_ptz_target else ""
            label = f"{detection['class_name']}:{confidence:.2f} ({int(width)}px){ptz_status}"
        
        self._draw_text_with_background(frame, label, x1, y1)
        
        # Draw area
        area = detection.get('width', 0) * detection.get('height', 0)
        if area > 0:
            area_label = f"Area: {int(area)} px²"
            self._draw_text_with_background(frame, area_label, x1, y1 + 25)
    
    def _draw_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        scale: float = 0.6,
        thickness: int = 1
    ):
        """Draw text with background rectangle"""
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        text_width, text_height = text_size
        rect_x1, rect_y1 = x - 5, y - text_height - 10
        rect_x2, rect_y2 = x + text_width + 5, y
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        cv2.putText(frame, text, (x, y - 5), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    def _draw_center_crosshair(self, frame: np.ndarray):
        """Draw center crosshair for PTZ tracking"""
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (255, 0, 0), 2)
        cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (255, 0, 0), 2)
    
    def _draw_aircraft_crop_overlay(
        self,
        frame: np.ndarray,
        source_frame: np.ndarray,
        detection: Dict,
        position: str = 'top_right'
    ):
        """Draw real-time aircraft crop in corner with area info"""
        try:
            x1, y1, x2, y2 = map(int, detection['box'])
            
            # Add padding
            padding = 30
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(source_frame.shape[1], x2 + padding)
            crop_y2 = min(source_frame.shape[0], y2 + padding)
            
            aircraft_crop = source_frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            
            if aircraft_crop.size == 0:
                return
            
            # Resize to fixed size
            max_crop_width = 320
            max_crop_height = 240
            
            crop_h, crop_w = aircraft_crop.shape[:2]
            aspect_ratio = crop_w / crop_h if crop_h > 0 else 1
            
            if aspect_ratio > max_crop_width / max_crop_height:
                new_width = max_crop_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = max_crop_height
                new_width = int(new_height * aspect_ratio)
            
            if new_width <= 0 or new_height <= 0:
                return
            
            aircraft_crop_resized = cv2.resize(aircraft_crop, (new_width, new_height))
            
            # Calculate position
            margin = 10
            if position == 'top_right':
                overlay_x = frame.shape[1] - new_width - margin
                overlay_y = margin
            else:
                overlay_x = margin
                overlay_y = margin
            
            if overlay_x + new_width > frame.shape[1] or overlay_y + new_height > frame.shape[0]:
                return
            
            # Draw background with border
            bg_x1, bg_y1 = overlay_x - 5, overlay_y - 5
            bg_x2, bg_y2 = overlay_x + new_width + 5, overlay_y + new_height + 5
            
            # Semi-transparent dark background
            overlay_bg = frame[bg_y1:bg_y2, bg_x1:bg_x2].copy()
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(frame[bg_y1:bg_y2, bg_x1:bg_x2], 0.3, overlay_bg, 0.7, 0, frame[bg_y1:bg_y2, bg_x1:bg_x2])
            
            # Red border
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), 3)
            
            # Place crop
            frame[overlay_y:overlay_y+new_height, overlay_x:overlay_x+new_width] = aircraft_crop_resized
            
            # Add label
            area = detection.get('width', 0) * detection.get('height', 0)
            track_id = detection.get('track_id', 'N/A')
            label = f"PTZ LOCKED ID:{track_id} | Area: {int(area):,} px²"
            
            label_bg_y1 = bg_y1 - 30
            label_bg_y2 = bg_y1 - 2
            
            if label_bg_y1 >= 0:
                cv2.rectangle(frame, (bg_x1, label_bg_y1), (bg_x2, label_bg_y2), (0, 0, 255), -1)
                cv2.putText(frame, label, (bg_x1 + 5, label_bg_y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        except Exception as e:
            logger.debug(f"Error drawing aircraft crop overlay: {e}")
    
    def _draw_status_info(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        ptz_target: Optional[Dict]
    ):
        """Draw status information overlay"""
        tracker_stats = self.tracker.get_stats()
        
        lock_status = "🔒 LOCKED" if tracker_stats['is_locked'] else "🔓 OPEN"
        mode_status = "LOCK-ONLY" if tracker_stats['lock_only_mode'] else "NORMAL"
        
        ptz_status = "PTZ:ON" if self.enable_ptz_tracking else "PTZ:OFF"
        zoom_status = f"ZOOM:{'ON' if self.enable_zoom_control else 'OFF'}"
        
        status_lines = [
            f"Camera: {self.camera_name} | Mode: {mode_status} | Status: {lock_status}",
            f"Active Tracks: {tracker_stats['active_tracks']} | PTZ Target: {tracker_stats['currently_tracked_id']}",
            f"{zoom_status} | {ptz_status}",
            f"No Detection: {tracker_stats['frames_without_detection']}/{FRAMES_BEFORE_PRESET}",
        ]
        
        if ptz_target:
            target_area = ptz_target.get('width', 0) * ptz_target.get('height', 0)
            status_lines.append(f"PTZ Target ID:{ptz_target.get('track_id')} Area: {int(target_area)} px²")
        
        for i, status_text in enumerate(status_lines):
            y_pos = 60 + (i * 25)
            cv2.putText(frame, status_text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
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
    
    def set_ptz_tracking_enabled(self, enabled: bool):
        """Enable or disable PTZ tracking"""
        self.enable_ptz_tracking = enabled
        logger.info(f"PTZ tracking {'enabled' if enabled else 'disabled'} for camera {self.camera_id}")
    
    def set_zoom_control_enabled(self, enabled: bool):
        """Enable or disable zoom control"""
        self.enable_zoom_control = enabled
        if self.ptz_controller:
            self.ptz_controller.enable_zoom_control = enabled
        logger.info(f"Zoom control {'enabled' if enabled else 'disabled'} for camera {self.camera_id}")
    
    def clear_tracking_lock(self):
        """Clear current PTZ tracking lock"""
        self.tracker.clear_lock()
        self.current_ptz_target = None
    
    def go_to_preset(self, preset_number: Optional[int] = None):
        """Go to PTZ preset position"""
        if self.ptz_controller:
            self.ptz_controller.go_to_preset(preset_number)
            self.preset_triggered = False
    
    def emergency_stop_ptz(self):
        """Emergency stop PTZ movement"""
        if self.ptz_controller:
            self.ptz_controller.emergency_stop()
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        stats = {
            'camera_id': self.camera_id,
            'frame_counter': self.frame_counter,
            'fps': round(self.fps, 1),
            'processing_enabled': self.processing_enabled,
            'queue_size': self.frame_queue.qsize(),
            'active_tracks': len(self.current_track_ids),
            'ptz_tracking_enabled': self.enable_ptz_tracking,
            'zoom_control_enabled': self.enable_zoom_control,
            'current_ptz_target': self.current_ptz_target.get('track_id') if self.current_ptz_target else None,
        }
        
        # Add tracker stats
        stats['tracker'] = self.tracker.get_stats()
        
        # Add PTZ stats
        if self.ptz_controller:
            stats['ptz'] = self.ptz_controller.get_status()
        
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
