"""
Enhanced Aircraft Tracker with PTZ support
Based on the ML team's aircraft_det.py demo script
Includes:
- TrackedAircraft class with detailed tracking state
- IOU and distance-based track association
- Lock-only mode for stable PTZ tracking
- Edge filtering to avoid tracking aircraft at frame edges
- Minimum consecutive detections before PTZ tracking
"""
import numpy as np
import time
import logging
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


# Tracking Configuration Constants
TRACKING_THRESHOLD = 80  # Max distance for track association
CENTER_TOLERANCE = 25
MAX_FRAMES_WITHOUT_DETECTION = 120
MIN_CONSECUTIVE_DETECTIONS = 5  # Minimum detections before PTZ tracking
ENABLE_SIZE_FILTER = True
MIN_AIRCRAFT_WIDTH = 10
ENABLE_EDGE_FILTERING = True
EDGE_MARGIN_PERCENT = 7
LOCK_ONLY_MODE = True


class TrackedAircraft:
    """
    Enhanced class to maintain track information for each aircraft
    Provides detailed tracking state, velocity history, and stability metrics
    """
    
    def __init__(self, track_id: int, detection: dict, frame_count: int):
        """
        Initialize a new tracked aircraft
        
        Args:
            track_id: Unique track identifier
            detection: Initial detection dict with 'box', 'confidence', 'size_ratio', 'timestamp'
            frame_count: Current frame number
        """
        self.track_id = track_id
        self.first_detection_frame = frame_count
        self.last_detection_frame = frame_count
        self.detection_count = 1
        self.consecutive_misses = 0
        self.is_active = True
        
        # History tracking
        self.confidence_history: List[float] = [detection.get('confidence', 0.0)]
        self.position_history: List[List[float]] = [detection['box']]
        self.size_history: List[float] = [detection.get('size_ratio', 0.0)]
        self.velocity_history: List[Tuple[float, float]] = []
        
        # Current state
        self.last_position = detection['box']
        self.last_timestamp = detection.get('timestamp', time.time())
        self.latest_detection = detection
        
        # Track metadata
        self.class_id = detection.get('class_id', 0)
        self.class_name = detection.get('class_name', 'Aircraft')
        
    def update_detection(self, detection: dict, frame_count: int):
        """Update track with new detection"""
        self.last_detection_frame = frame_count
        self.detection_count += 1
        self.consecutive_misses = 0
        self.latest_detection = detection
        
        # Update histories
        self.confidence_history.append(detection.get('confidence', 0.0))
        self.position_history.append(detection['box'])
        self.size_history.append(detection.get('size_ratio', 0.0))
        
        # Limit history size
        max_history = 20
        if len(self.confidence_history) > max_history:
            self.confidence_history.pop(0)
        if len(self.position_history) > max_history:
            self.position_history.pop(0)
        if len(self.size_history) > max_history:
            self.size_history.pop(0)
        
        # Calculate velocity
        current_timestamp = detection.get('timestamp', time.time())
        if self.last_position is not None and self.last_timestamp is not None:
            time_diff = current_timestamp - self.last_timestamp
            if time_diff > 0:
                current_center = self.get_center_from_box(detection['box'])
                last_center = self.get_center_from_box(self.last_position)
                
                velocity_x = (current_center[0] - last_center[0]) / time_diff
                velocity_y = (current_center[1] - last_center[1]) / time_diff
                
                self.velocity_history.append((velocity_x, velocity_y))
                if len(self.velocity_history) > 10:
                    self.velocity_history.pop(0)
        
        self.last_position = detection['box']
        self.last_timestamp = current_timestamp
    
    def miss_detection(self, frame_count: int):
        """Mark track as missed in current frame"""
        self.consecutive_misses += 1
        if self.consecutive_misses >= MAX_FRAMES_WITHOUT_DETECTION:
            self.is_active = False
    
    def get_center_from_box(self, box: List[float]) -> Tuple[float, float]:
        """Get center coordinates from bounding box"""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def get_center(self) -> Tuple[float, float]:
        """Get current center position"""
        return self.get_center_from_box(self.last_position)
    
    def get_average_confidence(self) -> float:
        """Get average confidence over recent detections"""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history[-10:]) / len(self.confidence_history[-10:])
    
    def get_tracking_stability(self) -> float:
        """
        Get tracking stability score (0-1)
        Based on detection consistency and miss rate
        """
        if self.detection_count < 3:
            return 0.0
        
        # Base stability on detection consistency
        stability = min(1.0, self.detection_count / 10.0)
        
        # Reduce stability for missed detections
        miss_penalty = self.consecutive_misses * 0.1
        stability = max(0.0, stability - miss_penalty)
        
        return stability
    
    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """Get current velocity (average of recent history)"""
        if not self.velocity_history:
            return None
        
        # Average recent velocities
        recent = self.velocity_history[-3:]
        vx = sum(v[0] for v in recent) / len(recent)
        vy = sum(v[1] for v in recent) / len(recent)
        return (vx, vy)
    
    def get_direction(self) -> str:
        """Determine movement direction"""
        velocity = self.get_velocity()
        if velocity is None:
            return "stationary"
        
        vx, vy = velocity
        threshold = 5.0
        
        if abs(vx) < threshold and abs(vy) < threshold:
            return "stationary"
        
        if abs(vy) > abs(vx):
            return "descending" if vy > 0 else "ascending"
        else:
            return "moving_right" if vx > 0 else "moving_left"
    
    def get_size(self) -> Tuple[float, float]:
        """Get current width and height"""
        box = self.last_position
        width = box[2] - box[0]
        height = box[3] - box[1]
        return (width, height)
    
    def get_area(self) -> float:
        """Get current bounding box area"""
        width, height = self.get_size()
        return width * height
    
    def to_dict(self) -> dict:
        """Convert track to dictionary for serialization"""
        return {
            'track_id': self.track_id,
            'bbox': self.last_position,
            'confidence': self.get_average_confidence(),
            'class_id': self.class_id,
            'class_name': self.class_name,
            'detection_count': self.detection_count,
            'stability': self.get_tracking_stability(),
            'direction': self.get_direction(),
            'is_active': self.is_active,
        }


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_distance(box1: List[float], box2: List[float]) -> float:
    """Calculate distance between centers of two boxes"""
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    
    return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)


class AircraftTracker:
    """
    Enhanced Aircraft Tracker with PTZ support
    
    Features:
    - IOU and distance-based track association
    - Lock-only mode for stable PTZ tracking
    - Edge filtering to avoid tracking aircraft at frame edges
    - Minimum consecutive detections before PTZ tracking starts
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 2,
        enable_size_filter: bool = ENABLE_SIZE_FILTER,
        min_aircraft_width: int = MIN_AIRCRAFT_WIDTH,
        enable_edge_filtering: bool = ENABLE_EDGE_FILTERING,
        edge_margin_percent: float = EDGE_MARGIN_PERCENT,
        lock_only_mode: bool = LOCK_ONLY_MODE,
        min_consecutive_detections: int = MIN_CONSECUTIVE_DETECTIONS,
    ):
        """
        Initialize the aircraft tracker
        
        Args:
            iou_threshold: Minimum IOU for track association
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits before track is confirmed
            enable_size_filter: Filter out small detections
            min_aircraft_width: Minimum width for size filter
            enable_edge_filtering: Filter detections at frame edges
            edge_margin_percent: Edge margin as percentage of frame
            lock_only_mode: Only track currently locked aircraft
            min_consecutive_detections: Min detections before PTZ tracking
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        
        # Filtering settings
        self.enable_size_filter = enable_size_filter
        self.min_aircraft_width = min_aircraft_width
        self.enable_edge_filtering = enable_edge_filtering
        self.edge_margin_percent = edge_margin_percent
        
        # PTZ tracking settings
        self.lock_only_mode = lock_only_mode
        self.min_consecutive_detections = min_consecutive_detections
        
        # Track state
        self.active_tracks: Dict[int, TrackedAircraft] = {}
        self.next_track_id = 1
        self.frame_count = 0
        
        # PTZ tracking state
        self.currently_tracked_id: Optional[int] = None
        self.track_lock_frames = 0
        self.min_track_lock_frames = 5
        self.frames_without_detection = 0
        
        logger.info(f"AircraftTracker initialized: lock_only={lock_only_mode}, "
                   f"min_detections={min_consecutive_detections}")
    
    def is_aircraft_at_frame_edge(
        self,
        box: List[float],
        frame_shape: Tuple[int, int, int]
    ) -> bool:
        """Check if aircraft is at the edge of the frame"""
        if not self.enable_edge_filtering:
            return False
        
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = box
        
        edge_margin_x = (frame_width * self.edge_margin_percent) / 100
        edge_margin_y = (frame_height * self.edge_margin_percent) / 100
        
        aircraft_center_x = (x1 + x2) / 2
        aircraft_center_y = (y1 + y2) / 2
        
        at_left_edge = aircraft_center_x < edge_margin_x
        at_right_edge = aircraft_center_x > (frame_width - edge_margin_x)
        at_top_edge = aircraft_center_y < edge_margin_y
        at_bottom_edge = aircraft_center_y > (frame_height - edge_margin_y)
        
        is_at_edge = at_left_edge or at_right_edge or at_top_edge or at_bottom_edge
        
        if is_at_edge:
            edge_description = []
            if at_left_edge:
                edge_description.append("LEFT")
            if at_right_edge:
                edge_description.append("RIGHT")
            if at_top_edge:
                edge_description.append("TOP")
            if at_bottom_edge:
                edge_description.append("BOTTOM")
            logger.debug(f"[EDGE FILTER] Aircraft at frame edge: {'/'.join(edge_description)}")
        
        return is_at_edge
    
    def calculate_aircraft_size_ratio(
        self,
        box: List[float],
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[float, float, float]:
        """Calculate aircraft size as ratio of frame"""
        aircraft_width = box[2] - box[0]
        aircraft_height = box[3] - box[1]
        aircraft_area = aircraft_width * aircraft_height
        
        frame_area = frame_shape[0] * frame_shape[1]
        size_ratio = aircraft_area / frame_area if frame_area > 0 else 0
        
        return size_ratio, aircraft_width, aircraft_height
    
    def is_tracking_locked(self) -> bool:
        """Check if we are currently locked onto a target"""
        return (
            self.currently_tracked_id is not None and
            self.track_lock_frames > 0 and
            self.currently_tracked_id in self.active_tracks and
            self.active_tracks[self.currently_tracked_id].is_active
        )
    
    def should_track_aircraft(self, track_id: int) -> bool:
        """Check if aircraft should be tracked (has enough consecutive detections)"""
        if track_id not in self.active_tracks:
            return False
        
        track = self.active_tracks[track_id]
        return track.detection_count >= self.min_consecutive_detections
    
    def assign_track_ids(
        self,
        detections: List[dict],
        frame_shape: Tuple[int, int, int]
    ) -> List[dict]:
        """
        Assign track IDs to detections using IOU and distance matching
        
        Args:
            detections: List of detection dicts with 'box', 'confidence', etc.
            frame_shape: Frame shape (height, width, channels)
            
        Returns:
            List of detections with track_id assigned
        """
        if not detections:
            self.frames_without_detection += 1
            for track in self.active_tracks.values():
                track.miss_detection(self.frame_count)
            return []
        else:
            self.frames_without_detection = 0
        
        # Prepare detections with additional info
        current_time = time.time()
        for det in detections:
            size_ratio, width, height = self.calculate_aircraft_size_ratio(det['box'], frame_shape)
            det['size_ratio'] = size_ratio
            det['width'] = width
            det['height'] = height
            det['timestamp'] = current_time
            det['track_id'] = None
        
        # Lock-only mode filtering
        if self.lock_only_mode and self.is_tracking_locked():
            logger.debug(f"[LOCK MODE] Currently tracking ID:{self.currently_tracked_id}")
            
            for track in self.active_tracks.values():
                track.miss_detection(self.frame_count)
            
            tracked_aircraft = self.active_tracks[self.currently_tracked_id]
            best_detection = None
            best_score = 0.0
            
            for detection in detections:
                iou = calculate_iou(tracked_aircraft.last_position, detection['box'])
                distance = calculate_distance(tracked_aircraft.last_position, detection['box'])
                
                if iou > 0.1:
                    distance_score = max(0, 1 - distance / TRACKING_THRESHOLD)
                    combined_score = iou * 0.7 + distance_score * 0.3
                else:
                    combined_score = 0.0
                
                if combined_score > best_score and combined_score > 0.3:
                    best_score = combined_score
                    best_detection = detection
            
            if best_detection:
                best_detection['track_id'] = self.currently_tracked_id
                tracked_aircraft.update_detection(best_detection, self.frame_count)
                return [best_detection]
            else:
                return []
        
        # Normal mode processing
        for track in self.active_tracks.values():
            track.miss_detection(self.frame_count)
        
        matched_detections = []
        unmatched_detections = list(detections)
        
        # Match detections to existing tracks
        track_ids = list(self.active_tracks.keys())
        assignment_matrix = []
        
        for track_id in track_ids:
            track = self.active_tracks[track_id]
            if not track.is_active:
                continue
            
            track_scores = []
            for detection in unmatched_detections:
                iou = calculate_iou(track.last_position, detection['box'])
                distance = calculate_distance(track.last_position, detection['box'])
                
                if iou > 0.1:
                    distance_score = max(0, 1 - distance / TRACKING_THRESHOLD)
                    combined_score = iou * 0.7 + distance_score * 0.3
                else:
                    combined_score = 0.0
                
                track_scores.append(combined_score)
            
            assignment_matrix.append(track_scores)
        
        # Simple greedy assignment
        used_detections: Set[int] = set()
        
        for track_idx, track_id in enumerate(track_ids):
            if track_id not in self.active_tracks or not self.active_tracks[track_id].is_active:
                continue
            
            if track_idx >= len(assignment_matrix):
                continue
            
            track_scores = assignment_matrix[track_idx]
            
            best_score = 0.0
            best_detection_idx = -1
            
            for det_idx, score in enumerate(track_scores):
                if det_idx not in used_detections and score > best_score and score > 0.3:
                    best_score = score
                    best_detection_idx = det_idx
            
            if best_detection_idx >= 0:
                detection = unmatched_detections[best_detection_idx]
                detection['track_id'] = track_id
                
                self.active_tracks[track_id].update_detection(detection, self.frame_count)
                
                matched_detections.append(detection)
                used_detections.add(best_detection_idx)
        
        # Assign new track IDs to unmatched detections
        for det_idx, detection in enumerate(unmatched_detections):
            if det_idx not in used_detections:
                # Skip edge detections for new tracks
                if self.is_aircraft_at_frame_edge(detection['box'], frame_shape):
                    continue
                
                new_track_id = self.next_track_id
                self.next_track_id += 1
                
                detection['track_id'] = new_track_id
                
                new_track = TrackedAircraft(new_track_id, detection, self.frame_count)
                self.active_tracks[new_track_id] = new_track
                
                matched_detections.append(detection)
        
        # Clean up inactive tracks
        inactive_tracks = [
            track_id for track_id, track in self.active_tracks.items()
            if not track.is_active
        ]
        for track_id in inactive_tracks:
            logger.debug(f"Removing inactive track ID: {track_id}")
            del self.active_tracks[track_id]
            
            if self.currently_tracked_id == track_id:
                self.currently_tracked_id = None
                self.track_lock_frames = 0
                logger.info(f"[LOCK CLEARED] Track ID:{track_id} became inactive")
        
        return matched_detections
    
    def select_aircraft_for_ptz_tracking(
        self,
        detections_with_tracks: List[dict],
        frame_shape: Tuple[int, int, int]
    ) -> Optional[dict]:
        """
        Select which aircraft to track with PTZ based on stability and consecutive detections
        
        Args:
            detections_with_tracks: List of detections with track_id assigned
            frame_shape: Frame shape (height, width, channels)
            
        Returns:
            Best detection for PTZ tracking, or None
        """
        if not detections_with_tracks:
            if self.track_lock_frames > 0:
                self.track_lock_frames -= 1
            
            if self.track_lock_frames == 0:
                self.currently_tracked_id = None
                logger.debug("[LOCK CLEARED] No detections, track lock expired")
            
            return None
        
        # Continue tracking current aircraft if it's still detected
        if self.currently_tracked_id is not None and self.track_lock_frames > 0:
            current_track_detection = None
            for detection in detections_with_tracks:
                if detection.get('track_id') == self.currently_tracked_id:
                    current_track_detection = detection
                    break
            
            if current_track_detection is not None:
                self.track_lock_frames -= 1
                logger.debug(f"[LOCKED] Continuing track ID:{self.currently_tracked_id}")
                return current_track_detection
            else:
                logger.info(f"[LOCK LOST] Track ID: {self.currently_tracked_id}")
                self.currently_tracked_id = None
                self.track_lock_frames = 0
        
        # Only select new aircraft if we're not locked
        if self.lock_only_mode and self.is_tracking_locked():
            return None
        
        # Filter valid detections for PTZ tracking
        valid_detections = []
        for detection in detections_with_tracks:
            width = detection.get('width', 0)
            track_id = detection.get('track_id')
            
            # Apply size filter if enabled
            if self.enable_size_filter and width < self.min_aircraft_width:
                continue
            
            # Apply edge filter if enabled
            if self.is_aircraft_at_frame_edge(detection['box'], frame_shape):
                continue
            
            # Check if aircraft has enough consecutive detections
            if not self.should_track_aircraft(track_id):
                continue
            
            valid_detections.append(detection)
        
        if not valid_detections:
            return None
        
        # Select best aircraft for tracking
        best_aircraft = None
        best_score = -1
        
        frame_center_x = frame_shape[1] // 2
        frame_center_y = frame_shape[0] // 2
        
        for detection in valid_detections:
            track_id = detection.get('track_id')
            track = self.active_tracks.get(track_id)
            
            if not track:
                continue
            
            # Calculate center distance from frame center
            aircraft_center_x = (detection['box'][0] + detection['box'][2]) / 2
            aircraft_center_y = (detection['box'][1] + detection['box'][3]) / 2
            center_distance = np.sqrt(
                (aircraft_center_x - frame_center_x)**2 +
                (aircraft_center_y - frame_center_y)**2
            )
            max_distance = np.sqrt(frame_center_x**2 + frame_center_y**2)
            center_score = 1.0 - (center_distance / max_distance) if max_distance > 0 else 0
            
            # Track stability score
            stability_score = track.get_tracking_stability()
            
            # Size score (prefer larger aircraft)
            size_score = min(1.0, detection.get('size_ratio', 0) * 10)
            
            # Confidence score
            confidence_score = detection.get('confidence', 0)
            
            # Combined score
            combined_score = (
                center_score * 0.3 +
                stability_score * 0.4 +
                size_score * 0.2 +
                confidence_score * 0.1
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_aircraft = detection
        
        if best_aircraft:
            self.currently_tracked_id = best_aircraft.get('track_id')
            self.track_lock_frames = self.min_track_lock_frames
            logger.info(f"[NEW LOCK] Started tracking aircraft ID: {self.currently_tracked_id}")
        
        return best_aircraft
    
    def update(
        self,
        detections: List[dict],
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[List[dict], Optional[dict]]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with 'box', 'confidence', 'class_id', 'class_name'
            frame_shape: Frame shape (height, width, channels)
            
        Returns:
            Tuple of (detections_with_tracks, ptz_target)
        """
        self.frame_count += 1
        
        # Assign track IDs
        detections_with_tracks = self.assign_track_ids(detections, frame_shape)
        
        # Select aircraft for PTZ tracking
        ptz_target = self.select_aircraft_for_ptz_tracking(detections_with_tracks, frame_shape)
        
        return detections_with_tracks, ptz_target
    
    def should_go_to_preset(self, frames_before_preset: int = 120) -> bool:
        """Check if we should go to preset (no detection for too long)"""
        return self.frames_without_detection >= frames_before_preset
    
    def clear_lock(self):
        """Clear current PTZ tracking lock"""
        self.currently_tracked_id = None
        self.track_lock_frames = 0
        logger.info("[LOCK CLEARED] Manual clear")
    
    def get_track(self, track_id: int) -> Optional[TrackedAircraft]:
        """Get a track by ID"""
        return self.active_tracks.get(track_id)
    
    def get_active_tracks(self) -> List[TrackedAircraft]:
        """Get all active tracks"""
        return [t for t in self.active_tracks.values() if t.is_active]
    
    def get_track_direction(self, track_id: int) -> str:
        """Get the movement direction of a track"""
        track = self.active_tracks.get(track_id)
        if track:
            return track.get_direction()
        return "unknown"
    
    def clear(self):
        """Clear all tracks"""
        self.active_tracks.clear()
        self.frame_count = 0
        self.currently_tracked_id = None
        self.track_lock_frames = 0
        self.frames_without_detection = 0
    
    def get_stats(self) -> dict:
        """Get tracker statistics"""
        return {
            'frame_count': self.frame_count,
            'active_tracks': len([t for t in self.active_tracks.values() if t.is_active]),
            'total_tracks': len(self.active_tracks),
            'currently_tracked_id': self.currently_tracked_id,
            'is_locked': self.is_tracking_locked(),
            'lock_only_mode': self.lock_only_mode,
            'frames_without_detection': self.frames_without_detection,
        }
