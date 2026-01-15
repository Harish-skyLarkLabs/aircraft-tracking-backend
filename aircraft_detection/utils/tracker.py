"""
Simple IOU Tracker for aircraft tracking
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class Track:
    """Represents a single tracked object"""
    
    def __init__(self, track_id: int, bbox: List[float], class_id: int = 0):
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history = [bbox]
        self.max_history = 30
        
    def update(self, bbox: List[float]):
        """Update track with new detection"""
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def predict(self):
        """Simple prediction - just return last bbox"""
        self.age += 1
        self.time_since_update += 1
        return self.bbox
    
    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """Calculate velocity based on history"""
        if len(self.history) < 2:
            return None
        
        # Calculate center points
        curr = self.history[-1]
        prev = self.history[-2]
        
        curr_center = ((curr[0] + curr[2]) / 2, (curr[1] + curr[3]) / 2)
        prev_center = ((prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2)
        
        vx = curr_center[0] - prev_center[0]
        vy = curr_center[1] - prev_center[1]
        
        return (vx, vy)
    
    def get_direction(self) -> str:
        """Determine movement direction"""
        velocity = self.get_velocity()
        if velocity is None:
            return "stationary"
        
        vx, vy = velocity
        
        # Threshold for considering movement
        threshold = 5.0
        
        if abs(vx) < threshold and abs(vy) < threshold:
            return "stationary"
        
        if abs(vy) > abs(vx):
            return "descending" if vy > 0 else "ascending"
        else:
            return "moving_right" if vx > 0 else "moving_left"


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union between two boxes"""
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
    
    if union <= 0:
        return 0.0
    
    return intersection / union


class IOUTracker:
    """Simple IOU-based tracker for aircraft detection"""
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0
    
    def update(self, detections: List[List[float]], class_ids: Optional[List[int]] = None) -> List[Tuple[int, List[float]]]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of [x1, y1, x2, y2] bounding boxes
            class_ids: Optional list of class IDs for each detection
        
        Returns:
            List of (track_id, bbox) tuples for active tracks
        """
        self.frame_count += 1
        
        if class_ids is None:
            class_ids = [0] * len(detections)
        
        # Predict new locations for existing tracks
        for track in self.tracks.values():
            track.predict()
        
        # Match detections to tracks
        if len(detections) > 0 and len(self.tracks) > 0:
            matched, unmatched_dets, unmatched_tracks = self._associate_detections_to_tracks(
                detections, list(self.tracks.values())
            )
            
            # Update matched tracks
            for det_idx, track_id in matched:
                self.tracks[track_id].update(detections[det_idx])
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                self._create_track(detections[det_idx], class_ids[det_idx])
            
            # Mark unmatched tracks
            # (they will be removed if too old)
        
        elif len(detections) > 0:
            # No existing tracks, create new ones
            for i, det in enumerate(detections):
                self._create_track(det, class_ids[i])
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Return active tracks that have been confirmed
        results = []
        for track_id, track in self.tracks.items():
            if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                results.append((track_id, track.bbox))
        
        return results
    
    def _associate_detections_to_tracks(
        self,
        detections: List[List[float]],
        tracks: List[Track],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to existing tracks using IOU"""
        if len(detections) == 0 or len(tracks) == 0:
            return [], list(range(len(detections))), [t.track_id for t in tracks]
        
        # Calculate IOU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(tracks):
                iou_matrix[d, t] = calculate_iou(det, track.bbox)
        
        # Greedy matching
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = [t.track_id for t in tracks]
        
        while True:
            # Find best match
            if iou_matrix.size == 0:
                break
            
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
            
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            det_idx, track_idx = max_idx
            
            matched.append((det_idx, tracks[track_idx].track_id))
            
            if det_idx in unmatched_dets:
                unmatched_dets.remove(det_idx)
            if tracks[track_idx].track_id in unmatched_tracks:
                unmatched_tracks.remove(tracks[track_idx].track_id)
            
            # Remove matched row and column
            iou_matrix[det_idx, :] = 0
            iou_matrix[:, track_idx] = 0
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _create_track(self, bbox: List[float], class_id: int = 0):
        """Create a new track"""
        track = Track(self.next_track_id, bbox, class_id)
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get a track by ID"""
        return self.tracks.get(track_id)
    
    def get_track_direction(self, track_id: int) -> str:
        """Get the movement direction of a track"""
        track = self.tracks.get(track_id)
        if track:
            return track.get_direction()
        return "unknown"
    
    def clear(self):
        """Clear all tracks"""
        self.tracks.clear()
        self.frame_count = 0



