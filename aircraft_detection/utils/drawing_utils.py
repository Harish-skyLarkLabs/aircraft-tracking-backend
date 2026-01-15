"""
Utility functions for drawing on frames - bounding boxes, labels, etc.
"""
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional


def get_contrasting_color(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Get a contrasting (darker) color for backgrounds"""
    return tuple(max(0, c - 80) for c in color)


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.6,
    thickness: int = 1,
    alpha: float = 0.8,
) -> np.ndarray:
    """
    Draw text with a semi-transparent background rectangle
    
    Args:
        frame: The image to draw on
        text: Text to display
        position: (x, y) position for text
        color: Color for the background rectangle
        font: OpenCV font
        scale: Font scale
        thickness: Line thickness
        alpha: Transparency of the background rectangle
    
    Returns:
        Modified frame
    """
    x, y = int(position[0]), int(position[1])
    
    overlay = frame.copy()
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    rect_x1, rect_y1 = x, y - text_size[1] - 5
    rect_x2, rect_y2 = x + text_size[0] + 10, y + 5
    
    # Ensure coordinates are within frame bounds
    rect_y1 = max(0, rect_y1)
    rect_x2 = min(frame.shape[1], rect_x2)
    
    dark_color = get_contrasting_color(color)
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), dark_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x + 5, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return frame


def draw_bounding_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_label: bool = True,
) -> np.ndarray:
    """
    Draw a bounding box with optional label
    
    Args:
        frame: The image to draw on
        bbox: (x1, y1, x2, y2) bounding box coordinates
        label: Label text to display
        color: Box color
        thickness: Line thickness
        show_label: Whether to show the label
    
    Returns:
        Modified frame
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure coordinates are within frame bounds
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if show_label and label:
        draw_text_with_background(frame, label, (x1, y1 - 10), color)
    
    return frame


def draw_aircraft_detection(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    detection_type: str = "aircraft",
    action: str = "unknown",
    track_id: int = 0,
    confidence: float = 0.0,
    flight_number: Optional[str] = None,
) -> np.ndarray:
    """
    Draw aircraft detection with all relevant information
    
    Args:
        frame: The image to draw on
        bbox: (x1, y1, x2, y2) bounding box coordinates
        detection_type: Type of detection (aircraft, helicopter, drone)
        action: Action being performed (landing, takeoff, taxiing, etc.)
        track_id: Tracking ID
        confidence: Detection confidence
        flight_number: Flight number if available
    
    Returns:
        Modified frame
    """
    # Color based on action
    action_colors = {
        'landing': (0, 255, 0),      # Green
        'takeoff': (0, 165, 255),    # Orange
        'taxiing': (255, 255, 0),    # Cyan
        'parked': (128, 128, 128),   # Gray
        'flying': (255, 0, 0),       # Blue
        'hovering': (255, 0, 255),   # Magenta
        'unknown': (0, 255, 255),    # Yellow
    }
    
    color = action_colors.get(action, (0, 255, 255))
    
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw corner accents for better visibility
    corner_length = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, 3)
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, 3)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, 3)
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, 3)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, 3)
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, 3)
    
    # Build label
    label_parts = [f"{detection_type.upper()}-{track_id}"]
    if confidence > 0:
        label_parts.append(f"{confidence:.0%}")
    label = " ".join(label_parts)
    
    # Draw main label
    draw_text_with_background(frame, label, (x1, y1 - 10), color)
    
    # Draw action label
    action_label = action.upper()
    draw_text_with_background(frame, action_label, (x1, y2 + 20), color)
    
    # Draw flight number if available
    if flight_number:
        draw_text_with_background(frame, flight_number, (x1, y1 - 35), color)
    
    return frame


def draw_polygon_roi(
    frame: np.ndarray,
    roi_points: List[List[int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    fill_alpha: float = 0.1,
) -> np.ndarray:
    """
    Draw a polygon region of interest on the frame
    
    Args:
        frame: The image to draw on
        roi_points: List of [x, y] coordinates defining the polygon
        color: Line color
        thickness: Line thickness
        fill_alpha: Alpha for fill (0 for no fill)
    
    Returns:
        Modified frame
    """
    if not roi_points or len(roi_points) < 3:
        return frame
    
    points = np.array(roi_points, np.int32)
    points = points.reshape((-1, 1, 2))
    
    # Draw filled polygon with transparency
    if fill_alpha > 0:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
    
    # Draw polygon outline
    cv2.polylines(frame, [points], True, color, thickness)
    
    return frame


def draw_timestamp(
    frame: np.ndarray,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Draw current timestamp on the frame
    
    Args:
        frame: The image to draw on
        position: (x, y) position for timestamp
        color: Text color
    
    Returns:
        Modified frame
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    draw_text_with_background(frame, timestamp, position, color)
    return frame


def draw_info_overlay(
    frame: np.ndarray,
    fps: float = 0.0,
    detection_count: int = 0,
    camera_name: str = "",
) -> np.ndarray:
    """
    Draw information overlay on the frame
    
    Args:
        frame: The image to draw on
        fps: Current FPS
        detection_count: Number of detections
        camera_name: Name of the camera
    
    Returns:
        Modified frame
    """
    h, w = frame.shape[:2]
    
    # Draw semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Draw timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw FPS
    if fps > 0:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Draw detection count
    if detection_count > 0:
        det_text = f"Detections: {detection_count}"
        cv2.putText(frame, det_text, (w // 2 - 60, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    
    # Draw camera name
    if camera_name:
        cv2.putText(frame, camera_name, (200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame


def convert_roi_to_pixels(
    roi_points: List[List[float]],
    frame_width: int,
    frame_height: int,
) -> List[List[int]]:
    """
    Convert ROI points from percentage-based coordinates to absolute pixel coordinates.
    
    Args:
        roi_points: List of [x, y] points in percentage format (0-100%)
        frame_width: Width of the frame in pixels
        frame_height: Height of the frame in pixels
        
    Returns:
        List of [x, y] points in absolute pixel coordinates
    """
    if not roi_points:
        return []
    
    pixel_points = []
    
    for point in roi_points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        
        # Check if already in pixel format (large numbers)
        if isinstance(point[0], (int, float)) and point[0] > 100:
            pixel_points.append([int(point[0]), int(point[1])])
        else:
            # Convert from percentage to pixels
            x = int((float(point[0]) / 100.0) * frame_width)
            y = int((float(point[1]) / 100.0) * frame_height)
            pixel_points.append([x, y])
    
    return pixel_points

