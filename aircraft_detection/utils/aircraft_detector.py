"""
Aircraft Detection using YOLO model
"""
import cv2
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AircraftDetector:
    """YOLO-based aircraft detector"""
    
    # Class names for aircraft detection
    CLASS_NAMES = {
        0: 'aircraft',
        1: 'helicopter',
        2: 'drone',
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'auto',
    ):
        """
        Initialize the aircraft detector
        
        Args:
            model_path: Path to the YOLO model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = self._get_device(device)
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning(f"Model path not found: {model_path}. Using default YOLO model.")
            self._load_default_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the device to use"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model(self, model_path: str):
        """Load a YOLO model from path"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"Loaded aircraft detection model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._load_default_model()
    
    def _load_default_model(self):
        """Load default YOLO model for general object detection"""
        try:
            from ultralytics import YOLO
            # Use YOLOv8n as default - can detect airplanes from COCO classes
            self.model = YOLO('yolov8n.pt')
            self.model.to(self.device)
            logger.info("Loaded default YOLOv8n model")
        except Exception as e:
            logger.error(f"Error loading default model: {e}")
            self.model = None
    
    def detect(
        self,
        frame: np.ndarray,
        roi_points: Optional[List[List[int]]] = None,
    ) -> List[Dict]:
        """
        Detect aircraft in a frame
        
        Args:
            frame: Input frame (BGR format)
            roi_points: Optional ROI polygon to filter detections
            
        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_id: int
                - class_name: str
        """
        if self.model is None:
            return []
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Map class ID to aircraft types
                    # COCO class 4 is 'airplane', we'll treat it as 'aircraft'
                    # For custom models, use the CLASS_NAMES mapping
                    if class_id == 4:  # COCO airplane class
                        class_name = 'aircraft'
                        class_id = 0
                    elif class_id in self.CLASS_NAMES:
                        class_name = self.CLASS_NAMES[class_id]
                    else:
                        # Skip non-aircraft detections for default model
                        continue
                    
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    
                    # Check if detection is within ROI
                    if roi_points and not self._is_in_roi(bbox, roi_points):
                        continue
                    
                    detections.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def _is_in_roi(self, bbox: List[int], roi_points: List[List[int]]) -> bool:
        """Check if bbox center is within ROI polygon"""
        if not roi_points or len(roi_points) < 3:
            return True
        
        # Calculate bbox center
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        
        # Use OpenCV to check if point is in polygon
        roi_array = np.array(roi_points, np.int32)
        result = cv2.pointPolygonTest(roi_array, (center_x, center_y), False)
        
        return result >= 0
    
    def determine_action(
        self,
        direction: str,
        position_in_frame: str,
        velocity: Optional[Tuple[float, float]] = None,
    ) -> str:
        """
        Determine the action of an aircraft based on movement
        
        Args:
            direction: Movement direction from tracker
            position_in_frame: 'top', 'middle', 'bottom'
            velocity: (vx, vy) velocity tuple
            
        Returns:
            Action string: 'landing', 'takeoff', 'taxiing', 'parked', 'flying'
        """
        if direction == "stationary":
            return "parked"
        
        if direction == "descending":
            return "landing"
        
        if direction == "ascending":
            return "takeoff"
        
        if direction in ["moving_left", "moving_right"]:
            # Check velocity magnitude for taxiing vs flying
            if velocity:
                speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                if speed < 50:  # Threshold for taxiing
                    return "taxiing"
                else:
                    return "flying"
            return "taxiing"
        
        return "unknown"
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
        }


# Global detector instance
_detector_instance: Optional[AircraftDetector] = None


def get_detector(
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
) -> AircraftDetector:
    """Get or create the global detector instance"""
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = AircraftDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
    
    return _detector_instance

