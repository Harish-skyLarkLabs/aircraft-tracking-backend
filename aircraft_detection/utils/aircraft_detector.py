"""
Aircraft Detection using YOLO model (Aircraft.pt)
Custom model trained for aircraft detection
"""
import cv2
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)

# Default model path - Aircraft.pt 
DEFAULT_MODEL_PATH = Path(settings.BASE_DIR) / 'models' / 'aircraft_new_jun22.pt'

# Detection thresholds 
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


class AircraftDetector:
    """YOLO-based aircraft detector using custom aircraft_new_jun22.pt model"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        device: str = 'auto',
    ):
        """
        Initialize the aircraft detector
        
        Args:
            model_path: Path to the YOLO model weights (defaults to Aircraft.pt)
            confidence_threshold: Minimum confidence for detections (default: 0.25)
            iou_threshold: IOU threshold for NMS (default: 0.45)
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = self._get_device(device)
        self.class_names = {}  # Will be populated from model
        
        # Use provided path or default to Aircraft.pt
        if model_path is None:
            model_path = str(DEFAULT_MODEL_PATH)
        
        if Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.error(f"Model not found at {model_path}")
            # Try default location
            if DEFAULT_MODEL_PATH.exists():
                self.load_model(str(DEFAULT_MODEL_PATH))
            else:
                logger.error("Aircraft.pt model not found. Please ensure the model file exists.")
    
    def _get_device(self, device: str) -> str:
        """Determine the device to use"""
        if device == 'auto':
            if torch.cuda.is_available():
                logger.info("CUDA available - using GPU")
                return 'cuda'
            else:
                logger.info("CUDA not available - using CPU")
                return 'cpu'
        return device
    
    def load_model(self, model_path: str):
        """Load the YOLO model from path"""
        try:
            from ultralytics import YOLO
            
            logger.info(f"Loading Aircraft detection model from {model_path}...")
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Get class names from model
            self.class_names = self.model.names
            logger.info(f"Model loaded successfully! Classes: {self.class_names}")
            logger.info(f"Running on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
            # Run inference with same parameters as inference.py
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                imgsz=1280,
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get class name from model's class names
                    class_name = self.class_names.get(class_id, 'unknown')
                    
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
    
    def detect_and_plot(
        self,
        frame: np.ndarray,
        roi_points: Optional[List[List[int]]] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect aircraft and return annotated frame (like inference.py)
        
        Args:
            frame: Input frame (BGR format)
            roi_points: Optional ROI polygon to filter detections
            
        Returns:
            Tuple of (annotated_frame, detections_list)
        """
        if self.model is None:
            return frame, []
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            
            detections = []
            # Don't draw anything - keep frame clean, we'll draw only PTZ-locked aircraft in frame_processor
            annotated_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.class_names.get(class_id, 'unknown')
                    
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
            
            return annotated_frame, detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return frame, []
    
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
            'class_names': self.class_names,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
        }


# Global detector instance
_detector_instance: Optional[AircraftDetector] = None


def get_detector(
    model_path: Optional[str] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
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
