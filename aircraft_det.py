#!/usr/bin/env python3
"""
Enhanced Dual Camera Aircraft Detection and PTZ Tracking System
- Seamless switching between two cameras
- Priority to camera 1 (92) when no aircraft detected
- Smooth transition to camera 2 (93) when aircraft appears there
- Maintains tracking consistency across cameras
"""

import requests
import time
import json
import cv2
import numpy as np
from datetime import datetime
from requests.auth import HTTPDigestAuth
from urllib3.exceptions import InsecureRequestWarning
import urllib3
from ultralytics import YOLO
import threading
from queue import Queue, Empty
import xml.etree.ElementTree as ET
import logging
from collections import deque, defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(InsecureRequestWarning)

# Global configuration variables
ENABLE_SIZE_FILTER = True
MIN_AIRCRAFT_WIDTH = 10
TRACKING_THRESHOLD = 80
CENTER_TOLERANCE = 25
PTZ_SPEED = 120
ZOOM_SPEED = 10
TARGET_AIRCRAFT_RATIO = 0.0025
ZOOM_TOLERANCE = 0.03
MAX_ZOOM_LEVEL = 2.0
MIN_ZOOM_LEVEL = -2.0
TRACKING_CONSISTENCY_FRAMES = 10
MAX_FRAMES_WITHOUT_DETECTION = 120
MAX_PTZ_VALUE = 120
PRESET_NUMBER = 20
TILT_SENSITIVITY = 0.9
FRAMES_BEFORE_PRESET = 120
LOCK_ONLY_MODE = True

# Global zoom control variables
ENABLE_ZOOM_CONTROL = True  # Set to False to disable all zoom controls
ENABLE_ZOOM_IN = True       # Set to False to disable zoom in
ENABLE_ZOOM_OUT = True      # Set to False to disable zoom out

# Edge detection filtering variables
ENABLE_EDGE_FILTERING = True
EDGE_MARGIN_PERCENT = 7
#__-----------------------------------------------------------------------------------------
# PTZ Tracking control
PTZ_TRACKING = True  # Set to False to disable all PTZ movement tracking
#__-----------------------------------------------------------------------------------------
# Camera configuration
CAMERA_CONFIG_1 = {
    'ip': '192.168.1.106',
    'username': 'admin',
    'password': 'skylark123',
    'name': 'Camera 1 (150)'
}

# Single camera configuration - Camera 2 removed

class RTSPStreamHandler:
    """Enhanced RTSP stream handler with GStreamer support and improved error recovery"""
    
    def __init__(self, rtsp_url, buffer_size=10):
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.frame_queue = Queue(maxsize=buffer_size)
        self.cap = None
        self.is_running = False
        self.reconnect_delay = 5
        self.last_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.error_count = 0
        self.max_errors = 50
        self.consecutive_failures = 0
        
    def start(self):
        """Start the stream capture thread"""
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
    def stop(self):
        """Stop the stream capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            
    def _create_capture_with_gstreamer(self):
        """Create capture using GStreamer pipeline for better performance"""
        try:
            gst_pipeline = (
                f'rtspsrc location={self.rtsp_url} latency=0 buffer-mode=0 ! '
                'rtph264depay ! h264parse ! avdec_h264 ! '
                'videoconvert ! appsink drop=true sync=false'
            )
            
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                logger.info("Successfully connected using GStreamer")
                return cap
            else:
                logger.warning("GStreamer failed, falling back to default backend")
                return None
        except Exception as e:
            logger.debug(f"GStreamer not available: {e}")
            return None
            
    def _create_capture_with_ffmpeg(self):
        """Create capture using FFmpeg backend with optimized settings"""
        try:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 25)
                
                # Try to set timeout properties if available
                try:
                    cap.set(cv2.CAP_PROP_STREAM_OPEN_TIME_USEC, 5000000)
                    cap.set(cv2.CAP_PROP_STREAM_TIMEOUT_USEC, 5000000)
                except AttributeError:
                    # These properties may not be available in all OpenCV versions
                    pass
                
                logger.info("Successfully connected using FFmpeg backend")
                return cap
            else:
                return None
        except Exception as e:
            logger.error(f"FFmpeg backend failed: {e}")
            return None
            
    def _connect(self):
        """Connect to RTSP stream with multiple fallback options"""
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Try GStreamer first
        self.cap = self._create_capture_with_gstreamer()
        if self.cap and self.cap.isOpened():
            return True
            
        # Try FFmpeg backend
        self.cap = self._create_capture_with_ffmpeg()
        if self.cap and self.cap.isOpened():
            return True
            
        # Try default OpenCV backend
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 25)
                logger.info("Connected using default OpenCV backend")
                return True
        except Exception as e:
            logger.error(f"Default backend failed: {e}")
            
        return False
        
    def _capture_frames(self):
        """Continuously capture frames from RTSP stream with improved error handling"""
        frame_skip_counter = 0
        skip_frames = 1
        
        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.info(f"Attempting to connect to RTSP stream...")
                    if self._connect():
                        logger.info("Successfully connected to RTSP stream")
                        self.consecutive_failures = 0
                        self.error_count = 0
                    else:
                        logger.warning(f"Failed to connect. Retrying in {self.reconnect_delay} seconds...")
                        time.sleep(self.reconnect_delay)
                        continue
                        
                ret = self.cap.grab()
                
                if not ret:
                    self.consecutive_failures += 1
                    if self.consecutive_failures > 30:
                        logger.warning("Too many grab failures. Reconnecting...")
                        self._connect()
                        self.consecutive_failures = 0
                    continue
                
                frame_skip_counter += 1
                if frame_skip_counter % skip_frames != 0:
                    continue
                    
                ret, frame = self.cap.retrieve()
                
                if ret and frame is not None:
                    self.consecutive_failures = 0
                    
                    if frame.shape[0] > 0 and frame.shape[1] > 0:
                        with self.frame_lock:
                            self.last_frame = frame.copy()
                        
                        # Clear old frames if queue is full
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except Empty:
                                pass
                                
                        self.frame_queue.put(frame)
                    else:
                        logger.warning("Invalid frame dimensions")
                        
                else:
                    self.consecutive_failures += 1
                    self.error_count += 1
                    
                    if self.error_count > self.max_errors:
                        logger.warning("Max errors reached. Reconnecting...")
                        self._connect()
                        self.error_count = 0
                        
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                self.consecutive_failures += 1
                if self.consecutive_failures > 10:
                    time.sleep(1)
                    
    def get_frame(self):
        """Get the latest frame from the buffer"""
        try:
            frame = self.frame_queue.get(timeout=0.05)
            return True, frame
        except Empty:
            with self.frame_lock:
                if self.last_frame is not None:
                    return True, self.last_frame.copy()
            return False, None

class TrackedAircraft:
    """Enhanced class to maintain track information for each aircraft"""
    def __init__(self, track_id, detection, frame_count):
        self.track_id = track_id
        self.first_detection_frame = frame_count
        self.last_detection_frame = frame_count
        self.detection_count = 1
        self.consecutive_misses = 0
        self.is_active = True
        self.confidence_history = [detection['confidence']]
        self.position_history = [detection['box']]
        self.size_history = [detection['size_ratio']]
        self.velocity_history = []
        self.last_position = detection['box']
        self.last_timestamp = detection['timestamp']
        
        # Store latest detection info
        self.latest_detection = detection
        
    def update_detection(self, detection, frame_count):
        """Update track with new detection"""
        self.last_detection_frame = frame_count
        self.detection_count += 1
        self.consecutive_misses = 0
        self.latest_detection = detection
        
        # Update histories
        self.confidence_history.append(detection['confidence'])
        self.position_history.append(detection['box'])
        self.size_history.append(detection['size_ratio'])
        
        # Limit history size
        if len(self.confidence_history) > 20:
            self.confidence_history.pop(0)
        if len(self.position_history) > 20:
            self.position_history.pop(0)
        if len(self.size_history) > 20:
            self.size_history.pop(0)
        
        # Calculate velocity
        if self.last_position is not None and self.last_timestamp is not None:
            time_diff = detection['timestamp'] - self.last_timestamp
            if time_diff > 0:
                current_center = self.get_center_from_box(detection['box'])
                last_center = self.get_center_from_box(self.last_position)
                
                velocity_x = (current_center[0] - last_center[0]) / time_diff
                velocity_y = (current_center[1] - last_center[1]) / time_diff
                
                self.velocity_history.append((velocity_x, velocity_y))
                if len(self.velocity_history) > 10:
                    self.velocity_history.pop(0)
        
        self.last_position = detection['box']
        self.last_timestamp = detection['timestamp']
    
    def miss_detection(self, frame_count):
        """Mark track as missed in current frame"""
        self.consecutive_misses += 1
        if self.consecutive_misses >= MAX_FRAMES_WITHOUT_DETECTION:
            self.is_active = False
    
    def get_center_from_box(self, box):
        """Get center coordinates from bounding box"""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def get_average_confidence(self):
        """Get average confidence over recent detections"""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history[-10:]) / len(self.confidence_history[-10:])
    
    def get_tracking_stability(self):
        """Get tracking stability score (0-1)"""
        if self.detection_count < 3:
            return 0.0
        
        # Base stability on detection consistency
        stability = min(1.0, self.detection_count / 10.0)
        
        # Reduce stability for missed detections
        miss_penalty = self.consecutive_misses * 0.1
        stability = max(0.0, stability - miss_penalty)
        
        return stability

class PTZCommand:
    def __init__(self, action, speed, duration=0.05, pan_value=0, tilt_value=0, zoom_value=0):
        self.action = action
        self.speed = speed
        self.duration = duration
        self.timestamp = time.time()
        self.pan_value = pan_value
        self.tilt_value = tilt_value
        self.zoom_value = zoom_value

class HikvisionPTZ:
    def __init__(self, ip, username, password, channel=1):
        """Initialize PTZ camera controller with smooth movement"""
        self.ip = ip
        self.username = username
        self.password = password
        self.channel = channel
        self.base_url = f"http://{ip}/ISAPI"
        self.auth = HTTPDigestAuth(username, password)
        self.current_zoom_level = 1.0
        self.current_pan = 0
        self.current_tilt = 0
        self.command_queue = Queue()
        self.ptz_thread = None
        self.stop_thread = False
        self.is_moving = False
        self.last_command_time = 0
        
        # Enhanced smoothing parameters
        self.movement_history = deque(maxlen=10)
        self.movement_dampening = 0.50  # Reduce movement speed by 50%
        self.dead_zone = 0.001  # Increased dead zone for less aggressive tracking
        self.move_interval = 0.001  # Minimum time between movements
        
        if self.test_connection():
            print(f"✓ Successfully connected to camera at {ip}")
            self.get_current_ptz_position()
            self.start_ptz_thread()
            # Go to preset on startup
            self.go_to_preset(PRESET_NUMBER)
        else:
            print(f"✗ Failed to connect to camera at {ip}")
    
    def test_connection(self):
        """Test if camera is accessible"""
        try:
            url = f"{self.base_url}/System/deviceInfo"
            response = requests.get(url, auth=self.auth, timeout=2, verify=False)
            return response.status_code == 200
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def get_current_ptz_position(self):
        """Get current PTZ position from camera"""
        try:
            url = f"{self.base_url}/PTZCtrl/channels/{self.channel}/status"
            response = requests.get(url, auth=self.auth, timeout=3, verify=False)
            
            if response.status_code == 200:
                root = ET.fromstring(response.text)
                
                pan_elem = root.find('.//azimuth')
                tilt_elem = root.find('.//elevation') 
                zoom_elem = root.find('.//absoluteZoom')
                
                if pan_elem is not None:
                    self.current_pan = float(pan_elem.text) / 10.0
                
                if tilt_elem is not None:
                    self.current_tilt = float(tilt_elem.text) / 10.0
                
                if zoom_elem is not None:
                    self.current_zoom_level = float(zoom_elem.text) / 10.0
                
                print(f"✓ Retrieved current PTZ position: Pan={self.current_pan}, Tilt={self.current_tilt}, Zoom={self.current_zoom_level}x")
                return True
            else:
                print(f"Failed to get PTZ status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error getting PTZ position: {e}")
            return False
    
    def go_to_preset(self, preset_number):
        """Go to specified preset position"""
        try:
            url = f"{self.base_url}/PTZCtrl/channels/{self.channel}/presets/{preset_number}/goto"
            response = requests.put(url, auth=self.auth, timeout=5, verify=False)
            
            if response.status_code == 200:
                print(f"✓ Moving to preset {preset_number}")
                time.sleep(3)
                self.get_current_ptz_position()
                return True
            else:
                print(f"Failed to go to preset {preset_number}: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error going to preset {preset_number}: {e}")
            return False
    
    def start_ptz_thread(self):
        """Start background thread for PTZ commands"""
        self.ptz_thread = threading.Thread(target=self._ptz_worker, daemon=True)
        self.ptz_thread.start()
    
    def _ptz_worker(self):
        """Background worker for PTZ commands with smooth execution"""
        while not self.stop_thread:
            try:
                command = self.command_queue.get(timeout=0.1)
                
                if command.action == 'PROPORTIONAL':
                    self.is_moving = True
                    self._execute_ptz_command('PROPORTIONAL', command.speed, 
                                            command.pan_value, command.tilt_value, command.zoom_value)
                    
                    time.sleep(command.duration)
                    self._execute_ptz_command('STOP', 0)
                    self.is_moving = False
                    
                elif command.action != 'STOP':
                    self.is_moving = True
                    self._execute_ptz_command(command.action, command.speed)
                    time.sleep(command.duration)
                    self._execute_ptz_command('STOP', 0)
                    self.is_moving = False
                    
                else:
                    self._execute_ptz_command('STOP', 0)
                    self.is_moving = False
                
                self.command_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"PTZ worker error: {e}")
                self.is_moving = False
    
    def _execute_ptz_command(self, action, speed, pan_value=0, tilt_value=0, zoom_value=0):
        """Execute PTZ command with enhanced smoothing and zoom control checks"""
        try:
            # Check zoom control permissions
            if not ENABLE_ZOOM_CONTROL:
                if action in ['ZOOM_IN', 'ZOOM_OUT'] or zoom_value != 0:
                    return False
            
            if action == 'ZOOM_IN' and not ENABLE_ZOOM_IN:
                return False
                
            if action == 'ZOOM_OUT' and not ENABLE_ZOOM_OUT:
                return False
                
            if zoom_value > 0 and not ENABLE_ZOOM_IN:
                zoom_value = 0
                
            if zoom_value < 0 and not ENABLE_ZOOM_OUT:
                zoom_value = 0
            
            url = f"{self.base_url}/PTZCtrl/channels/{self.channel}/continuous"
            
            # Apply movement dampening for smoother operation
            if pan_value != 0 or tilt_value != 0:
                pan_value = int(pan_value * self.movement_dampening)
                tilt_value = int(tilt_value * self.movement_dampening)
                command_data = f'<PTZData><pan>{pan_value}</pan><tilt>{tilt_value}</tilt><zoom>{zoom_value}</zoom></PTZData>'
            else:
                # Traditional discrete commands
                commands = {
                    'UP': f'<PTZData><pan>0</pan><tilt>{speed}</tilt><zoom>0</zoom></PTZData>',
                    'DOWN': f'<PTZData><pan>0</pan><tilt>-{speed}</tilt><zoom>0</zoom></PTZData>',
                    'LEFT': f'<PTZData><pan>-{speed}</pan><tilt>0</tilt><zoom>0</zoom></PTZData>',
                    'RIGHT': f'<PTZData><pan>{speed}</pan><tilt>0</tilt><zoom>0</zoom></PTZData>',
                    'ZOOM_IN': f'<PTZData><pan>0</pan><tilt>0</tilt><zoom>{speed}</zoom></PTZData>',
                    'ZOOM_OUT': f'<PTZData><pan>0</pan><tilt>0</tilt><zoom>-{speed}</zoom></PTZData>',
                    'STOP': '<PTZData><pan>0</pan><tilt>0</tilt><zoom>0</zoom></PTZData>'
                }
                
                if action not in commands:
                    return False
                
                command_data = commands[action]
            
            headers = {'Content-Type': 'application/xml'}
            response = requests.put(
                url, 
                data=command_data, 
                headers=headers, 
                auth=self.auth, 
                verify=False,
                timeout=0.5
            )
            
            # Update zoom level tracking
            if (action == 'ZOOM_IN' or zoom_value > 0) and ENABLE_ZOOM_IN and ENABLE_ZOOM_CONTROL:
                zoom_increment = 0.8 * (speed / 7.0) if zoom_value == 0 else abs(zoom_value) / 10
                self.current_zoom_level = min(self.current_zoom_level + zoom_increment, MAX_ZOOM_LEVEL)
            elif (action == 'ZOOM_OUT' or zoom_value < 0) and ENABLE_ZOOM_OUT and ENABLE_ZOOM_CONTROL:
                zoom_decrement = 1.2 * (speed / 7.0) if zoom_value == 0 else abs(zoom_value) / 8
                self.current_zoom_level = max(self.current_zoom_level - zoom_decrement, MIN_ZOOM_LEVEL)
            
            self.last_command_time = time.time()
            return response.status_code == 200
                
        except Exception as e:
            print(f"PTZ execute error: {e}")
            return False
    
    def track_object_smooth(self, frame_width, frame_height, object_x, object_y, object_width=None, object_height=None):
        """Enhanced smooth object tracking - PAN and TILT ONLY, NO ZOOM"""
        current_time = time.time()
        
        # Check if enough time has passed since last movement
        if current_time - self.last_command_time < self.move_interval:
            return
        
        # Calculate normalized position (-1 to 1)
        norm_x = (object_x / frame_width - 0.5) * 2
        norm_y = (object_y / frame_height - 0.5) * 2
        
        # Add to movement history for smoothing
        self.movement_history.append((norm_x, norm_y))
        
        # Calculate weighted average movement (recent values have more weight)
        if len(self.movement_history) > 0:
            weights = np.linspace(0.5, 1.0, len(self.movement_history))
            weights = weights / weights.sum()
            
            x_values = [m[0] for m in self.movement_history]
            y_values = [m[1] for m in self.movement_history]
            
            avg_x = np.average(x_values, weights=weights)
            avg_y = np.average(y_values, weights=weights)
        else:
            avg_x, avg_y = norm_x, norm_y
        
        # Check if object is outside dead zone - more sensitive for Camera 1
        dead_zone = 0.001  # Very small dead zone for aggressive centering
        if abs(avg_x) > dead_zone or abs(avg_y) > dead_zone:
            # Calculate movement speeds with exponential decay for smoother movement
            x_factor = np.sign(avg_x) * (1 - np.exp(-abs(avg_x) * 3))  # Increased sensitivity
            y_factor = np.sign(avg_y) * (1 - np.exp(-abs(avg_y) * 3))  # Increased sensitivity
            
            # Apply tilt sensitivity
            y_factor = y_factor * TILT_SENSITIVITY
            
            # Scale to PTZ range
            pan_speed = int(x_factor * MAX_PTZ_VALUE)
            tilt_speed = int(-y_factor * MAX_PTZ_VALUE)  # Negative because camera Y is inverted
            
            # Clamp values
            pan_speed = max(-MAX_PTZ_VALUE, min(MAX_PTZ_VALUE, pan_speed))
            tilt_speed = max(-MAX_PTZ_VALUE, min(MAX_PTZ_VALUE, tilt_speed))
            
            # Use proportional control for smooth movement - NO ZOOM (always 0)
            if self.command_queue.qsize() < 2:
                command = PTZCommand('PROPORTIONAL', PTZ_SPEED, 0.05, pan_speed, tilt_speed, 0)  # zoom_value = 0
                self.command_queue.put(command, block=False)
                self.last_command_time = current_time
                print(f"[PTZ SMOOTH] Pan: {pan_speed}, Tilt: {tilt_speed}, Zoom: 0 (disabled)")
        else:
            # Object is centered, stop movement
            self.emergency_stop()
    
    def ptz_control(self, action, speed=7, duration=0.05):
        """Non-blocking PTZ control with automatic STOP"""
        try:
            if self.command_queue.qsize() < 2:
                command = PTZCommand(action, speed, duration)
                self.command_queue.put(command, block=False)
                return True
        except Exception as e:
            print(f"PTZ control queue error: {e}")
        return False
    
    def ptz_proportional_control(self, pan_value, tilt_value, zoom_value=0, duration=0.05):
        """Proportional PTZ control with normalized values (-1 to 1)"""
        try:
            # Apply tilt sensitivity reduction
            tilt_value = tilt_value * TILT_SENSITIVITY
            
            # Scale values to PTZ range
            scaled_pan = int(pan_value * MAX_PTZ_VALUE)
            scaled_tilt = int(tilt_value * MAX_PTZ_VALUE)  
            scaled_zoom = int(zoom_value * MAX_PTZ_VALUE)
            
            # Clamp values
            scaled_pan = max(-MAX_PTZ_VALUE, min(MAX_PTZ_VALUE, scaled_pan))
            scaled_tilt = max(-MAX_PTZ_VALUE, min(MAX_PTZ_VALUE, scaled_tilt))
            scaled_zoom = max(-MAX_PTZ_VALUE, min(MAX_PTZ_VALUE, scaled_zoom))
            
            if self.command_queue.qsize() < 2:
                command = PTZCommand('PROPORTIONAL', PTZ_SPEED, duration, scaled_pan, scaled_tilt, scaled_zoom)
                self.command_queue.put(command, block=False)
                return True
        except Exception as e:
            print(f"PTZ proportional control error: {e}")
        return False

    def emergency_stop(self):
        """Emergency stop - clear queue and stop immediately"""
        try:
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except Empty:
                    break
            
            self.command_queue.put(PTZCommand('STOP', 0, 0), block=False)
        except Exception as e:
            print(f"Emergency stop error: {e}")
    
    def get_zoom_percentage(self):
        """Get current zoom as percentage of maximum"""
        return (self.current_zoom_level - MIN_ZOOM_LEVEL) / (MAX_ZOOM_LEVEL - MIN_ZOOM_LEVEL) * 100
    
    def stop(self):
        """Stop PTZ thread"""
        self.emergency_stop()
        self.stop_thread = True
        if self.ptz_thread and self.ptz_thread.is_alive():
            self.ptz_thread.join(timeout=2)

class AircraftTracker:
    def __init__(self, model_path):
        """Initialize aircraft tracker with enhanced tracking capabilities"""
        self.model = YOLO(model_path)
        self.frame_count = 0
        self.next_track_id = 1
        self.active_tracks = {}
        self.currently_tracked_id = None
        self.track_lock_frames = 0
        self.min_track_lock_frames = 5
        self.frames_without_detection = 0
        self.lock_only_mode = LOCK_ONLY_MODE
        
        # Minimum consecutive detections before tracking (from second file)
        self.min_consecutive_detections = 5
        
    def is_aircraft_at_frame_edge(self, box, frame_shape):
        """Check if aircraft is at the edge of the frame"""
        if not ENABLE_EDGE_FILTERING:
            return False
        
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = box
        
        edge_margin_x = (frame_width * EDGE_MARGIN_PERCENT) / 100
        edge_margin_y = (frame_height * EDGE_MARGIN_PERCENT) / 100
        
        aircraft_center_x = (x1 + x2) / 2
        aircraft_center_y = (y1 + y2) / 2
        
        at_left_edge = aircraft_center_x < edge_margin_x
        at_right_edge = aircraft_center_x > (frame_width - edge_margin_x)
        at_top_edge = aircraft_center_y < edge_margin_y
        at_bottom_edge = aircraft_center_y > (frame_height - edge_margin_y)
        
        is_at_edge = at_left_edge or at_right_edge or at_top_edge or at_bottom_edge
        
        if is_at_edge:
            edge_description = []
            if at_left_edge: edge_description.append("LEFT")
            if at_right_edge: edge_description.append("RIGHT")
            if at_top_edge: edge_description.append("TOP")
            if at_bottom_edge: edge_description.append("BOTTOM")
            
            print(f"[EDGE FILTER] Aircraft at frame edge: {'/'.join(edge_description)}")
        
        return is_at_edge
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes"""
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
    
    def calculate_distance(self, box1, box2):
        """Calculate distance between centers of two boxes"""
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        center2_x = (box2[0] + box2[2]) / 2
        center2_y = (box2[1] + box2[3]) / 2
        
        return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def calculate_aircraft_size_ratio(self, box, frame_shape):
        """Calculate aircraft size as ratio of frame"""
        aircraft_width = box[2] - box[0]
        aircraft_height = box[3] - box[1]
        aircraft_area = aircraft_width * aircraft_height
        
        frame_area = frame_shape[0] * frame_shape[1]
        size_ratio = aircraft_area / frame_area
        
        return size_ratio, aircraft_width, aircraft_height
    
    def is_tracking_locked(self):
        """Check if we are currently locked onto a target"""
        return (self.currently_tracked_id is not None and 
                self.track_lock_frames > 0 and 
                self.currently_tracked_id in self.active_tracks and
                self.active_tracks[self.currently_tracked_id].is_active)
    
    def assign_track_ids(self, detections):
        """Assign track IDs to detections using IoU and distance matching"""
        if not detections:
            self.frames_without_detection += 1
            for track in self.active_tracks.values():
                track.miss_detection(self.frame_count)
            return []
        else:
            self.frames_without_detection = 0
        
        # Lock-only mode filtering
        if self.lock_only_mode and self.is_tracking_locked():
            print(f"[LOCK MODE] Currently tracking ID:{self.currently_tracked_id}, filtering detections...")
            
            for track in self.active_tracks.values():
                track.miss_detection(self.frame_count)
            
            tracked_aircraft = self.active_tracks[self.currently_tracked_id]
            best_detection = None
            best_score = 0.0
            
            for detection in detections:
                iou = self.calculate_iou(tracked_aircraft.last_position, detection['box'])
                distance = self.calculate_distance(tracked_aircraft.last_position, detection['box'])
                
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
                iou = self.calculate_iou(track.last_position, detection['box'])
                distance = self.calculate_distance(track.last_position, detection['box'])
                
                if iou > 0.1:
                    distance_score = max(0, 1 - distance / TRACKING_THRESHOLD)
                    combined_score = iou * 0.7 + distance_score * 0.3
                else:
                    combined_score = 0.0
                
                track_scores.append(combined_score)
            
            assignment_matrix.append(track_scores)
        
        # Simple greedy assignment
        used_detections = set()
        
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
                new_track_id = self.next_track_id
                self.next_track_id += 1
                
                detection['track_id'] = new_track_id
                
                new_track = TrackedAircraft(new_track_id, detection, self.frame_count)
                self.active_tracks[new_track_id] = new_track
                
                matched_detections.append(detection)
        
        # Clean up inactive tracks
        inactive_tracks = [track_id for track_id, track in self.active_tracks.items() if not track.is_active]
        for track_id in inactive_tracks:
            print(f"Removing inactive track ID: {track_id}")
            del self.active_tracks[track_id]
            
            if self.currently_tracked_id == track_id:
                self.currently_tracked_id = None
                self.track_lock_frames = 0
                print(f"[LOCK CLEARED] Track ID:{track_id} became inactive")
        
        return matched_detections
    
    def should_track_aircraft(self, track_id):
        """Check if aircraft should be tracked (has enough consecutive detections)"""
        if track_id not in self.active_tracks:
            return False
            
        track = self.active_tracks[track_id]
        return track.detection_count >= self.min_consecutive_detections
    
    def select_aircraft_for_ptz_tracking(self, detections_with_tracks, frame_shape):
        """Select which aircraft to track with PTZ based on stability and consecutive detections"""
        if not detections_with_tracks:
            if self.track_lock_frames > 0:
                self.track_lock_frames -= 1
            
            if self.track_lock_frames == 0:
                self.currently_tracked_id = None
                print("[LOCK CLEARED] No detections, track lock expired")
            
            return None
        
        # Continue tracking current aircraft if it's still detected
        if self.currently_tracked_id is not None and self.track_lock_frames > 0:
            current_track_detection = None
            for detection in detections_with_tracks:
                if detection['track_id'] == self.currently_tracked_id:
                    current_track_detection = detection
                    break
            
            if current_track_detection is not None:
                self.track_lock_frames -= 1
                print(f"[LOCKED] Continuing track ID:{self.currently_tracked_id}")
                return current_track_detection
            else:
                print(f"[LOCK LOST] Track ID: {self.currently_tracked_id}")
                self.currently_tracked_id = None
                self.track_lock_frames = 0
        
        # Only select new aircraft if we're not locked
        if self.lock_only_mode and self.is_tracking_locked():
            return None
        
        # Filter valid detections for PTZ tracking
        valid_detections = []
        for detection in detections_with_tracks:
            width = detection['width']
            
            # Apply size filter if enabled
            if ENABLE_SIZE_FILTER and width < MIN_AIRCRAFT_WIDTH:
                continue
            
            # Apply edge filter if enabled
            if self.is_aircraft_at_frame_edge(detection['box'], frame_shape):
                continue
            
            # Check if aircraft has enough consecutive detections
            if not self.should_track_aircraft(detection['track_id']):
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
            track = self.active_tracks[detection['track_id']]
            
            # Calculate center distance from frame center
            aircraft_center_x = (detection['box'][0] + detection['box'][2]) / 2
            aircraft_center_y = (detection['box'][1] + detection['box'][3]) / 2
            center_distance = np.sqrt((aircraft_center_x - frame_center_x)**2 + (aircraft_center_y - frame_center_y)**2)
            max_distance = np.sqrt(frame_center_x**2 + frame_center_y**2)
            center_score = 1.0 - (center_distance / max_distance)
            
            # Track stability score
            stability_score = track.get_tracking_stability()
            
            # Size score (prefer larger aircraft)
            size_score = min(1.0, detection['size_ratio'] * 10)
            
            # Confidence score
            confidence_score = detection['confidence']
            
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
            self.currently_tracked_id = best_aircraft['track_id']
            self.track_lock_frames = self.min_track_lock_frames
            print(f"[NEW LOCK] Started tracking aircraft ID: {self.currently_tracked_id}")
        
        return best_aircraft
    
    def detect_and_track(self, frame):
        """Detect aircraft and return tracking information with track IDs"""
        self.frame_count += 1
        current_time = time.time()
        
        try:
            # Run YOLO detection
            results = self.model.predict(frame, conf=0.25, verbose=False, imgsz=1280)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            
            # Process detections
            all_detections = []
            edge_filtered_count = 0
            
            for i, box in enumerate(boxes):
                size_ratio, width, height = self.calculate_aircraft_size_ratio(box, frame.shape)
                
                # Check if aircraft is at frame edge
                if self.is_aircraft_at_frame_edge(box, frame.shape):
                    edge_filtered_count += 1
                    continue
                
                detection = {
                    'box': box,
                    'class': int(classes[i]),
                    'confidence': scores[i],
                    'class_name': self.model.names[int(classes[i])],
                    'width': width,
                    'height': height,
                    'size_ratio': size_ratio,
                    'timestamp': current_time,
                    'track_id': None
                }
                all_detections.append(detection)
            
            if edge_filtered_count > 0:
                print(f"[EDGE FILTER] Filtered out {edge_filtered_count} aircraft at frame edges")
            
            # Assign track IDs
            detections_with_tracks = self.assign_track_ids(all_detections)
            
            # Select aircraft for PTZ tracking
            ptz_target = self.select_aircraft_for_ptz_tracking(detections_with_tracks, frame.shape)
            
            return detections_with_tracks, ptz_target
            
        except Exception as e:
            print(f"Detection error: {e}")
            return [], None
    
    def should_go_to_preset(self):
        """Check if we should go to preset (no detection for too long)"""
        return self.frames_without_detection >= FRAMES_BEFORE_PRESET

class CameraSystem:
    def __init__(self, camera_config):
        """Initialize camera system with PTZ and stream handlers"""
        self.config = camera_config
        self.ptz = HikvisionPTZ(
            camera_config['ip'],
            camera_config['username'],
            camera_config['password']
        )
        self.rtsp_url = f"rtsp://{camera_config['username']}:{camera_config['password']}@{camera_config['ip']}"
        self.stream_handler = RTSPStreamHandler(self.rtsp_url, buffer_size=5)
        self.stream_handler.start()
        
    def get_frame(self):
        """Get frame from camera"""
        return self.stream_handler.get_frame()
    
    def stop(self):
        """Stop camera system"""
        self.stream_handler.stop()
        self.ptz.stop()

class SingleCameraAircraftTracker:
    def __init__(self, model_path):
        """Initialize single camera aircraft tracking system"""
        self.model_path = model_path
        self.camera = CameraSystem(CAMERA_CONFIG_1)
        self.tracker = AircraftTracker(model_path)
        
        self.zoom_stabilization_counter = 0
        self.last_ptz_time = 0
        self.ptz_cooldown = 0.1
        self.preset_triggered = False
        
        # PTZ tracking state
        self.current_tracked_id = None
        self.lost_track_time = None
        self.return_home_delay = 1  # Reduced delay for faster return
        
    def draw_text_with_rectangle(self, frame, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, thickness=1):
        """Draw text with background rectangle"""
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        text_width, text_height = text_size
        rect_x1, rect_y1 = x - 5, y - text_height - 10
        rect_x2, rect_y2 = x + text_width + 5, y
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        cv2.putText(frame, text, (x, y - 5), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    def draw_aircraft_crop_overlay(self, frame, source_frame, detection, position='top_right'):
        """Draw real-time aircraft crop in upper right corner with area info"""
        try:
            # Get aircraft bounding box
            x1, y1, x2, y2 = map(int, detection['box'])
            
            # Add padding to crop for better visualization
            padding = 30
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(source_frame.shape[1], x2 + padding)
            crop_y2 = min(source_frame.shape[0], y2 + padding)
            
            # Crop aircraft from source frame
            aircraft_crop = source_frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            
            if aircraft_crop.size == 0:
                return
            
            # Resize crop to fixed size for overlay (maintain aspect ratio)
            max_crop_width = 320
            max_crop_height = 240
            
            crop_h, crop_w = aircraft_crop.shape[:2]
            aspect_ratio = crop_w / crop_h
            
            if aspect_ratio > max_crop_width / max_crop_height:
                # Width is limiting factor
                new_width = max_crop_width
                new_height = int(new_width / aspect_ratio)
            else:
                # Height is limiting factor
                new_height = max_crop_height
                new_width = int(new_height * aspect_ratio)
            
            # Ensure dimensions are valid
            if new_width <= 0 or new_height <= 0:
                return
            
            aircraft_crop_resized = cv2.resize(aircraft_crop, (new_width, new_height))
            
            # Calculate overlay position (upper right corner)
            margin = 10
            if position == 'top_right':
                overlay_x = frame.shape[1] - new_width - margin
                overlay_y = margin
            else:  # top_left
                overlay_x = margin
                overlay_y = margin
            
            # Ensure overlay fits in frame
            if overlay_x + new_width > frame.shape[1] or overlay_y + new_height > frame.shape[0]:
                return
            
            # Draw background rectangle with border
            bg_x1, bg_y1 = overlay_x - 5, overlay_y - 5
            bg_x2, bg_y2 = overlay_x + new_width + 5, overlay_y + new_height + 5
            
            # Semi-transparent dark background
            overlay_bg = frame[bg_y1:bg_y2, bg_x1:bg_x2].copy()
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(frame[bg_y1:bg_y2, bg_x1:bg_x2], 0.3, overlay_bg, 0.7, 0, frame[bg_y1:bg_y2, bg_x1:bg_x2])
            
            # Draw red border around crop
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), 3)
            
            # Place aircraft crop
            frame[overlay_y:overlay_y+new_height, overlay_x:overlay_x+new_width] = aircraft_crop_resized
            
            # Add label with area information
            area = detection['width'] * detection.get('height', 0)
            track_id = detection.get('track_id', 'N/A')
            label = f"PTZ LOCKED ID:{track_id} | Area: {int(area):,} px²"
            
            # Draw label background above the crop
            label_bg_y1 = bg_y1 - 30
            label_bg_y2 = bg_y1 - 2
            
            if label_bg_y1 >= 0:
                cv2.rectangle(frame, (bg_x1, label_bg_y1), (bg_x2, label_bg_y2), (0, 0, 255), -1)
                cv2.putText(frame, label, (bg_x1 + 5, label_bg_y1 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        except Exception as e:
            print(f"Error drawing aircraft crop overlay: {e}")
    
    def control_zoom_for_aircraft_by_area(self, ptz_controller, aircraft_width, aircraft_height):
        """Smart progressive area-based ZOOM OUT control with multiple thresholds:
        - Area < 10K: No change (aircraft is small enough)
        - Area 10K-20K: Zoom out 2x (very initial) ✨ NEW
        - Area 20K-30K: Zoom out 3x (early zoom) ✨ NEW
        - Area 30K-40K: Zoom out 3x (initial zoom out) ✨ ENHANCED
        - Area 40K-60K: Zoom out 3x (moderate zoom out) ✨ ENHANCED
        - Area 60K-75K: Zoom out 4x (aggressive zoom out)
        - Area 75K-85K: Zoom out 4x (more aggressive)
        - Area 85K-95K: Zoom out 2x (additional adjustment)
        - Area 95K-120K: Zoom out 2x (additional adjustment)
        - Area > 120K: Zoom out 2x (maximum adjustment)
        """
        if not ENABLE_ZOOM_CONTROL:
            print(f"[ZOOM DISABLED] Zoom control is disabled")
            return
            
        if aircraft_width is None or aircraft_height is None or self.zoom_stabilization_counter > 0:
            return
        
        # Calculate area
        area = aircraft_width * aircraft_height
        
        current_time = time.time()
        if current_time - self.last_ptz_time < self.ptz_cooldown:
            return
        
        # Progressive zoom out based on area thresholds
        zoom_action = None
        zoom_duration = 0
        zoom_speed = 5
        zoom_label = ""
        
        if area < 10000:
            # Aircraft is small enough - no zoom needed
            print(f"[AREA OK] Area: {int(area)} < 10K - No zoom needed")
            return
        
        elif area > 120000:
            # Extremely large aircraft - progressive zoom out 2x
            if ENABLE_ZOOM_OUT:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.5
                zoom_speed = 6
                zoom_label = "EXTREME SIZE > 120K"
                print(f"🔍 [ZOOM OUT 2X] Area: {int(area)} > 120K - {zoom_label}")
            else:
                print(f"[ZOOM BLOCKED] Area: {int(area)} > 120K needs zoom out but disabled")
                return
        
        elif area > 95000:
            # Very large aircraft 95K-120K - zoom out 2x
            if ENABLE_ZOOM_OUT:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.5
                zoom_speed = 6
                zoom_label = "VERY LARGE 95K-120K"
                print(f"🔍 [ZOOM OUT 2X] Area: {int(area)} (95K-120K) - {zoom_label}")
            else:
                print(f"[ZOOM BLOCKED] Area: {int(area)} (95K-120K) needs zoom out but disabled")
                return
        
        elif area > 85000:
            # Large aircraft 85K-95K - zoom out 2x
            if ENABLE_ZOOM_OUT:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.5
                zoom_speed = 6
                zoom_label = "LARGE 85K-95K"
                print(f"🔍 [ZOOM OUT 2X] Area: {int(area)} (85K-95K) - {zoom_label}")
            else:
                print(f"[ZOOM BLOCKED] Area: {int(area)} (85K-95K) needs zoom out but disabled")
                return
        
        elif area > 75000:
            # Aircraft 75K-85K - zoom out 4x aggressively
            if ENABLE_ZOOM_OUT:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.10
                zoom_speed = 8
                zoom_label = "AGGRESSIVE 75K-85K"
                print(f"🔍 [ZOOM OUT 4X] Area: {int(area)} (75K-85K) - {zoom_label}")
            else:
                print(f"[ZOOM BLOCKED] Area: {int(area)} (75K-85K) needs 4x zoom out but disabled")
                return
        
        elif area >= 60000:
            # Aircraft 60K-75K - zoom out 4x
            if ENABLE_ZOOM_OUT:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.10
                zoom_speed = 8
                zoom_label = "HEAVY 60K-75K"
                print(f"🔍 [ZOOM OUT 4X] Area: {int(area)} (60K-75K) - {zoom_label}")
            else:
                print(f"[ZOOM BLOCKED] Area: {int(area)} (60K-75K) needs 4x zoom out but disabled")
                return
        
        elif area >= 40000:
            # Aircraft 40K-60K - zoom out 3x (enhanced from 2x)
            if ENABLE_ZOOM_OUT:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.6  # Longer duration for 3x zoom out
                zoom_speed = 6       # Increased speed for 3x zoom out
                zoom_label = "MODERATE 40K-60K"
                print(f"🔍 [ZOOM OUT 3X] Area: {int(area)} (40K-60K) - {zoom_label}")
            else:
                print(f"[ZOOM BLOCKED] Area: {int(area)} (40K-60K) needs 3x zoom out but disabled")
                return
        
        elif area >= 30000:
            # Aircraft 30K-40K - initial zoom out 3x
            if ENABLE_ZOOM_OUT:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.6  # Duration for 3x zoom out
                zoom_speed = 6       # Speed for 3x zoom out
                zoom_label = "INITIAL 30K-40K"
                print(f"🔍 [ZOOM OUT 3X] Area: {int(area)} (30K-40K) - {zoom_label}")
            else:
                print(f"[ZOOM BLOCKED] Area: {int(area)} (30K-40K) needs 3x zoom out but disabled")
                return
        
        elif area >= 20000:
            # Aircraft 20K-30K - early zoom out 3x ✨ NEW
            if ENABLE_ZOOM_OUT:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.8  # Duration for 3x zoom out
                zoom_speed = 7      # Speed for 3x zoom out
                zoom_label = "EARLY 20K-30K"
                print(f"🔍 [ZOOM OUT 3X] Area: {int(area)} (20K-30K) - {zoom_label} ✨ NEW")
            else:
                print(f"[ZOOM BLOCKED] Area: {int(area)} (20K-30K) needs 3x zoom out but disabled")
                return
        
        elif area >= 10000:
            # Aircraft 10K-20K - very initial zoom out 2x ✨ NEW
            if ENABLE_ZOOM_OUT:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.8  # Duration for 2x zoom out
                zoom_speed = 7       # Speed for 2x zoom out
                zoom_label = "VERY INITIAL 10K-20K"
                print(f"🔍 [ZOOM OUT 2X] Area: {int(area)} (10K-20K) - {zoom_label} ✨ NEW")
            else:
                print(f"[ZOOM BLOCKED] Area: {int(area)} (10K-20K) needs 2x zoom out but disabled")
                return
        
        else:
            # Aircraft < 10K - no action needed
            print(f"[SMALL AIRCRAFT] Area: {int(area)} < 10K - No zoom needed")
            return
        
        # Execute zoom action
        if zoom_action == 'ZOOM_OUT':
            success = ptz_controller.ptz_control('ZOOM_OUT', speed=zoom_speed, duration=zoom_duration)
            if success:
                print(f"✓ ZOOM OUT executed - {zoom_label} - Speed: {zoom_speed}, Duration: {zoom_duration}s")
                self.zoom_stabilization_counter = 25  # Wait longer before next zoom adjustment
                self.last_ptz_time = current_time
            else:
                print(f"✗ ZOOM OUT failed - {zoom_label}")
    
    def draw_edge_margins(self, frame):
        """Draw edge margins on frame for visualization"""
        if not ENABLE_EDGE_FILTERING:
            return
            
        frame_height, frame_width = frame.shape[:2]
        edge_margin_x = int((frame_width * EDGE_MARGIN_PERCENT) / 100)
        edge_margin_y = int((frame_height * EDGE_MARGIN_PERCENT) / 100)
        
        color = (100, 100, 100)
        thickness = 1
        
        # Draw edge lines
        cv2.line(frame, (edge_margin_x, 0), (edge_margin_x, frame_height), color, thickness)
        cv2.line(frame, (frame_width - edge_margin_x, 0), (frame_width - edge_margin_x, frame_height), color, thickness)
        cv2.line(frame, (0, edge_margin_y), (frame_width, edge_margin_y), color, thickness)
        cv2.line(frame, (0, frame_height - edge_margin_y), (frame_width, frame_height - edge_margin_y), color, thickness)
        
        cv2.putText(frame, f"EDGE {EDGE_MARGIN_PERCENT}%", (10, edge_margin_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def run(self):
        """Main execution loop with single camera support"""
        global PTZ_TRACKING
        
        # Initialize video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"single_camera_aircraft_tracking_{timestamp}.avi"
        writer = None
        
        print(f"Starting single camera aircraft tracking system...")
        print(f"Camera: {CAMERA_CONFIG_1['ip']}")
        print(f"Auto Preset: {PRESET_NUMBER}")
        print(f"Lock-Only Mode: {'ENABLED' if LOCK_ONLY_MODE else 'DISABLED'}")
        print(f"Edge Filtering: {'ENABLED' if ENABLE_EDGE_FILTERING else 'DISABLED'}")
        print(f"GStreamer Support: Available")
        print("Controls:")
        print("  'q' - quit")
        print("  'p' - go to preset")
        print("  's' - emergency stop PTZ")
        print("  'c' - clear tracking lock")
        print("  'l' - toggle lock mode")
        print("  't' - toggle PTZ tracking")
        
        # Performance monitoring
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0
        
        try:
            while True:
                # Get frame from camera
                ret, frame = self.camera.get_frame()
                
                if not ret:
                    logger.warning("No frame available from camera, waiting...")
                    time.sleep(0.1)
                    continue
                
                # Initialize video writer on first successful frame
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter(
                        output_filename, fourcc, 20, (frame.shape[1], frame.shape[0])
                    )
                
                # Process camera
                all_detections, ptz_target = self.tracker.detect_and_track(frame)
                
                display_frame = frame.copy()
                current_tracker = self.tracker
                current_ptz = self.camera.ptz
                current_ptz_target = ptz_target
                
                # Draw edge margins for visualization
                self.draw_edge_margins(display_frame)
                
                # Add camera indicator overlay
                cv2.putText(display_frame, f"ACTIVE: {CAMERA_CONFIG_1['name']}", (display_frame.shape[1] - 300, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Check if we should go to preset
                if current_tracker.should_go_to_preset() and not self.preset_triggered:
                    print(f"No detection for {FRAMES_BEFORE_PRESET} frames, going to preset {PRESET_NUMBER}")
                    current_ptz.go_to_preset(PRESET_NUMBER)
                    self.preset_triggered = True
                elif ptz_target:
                    self.preset_triggered = False
                
                # Draw all detections with track IDs and area
                for detection in all_detections:
                    x1, y1, x2, y2 = map(int, detection['box'])
                    width = detection['width']
                    height = detection['height']
                    track_id = detection['track_id']
                    
                    # Calculate area (in pixels)
                    area = width * height
                    
                    # Choose box color based on tracking status and consecutive detections
                    track = current_tracker.active_tracks.get(track_id)
                    has_enough_detections = track and track.detection_count >= current_tracker.min_consecutive_detections
                    
                    if ENABLE_SIZE_FILTER and width < MIN_AIRCRAFT_WIDTH:
                        color = (255, 255, 255)  # White for small aircraft
                        thickness = 1
                    elif current_ptz_target and track_id == current_ptz_target['track_id']:
                        color = (0, 0, 255)  # Red for PTZ tracked aircraft
                        thickness = 3
                    elif has_enough_detections:
                        color = (0, 255, 0)  # Green for confirmed aircraft
                        thickness = 2
                    else:
                        color = (0, 255, 255)  # Yellow for confirming aircraft
                        thickness = 2
                    
                    #add padding to the box
                    x1 -= 10
                    y1 -= 10
                    x2 += 10
                    y2 += 10

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw track ID and info
                    stability = track.get_tracking_stability() if track else 0.0
                    size_pct = detection['size_ratio'] * 100
                    
                    ptz_status = " [PTZ-LOCKED]" if (current_ptz_target and track_id == current_ptz_target['track_id']) else ""
                    detection_count = track.detection_count if track else 0
                    
                    if not has_enough_detections:
                        label = f"Confirming {detection_count}/{current_tracker.min_consecutive_detections}"
                    else:
                        label = f"{detection['class_name']}:{detection['confidence']:.2f} ({int(width)}px){ptz_status}"
                    
                    self.draw_text_with_rectangle(display_frame, label, x1, y1)
                    
                    # Draw area below the main label
                    area_label = f"Area: {int(area)} px²"
                    self.draw_text_with_rectangle(display_frame, area_label, x1, y1 + 25)
                
                # Control PTZ for tracked aircraft with smooth movement - PAN/TILT ONLY
                if current_ptz_target and PTZ_TRACKING:
                    # Get aircraft center
                    box = current_ptz_target['box']
                    aircraft_center_x = (box[0] + box[2]) / 2
                    aircraft_center_y = (box[1] + box[3]) / 2
                    
                    # Calculate distance from frame center
                    frame_center_x = display_frame.shape[1] // 2
                    frame_center_y = display_frame.shape[0] // 2
                    
                    # Calculate normalized distance from center (-1 to 1)
                    norm_x = (aircraft_center_x - frame_center_x) / frame_center_x
                    norm_y = (aircraft_center_y - frame_center_y) / frame_center_y
                    
                    # Aggressive centering
                    if abs(norm_x) > 0.05 or abs(norm_y) > 0.05:  # Smaller tolerance for aggressive centering
                        # Use smooth tracking function with enhanced centering
                        current_ptz.track_object_smooth(
                            display_frame.shape[1], display_frame.shape[0],
                            aircraft_center_x, aircraft_center_y,
                            current_ptz_target['width'], current_ptz_target.get('height')
                        )
                        print(f"[CENTERING] Object at ({norm_x:.2f}, {norm_y:.2f}) - Centering...")
                    
                    # Only call zoom control if zoom is enabled
                    if ENABLE_ZOOM_CONTROL:
                        self.control_zoom_for_aircraft_by_area(current_ptz, current_ptz_target['width'], current_ptz_target.get('height'))
                    else:
                        # Show that zoom is disabled
                        area = current_ptz_target['width'] * current_ptz_target.get('height', 0)
                        print(f"[ZOOM DISABLED] Aircraft area: {int(area)} px² but zoom control is disabled")
                    
                    # Draw center crosshair
                    center_x, center_y = display_frame.shape[1] // 2, display_frame.shape[0] // 2
                    cv2.line(display_frame, (center_x - 15, center_y), (center_x + 15, center_y), (255, 0, 0), 2)
                    cv2.line(display_frame, (center_x, center_y - 15), (center_x, center_y + 15), (255, 0, 0), 2)
                    
                    # Draw real-time aircraft crop in upper right corner
                    self.draw_aircraft_crop_overlay(display_frame, frame, current_ptz_target, position='top_right')
                    
                    # Reset lost track time
                    self.current_tracked_id = current_ptz_target['track_id']
                    self.lost_track_time = None
                elif current_ptz_target and not PTZ_TRACKING:
                    # PTZ tracking is disabled, just draw center crosshair without moving camera
                    center_x, center_y = display_frame.shape[1] // 2, display_frame.shape[0] // 2
                    cv2.line(display_frame, (center_x - 15, center_y), (center_x + 15, center_y), (255, 0, 0), 2)
                    cv2.line(display_frame, (center_x, center_y - 15), (center_x, center_y + 15), (255, 0, 0), 2)
                    
                    # Draw real-time aircraft crop in upper right corner
                    self.draw_aircraft_crop_overlay(display_frame, frame, current_ptz_target, position='top_right')
                    
                    # Reset lost track time
                    self.current_tracked_id = current_ptz_target['track_id']
                    self.lost_track_time = None
                else:
                    # Check if we lost track
                    if self.current_tracked_id is not None:
                        if self.lost_track_time is None:
                            self.lost_track_time = time.time()
                            logger.info("Lost track of aircraft")
                            current_ptz.emergency_stop()
                        elif time.time() - self.lost_track_time > self.return_home_delay:
                            # Return to preset after delay
                            current_ptz.go_to_preset(PRESET_NUMBER)
                            self.current_tracked_id = None
                            self.lost_track_time = None
                
                # Decrease stabilization counter
                if self.zoom_stabilization_counter > 0:
                    self.zoom_stabilization_counter -= 1
                
                # Get tracking summary
                active_tracks = len([t for t in self.tracker.active_tracks.values() if t.is_active])
                
                # Add status information
                lock_status = "🔒 LOCKED" if current_tracker.is_tracking_locked() else "🔓 OPEN"
                mode_status = "LOCK-ONLY" if current_tracker.lock_only_mode else "NORMAL"
                zoom_status = f"ZOOM:{'OFF' if not ENABLE_ZOOM_CONTROL else 'ON'}"
                ptz_status = f"PTZ:{'ON' if PTZ_TRACKING else 'OFF'}"
                
                status_lines = [
                    f"Camera: {CAMERA_CONFIG_1['name']} | Mode: {mode_status} | Status: {lock_status}",
                    f"Active Tracks: {active_tracks} | PTZ Target: {current_tracker.currently_tracked_id}",
                    f"Zoom: {current_ptz.current_zoom_level:.1f}x | Moving: {'YES' if current_ptz.is_moving else 'NO'} | {zoom_status} | {ptz_status}",
                    f"No Detection: {current_tracker.frames_without_detection}/{FRAMES_BEFORE_PRESET} | Min Detections: {current_tracker.min_consecutive_detections}"
                ]
                
                if current_ptz_target:
                    target_area = current_ptz_target['width'] * current_ptz_target.get('height', 0)
                    status_lines.append(f"PTZ Target ID:{current_ptz_target['track_id']} Area: {int(target_area)} px²")
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                
                # Draw status
                # cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                for i, status_text in enumerate(status_lines):
                    y_pos = 60 + (i * 25)
                    cv2.putText(display_frame, status_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write frame and display
                if writer:
                    writer.write(display_frame)
                
                cv2.imshow("Single Camera Aircraft PTZ Tracking", display_frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # Emergency stop
                    print("Emergency PTZ stop!")
                    current_ptz.emergency_stop()
                elif key == ord('c'):  # Clear current tracking
                    print("Clearing current PTZ tracking lock...")
                    current_tracker.currently_tracked_id = None
                    current_tracker.track_lock_frames = 0
                    self.current_tracked_id = None
                    self.lost_track_time = None
                elif key == ord('p'):  # Go to preset
                    print(f"Manually going to preset {PRESET_NUMBER}")
                    current_ptz.go_to_preset(PRESET_NUMBER)
                    self.preset_triggered = False
                elif key == ord('l'):  # Toggle lock mode
                    current_tracker.lock_only_mode = not current_tracker.lock_only_mode
                    print(f"Lock-only mode: {'ENABLED' if current_tracker.lock_only_mode else 'DISABLED'}")
                elif key == ord('t'):  # Toggle PTZ tracking
                    PTZ_TRACKING = not PTZ_TRACKING
                    print(f"PTZ Tracking: {'ENABLED' if PTZ_TRACKING else 'DISABLED'}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        finally:
            # Cleanup
            print("Cleaning up...")
            self.camera.ptz.emergency_stop()
            self.camera.ptz.go_to_preset(PRESET_NUMBER)
            
            self.camera.stop()
            
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print(f"Recording saved as: {output_filename}")

def main():
    """Main function"""
    model_path = "/home/skylark/Desktop/Aircraft-Tracking/aircraft-tracking-backend/models/Aircraft.pt"
    
    print("=== Single Camera Aircraft PTZ Tracking System ===")
    print(f"Configuration:")
    print(f"  Camera: {CAMERA_CONFIG_1['ip']} ({CAMERA_CONFIG_1['name']})")
    print(f"  GStreamer Support: Available for improved RTSP streaming")
    print(f"  Smooth PTZ Movement: Enhanced with movement dampening")
    print(f"  Minimum Consecutive Detections: 5 (before PTZ tracking starts)")
    print(f"  Size Filter: {'ENABLED' if ENABLE_SIZE_FILTER else 'DISABLED'}")
    print(f"  Lock-Only Mode: {'ENABLED' if LOCK_ONLY_MODE else 'DISABLED'}")
    print(f"  Edge Filtering: {'ENABLED' if ENABLE_EDGE_FILTERING else 'DISABLED'}")
    print(f"  Zoom Control: {'ENABLED' if ENABLE_ZOOM_CONTROL else 'DISABLED'}")
    print(f"  Zoom In: {'ENABLED' if ENABLE_ZOOM_IN else 'DISABLED'}")  
    print(f"  Zoom Out: {'ENABLED' if ENABLE_ZOOM_OUT else 'DISABLED'}")
    print(f"  PTZ Tracking: {'ENABLED' if PTZ_TRACKING else 'DISABLED'}")
    print(f"  Progressive Area-Based Zoom OUT:")
    print(f"    • < 10K: No zoom")
    print(f"    • 10K-20K: 2x zoom out (very initial) ✨ NEW")
    print(f"    • 20K-30K: 3x zoom out (early) ✨ NEW")
    print(f"    • 30K-40K: 3x zoom out (initial) ✨ ENHANCED")
    print(f"    • 40K-60K: 3x zoom out (moderate) ✨ ENHANCED")
    print(f"    • 60K-75K: 4x zoom out (aggressive)")
    print(f"    • 75K-85K: 4x zoom out (more aggressive)")
    print(f"    • 85K-95K: 2x zoom out (additional)")
    print(f"    • 95K-120K: 2x zoom out (additional)")
    print(f"    • > 120K: 2x zoom out (extreme)")
    print(f"  Real-time Aircraft Crop: Upper Right Corner ✨")
    print(f"  Auto Preset: {PRESET_NUMBER}")
    
    # Show zoom control warnings
    if not ENABLE_ZOOM_CONTROL:
        print("  ⚠️  WARNING: ALL ZOOM CONTROL IS DISABLED - Camera will only PAN and TILT")
    elif not ENABLE_ZOOM_IN and not ENABLE_ZOOM_OUT:
        print("  ⚠️  WARNING: Both zoom in and zoom out are disabled")
    elif not ENABLE_ZOOM_OUT:
        print("  ⚠️  WARNING: Zoom out is disabled - system may not handle large aircraft well")
    
    print("\n" + "="*60)
    
    # Create and run the single camera aircraft tracking system
    system = SingleCameraAircraftTracker(model_path)
    system.run()

if __name__ == "__main__":
    main()