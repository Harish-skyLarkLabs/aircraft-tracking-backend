"""
PTZ Controller for Hikvision cameras
Based on the ML team's aircraft_det.py demo script
Provides smooth PTZ tracking for aircraft detection
"""
import requests
import time
import threading
import logging
import numpy as np
import xml.etree.ElementTree as ET
from queue import Queue, Empty
from collections import deque
from typing import Optional, Tuple
from requests.auth import HTTPDigestAuth
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(InsecureRequestWarning)

logger = logging.getLogger(__name__)


# PTZ Configuration Constants
PTZ_SPEED = 120
ZOOM_SPEED = 10
MAX_PTZ_VALUE = 120
MAX_ZOOM_LEVEL = 2.0
MIN_ZOOM_LEVEL = -2.0
TILT_SENSITIVITY = 0.9


class PTZCommand:
    """Represents a PTZ command to be executed"""
    
    def __init__(
        self,
        action: str,
        speed: int,
        duration: float = 0.05,
        pan_value: int = 0,
        tilt_value: int = 0,
        zoom_value: int = 0
    ):
        self.action = action
        self.speed = speed
        self.duration = duration
        self.timestamp = time.time()
        self.pan_value = pan_value
        self.tilt_value = tilt_value
        self.zoom_value = zoom_value


class PTZController:
    """
    Hikvision PTZ Camera Controller
    
    Provides smooth PTZ tracking with:
    - Proportional control for smooth movement
    - Movement dampening to reduce jitter
    - Dead zone to prevent micro-movements
    - Command queuing for non-blocking operation
    - Zoom control with limits
    """
    
    def __init__(
        self,
        ip: str,
        username: str,
        password: str,
        channel: int = 1,
        enable_ptz_tracking: bool = True,
        enable_zoom_control: bool = True,
        enable_zoom_in: bool = True,
        enable_zoom_out: bool = True,
        preset_number: int = 20,
    ):
        """
        Initialize PTZ camera controller
        
        Args:
            ip: Camera IP address
            username: Camera username
            password: Camera password
            channel: PTZ channel number
            enable_ptz_tracking: Enable/disable PTZ movement tracking
            enable_zoom_control: Enable/disable all zoom controls
            enable_zoom_in: Enable/disable zoom in
            enable_zoom_out: Enable/disable zoom out
            preset_number: Preset position number for home position
        """
        self.ip = ip
        self.username = username
        self.password = password
        self.channel = channel
        self.base_url = f"http://{ip}/ISAPI"
        self.auth = HTTPDigestAuth(username, password)
        
        # PTZ state
        self.current_zoom_level = 1.0
        self.current_pan = 0
        self.current_tilt = 0
        
        # Control flags
        self.enable_ptz_tracking = enable_ptz_tracking
        self.enable_zoom_control = enable_zoom_control
        self.enable_zoom_in = enable_zoom_in
        self.enable_zoom_out = enable_zoom_out
        self.preset_number = preset_number
        
        # Command queue for non-blocking PTZ control
        self.command_queue: Queue = Queue()
        self.ptz_thread: Optional[threading.Thread] = None
        self.stop_thread = False
        self.is_moving = False
        self.last_command_time = 0
        
        # Movement smoothing parameters
        self.movement_history: deque = deque(maxlen=10)
        self.movement_dampening = 0.50  # Reduce movement speed by 50%
        self.dead_zone = 0.001  # Dead zone for less aggressive tracking
        self.move_interval = 0.001  # Minimum time between movements
        
        # Connection state
        self.is_connected = False
        
        logger.info(f"PTZController initialized for camera at {ip}")
    
    def connect(self) -> bool:
        """Test connection and initialize PTZ"""
        if self.test_connection():
            logger.info(f"✓ Successfully connected to PTZ camera at {self.ip}")
            self.is_connected = True
            print(f"ip: {self.ip} connected")
            print(f"username: {self.username}")
            print(f"password: {self.password}")
            print(f"channel: {self.channel}")
            print(f"base_url: {self.base_url}")
            print(f"auth: {self.auth}")
            print(f"is_connected: {self.is_connected}")
            print(f"is_moving: {self.is_moving}")
            print(f"last_command_time: {self.last_command_time}")
            print(f"movement_history: {self.movement_history}")
            # self.get_current_ptz_position()
            self.start_ptz_thread()
            return True
        else:
            logger.error(f"✗ Failed to connect to PTZ camera at {self.ip}")
            return False
    
    def test_connection(self) -> bool:
        """Test if camera is accessible"""
        try:
            url = f"{self.base_url}/System/deviceInfo"
            response = requests.get(url, auth=self.auth, timeout=5, verify=False)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"PTZ connection error: {e}")
            return False
    
    def get_current_ptz_position(self) -> bool:
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
                
                logger.info(f"PTZ position: Pan={self.current_pan}, Tilt={self.current_tilt}, Zoom={self.current_zoom_level}x")
                return True
            else:
                logger.warning(f"Failed to get PTZ status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error getting PTZ position: {e}")
            return False
    
    def go_to_preset(self, preset_number: Optional[int] = None) -> bool:
        """Go to specified preset position"""
        if preset_number is None:
            preset_number = self.preset_number
            
        try:
            url = f"{self.base_url}/PTZCtrl/channels/{self.channel}/presets/{preset_number}/goto"
            response = requests.put(url, auth=self.auth, timeout=5, verify=False)
            
            if response.status_code == 200:
                logger.info(f"✓ Moving to preset {preset_number}")
                time.sleep(3)
                self.get_current_ptz_position()
                return True
            else:
                logger.warning(f"Failed to go to preset {preset_number}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error going to preset {preset_number}: {e}")
            return False
    
    def start_ptz_thread(self):
        """Start background thread for PTZ commands"""
        if self.ptz_thread is not None and self.ptz_thread.is_alive():
            return
            
        self.stop_thread = False
        self.ptz_thread = threading.Thread(target=self._ptz_worker, daemon=True)
        self.ptz_thread.start()
        logger.info("PTZ worker thread started")
    
    def _ptz_worker(self):
        """Background worker for PTZ commands with smooth execution"""
        while not self.stop_thread:
            try:
                command = self.command_queue.get(timeout=0.1)
                
                if command.action == 'PROPORTIONAL':
                    self.is_moving = True
                    self._execute_ptz_command(
                        'PROPORTIONAL', command.speed,
                        command.pan_value, command.tilt_value, command.zoom_value
                    )
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
                logger.error(f"PTZ worker error: {e}")
                self.is_moving = False
    
    def _execute_ptz_command(
        self,
        action: str,
        speed: int,
        pan_value: int = 0,
        tilt_value: int = 0,
        zoom_value: int = 0
    ) -> bool:
        """Execute PTZ command with enhanced smoothing and zoom control checks"""
        try:
            # Check zoom control permissions
            if not self.enable_zoom_control:
                if action in ['ZOOM_IN', 'ZOOM_OUT'] or zoom_value != 0:
                    return False
            
            if action == 'ZOOM_IN' and not self.enable_zoom_in:
                return False
                
            if action == 'ZOOM_OUT' and not self.enable_zoom_out:
                return False
                
            if zoom_value > 0 and not self.enable_zoom_in:
                zoom_value = 0
                
            if zoom_value < 0 and not self.enable_zoom_out:
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
            if (action == 'ZOOM_IN' or zoom_value > 0) and self.enable_zoom_in and self.enable_zoom_control:
                zoom_increment = 0.8 * (speed / 7.0) if zoom_value == 0 else abs(zoom_value) / 10
                self.current_zoom_level = min(self.current_zoom_level + zoom_increment, MAX_ZOOM_LEVEL)
            elif (action == 'ZOOM_OUT' or zoom_value < 0) and self.enable_zoom_out and self.enable_zoom_control:
                zoom_decrement = 1.2 * (speed / 7.0) if zoom_value == 0 else abs(zoom_value) / 8
                self.current_zoom_level = max(self.current_zoom_level - zoom_decrement, MIN_ZOOM_LEVEL)
            
            self.last_command_time = time.time()
            return response.status_code == 200
                
        except Exception as e:
            logger.error(f"PTZ execute error: {e}")
            return False
    
    def track_object_smooth(
        self,
        frame_width: int,
        frame_height: int,
        object_x: float,
        object_y: float,
        object_width: Optional[float] = None,
        object_height: Optional[float] = None
    ):
        """
        Enhanced smooth object tracking - PAN and TILT ONLY, NO ZOOM
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            object_x: Object center X coordinate
            object_y: Object center Y coordinate
            object_width: Object width (optional, not used for zoom)
            object_height: Object height (optional, not used for zoom)
        """
        if not self.enable_ptz_tracking:
            return
            
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
        
        # Check if object is outside dead zone
        if abs(avg_x) > self.dead_zone or abs(avg_y) > self.dead_zone:
            # Calculate movement speeds with exponential decay for smoother movement
            x_factor = np.sign(avg_x) * (1 - np.exp(-abs(avg_x) * 3))
            y_factor = np.sign(avg_y) * (1 - np.exp(-abs(avg_y) * 3))
            
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
                command = PTZCommand('PROPORTIONAL', PTZ_SPEED, 0.05, pan_speed, tilt_speed, 0)
                self.command_queue.put(command, block=False)
                self.last_command_time = current_time
                logger.debug(f"[PTZ SMOOTH] Pan: {pan_speed}, Tilt: {tilt_speed}")
        else:
            # Object is centered, stop movement
            self.emergency_stop()
    
    def ptz_control(self, action: str, speed: int = 7, duration: float = 0.05) -> bool:
        """Non-blocking PTZ control with automatic STOP"""
        try:
            if self.command_queue.qsize() < 2:
                command = PTZCommand(action, speed, duration)
                self.command_queue.put(command, block=False)
                return True
        except Exception as e:
            logger.error(f"PTZ control queue error: {e}")
        return False
    
    def ptz_proportional_control(
        self,
        pan_value: float,
        tilt_value: float,
        zoom_value: float = 0,
        duration: float = 0.05
    ) -> bool:
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
            logger.error(f"PTZ proportional control error: {e}")
        return False
    
    def control_zoom_for_aircraft_by_area(
        self,
        aircraft_width: float,
        aircraft_height: float,
        ptz_cooldown: float = 0.1
    ) -> bool:
        """
        Smart progressive area-based ZOOM OUT control with multiple thresholds
        
        Area thresholds:
        - < 10K: No change (aircraft is small enough)
        - 10K-20K: Zoom out 2x
        - 20K-30K: Zoom out 3x
        - 30K-40K: Zoom out 3x
        - 40K-60K: Zoom out 3x
        - 60K-75K: Zoom out 4x
        - 75K-85K: Zoom out 4x
        - 85K-95K: Zoom out 2x
        - 95K-120K: Zoom out 2x
        - > 120K: Zoom out 2x
        """
        if not self.enable_zoom_control:
            logger.debug("[ZOOM DISABLED] Zoom control is disabled")
            return False
            
        if aircraft_width is None or aircraft_height is None:
            return False
        
        # Calculate area
        area = aircraft_width * aircraft_height
        
        current_time = time.time()
        if current_time - self.last_command_time < ptz_cooldown:
            return False
        
        # Progressive zoom out based on area thresholds
        zoom_action = None
        zoom_duration = 0
        zoom_speed = 5
        zoom_label = ""
        
        if area < 10000:
            logger.debug(f"[AREA OK] Area: {int(area)} < 10K - No zoom needed")
            return False
        
        elif area > 120000:
            if self.enable_zoom_out:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.5
                zoom_speed = 6
                zoom_label = "EXTREME SIZE > 120K"
            else:
                return False
        
        elif area > 95000:
            if self.enable_zoom_out:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.5
                zoom_speed = 6
                zoom_label = "VERY LARGE 95K-120K"
            else:
                return False
        
        elif area > 85000:
            if self.enable_zoom_out:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.5
                zoom_speed = 6
                zoom_label = "LARGE 85K-95K"
            else:
                return False
        
        elif area > 75000:
            if self.enable_zoom_out:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.10
                zoom_speed = 8
                zoom_label = "AGGRESSIVE 75K-85K"
            else:
                return False
        
        elif area >= 60000:
            if self.enable_zoom_out:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.10
                zoom_speed = 8
                zoom_label = "HEAVY 60K-75K"
            else:
                return False
        
        elif area >= 40000:
            if self.enable_zoom_out:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.6
                zoom_speed = 6
                zoom_label = "MODERATE 40K-60K"
            else:
                return False
        
        elif area >= 30000:
            if self.enable_zoom_out:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.6
                zoom_speed = 6
                zoom_label = "INITIAL 30K-40K"
            else:
                return False
        
        elif area >= 20000:
            if self.enable_zoom_out:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.8
                zoom_speed = 7
                zoom_label = "EARLY 20K-30K"
            else:
                return False
        
        elif area >= 10000:
            if self.enable_zoom_out:
                zoom_action = 'ZOOM_OUT'
                zoom_duration = 0.8
                zoom_speed = 7
                zoom_label = "VERY INITIAL 10K-20K"
            else:
                return False
        
        else:
            return False
        
        # Execute zoom action
        if zoom_action == 'ZOOM_OUT':
            success = self.ptz_control('ZOOM_OUT', speed=zoom_speed, duration=zoom_duration)
            if success:
                logger.info(f"✓ ZOOM OUT executed - {zoom_label} - Speed: {zoom_speed}, Duration: {zoom_duration}s")
                return True
            else:
                logger.warning(f"✗ ZOOM OUT failed - {zoom_label}")
        
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
            logger.error(f"Emergency stop error: {e}")
    
    def get_zoom_percentage(self) -> float:
        """Get current zoom as percentage of maximum"""
        return (self.current_zoom_level - MIN_ZOOM_LEVEL) / (MAX_ZOOM_LEVEL - MIN_ZOOM_LEVEL) * 100
    
    def stop(self):
        """Stop PTZ thread and cleanup"""
        self.emergency_stop()
        self.stop_thread = True
        if self.ptz_thread and self.ptz_thread.is_alive():
            self.ptz_thread.join(timeout=2)
        self.is_connected = False
        logger.info(f"PTZ controller stopped for {self.ip}")
    
    def get_status(self) -> dict:
        """Get PTZ controller status"""
        return {
            'ip': self.ip,
            'is_connected': self.is_connected,
            'is_moving': self.is_moving,
            'current_zoom_level': self.current_zoom_level,
            'current_pan': self.current_pan,
            'current_tilt': self.current_tilt,
            'zoom_percentage': self.get_zoom_percentage(),
            'ptz_tracking_enabled': self.enable_ptz_tracking,
            'zoom_control_enabled': self.enable_zoom_control,
            'preset_number': self.preset_number,
            'queue_size': self.command_queue.qsize(),
        }


# Global PTZ controllers registry
_ptz_controllers: dict = {}


def get_ptz_controller(
    camera_id: str,
    ip: str = None,
    username: str = None,
    password: str = None,
    **kwargs
) -> Optional[PTZController]:
    """
    Get or create a PTZ controller for a camera
    
    Args:
        camera_id: Camera identifier
        ip: Camera IP address (required for new controller)
        username: Camera username (required for new controller)
        password: Camera password (required for new controller)
        **kwargs: Additional PTZ configuration options
        
    Returns:
        PTZController instance or None if creation fails
    """
    global _ptz_controllers
    
    if camera_id in _ptz_controllers:
        return _ptz_controllers[camera_id]
    
    if ip and username and password:
        controller = PTZController(ip, username, password, **kwargs)
        if controller.connect():
            _ptz_controllers[camera_id] = controller
            return controller
        else:
            logger.error(f"Failed to connect PTZ controller for camera {camera_id}")
            return None
    
    return None


def remove_ptz_controller(camera_id: str):
    """Remove and stop a PTZ controller"""
    global _ptz_controllers
    
    if camera_id in _ptz_controllers:
        _ptz_controllers[camera_id].stop()
        del _ptz_controllers[camera_id]
        logger.info(f"Removed PTZ controller for camera {camera_id}")


def get_all_ptz_status() -> dict:
    """Get status of all PTZ controllers"""
    return {
        camera_id: controller.get_status()
        for camera_id, controller in _ptz_controllers.items()
    }

