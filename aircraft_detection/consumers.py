"""
WebSocket consumers for real-time video feed and alerts.

Supports two streaming modes:
1. WebSocket frames (legacy) - Sends Base64-encoded JPEG frames
2. LiveKit mode - Sends only detection data, video comes from LiveKit WebRTC

Set USE_LIVEKIT=true in environment to enable LiveKit mode.
"""
import json
import asyncio
import os
import cv2
import base64
import time
import uuid
import logging
import traceback
import numpy as np
from uuid import UUID
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async

from .utils.camera_manager import camera_manager

logger = logging.getLogger(__name__)

# Check if LiveKit mode is enabled
USE_LIVEKIT = os.getenv("USE_LIVEKIT", "false").lower() == "true"


def convert_to_serializable(obj):
    """Convert numpy/torch objects to JSON serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    return obj


class UUIDEncoder(json.JSONEncoder):
    """JSON encoder that handles UUID objects"""
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


def encode_frame_to_base64(frame: np.ndarray, quality: int = 70) -> str:
    """Encode a frame as base64 JPEG for websocket transmission"""
    try:
        # Validate frame is a proper numpy array
        if frame is None:
            logger.warning("Frame is None, cannot encode")
            return None
        
        if not isinstance(frame, np.ndarray):
            logger.warning(f"Frame is not a numpy array, got {type(frame)}")
            return None
        
        if frame.size == 0 or len(frame.shape) < 2:
            logger.warning(f"Invalid frame shape: {frame.shape if hasattr(frame, 'shape') else 'unknown'}")
            return None
        
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        success, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        if not success:
            logger.warning("cv2.imencode returned False")
            return None
        
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{frame_base64}"
    except Exception as e:
        logger.error(f"Error encoding frame: {e}")
        return None


class VideoFeedConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time video feed.
    
    Supports two modes:
    - Legacy mode: Sends Base64-encoded JPEG frames (USE_LIVEKIT=false)
    - LiveKit mode: Sends only detection data (USE_LIVEKIT=true)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_id = None
        self.client_id = str(uuid.uuid4())
        self.is_connected = False
        self.send_frames_task = None
        self.send_detections_task = None
        self.check_detections_task = None
        
        self.last_frame_id = 0
        self.last_frame_time = 0
        self.target_fps = 15
        self.frame_interval = 1.0 / self.target_fps
        
        self.last_seen_detections = set()
        self.last_sent_detections = []
        
        # Direct stream handler for when processing is not active
        self.direct_stream = None
        self.camera_rtsp_link = None
        
        # LiveKit mode flag
        self.use_livekit = USE_LIVEKIT
    
    async def connect(self):
        """Handle WebSocket connection"""
        camera_manager.initialize()
        
        self.camera_id = self.scope['url_route']['kwargs']['camera_id']
        self.camera_id = UUID(self.camera_id)
        
        await self.accept()
        self.is_connected = True
        
        # Get camera RTSP link from database
        await self._load_camera_info()
        
        # Register client
        camera_manager.register_client(self.camera_id, self.client_id)
        
        logger.info(f"WebSocket connected: client {self.client_id} on camera {self.camera_id} (LiveKit: {self.use_livekit})")
        
        # Send initial status with LiveKit info
        camera_status = camera_manager.get_camera_status(self.camera_id)
        
        # Get LiveKit info if enabled
        livekit_info = None
        if self.use_livekit:
            try:
                from .utils.livekit_client import get_livekit_url, get_room_name, is_livekit_enabled
                if is_livekit_enabled():
                    livekit_info = {
                        "enabled": True,
                        "url": get_livekit_url(),
                        "room_name": get_room_name(str(self.camera_id)),
                        "token_endpoint": f"/api/cameras/{self.camera_id}/stream/token/",
                    }
            except ImportError:
                pass
        
        await self.send(text_data=json.dumps({
            "type": "connection_status",
            "camera_id": str(self.camera_id),
            "processing_active": camera_status.get('active', False),
            "use_livekit": self.use_livekit,
            "livekit": livekit_info,
            "timestamp": time.time()
        }, cls=UUIDEncoder))
        
        # Start appropriate tasks based on mode
        if self.use_livekit:
            # LiveKit mode: only send detection data, no frames
            self.send_detections_task = asyncio.create_task(self.send_detections_periodically())
        else:
            # Legacy mode: send frames via WebSocket
            self.send_frames_task = asyncio.create_task(self.send_frames_periodically())
        
        # Start detection checking task (for alerts)
        self.check_detections_task = asyncio.create_task(self.check_new_detections())
    
    async def _load_camera_info(self):
        """Load camera info from database"""
        @sync_to_async
        def get_camera():
            from cameras.models import Camera
            try:
                camera = Camera.objects.get(camera_id=self.camera_id)
                return camera.rtsp_link
            except Camera.DoesNotExist:
                return None
        
        self.camera_rtsp_link = await get_camera()
    
    def _start_direct_stream(self):
        """Start direct RTSP stream when processing is not active"""
        if self.direct_stream is None and self.camera_rtsp_link:
            from .utils.stream_handler import StreamHandler
            self.direct_stream = StreamHandler(self.camera_rtsp_link)
            if self.direct_stream.start():
                logger.info(f"Started direct stream for camera {self.camera_id}")
            else:
                logger.error(f"Failed to start direct stream for camera {self.camera_id}")
                self.direct_stream = None
    
    def _stop_direct_stream(self):
        """Stop direct RTSP stream"""
        if self.direct_stream:
            self.direct_stream.stop()
            self.direct_stream = None
            logger.info(f"Stopped direct stream for camera {self.camera_id}")
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        logger.info(f"WebSocket disconnecting: client {self.client_id}")
        
        # Cancel tasks
        if self.send_frames_task:
            self.send_frames_task.cancel()
            try:
                await self.send_frames_task
            except asyncio.CancelledError:
                pass
        
        if self.send_detections_task:
            self.send_detections_task.cancel()
            try:
                await self.send_detections_task
            except asyncio.CancelledError:
                pass
        
        if self.check_detections_task:
            self.check_detections_task.cancel()
            try:
                await self.check_detections_task
            except asyncio.CancelledError:
                pass
        
        # Stop direct stream if running
        self._stop_direct_stream()
        
        # Unregister client
        if self.camera_id:
            camera_manager.unregister_client(self.camera_id, self.client_id)
        
        self.is_connected = False
        logger.info(f"WebSocket disconnected: client {self.client_id}")
    
    async def send_frames_periodically(self):
        """Task to send frames at regular intervals"""
        try:
            logger.info(f"Started frame sending task for client {self.client_id}")
            await asyncio.sleep(0.1)
            
            next_frame_time = time.time()
            direct_frame_id = 0
            
            while self.is_connected:
                current_time = time.time()
                
                # Maintain frame rate
                sleep_time = next_frame_time - current_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                elif sleep_time < -3 * self.frame_interval:
                    next_frame_time = current_time
                
                next_frame_time += self.frame_interval
                
                # Check if camera processing is active
                camera_status = camera_manager.get_camera_status(self.camera_id)
                processing_active = camera_status.get('active', False)
                
                frame = None
                detections = []
                process_time = 0
                frame_id = 0
                
                if processing_active:
                    # Processing is active - get frames from camera manager (with AI)
                    # Stop direct stream if it was running
                    if self.direct_stream:
                        self._stop_direct_stream()
                    
                    frame_data = camera_manager.get_latest_frame(self.camera_id)
                    
                    # Skip if same frame
                    if frame_data.get('frame_id', 0) == self.last_frame_id:
                        continue
                    
                    frame = frame_data.get('frame')
                    detections = frame_data.get('detections', [])
                    process_time = frame_data.get('process_time', 0)
                    frame_id = frame_data.get('frame_id', 0)
                    self.last_frame_id = frame_id
                else:
                    # Processing is NOT active - stream raw RTSP directly
                    # Start direct stream if not already running
                    if self.direct_stream is None and self.camera_rtsp_link:
                        self._start_direct_stream()
                    
                    if self.direct_stream:
                        # get_frame returns (success, frame) tuple
                        ret, raw_frame = self.direct_stream.get_frame()
                        if ret and raw_frame is not None:
                            frame = raw_frame
                            direct_frame_id += 1
                            frame_id = direct_frame_id
                        else:
                            # No frame available yet
                            await asyncio.sleep(0.05)
                            continue
                    else:
                        # No direct stream available
                        await asyncio.sleep(0.1)
                        continue
                
                if frame is None:
                    continue
                
                try:
                    # Encode frame
                    frame_base64 = encode_frame_to_base64(frame)
                    if not frame_base64:
                        continue
                    
                    # Convert detections to serializable format
                    detections = convert_to_serializable(detections)
                    camera_status = convert_to_serializable(camera_status)
                    
                    # Prepare response
                    response = {
                        "type": "frame",
                        "frame": frame_base64,
                        "processing_active": processing_active,
                        "detections": detections,
                        "timestamp": time.time(),
                        "frame_id": frame_id,
                        "stats": {
                            "process_time_ms": round(process_time * 1000, 2),
                            "fps": round(1.0 / max(0.001, time.time() - self.last_frame_time), 1) if self.last_frame_time > 0 else 0,
                            "active_tracks": camera_status.get('active_tracks', 0) if processing_active else 0
                        }
                    }
                    
                    await self.send(text_data=json.dumps(response, cls=UUIDEncoder))
                    self.last_frame_time = time.time()
                    
                except Exception as e:
                    logger.error(f"Error preparing frame: {e}")
                    await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            logger.info(f"Frame sending task cancelled for client {self.client_id}")
            self._stop_direct_stream()
        except Exception as e:
            logger.error(f"Error in send_frames_periodically: {e}")
            traceback.print_exc()
            self._stop_direct_stream()
    
    async def send_detections_periodically(self):
        """
        Task to send only detection data (LiveKit mode).
        
        In LiveKit mode, video comes from WebRTC, so we only need to send
        detection bounding boxes and metadata for the frontend to overlay.
        """
        try:
            logger.info(f"Started detection sending task for client {self.client_id} (LiveKit mode)")
            await asyncio.sleep(0.1)
            
            detection_interval = 0.1  # 10 Hz detection updates
            
            while self.is_connected:
                try:
                    # Check if camera processing is active
                    camera_status = camera_manager.get_camera_status(self.camera_id)
                    processing_active = camera_status.get('active', False)
                    
                    if processing_active:
                        # Get latest frame data (for detections only)
                        frame_data = camera_manager.get_latest_frame(self.camera_id)
                        
                        # Skip if same frame
                        if frame_data.get('frame_id', 0) == self.last_frame_id:
                            await asyncio.sleep(detection_interval)
                            continue
                        
                        detections = frame_data.get('detections', [])
                        process_time = frame_data.get('process_time', 0)
                        frame_id = frame_data.get('frame_id', 0)
                        self.last_frame_id = frame_id
                        
                        # Convert detections to serializable format
                        detections = convert_to_serializable(detections)
                        
                        # Send detection data (no frame)
                        response = {
                            "type": "detections",
                            "processing_active": True,
                            "detections": detections,
                            "timestamp": time.time(),
                            "frame_id": frame_id,
                            "stats": {
                                "process_time_ms": round(process_time * 1000, 2),
                                "fps": round(1.0 / max(0.001, time.time() - self.last_frame_time), 1) if self.last_frame_time > 0 else 0,
                                "active_tracks": len(detections)
                            }
                        }
                        
                        await self.send(text_data=json.dumps(response, cls=UUIDEncoder))
                        self.last_frame_time = time.time()
                        self.last_sent_detections = detections
                    else:
                        # AI not active - send empty detections
                        if self.last_sent_detections:
                            await self.send(text_data=json.dumps({
                                "type": "detections",
                                "processing_active": False,
                                "detections": [],
                                "timestamp": time.time(),
                                "frame_id": 0,
                                "stats": {
                                    "process_time_ms": 0,
                                    "fps": 0,
                                    "active_tracks": 0
                                }
                            }, cls=UUIDEncoder))
                            self.last_sent_detections = []
                    
                    await asyncio.sleep(detection_interval)
                    
                except Exception as e:
                    logger.error(f"Error sending detections: {e}")
                    await asyncio.sleep(0.5)
        
        except asyncio.CancelledError:
            logger.info(f"Detection sending task cancelled for client {self.client_id}")
        except Exception as e:
            logger.error(f"Error in send_detections_periodically: {e}")
            traceback.print_exc()
    
    async def check_new_detections(self):
        """Task to check for new detections and send alerts"""
        try:
            logger.info(f"Started detection check task for client {self.client_id}")
            
            while self.is_connected:
                try:
                    # Get recent detections from database
                    @sync_to_async
                    def get_recent_detections():
                        from aircraft_detection.models import AircraftDetection
                        
                        detections = AircraftDetection.objects.filter(
                            camera_id=str(self.camera_id),
                            status='pending'
                        ).order_by('-detection_time')[:10]
                        
                        new_detections = []
                        new_ids = set()
                        
                        for det in detections:
                            det_id = str(det.detection_id)
                            if det_id not in self.last_seen_detections:
                                new_detections.append({
                                    'id': det_id,
                                    'detection_type': det.detection_type,
                                    'action': det.action,
                                    'confidence': det.confidence,
                                    'title': det.title,
                                    'severity': det.severity,
                                    'camera_id': det.camera_id,
                                    'detection_time': det.detection_time.isoformat(),
                                    'image_url': det.image_url,
                                })
                                new_ids.add(det_id)
                        
                        return new_detections, new_ids
                    
                    new_detections, new_ids = await get_recent_detections()
                    self.last_seen_detections.update(new_ids)
                    
                    if new_detections:
                        await self.send(text_data=json.dumps({
                            'type': 'new_detections',
                            'detections': new_detections,
                            'timestamp': time.time()
                        }, cls=UUIDEncoder))
                    
                    # Limit set size
                    if len(self.last_seen_detections) > 100:
                        self.last_seen_detections = set(list(self.last_seen_detections)[-100:])
                    
                    await asyncio.sleep(2.0)
                    
                except Exception as e:
                    logger.error(f"Error checking detections: {e}")
                    await asyncio.sleep(5.0)
        
        except asyncio.CancelledError:
            logger.info(f"Detection check task cancelled for client {self.client_id}")
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            
            if 'command' in data:
                command = data['command']
                
                if command == 'check_status':
                    camera_status = camera_manager.get_camera_status(self.camera_id)
                    camera_status = convert_to_serializable(camera_status)
                    
                    await self.send(text_data=json.dumps({
                        "type": "status_update",
                        "camera_id": str(self.camera_id),
                        "status": camera_status,
                        "timestamp": time.time()
                    }, cls=UUIDEncoder))
                
                elif command == 'request_high_quality_frame':
                    frame_data = camera_manager.get_latest_frame(self.camera_id)
                    frame_base64 = encode_frame_to_base64(frame_data['frame'], quality=90)
                    
                    if frame_base64:
                        await self.send(text_data=json.dumps({
                            "type": "high_quality_frame",
                            "frame": frame_base64,
                            "timestamp": time.time(),
                            "frame_id": frame_data.get('frame_id', 0)
                        }, cls=UUIDEncoder))
                
                elif command == 'get_detections':
                    count = int(data.get('count', 10))
                    
                    @sync_to_async
                    def get_detections():
                        from aircraft_detection.models import AircraftDetection
                        
                        detections = AircraftDetection.objects.filter(
                            camera_id=str(self.camera_id)
                        ).order_by('-detection_time')[:count]
                        
                        return [{
                            'id': str(d.detection_id),
                            'detection_type': d.detection_type,
                            'action': d.action,
                            'confidence': d.confidence,
                            'title': d.title,
                            'severity': d.severity,
                            'detection_time': d.detection_time.isoformat(),
                            'image_url': d.image_url,
                        } for d in detections]
                    
                    detections = await get_detections()
                    
                    await self.send(text_data=json.dumps({
                        'type': 'detections',
                        'detections': detections,
                        'timestamp': time.time()
                    }, cls=UUIDEncoder))
                
                else:
                    await self.send(text_data=json.dumps({
                        "type": "error",
                        "message": f"Unknown command: {command}",
                        "timestamp": time.time()
                    }, cls=UUIDEncoder))
            
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                "type": "error",
                "message": "Invalid JSON format",
                "timestamp": time.time()
            }, cls=UUIDEncoder))
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            traceback.print_exc()


class DetectionsConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time AI detection data only.
    
    This is a lightweight consumer specifically for LiveKit mode where
    video comes from WebRTC and we only need to send detection overlays.
    
    Endpoint: /ws/detections/<camera_id>/
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_id = None
        self.client_id = str(uuid.uuid4())
        self.is_connected = False
        self.send_detections_task = None
        self.last_frame_id = 0
        self.last_frame_time = 0
    
    async def connect(self):
        """Handle WebSocket connection"""
        camera_manager.initialize()
        
        self.camera_id = self.scope['url_route']['kwargs']['camera_id']
        self.camera_id = UUID(self.camera_id)
        
        await self.accept()
        self.is_connected = True
        
        logger.info(f"Detections WebSocket connected: client {self.client_id} on camera {self.camera_id}")
        
        # Send initial status
        camera_status = camera_manager.get_camera_status(self.camera_id)
        await self.send(text_data=json.dumps({
            "type": "connection_status",
            "camera_id": str(self.camera_id),
            "processing_active": camera_status.get('active', False),
            "timestamp": time.time()
        }, cls=UUIDEncoder))
        
        # Start detection sending task
        self.send_detections_task = asyncio.create_task(self._send_detections_loop())
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        logger.info(f"Detections WebSocket disconnecting: client {self.client_id}")
        
        if self.send_detections_task:
            self.send_detections_task.cancel()
            try:
                await self.send_detections_task
            except asyncio.CancelledError:
                pass
        
        self.is_connected = False
        logger.info(f"Detections WebSocket disconnected: client {self.client_id}")
    
    async def _send_detections_loop(self):
        """Send detection data at regular intervals"""
        try:
            detection_interval = 0.1  # 10 Hz
            
            while self.is_connected:
                try:
                    camera_status = camera_manager.get_camera_status(self.camera_id)
                    processing_active = camera_status.get('active', False)
                    
                    if processing_active:
                        frame_data = camera_manager.get_latest_frame(self.camera_id)
                        
                        # Skip if same frame
                        if frame_data.get('frame_id', 0) == self.last_frame_id:
                            await asyncio.sleep(detection_interval)
                            continue
                        
                        detections = frame_data.get('detections', [])
                        process_time = frame_data.get('process_time', 0)
                        frame_id = frame_data.get('frame_id', 0)
                        self.last_frame_id = frame_id
                        
                        # Convert to serializable format
                        detections = convert_to_serializable(detections)
                        
                        response = {
                            "type": "detections",
                            "processing_active": True,
                            "detections": detections,
                            "timestamp": time.time(),
                            "frame_id": frame_id,
                            "stats": {
                                "process_time_ms": round(process_time * 1000, 2),
                                "fps": round(1.0 / max(0.001, time.time() - self.last_frame_time), 1) if self.last_frame_time > 0 else 0,
                                "active_tracks": len(detections)
                            }
                        }
                        
                        await self.send(text_data=json.dumps(response, cls=UUIDEncoder))
                        self.last_frame_time = time.time()
                    
                    await asyncio.sleep(detection_interval)
                    
                except Exception as e:
                    logger.error(f"Error in detection loop: {e}")
                    await asyncio.sleep(0.5)
        
        except asyncio.CancelledError:
            logger.info(f"Detection loop cancelled for client {self.client_id}")
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            command = data.get('command')
            
            if command == 'check_status':
                camera_status = camera_manager.get_camera_status(self.camera_id)
                camera_status = convert_to_serializable(camera_status)
                
                await self.send(text_data=json.dumps({
                    "type": "status_update",
                    "camera_id": str(self.camera_id),
                    "status": camera_status,
                    "timestamp": time.time()
                }, cls=UUIDEncoder))
            else:
                await self.send(text_data=json.dumps({
                    "type": "error",
                    "message": f"Unknown command: {command}",
                    "timestamp": time.time()
                }, cls=UUIDEncoder))
        
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                "type": "error",
                "message": "Invalid JSON format",
                "timestamp": time.time()
            }, cls=UUIDEncoder))
        except Exception as e:
            logger.error(f"Error handling message: {e}")


class AlertsConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time alerts.
    
    Receives alerts from:
    1. AlertService broadcast via channel layer (real-time)
    2. Periodic database polling (backup)
    
    Group name: "alerts"
    """
    
    async def connect(self):
        """Handle WebSocket connection"""
        # Join the alerts group for real-time broadcasts
        await self.channel_layer.group_add("alerts", self.channel_name)
        await self.accept()
        
        self.is_connected = True
        self.client_id = str(uuid.uuid4())
        self.last_seen_alerts = set()
        
        logger.info(f"Alerts WebSocket connected: client {self.client_id}")
        
        # Send connection confirmation
        await self.send(text_data=json.dumps({
            "type": "connection_status",
            "status": "connected",
            "client_id": self.client_id,
            "timestamp": time.time()
        }, cls=UUIDEncoder))
        
        # Send recent alerts on connect
        await self._send_recent_alerts()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        await self.channel_layer.group_discard("alerts", self.channel_name)
        self.is_connected = False
        logger.info(f"Alerts WebSocket disconnected: client {self.client_id}")
    
    async def alert_message(self, event):
        """
        Handle alert broadcast from AlertService.
        This is called when AlertService broadcasts a new alert.
        """
        try:
            event_type = event.get('event', 'new_alert')
            data = event.get('data', {})
            data = convert_to_serializable(data)
            
            await self.send(text_data=json.dumps({
                'type': event_type,
                'data': data,
                'timestamp': time.time()
            }, cls=UUIDEncoder))
            
            logger.debug(f"Sent {event_type} to client {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error sending alert message: {e}")
    
    async def _send_recent_alerts(self):
        """Send recent alerts to newly connected client"""
        try:
            @sync_to_async
            def get_recent_alerts():
                from aircraft_detection.models import AircraftDetection
                
                alerts = AircraftDetection.objects.filter(
                    is_read=False
                ).order_by('-detection_time')[:20]
                
                return [{
                    'id': str(a.detection_id),
                    'detection_type': a.detection_type,
                    'confidence': a.confidence,
                    'title': a.title,
                    'description': a.description,
                    'severity': a.severity,
                    'is_read': a.is_read,
                    'camera_id': a.camera_id,
                    'camera_name': a.camera_name,
                    'image_url': a.image_url,
                    'video_url': a.video_url,
                    'timestamp': a.detection_time.isoformat(),
                } for a in alerts]
            
            alerts = await get_recent_alerts()
            
            await self.send(text_data=json.dumps({
                'type': 'initial_alerts',
                'data': alerts,
                'count': len(alerts),
                'timestamp': time.time()
            }, cls=UUIDEncoder))
            
        except Exception as e:
            logger.error(f"Error sending recent alerts: {e}")
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            command = data.get('command')
            
            if command == 'get_alerts':
                # Get alerts with optional filters
                count = int(data.get('count', 20))
                camera_id = data.get('camera_id')
                unread_only = data.get('unread_only', True)
                
                @sync_to_async
                def get_alerts():
                    from aircraft_detection.models import AircraftDetection
                    
                    queryset = AircraftDetection.objects.all()
                    
                    if unread_only:
                        queryset = queryset.filter(is_read=False)
                    
                    if camera_id:
                        queryset = queryset.filter(camera_id=camera_id)
                    
                    alerts = queryset.order_by('-detection_time')[:count]
                    
                    return [{
                        'id': str(a.detection_id),
                        'detection_type': a.detection_type,
                        'confidence': a.confidence,
                        'title': a.title,
                        'description': a.description,
                        'severity': a.severity,
                        'is_read': a.is_read,
                        'camera_id': a.camera_id,
                        'camera_name': a.camera_name,
                        'image_url': a.image_url,
                        'video_url': a.video_url,
                        'timestamp': a.detection_time.isoformat(),
                    } for a in alerts]
                
                alerts = await get_alerts()
                
                await self.send(text_data=json.dumps({
                    'type': 'alerts',
                    'data': alerts,
                    'count': len(alerts),
                    'timestamp': time.time()
                }, cls=UUIDEncoder))
            
            elif command == 'mark_read':
                # Mark alert as read
                alert_id = data.get('alert_id')
                
                if alert_id:
                    @sync_to_async
                    def mark_read():
                        from aircraft_detection.models import AircraftDetection
                        AircraftDetection.objects.filter(
                            detection_id=alert_id
                        ).update(is_read=True)
                    
                    await mark_read()
                    
                    await self.send(text_data=json.dumps({
                        'type': 'alert_marked_read',
                        'alert_id': alert_id,
                        'timestamp': time.time()
                    }, cls=UUIDEncoder))
            
            elif command == 'mark_all_read':
                # Mark all alerts as read
                camera_id = data.get('camera_id')
                
                @sync_to_async
                def mark_all_read():
                    from aircraft_detection.models import AircraftDetection
                    queryset = AircraftDetection.objects.filter(is_read=False)
                    if camera_id:
                        queryset = queryset.filter(camera_id=camera_id)
                    count = queryset.update(is_read=True)
                    return count
                
                count = await mark_all_read()
                
                await self.send(text_data=json.dumps({
                    'type': 'all_alerts_marked_read',
                    'count': count,
                    'timestamp': time.time()
                }, cls=UUIDEncoder))
            
            elif command == 'get_unread_count':
                # Get unread alert count
                @sync_to_async
                def get_count():
                    from aircraft_detection.models import AircraftDetection
                    return AircraftDetection.objects.filter(is_read=False).count()
                
                count = await get_count()
                
                await self.send(text_data=json.dumps({
                    'type': 'unread_count',
                    'count': count,
                    'timestamp': time.time()
                }, cls=UUIDEncoder))
            
            else:
                await self.send(text_data=json.dumps({
                    "type": "error",
                    "message": f"Unknown command: {command}",
                    "timestamp": time.time()
                }, cls=UUIDEncoder))
        
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                "type": "error",
                "message": "Invalid JSON format",
                "timestamp": time.time()
            }, cls=UUIDEncoder))
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            traceback.print_exc()


