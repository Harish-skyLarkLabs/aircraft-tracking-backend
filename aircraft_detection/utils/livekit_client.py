"""
LiveKit Client for managing video streams via WebRTC.

This module provides integration with LiveKit for low-latency video streaming,
replacing the WebSocket-based Base64 frame streaming approach.

Features:
- RTSP stream ingestion via LiveKit Ingress
- Token generation for frontend viewers
- Stream status monitoring
- Automatic cleanup on camera deletion
"""
import os
import time
import logging
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Check if livekit SDK is available
try:
    from livekit import api
    from livekit.api import LiveKitAPI
    from livekit.protocol import ingress as ingress_proto
    from livekit.protocol import room as room_proto
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    logger.warning("LiveKit SDK not installed. Run: pip install livekit livekit-api")


@dataclass
class LiveKitConfig:
    """LiveKit configuration settings"""
    url: str = "ws://localhost:7880"
    api_key: str = "devkey"
    api_secret: str = "secret"
    
    @classmethod
    def from_env(cls) -> 'LiveKitConfig':
        """Load configuration from environment variables"""
        return cls(
            url=os.getenv("LIVEKIT_URL", "ws://localhost:7880"),
            api_key=os.getenv("LIVEKIT_API_KEY", "devkey"),
            api_secret=os.getenv("LIVEKIT_API_SECRET", "secret"),
        )


@dataclass
class IngressInfo:
    """Information about a LiveKit ingress stream"""
    ingress_id: str
    stream_key: str
    room_name: str
    participant_identity: str
    url: str
    status: str
    started_at: Optional[datetime] = None


class LiveKitClient:
    """
    Client for managing LiveKit streams and tokens.
    
    Provides methods for:
    - Creating RTSP ingress streams
    - Generating viewer tokens
    - Monitoring stream status
    - Cleaning up streams
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for LiveKit client"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.config = LiveKitConfig.from_env()
        self._api: Optional[LiveKitAPI] = None
        self._ingresses: Dict[str, IngressInfo] = {}  # camera_id -> IngressInfo
        self._enabled = LIVEKIT_AVAILABLE and os.getenv("USE_LIVEKIT", "false").lower() == "true"
        
        if self._enabled:
            self._initialize_api()
        else:
            logger.info("LiveKit is disabled. Set USE_LIVEKIT=true to enable.")
    
    def _initialize_api(self):
        """Initialize the LiveKit API client"""
        if not LIVEKIT_AVAILABLE:
            logger.error("Cannot initialize LiveKit API - SDK not installed")
            return
        
        try:
            self._api = LiveKitAPI(
                url=self.config.url,
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
            )
            logger.info(f"LiveKit API initialized: {self.config.url}")
        except Exception as e:
            logger.error(f"Failed to initialize LiveKit API: {e}")
            self._api = None
    
    @property
    def is_enabled(self) -> bool:
        """Check if LiveKit is enabled and available"""
        return self._enabled and self._api is not None
    
    async def create_ingress(
        self,
        camera_id: str,
        rtsp_url: str,
        camera_name: str = "",
    ) -> Optional[IngressInfo]:
        """
        Create an RTSP ingress for a camera stream.
        
        Args:
            camera_id: Unique camera identifier
            rtsp_url: RTSP stream URL
            camera_name: Display name for the camera
            
        Returns:
            IngressInfo if successful, None otherwise
        """
        if not self.is_enabled:
            logger.warning("LiveKit is not enabled, skipping ingress creation")
            return None
        
        room_name = f"camera-{camera_id}"
        participant_identity = f"camera-{camera_id}"
        
        try:
            # Create ingress request
            request = ingress_proto.CreateIngressRequest(
                input_type=ingress_proto.IngressInput.URL_INPUT,
                url=rtsp_url,
                name=camera_name or f"Camera {camera_id}",
                room_name=room_name,
                participant_identity=participant_identity,
                participant_name=camera_name or f"Camera {camera_id}",
                # Video encoding settings for low latency
                video=ingress_proto.IngressVideoOptions(
                    preset=ingress_proto.IngressVideoEncodingPreset.H264_1080P_30,
                ),
                # Audio settings (muted for security cameras)
                audio=ingress_proto.IngressAudioOptions(
                    preset=ingress_proto.IngressAudioEncodingPreset.OPUS_MONO_64KBS,
                ),
            )
            
            # Create the ingress
            ingress = await self._api.ingress.create_ingress(request)
            
            ingress_info = IngressInfo(
                ingress_id=ingress.ingress_id,
                stream_key=ingress.stream_key,
                room_name=room_name,
                participant_identity=participant_identity,
                url=ingress.url,
                status="active",
                started_at=datetime.now(),
            )
            
            self._ingresses[camera_id] = ingress_info
            logger.info(f"Created LiveKit ingress for camera {camera_id}: {ingress.ingress_id}")
            
            return ingress_info
            
        except Exception as e:
            logger.error(f"Failed to create ingress for camera {camera_id}: {e}")
            return None
    
    async def delete_ingress(self, camera_id: str) -> bool:
        """
        Delete an RTSP ingress for a camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_enabled:
            return True
        
        ingress_info = self._ingresses.get(camera_id)
        if not ingress_info:
            logger.warning(f"No ingress found for camera {camera_id}")
            return True
        
        try:
            request = ingress_proto.DeleteIngressRequest(
                ingress_id=ingress_info.ingress_id,
            )
            await self._api.ingress.delete_ingress(request)
            
            del self._ingresses[camera_id]
            logger.info(f"Deleted LiveKit ingress for camera {camera_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete ingress for camera {camera_id}: {e}")
            return False
    
    def generate_viewer_token(
        self,
        camera_id: str,
        user_id: str,
        user_name: str = "",
        ttl_seconds: int = 3600,
    ) -> Optional[str]:
        """
        Generate a viewer token for accessing a camera stream.
        
        Args:
            camera_id: Camera identifier
            user_id: Unique user identifier
            user_name: Display name for the user
            ttl_seconds: Token time-to-live in seconds
            
        Returns:
            JWT token string if successful, None otherwise
        """
        if not LIVEKIT_AVAILABLE:
            logger.error("LiveKit SDK not available")
            return None
        
        room_name = f"camera-{camera_id}"
        
        try:
            # Create access token with new API (chained methods)
            token = (
                api.AccessToken(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                )
                .with_identity(user_id)
                .with_name(user_name or user_id)
                .with_ttl(timedelta(seconds=ttl_seconds))
                .with_grants(api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_subscribe=True,
                    can_publish=False,  # Viewers can't publish
                    can_publish_data=False,
                ))
            )
            
            jwt_token = token.to_jwt()
            logger.debug(f"Generated viewer token for camera {camera_id}, user {user_id}")
            
            return jwt_token
            
        except Exception as e:
            logger.error(f"Failed to generate viewer token: {e}")
            return None
    
    async def get_stream_status(self, camera_id: str) -> Dict[str, Any]:
        """
        Get the status of a camera stream.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Dictionary with stream status information
        """
        if not self.is_enabled:
            return {"enabled": False, "status": "disabled"}
        
        ingress_info = self._ingresses.get(camera_id)
        if not ingress_info:
            return {"enabled": True, "status": "not_found", "camera_id": camera_id}
        
        try:
            # List ingresses to get current status
            request = ingress_proto.ListIngressRequest(
                ingress_id=ingress_info.ingress_id,
            )
            response = await self._api.ingress.list_ingress(request)
            
            if response.items:
                ingress = response.items[0]
                return {
                    "enabled": True,
                    "status": "active" if ingress.state.status == ingress_proto.IngressState.Status.ENDPOINT_PUBLISHING else "inactive",
                    "camera_id": camera_id,
                    "ingress_id": ingress_info.ingress_id,
                    "room_name": ingress_info.room_name,
                    "started_at": ingress_info.started_at.isoformat() if ingress_info.started_at else None,
                }
            
            return {"enabled": True, "status": "not_found", "camera_id": camera_id}
            
        except Exception as e:
            logger.error(f"Failed to get stream status for camera {camera_id}: {e}")
            return {"enabled": True, "status": "error", "error": str(e)}
    
    def get_room_name(self, camera_id: str) -> str:
        """Get the LiveKit room name for a camera"""
        return f"camera-{camera_id}"
    
    def get_websocket_url(self) -> str:
        """Get the LiveKit WebSocket URL for frontend connection"""
        return self.config.url
    
    async def list_rooms(self) -> list:
        """List all active LiveKit rooms"""
        if not self.is_enabled:
            return []
        
        try:
            request = room_proto.ListRoomsRequest()
            response = await self._api.room.list_rooms(request)
            return [room.name for room in response.rooms]
        except Exception as e:
            logger.error(f"Failed to list rooms: {e}")
            return []
    
    async def cleanup_all(self):
        """Clean up all ingresses (called on shutdown)"""
        if not self.is_enabled:
            return
        
        camera_ids = list(self._ingresses.keys())
        for camera_id in camera_ids:
            await self.delete_ingress(camera_id)
        
        logger.info("Cleaned up all LiveKit ingresses")


# Singleton instance
livekit_client = LiveKitClient()


# Synchronous wrapper functions for use in non-async contexts
def get_viewer_token(camera_id: str, user_id: str, user_name: str = "") -> Optional[str]:
    """Synchronous wrapper to generate a viewer token"""
    return livekit_client.generate_viewer_token(camera_id, user_id, user_name)


def get_livekit_url() -> str:
    """Get the LiveKit WebSocket URL"""
    return livekit_client.get_websocket_url()


def get_room_name(camera_id: str) -> str:
    """Get the LiveKit room name for a camera"""
    return livekit_client.get_room_name(camera_id)


def is_livekit_enabled() -> bool:
    """Check if LiveKit is enabled"""
    return livekit_client.is_enabled

