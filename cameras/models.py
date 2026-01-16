"""
Camera models for Aircraft Tracking with PTZ support
"""
from django.db import models
from django.dispatch import receiver
from django.conf import settings
import uuid
import logging

logger = logging.getLogger(__name__)


class Camera(models.Model):
    """Model for camera devices with PTZ support"""
    CAMERA_TYPE_CHOICES = (
        ('ip', 'IP Camera'),
        ('rtsp', 'RTSP Stream'),
        ('usb', 'USB Camera'),
        ('file', 'Video File'),
        ('ptz', 'PTZ Camera'),
    )

    camera_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=256)
    camera_type = models.CharField(max_length=20, choices=CAMERA_TYPE_CHOICES, default='rtsp')
    rtsp_link = models.CharField(max_length=500)
    description = models.TextField(blank=True, null=True, default="")
    
    # Status
    is_healthy = models.BooleanField(default=True)
    is_active = models.BooleanField(default=True)
    is_streaming = models.BooleanField(default=True, help_text="Whether WebSocket streaming is enabled")
    ai_enabled = models.BooleanField(default=False, help_text="Whether AI inference is enabled for this camera")
    
    # MinIO Storage Paths (stores object path, not full URL)
    # Structure: cameras/{camera_id}/thumbnail/thumbnail.jpg
    thumbnail_path = models.CharField(max_length=500, blank=True, null=True)
    
    # Configuration
    roi_points = models.JSONField(default=list, blank=True, null=True, help_text="List of [x,y] coordinates defining the ROI polygon")
    config = models.JSONField(default=dict, blank=True, null=True)
    
    # Resolution
    resolution_width = models.IntegerField(default=1920, blank=True, null=True)
    resolution_height = models.IntegerField(default=1080, blank=True, null=True)
    fps = models.IntegerField(default=30, blank=True, null=True)
    
    # PTZ Configuration
    ptz_enabled = models.BooleanField(default=False, help_text="Whether PTZ control is enabled for this camera")
    ptz_ip = models.CharField(max_length=100, blank=True, null=True, help_text="PTZ camera IP address (if different from RTSP)")
    ptz_username = models.CharField(max_length=100, blank=True, null=True, help_text="PTZ camera username")
    ptz_password = models.CharField(max_length=100, blank=True, null=True, help_text="PTZ camera password")
    ptz_channel = models.IntegerField(default=1, help_text="PTZ channel number")
    ptz_preset_number = models.IntegerField(default=20, help_text="Default preset position number")
    
    # PTZ Tracking Settings
    ptz_tracking_enabled = models.BooleanField(default=True, help_text="Whether PTZ auto-tracking is enabled")
    ptz_zoom_enabled = models.BooleanField(default=True, help_text="Whether PTZ zoom control is enabled")
    ptz_zoom_in_enabled = models.BooleanField(default=True, help_text="Whether zoom in is enabled")
    ptz_zoom_out_enabled = models.BooleanField(default=True, help_text="Whether zoom out is enabled")
    
    # Tracking Settings
    tracking_lock_only_mode = models.BooleanField(default=True, help_text="Only track currently locked aircraft")
    tracking_min_consecutive_detections = models.IntegerField(default=5, help_text="Minimum detections before PTZ tracking")
    tracking_enable_size_filter = models.BooleanField(default=True, help_text="Filter out small aircraft")
    tracking_min_aircraft_width = models.IntegerField(default=10, help_text="Minimum aircraft width in pixels")
    tracking_enable_edge_filtering = models.BooleanField(default=True, help_text="Filter aircraft at frame edges")
    tracking_edge_margin_percent = models.FloatField(default=7.0, help_text="Edge margin as percentage of frame")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.camera_id})"
    
    @property
    def thumbnail_url(self) -> str | None:
        """Get the full MinIO URL for the thumbnail."""
        if self.thumbnail_path:
            return f"{settings.MINIO_PUBLIC_URL}/{settings.MINIO_BUCKET_NAME}/{self.thumbnail_path}"
        return None
    
    @property
    def stream_url(self) -> str:
        """Get the WebSocket stream URL for this camera."""
        backend_url = getattr(settings, 'BACKEND_PUBLIC_URL', 'http://localhost:8000')
        ws_url = backend_url.replace('http://', 'ws://').replace('https://', 'wss://')
        return f"{ws_url}/ws/video-feed/{self.camera_id}/"
    
    @property
    def ptz_config(self) -> dict:
        """Get PTZ configuration as dictionary."""
        return {
            'enabled': self.ptz_enabled,
            'ip': self.ptz_ip,
            'username': self.ptz_username,
            'password': self.ptz_password,
            'channel': self.ptz_channel,
            'preset_number': self.ptz_preset_number,
            'tracking_enabled': self.ptz_tracking_enabled,
            'zoom_enabled': self.ptz_zoom_enabled,
            'zoom_in_enabled': self.ptz_zoom_in_enabled,
            'zoom_out_enabled': self.ptz_zoom_out_enabled,
        }
    
    @property
    def tracking_config(self) -> dict:
        """Get tracking configuration as dictionary."""
        return {
            'lock_only_mode': self.tracking_lock_only_mode,
            'min_consecutive_detections': self.tracking_min_consecutive_detections,
            'enable_size_filter': self.tracking_enable_size_filter,
            'min_aircraft_width': self.tracking_min_aircraft_width,
            'enable_edge_filtering': self.tracking_enable_edge_filtering,
            'edge_margin_percent': self.tracking_edge_margin_percent,
        }
    
    def get_ptz_ip(self) -> str | None:
        """Get PTZ IP address, extracting from RTSP link if not specified."""
        if self.ptz_ip:
            return self.ptz_ip
        
        # Try to extract from RTSP link
        # Format: rtsp://username:password@ip_address/...
        try:
            import re
            match = re.search(r'@([\d.]+)', self.rtsp_link)
            if match:
                return match.group(1)
        except Exception:
            pass
        
        return None


@receiver(models.signals.post_delete, sender=Camera)
def post_delete_camera(sender, instance, *args, **kwargs):
    """Clean up MinIO files when camera is deleted"""
    try:
        from backend.storage import minio_storage
        minio_storage.delete_camera_files(str(instance.camera_id))
    except Exception as e:
        logger.warning(f"Failed to delete MinIO files for camera {instance.camera_id}: {e}")
