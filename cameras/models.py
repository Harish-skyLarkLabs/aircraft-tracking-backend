"""
Camera models for Aircraft Tracking
"""
from django.db import models
from django.dispatch import receiver
from django.conf import settings
import uuid
import logging

logger = logging.getLogger(__name__)


class Camera(models.Model):
    """Model for camera devices"""
    CAMERA_TYPE_CHOICES = (
        ('ip', 'IP Camera'),
        ('rtsp', 'RTSP Stream'),
        ('usb', 'USB Camera'),
        ('file', 'Video File'),
    )

    camera_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=256)
    camera_type = models.CharField(max_length=20, choices=CAMERA_TYPE_CHOICES, default='rtsp')
    rtsp_link = models.CharField(max_length=500)
    description = models.TextField(blank=True, null=True, default="")
    
    # Status
    is_healthy = models.BooleanField(default=True)
    is_active = models.BooleanField(default=True)
    is_streaming = models.BooleanField(default=True, help_text="Whether streaming is enabled (auto-enabled on creation)")
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


@receiver(models.signals.post_delete, sender=Camera)
def post_delete_camera(sender, instance, *args, **kwargs):
    """Clean up MinIO files when camera is deleted"""
    try:
        from backend.storage import minio_storage
        minio_storage.delete_camera_files(str(instance.camera_id))
    except Exception as e:
        logger.warning(f"Failed to delete MinIO files for camera {instance.camera_id}: {e}")
