"""
Models for Aircraft Detection
"""
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.dispatch import receiver
import uuid
import logging

logger = logging.getLogger(__name__)


class AircraftDetection(models.Model):
    """Model to store aircraft detection records"""

    # Detection type choices
    DETECTION_TYPE_CHOICES = (
        ('aircraft', 'Aircraft'),
        ('helicopter', 'Helicopter'),
        ('drone', 'Drone'),
        ('unknown', 'Unknown'),
    )

    # Action/Event type choices
    ACTION_TYPE_CHOICES = (
        ('landing', 'Landing'),
        ('takeoff', 'Takeoff'),
        ('taxiing', 'Taxiing'),
        ('parked', 'Parked'),
        ('flying', 'Flying'),
        ('hovering', 'Hovering'),
        ('unknown', 'Unknown'),
    )

    # Severity choices
    SEVERITY_CHOICES = (
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    )

    # Status choices
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('resolved', 'Resolved'),
        ('ignored', 'Ignored'),
    )

    detection_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    track_id = models.IntegerField(default=0, help_text="Tracking ID from the tracker")
    
    # Detection info
    detection_type = models.CharField(max_length=20, choices=DETECTION_TYPE_CHOICES, default='aircraft')
    action = models.CharField(max_length=20, choices=ACTION_TYPE_CHOICES, default='unknown')
    confidence = models.FloatField(default=0.0)
    
    # Aircraft details (optional)
    aircraft_type = models.CharField(max_length=100, blank=True, null=True)
    aircraft_registration = models.CharField(max_length=50, blank=True, null=True)
    flight_number = models.CharField(max_length=20, blank=True, null=True)
    airline = models.CharField(max_length=100, blank=True, null=True)
    runway = models.CharField(max_length=20, blank=True, null=True)
    
    # Flight data
    speed = models.FloatField(blank=True, null=True, help_text="Speed in knots")
    altitude = models.FloatField(blank=True, null=True, help_text="Altitude in feet")
    heading = models.FloatField(blank=True, null=True, help_text="Heading in degrees")
    
    # Alert info
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default='low')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    is_read = models.BooleanField(default=False, help_text="Whether the alert has been viewed")
    title = models.CharField(max_length=255, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    
    # Camera info
    camera_id = models.CharField(max_length=50)
    camera_name = models.CharField(max_length=256, blank=True, null=True)
    
    # MinIO Storage Paths (stores object path, not full URL)
    # Structure: cameras/{camera_id}/alerts/{detection_id}.jpg
    # Structure: cameras/{camera_id}/videos/{detection_id}.mp4
    image_path = models.CharField(max_length=500, blank=True, null=True)
    video_path = models.CharField(max_length=500, blank=True, null=True)
    
    # Bounding box
    bbox_x1 = models.IntegerField(default=0)
    bbox_y1 = models.IntegerField(default=0)
    bbox_x2 = models.IntegerField(default=0)
    bbox_y2 = models.IntegerField(default=0)
    
    # Metadata
    metadata = models.JSONField(default=dict, blank=True, null=True)
    
    # Timestamps
    detection_time = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-detection_time']
        indexes = [
            models.Index(fields=['detection_type']),
            models.Index(fields=['action']),
            models.Index(fields=['detection_time']),
            models.Index(fields=['camera_id']),
            models.Index(fields=['status']),
            models.Index(fields=['severity']),
        ]

    def __str__(self):
        return f"{self.detection_type} - {self.action} ({self.detection_time})"

    def save(self, *args, **kwargs):
        """Override save to auto-generate title and description"""
        if not self.title:
            action_display = dict(self.ACTION_TYPE_CHOICES).get(self.action, 'Unknown')
            type_display = dict(self.DETECTION_TYPE_CHOICES).get(self.detection_type, 'Unknown')
            self.title = f"{type_display} {action_display} Detected"
        
        if not self.description:
            self.description = f"{self.detection_type.title()} detected performing {self.action} action"
            if self.flight_number:
                self.description += f" - Flight {self.flight_number}"
            if self.aircraft_type:
                self.description += f" ({self.aircraft_type})"
        
        super().save(*args, **kwargs)

    @property
    def bbox(self):
        """Return bounding box as tuple"""
        return (self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2)

    @bbox.setter
    def bbox(self, value):
        """Set bounding box from tuple"""
        if value and len(value) == 4:
            self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2 = value
    
    @property
    def image_url(self) -> str | None:
        """Get the full MinIO URL for the detection image."""
        if self.image_path:
            # Images use public URL (small files, cacheable)
            return f"{settings.MINIO_PUBLIC_URL}/{settings.MINIO_BUCKET_NAME}/{self.image_path}"
        return None
    
    @property
    def video_url(self) -> str | None:
        """Get a presigned URL for the detection video (required for browser playback)."""
        if self.video_path:
            try:
                from backend.storage import minio_storage
                from datetime import timedelta
                # Use presigned URL for videos (larger files, need auth)
                presigned = minio_storage.get_presigned_url(
                    self.video_path, 
                    expires=timedelta(hours=24)
                )
                if presigned:
                    return presigned
                # Fallback to public URL
                return f"{settings.MINIO_PUBLIC_URL}/{settings.MINIO_BUCKET_NAME}/{self.video_path}"
            except Exception:
                return f"{settings.MINIO_PUBLIC_URL}/{settings.MINIO_BUCKET_NAME}/{self.video_path}"
        return None


@receiver(models.signals.post_delete, sender=AircraftDetection)
def post_delete_detection(sender, instance, *args, **kwargs):
    """Clean up MinIO files when detection is deleted"""
    try:
        from backend.storage import minio_storage
        
        if instance.image_path:
            minio_storage.delete_file(instance.image_path)
        if instance.video_path:
            minio_storage.delete_file(instance.video_path)
    except Exception as e:
        logger.warning(f"Failed to delete MinIO files for detection {instance.detection_id}: {e}")


class DetectionSession(models.Model):
    """Model to track detection sessions for cameras"""
    
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    camera_id = models.CharField(max_length=50)
    camera_name = models.CharField(max_length=256, blank=True, null=True)
    
    # Session info
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    
    # Statistics
    total_detections = models.IntegerField(default=0)
    total_landings = models.IntegerField(default=0)
    total_takeoffs = models.IntegerField(default=0)
    total_alerts = models.IntegerField(default=0)
    
    # Metadata
    metadata = models.JSONField(default=dict, blank=True, null=True)

    class Meta:
        ordering = ['-started_at']

    def __str__(self):
        return f"Session {self.session_id} - Camera {self.camera_name or self.camera_id}"


