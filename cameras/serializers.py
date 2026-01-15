"""
Serializers for cameras app
"""
from rest_framework import serializers
from .models import Camera


class CameraSerializer(serializers.ModelSerializer):
    """Serializer for Camera model"""
    
    # Computed fields for URLs
    thumbnail_url = serializers.SerializerMethodField()
    stream_url = serializers.SerializerMethodField()

    class Meta:
        model = Camera
        fields = [
            'camera_id', 'name', 'camera_type', 'rtsp_link', 'description',
            'is_healthy', 'is_active', 'is_streaming', 'ai_enabled',
            'thumbnail_path', 'thumbnail_url', 'stream_url',
            'roi_points', 'config', 'resolution_width', 'resolution_height',
            'fps', 'created_at', 'updated_at'
        ]
        read_only_fields = ['camera_id', 'created_at', 'updated_at', 'is_healthy', 'thumbnail_path', 'thumbnail_url', 'stream_url']
    
    def get_thumbnail_url(self, obj) -> str | None:
        """Get the full MinIO URL for the thumbnail."""
        return obj.thumbnail_url
    
    def get_stream_url(self, obj) -> str:
        """Get the WebSocket stream URL for this camera."""
        return obj.stream_url


class CameraCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating cameras"""

    class Meta:
        model = Camera
        fields = [
            'name', 'camera_type', 'rtsp_link', 'description',
            'roi_points', 'config', 'resolution_width', 'resolution_height', 'fps'
        ]

    def validate_rtsp_link(self, value):
        if not value:
            raise serializers.ValidationError("RTSP link is required.")
        return value
