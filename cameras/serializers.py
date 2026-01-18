"""
Serializers for cameras app with PTZ support
"""
from rest_framework import serializers
from .models import Camera


class CameraSerializer(serializers.ModelSerializer):
    """Serializer for Camera model with PTZ support"""
    
    # Computed fields for URLs
    thumbnail_url = serializers.SerializerMethodField()
    stream_url = serializers.SerializerMethodField()
    ptz_config = serializers.SerializerMethodField()
    tracking_config = serializers.SerializerMethodField()

    class Meta:
        model = Camera
        fields = [
            'camera_id', 'name', 'camera_type', 'rtsp_link', 'description',
            'is_healthy', 'is_active', 'is_streaming', 'ai_enabled',
            'thumbnail_path', 'thumbnail_url', 'stream_url',
            'roi_points', 'config', 'resolution_width', 'resolution_height', 'fps',
            # PTZ fields
            'ptz_enabled', 'ptz_ip', 'ptz_username', 'ptz_channel', 'ptz_preset_number',
            'ptz_tracking_enabled', 'ptz_zoom_enabled', 'ptz_zoom_in_enabled', 'ptz_zoom_out_enabled',
            'ptz_zoom_config',
            # Tracking fields
            'tracking_lock_only_mode', 'tracking_min_consecutive_detections',
            'tracking_enable_size_filter', 'tracking_min_aircraft_width',
            'tracking_enable_edge_filtering', 'tracking_edge_margin_percent',
            # Computed fields
            'ptz_config', 'tracking_config',
            # Timestamps
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'camera_id', 'created_at', 'updated_at', 'is_healthy',
            'thumbnail_path', 'thumbnail_url', 'stream_url',
            'ptz_config', 'tracking_config'
        ]
        extra_kwargs = {
            'ptz_password': {'write_only': True}  # Don't expose password in responses
        }
    
    def get_thumbnail_url(self, obj) -> str | None:
        """Get the full MinIO URL for the thumbnail."""
        return obj.thumbnail_url
    
    def get_stream_url(self, obj) -> str:
        """Get the WebSocket stream URL for this camera."""
        return obj.stream_url
    
    def get_ptz_config(self, obj) -> dict:
        """Get PTZ configuration (without password)."""
        config = obj.ptz_config
        # Remove password from response
        config.pop('password', None)
        return config
    
    def get_tracking_config(self, obj) -> dict:
        """Get tracking configuration."""
        return obj.tracking_config


class CameraCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating cameras with PTZ support"""

    class Meta:
        model = Camera
        fields = [
            'name', 'camera_type', 'rtsp_link', 'description',
            'roi_points', 'config', 'resolution_width', 'resolution_height', 'fps',
            # PTZ fields
            'ptz_enabled', 'ptz_ip', 'ptz_username', 'ptz_password',
            'ptz_channel', 'ptz_preset_number',
            'ptz_tracking_enabled', 'ptz_zoom_enabled',
            'ptz_zoom_in_enabled', 'ptz_zoom_out_enabled',
            'ptz_zoom_config',
            # Tracking fields
            'tracking_lock_only_mode', 'tracking_min_consecutive_detections',
            'tracking_enable_size_filter', 'tracking_min_aircraft_width',
            'tracking_enable_edge_filtering', 'tracking_edge_margin_percent',
        ]

    def validate_rtsp_link(self, value):
        if not value:
            raise serializers.ValidationError("RTSP link is required.")
        return value
    
    def validate(self, data):
        """Validate PTZ configuration if enabled."""
        if data.get('ptz_enabled'):
            # PTZ requires either ptz_ip or extractable IP from rtsp_link
            if not data.get('ptz_ip'):
                # Try to extract from RTSP link
                rtsp_link = data.get('rtsp_link', '')
                import re
                match = re.search(r'@([\d.]+)', rtsp_link)
                if not match:
                    raise serializers.ValidationError({
                        'ptz_ip': 'PTZ IP is required when PTZ is enabled and cannot be extracted from RTSP link.'
                    })
            
            # PTZ requires username and password
            if not data.get('ptz_username'):
                raise serializers.ValidationError({
                    'ptz_username': 'PTZ username is required when PTZ is enabled.'
                })
            if not data.get('ptz_password'):
                raise serializers.ValidationError({
                    'ptz_password': 'PTZ password is required when PTZ is enabled.'
                })
        
        return data


class CameraUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating cameras"""

    class Meta:
        model = Camera
        fields = [
            'name', 'camera_type', 'rtsp_link', 'description',
            'is_active', 'is_streaming', 'ai_enabled',
            'roi_points', 'config', 'resolution_width', 'resolution_height', 'fps',
            # PTZ fields
            'ptz_enabled', 'ptz_ip', 'ptz_username', 'ptz_password',
            'ptz_channel', 'ptz_preset_number',
            'ptz_tracking_enabled', 'ptz_zoom_enabled',
            'ptz_zoom_in_enabled', 'ptz_zoom_out_enabled',
            'ptz_zoom_config',
            # Tracking fields
            'tracking_lock_only_mode', 'tracking_min_consecutive_detections',
            'tracking_enable_size_filter', 'tracking_min_aircraft_width',
            'tracking_enable_edge_filtering', 'tracking_edge_margin_percent',
        ]
        extra_kwargs = {
            'ptz_password': {'write_only': True}
        }


class PTZControlSerializer(serializers.Serializer):
    """Serializer for PTZ control commands"""
    action = serializers.ChoiceField(choices=[
        ('go_to_preset', 'Go to Preset'),
        ('emergency_stop', 'Emergency Stop'),
        ('clear_lock', 'Clear Tracking Lock'),
        ('enable_tracking', 'Enable PTZ Tracking'),
        ('disable_tracking', 'Disable PTZ Tracking'),
        ('enable_zoom', 'Enable Zoom Control'),
        ('disable_zoom', 'Disable Zoom Control'),
    ])
    preset_number = serializers.IntegerField(required=False, min_value=1, max_value=255)
    
    def validate(self, data):
        action = data.get('action')
        if action == 'go_to_preset' and 'preset_number' not in data:
            # Use default preset if not specified
            pass
        return data
