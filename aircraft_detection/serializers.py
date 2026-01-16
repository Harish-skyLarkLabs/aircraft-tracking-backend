"""
Serializers for aircraft_detection app
"""
from rest_framework import serializers
from .models import AircraftDetection, DetectionSession


class AircraftDetectionSerializer(serializers.ModelSerializer):
    """Serializer for AircraftDetection model"""
    bbox = serializers.SerializerMethodField()
    image_url = serializers.SerializerMethodField()
    video_url = serializers.SerializerMethodField()
    crop_video_url = serializers.SerializerMethodField()

    class Meta:
        model = AircraftDetection
        fields = [
            'detection_id', 'track_id', 'detection_type', 'action', 'confidence',
            'aircraft_type', 'aircraft_registration', 'flight_number', 'airline', 'runway',
            'speed', 'altitude', 'heading',
            'severity', 'status', 'is_read', 'title', 'description',
            'camera_id', 'camera_name',
            'image_path', 'image_url', 'video_path', 'video_url',
            'crop_video_path', 'crop_video_url',
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'bbox',
            'metadata', 'detection_time', 'created_at', 'updated_at'
        ]
        read_only_fields = ['detection_id', 'created_at', 'updated_at', 'image_url', 'video_url', 'crop_video_url']

    def get_bbox(self, obj):
        return [obj.bbox_x1, obj.bbox_y1, obj.bbox_x2, obj.bbox_y2]

    def get_image_url(self, obj) -> str | None:
        """Get the full MinIO URL for the detection image."""
        return obj.image_url

    def get_video_url(self, obj) -> str | None:
        """Get the full MinIO URL for the detection video."""
        return obj.video_url
    
    def get_crop_video_url(self, obj) -> str | None:
        """Get the full MinIO URL for the cropped aircraft video."""
        return obj.crop_video_url


class AircraftDetectionListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for list views"""
    image_url = serializers.SerializerMethodField()
    video_url = serializers.SerializerMethodField()
    crop_video_url = serializers.SerializerMethodField()

    class Meta:
        model = AircraftDetection
        fields = [
            'detection_id', 'detection_type', 'action', 'confidence',
            'title', 'severity', 'status', 'is_read', 'camera_id', 'camera_name',
            'detection_time', 'image_path', 'image_url', 'video_path', 'video_url',
            'crop_video_path', 'crop_video_url',
            'flight_number', 'aircraft_type'
        ]

    def get_image_url(self, obj) -> str | None:
        """Get the full MinIO URL for the detection image."""
        return obj.image_url

    def get_video_url(self, obj) -> str | None:
        """Get the full MinIO URL for the detection video."""
        return obj.video_url
    
    def get_crop_video_url(self, obj) -> str | None:
        """Get the full MinIO URL for the cropped aircraft video."""
        return obj.crop_video_url


class AircraftDetectionCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating detections"""
    bbox = serializers.ListField(
        child=serializers.IntegerField(),
        min_length=4,
        max_length=4,
        required=False
    )

    class Meta:
        model = AircraftDetection
        fields = [
            'detection_type', 'action', 'confidence',
            'aircraft_type', 'aircraft_registration', 'flight_number',
            'airline', 'runway', 'speed', 'altitude', 'heading',
            'severity', 'title', 'description',
            'camera_id', 'camera_name', 'bbox', 'metadata',
            'image_path', 'video_path'
        ]

    def create(self, validated_data):
        bbox = validated_data.pop('bbox', None)
        instance = AircraftDetection(**validated_data)
        if bbox:
            instance.bbox = bbox
        instance.save()
        return instance


class DetectionSessionSerializer(serializers.ModelSerializer):
    """Serializer for DetectionSession model"""
    duration = serializers.SerializerMethodField()

    class Meta:
        model = DetectionSession
        fields = '__all__'
        read_only_fields = ['session_id', 'started_at']

    def get_duration(self, obj):
        if obj.ended_at:
            delta = obj.ended_at - obj.started_at
            return delta.total_seconds()
        return None


class DetectionStatsSerializer(serializers.Serializer):
    """Serializer for detection statistics"""
    total_detections = serializers.IntegerField()
    total_landings = serializers.IntegerField()
    total_takeoffs = serializers.IntegerField()
    total_alerts = serializers.IntegerField()
    pending_alerts = serializers.IntegerField()
    by_type = serializers.DictField()
    by_action = serializers.DictField()
    by_severity = serializers.DictField()


