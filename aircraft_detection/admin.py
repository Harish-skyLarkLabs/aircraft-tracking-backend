"""
Admin configuration for aircraft_detection app
"""
from django.contrib import admin
from .models import AircraftDetection, DetectionSession


@admin.register(AircraftDetection)
class AircraftDetectionAdmin(admin.ModelAdmin):
    list_display = [
        'detection_id', 'detection_type', 'action', 'confidence',
        'severity', 'status', 'camera_name', 'detection_time'
    ]
    list_filter = ['detection_type', 'action', 'severity', 'status', 'camera_id']
    search_fields = ['title', 'description', 'flight_number', 'aircraft_registration']
    readonly_fields = ['detection_id', 'created_at', 'updated_at']
    date_hierarchy = 'detection_time'


@admin.register(DetectionSession)
class DetectionSessionAdmin(admin.ModelAdmin):
    list_display = [
        'session_id', 'camera_name', 'is_active',
        'total_detections', 'started_at', 'ended_at'
    ]
    list_filter = ['is_active', 'camera_id']
    readonly_fields = ['session_id', 'started_at']



