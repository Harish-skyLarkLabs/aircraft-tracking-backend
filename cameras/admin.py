"""
Admin configuration for cameras app
"""
from django.contrib import admin
from .models import Camera


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ['name', 'camera_type', 'is_healthy', 'is_streaming', 'ai_enabled', 'is_active', 'created_at']
    list_filter = ['camera_type', 'is_healthy', 'is_streaming', 'ai_enabled', 'is_active']
    search_fields = ['name', 'rtsp_link', 'description']
    readonly_fields = ['camera_id', 'created_at', 'updated_at']
