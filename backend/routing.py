"""
WebSocket URL routing for Aircraft Tracking backend
"""
from django.urls import re_path
from aircraft_detection.consumers import VideoFeedConsumer, AlertsConsumer

websocket_urlpatterns = [
    re_path(r'ws/video-feed/(?P<camera_id>[0-9a-fA-F-]+)/?$', VideoFeedConsumer.as_asgi()),
    re_path(r'ws/alerts/?$', AlertsConsumer.as_asgi()),
]

