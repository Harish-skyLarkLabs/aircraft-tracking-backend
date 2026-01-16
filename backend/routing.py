"""
WebSocket URL routing for Aircraft Tracking backend

Routes:
- /ws/video-feed/<camera_id>/ - Full video feed (legacy mode) or detections only (LiveKit mode)
- /ws/detections/<camera_id>/ - Detection data only (for LiveKit WebRTC overlay)
- /ws/alerts/ - Real-time alerts
"""
from django.urls import re_path
from aircraft_detection.consumers import VideoFeedConsumer, DetectionsConsumer, AlertsConsumer

websocket_urlpatterns = [
    # Video feed - sends frames (legacy) or detections only (LiveKit mode based on USE_LIVEKIT env)
    re_path(r'ws/video-feed/(?P<camera_id>[0-9a-fA-F-]+)/?$', VideoFeedConsumer.as_asgi()),
    
    # Detections only - lightweight endpoint for LiveKit mode (always sends only detection data)
    re_path(r'ws/detections/(?P<camera_id>[0-9a-fA-F-]+)/?$', DetectionsConsumer.as_asgi()),
    
    # Alerts - real-time alert notifications
    re_path(r'ws/alerts/?$', AlertsConsumer.as_asgi()),
]

