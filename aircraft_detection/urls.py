"""
URL configuration for aircraft_detection app
"""
from rest_framework.routers import DefaultRouter
from .views import AircraftDetectionViewSet, DetectionSessionViewSet

router = DefaultRouter()
router.register(r'detections', AircraftDetectionViewSet, basename='detection')
router.register(r'sessions', DetectionSessionViewSet, basename='session')

urlpatterns = router.urls



