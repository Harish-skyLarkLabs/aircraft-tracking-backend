"""
Views for aircraft_detection app
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta

from .models import AircraftDetection, DetectionSession
from .serializers import (
    AircraftDetectionSerializer,
    AircraftDetectionListSerializer,
    AircraftDetectionCreateSerializer,
    DetectionSessionSerializer,
    DetectionStatsSerializer,
)


class AircraftDetectionViewSet(viewsets.ModelViewSet):
    """ViewSet for AircraftDetection model"""
    queryset = AircraftDetection.objects.all()
    permission_classes = [IsAuthenticated]

    def get_serializer_class(self):
        if self.action == 'create':
            return AircraftDetectionCreateSerializer
        elif self.action == 'list':
            return AircraftDetectionListSerializer
        return AircraftDetectionSerializer

    def get_queryset(self):
        queryset = AircraftDetection.objects.all()

        # Filter by camera
        camera_id = self.request.query_params.get('camera_id', None)
        if camera_id:
            queryset = queryset.filter(camera_id=camera_id)

        # Filter by detection type
        detection_type = self.request.query_params.get('type', None)
        if detection_type:
            queryset = queryset.filter(detection_type=detection_type)

        # Filter by action
        action_filter = self.request.query_params.get('action', None)
        if action_filter:
            queryset = queryset.filter(action=action_filter)

        # Filter by status
        status_filter = self.request.query_params.get('status', None)
        if status_filter:
            queryset = queryset.filter(status=status_filter)

        # Filter by severity
        severity = self.request.query_params.get('severity', None)
        if severity:
            queryset = queryset.filter(severity=severity)

        # Filter by date range
        start_date = self.request.query_params.get('start_date', None)
        end_date = self.request.query_params.get('end_date', None)
        if start_date:
            queryset = queryset.filter(detection_time__gte=start_date)
        if end_date:
            queryset = queryset.filter(detection_time__lte=end_date)

        return queryset

    @action(detail=False, methods=['get'], url_path='stats')
    def get_stats(self, request):
        """Get detection statistics"""
        # Time range filter
        hours = int(request.query_params.get('hours', 24))
        start_time = timezone.now() - timedelta(hours=hours)
        
        queryset = AircraftDetection.objects.filter(detection_time__gte=start_time)
        
        # Camera filter
        camera_id = request.query_params.get('camera_id', None)
        if camera_id:
            queryset = queryset.filter(camera_id=camera_id)

        stats = {
            'total_detections': queryset.count(),
            'total_landings': queryset.filter(action='landing').count(),
            'total_takeoffs': queryset.filter(action='takeoff').count(),
            'total_alerts': queryset.exclude(severity='low').count(),
            'pending_alerts': queryset.filter(status='pending').exclude(severity='low').count(),
            'by_type': dict(queryset.values('detection_type').annotate(count=Count('detection_id')).values_list('detection_type', 'count')),
            'by_action': dict(queryset.values('action').annotate(count=Count('detection_id')).values_list('action', 'count')),
            'by_severity': dict(queryset.values('severity').annotate(count=Count('detection_id')).values_list('severity', 'count')),
        }

        serializer = DetectionStatsSerializer(stats)
        return Response(serializer.data)

    @action(detail=True, methods=['post'], url_path='resolve')
    def resolve_detection(self, request, pk=None):
        """Mark a detection as resolved"""
        detection = self.get_object()
        detection.status = 'resolved'
        detection.save(update_fields=['status', 'updated_at'])
        return Response({'status': 'success', 'message': 'Detection marked as resolved'})

    @action(detail=True, methods=['post'], url_path='ignore')
    def ignore_detection(self, request, pk=None):
        """Mark a detection as ignored"""
        detection = self.get_object()
        detection.status = 'ignored'
        detection.save(update_fields=['status', 'updated_at'])
        return Response({'status': 'success', 'message': 'Detection marked as ignored'})

    @action(detail=False, methods=['get'], url_path='recent')
    def get_recent(self, request):
        """Get recent detections"""
        limit = int(request.query_params.get('limit', 10))
        camera_id = request.query_params.get('camera_id', None)
        
        queryset = AircraftDetection.objects.all()
        if camera_id:
            queryset = queryset.filter(camera_id=camera_id)
        
        queryset = queryset.order_by('-detection_time')[:limit]
        serializer = AircraftDetectionListSerializer(queryset, many=True, context={'request': request})
        return Response(serializer.data)

    @action(detail=False, methods=['get'], url_path='alerts')
    def get_alerts(self, request):
        """Get pending alerts (non-low severity detections)"""
        camera_id = request.query_params.get('camera_id', None)
        
        queryset = AircraftDetection.objects.filter(
            status='pending'
        ).exclude(severity='low')
        
        if camera_id:
            queryset = queryset.filter(camera_id=camera_id)
        
        queryset = queryset.order_by('-detection_time')
        serializer = AircraftDetectionListSerializer(queryset, many=True, context={'request': request})
        return Response(serializer.data)


class DetectionSessionViewSet(viewsets.ModelViewSet):
    """ViewSet for DetectionSession model"""
    queryset = DetectionSession.objects.all()
    serializer_class = DetectionSessionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        queryset = DetectionSession.objects.all()
        
        # Filter by camera
        camera_id = self.request.query_params.get('camera_id', None)
        if camera_id:
            queryset = queryset.filter(camera_id=camera_id)
        
        # Filter by active status
        is_active = self.request.query_params.get('is_active', None)
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active.lower() == 'true')
        
        return queryset

    @action(detail=True, methods=['post'], url_path='end')
    def end_session(self, request, pk=None):
        """End a detection session"""
        session = self.get_object()
        session.is_active = False
        session.ended_at = timezone.now()
        session.save(update_fields=['is_active', 'ended_at'])
        return Response({'status': 'success', 'message': 'Session ended'})



