"""
Views for cameras app
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.http import JsonResponse
import base64
import cv2
import logging

from .models import Camera
from .serializers import CameraSerializer, CameraCreateSerializer
from .utils import check_rtsp, get_stream_info

logger = logging.getLogger(__name__)


class CameraViewSet(viewsets.ModelViewSet):
    """ViewSet for Camera model"""
    queryset = Camera.objects.all()
    permission_classes = [IsAuthenticated]

    def get_serializer_class(self):
        if self.action == 'create':
            return CameraCreateSerializer
        return CameraSerializer

    def get_queryset(self):
        queryset = Camera.objects.all()
        
        # Filter by status
        is_active = self.request.query_params.get('is_active', None)
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active.lower() == 'true')
        
        ai_enabled = self.request.query_params.get('ai_enabled', None)
        if ai_enabled is not None:
            queryset = queryset.filter(ai_enabled=ai_enabled.lower() == 'true')
        
        return queryset

    def _upload_thumbnail_to_minio(self, camera, thumbnail_frame) -> bool:
        """Upload thumbnail to MinIO and update camera model."""
        try:
            from backend.storage import minio_storage
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", thumbnail_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_bytes = buffer.tobytes()
            
            # Upload to MinIO
            object_path = minio_storage.upload_thumbnail(str(camera.camera_id), image_bytes)
            
            if object_path:
                camera.thumbnail_path = object_path
                camera.save(update_fields=['thumbnail_path'])
                logger.info(f"Uploaded thumbnail for camera {camera.camera_id} to {object_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to upload thumbnail for camera {camera.camera_id}: {e}")
            return False

    def create(self, request, *args, **kwargs):
        """Override create to return full camera data after creation."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        camera = serializer.save()

        # Try to get stream info, capture thumbnail, and upload to MinIO
        if camera.rtsp_link:
            try:
                thumbnail = check_rtsp(camera.rtsp_link)
                if thumbnail is not None:
                    info = get_stream_info(camera.rtsp_link)
                    if info:
                        # Only update if not already set from request
                        if not camera.resolution_width:
                            camera.resolution_width = info.get('width', 1920)
                        if not camera.resolution_height:
                            camera.resolution_height = info.get('height', 1080)
                        if not camera.fps:
                            camera.fps = info.get('fps', 30)
                    camera.is_healthy = True
                    camera.save()
                    
                    # Upload thumbnail to MinIO
                    self._upload_thumbnail_to_minio(camera, thumbnail)
                else:
                    camera.is_healthy = False
                    camera.save()
            except Exception as e:
                logger.error(f"Error during camera creation: {e}")
                camera.is_healthy = False
                camera.save()

        # Return full camera data using CameraSerializer
        output_serializer = CameraSerializer(camera)
        headers = self.get_success_headers(output_serializer.data)
        return Response(output_serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    @action(detail=False, methods=["post"], url_path="check_rtsp")
    def check_rtsp_link(self, request):
        """Check if an RTSP link is valid and return a thumbnail."""
        rtsp_link = request.data.get("rtsp_link")
        
        if not rtsp_link:
            return JsonResponse({"message": "failed", "error": "RTSP link is required"})
        
        thumbnail = check_rtsp(rtsp_link)
        
        if thumbnail is None:
            return JsonResponse({
                "message": "failed",
                "image": None,
                "max_resolution": None
            })
        
        # Encode thumbnail as base64
        _, buffer = cv2.imencode(".jpg", thumbnail)
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        
        # Get stream info
        info = get_stream_info(rtsp_link)
        
        return JsonResponse({
            "message": "success",
            "image": image_base64,
            "width": info.get('width') if info else thumbnail.shape[1],
            "height": info.get('height') if info else thumbnail.shape[0],
            "fps": info.get('fps') if info else None,
        })

    @action(detail=True, methods=["post"], url_path="toggle_ai")
    def toggle_ai(self, request, pk=None):
        """Toggle AI processing for a specific camera."""
        camera = self.get_object()
        
        try:
            from aircraft_detection.utils.camera_manager import camera_manager
            
            if camera.ai_enabled:
                # Disable AI - stop processing
                success = camera_manager.stop_camera_processing(camera.camera_id)
                if success:
                    camera.ai_enabled = False
                    camera.save(update_fields=["ai_enabled"])
                    return Response(
                        {"status": "success", "message": f"AI disabled for camera {camera.camera_id}", "ai_enabled": False},
                        status=status.HTTP_200_OK
                    )
            else:
                # Enable AI - start processing
                success = camera_manager.start_camera_processing(
                    camera.camera_id,
                    camera.rtsp_link,
                    camera_name=camera.name,
                    roi_points=camera.roi_points
                )
                if success:
                    camera.ai_enabled = True
                    camera.save(update_fields=["ai_enabled"])
                    return Response(
                        {"status": "success", "message": f"AI enabled for camera {camera.camera_id}", "ai_enabled": True},
                        status=status.HTTP_200_OK
                    )
            
            return Response(
                {"status": "error", "message": f"Failed to toggle AI for camera {camera.camera_id}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except ImportError:
            return Response(
                {"status": "error", "message": "Camera manager not available"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=["post"], url_path="enable_ai")
    def enable_ai(self, request, pk=None):
        """Enable AI processing for a specific camera."""
        camera = self.get_object()
        
        if camera.ai_enabled:
            return Response(
                {"status": "success", "message": "AI already enabled", "ai_enabled": True},
                status=status.HTTP_200_OK
            )
        
        try:
            from aircraft_detection.utils.camera_manager import camera_manager
            
            success = camera_manager.start_camera_processing(
                camera.camera_id,
                camera.rtsp_link,
                camera_name=camera.name,
                roi_points=camera.roi_points
            )
            
            if success:
                camera.ai_enabled = True
                camera.save(update_fields=["ai_enabled"])
                return Response(
                    {"status": "success", "message": f"AI enabled for camera {camera.camera_id}", "ai_enabled": True},
                    status=status.HTTP_200_OK
                )
            else:
                return Response(
                    {"status": "error", "message": f"Failed to enable AI for camera {camera.camera_id}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        except ImportError:
            return Response(
                {"status": "error", "message": "Camera manager not available"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=["post"], url_path="disable_ai")
    def disable_ai(self, request, pk=None):
        """Disable AI processing for a specific camera."""
        camera = self.get_object()
        
        if not camera.ai_enabled:
            return Response(
                {"status": "success", "message": "AI already disabled", "ai_enabled": False},
                status=status.HTTP_200_OK
            )
        
        try:
            from aircraft_detection.utils.camera_manager import camera_manager
            
            success = camera_manager.stop_camera_processing(camera.camera_id)
            
            if success:
                camera.ai_enabled = False
                camera.save(update_fields=["ai_enabled"])
                return Response(
                    {"status": "success", "message": f"AI disabled for camera {camera.camera_id}", "ai_enabled": False},
                    status=status.HTTP_200_OK
                )
            else:
                return Response(
                    {"status": "error", "message": f"Failed to disable AI for camera {camera.camera_id}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        except ImportError:
            return Response(
                {"status": "error", "message": "Camera manager not available"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def destroy(self, request, *args, **kwargs):
        """Override destroy to stop processing and cleanup all data before deletion."""
        instance = self.get_object()
        camera_id = str(instance.camera_id)

        # 1. Stop AI processing if enabled
        if instance.ai_enabled:
            try:
                from aircraft_detection.utils.camera_manager import camera_manager
                camera_manager.stop_camera_processing(instance.camera_id)
                logger.info(f"Stopped AI processing for camera {camera_id}")
            except ImportError:
                logger.warning("Camera manager not available during camera deletion")

        # 2. Delete all detection records from database
        try:
            from aircraft_detection.models import AircraftDetection
            detection_count = AircraftDetection.objects.filter(camera_id=camera_id).count()
            deleted_detections = AircraftDetection.objects.filter(camera_id=camera_id).delete()
            logger.info(f"Deleted {detection_count} detection records for camera {camera_id}")
        except Exception as e:
            logger.error(f"Error deleting detection records for camera {camera_id}: {e}")

        # 3. Delete all files from MinIO (thumbnails, alert images, videos)
        try:
            from backend.storage import minio_storage
            success = minio_storage.delete_camera_files(camera_id)
            if success:
                logger.info(f"Deleted all MinIO files for camera {camera_id}")
            else:
                logger.warning(f"Some files may not have been deleted from MinIO for camera {camera_id}")
        except Exception as e:
            logger.error(f"Error deleting MinIO files for camera {camera_id}: {e}")

        # 4. Delete the camera record
        logger.info(f"Deleting camera {camera_id} - {instance.name}")
        return super().destroy(request, *args, **kwargs)
