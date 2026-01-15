"""
MinIO Object Storage Utility
Handles all object storage operations for the Aircraft Tracking system.

Bucket Structure:
- {bucket}/cameras/{camera_id}/thumbnail.jpg
- {bucket}/cameras/{camera_id}/alerts/{alert_id}.jpg
- {bucket}/cameras/{camera_id}/videos/{video_id}.mp4
"""

import io
import logging
from typing import Optional, BinaryIO, Union
from datetime import timedelta

from minio import Minio
from minio.error import S3Error
from django.conf import settings

logger = logging.getLogger(__name__)


class MinIOStorage:
    """MinIO storage client singleton"""
    
    _instance: Optional['MinIOStorage'] = None
    _client: Optional[Minio] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._client = Minio(
                endpoint=settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
            logger.info(f"MinIO client initialized for endpoint: {settings.MINIO_ENDPOINT}")
    
    @property
    def client(self) -> Minio:
        return self._client
    
    @property
    def bucket_name(self) -> str:
        return settings.MINIO_BUCKET_NAME
    
    def initialize_bucket(self) -> bool:
        """
        Initialize the bucket if it doesn't exist.
        Sets public read policy for the bucket.
        Returns True if successful.
        """
        try:
            if not self._client.bucket_exists(self.bucket_name):
                self._client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
                
                # Set public read policy
                policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": "*"},
                            "Action": ["s3:GetObject"],
                            "Resource": [f"arn:aws:s3:::{self.bucket_name}/*"]
                        }
                    ]
                }
                import json
                self._client.set_bucket_policy(self.bucket_name, json.dumps(policy))
                logger.info(f"Set public read policy for bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket already exists: {self.bucket_name}")
            return True
        except S3Error as e:
            logger.error(f"Failed to initialize bucket: {e}")
            return False
    
    def _get_object_path(self, camera_id: str, folder: str, filename: str) -> str:
        """
        Generate the object path for storage.
        Structure: cameras/{camera_id}/{folder}/{filename}
        """
        return f"cameras/{camera_id}/{folder}/{filename}"
    
    def upload_file(
        self,
        camera_id: str,
        folder: str,
        filename: str,
        data: Union[bytes, BinaryIO],
        content_type: str = "application/octet-stream"
    ) -> Optional[str]:
        """
        Upload a file to MinIO.
        
        Args:
            camera_id: The camera UUID
            folder: The folder name (thumbnail, alerts, videos)
            filename: The filename
            data: File data as bytes or file-like object
            content_type: MIME type of the file
            
        Returns:
            The object path if successful, None otherwise
        """
        object_path = self._get_object_path(camera_id, folder, filename)
        
        try:
            if isinstance(data, bytes):
                data_stream = io.BytesIO(data)
                data_length = len(data)
            else:
                # Assume it's a file-like object
                data.seek(0, 2)  # Seek to end
                data_length = data.tell()
                data.seek(0)  # Seek back to start
                data_stream = data
            
            self._client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_path,
                data=data_stream,
                length=data_length,
                content_type=content_type,
            )
            logger.info(f"Uploaded file to: {object_path}")
            return object_path
        except S3Error as e:
            logger.error(f"Failed to upload file {object_path}: {e}")
            return None
    
    def upload_thumbnail(self, camera_id: str, data: bytes) -> Optional[str]:
        """Upload camera thumbnail image."""
        return self.upload_file(
            camera_id=str(camera_id),
            folder="thumbnail",
            filename="thumbnail.jpg",
            data=data,
            content_type="image/jpeg"
        )
    
    def upload_alert_image(
        self,
        camera_id: str,
        alert_id: str,
        image_bytes: bytes,
        suffix: str = "",
    ) -> Optional[str]:
        """
        Upload alert detection snapshot image.
        
        Args:
            camera_id: Camera UUID
            alert_id: Alert/Detection UUID
            image_bytes: JPEG image bytes
            suffix: Optional suffix for filename (e.g., "_initial", "_final")
            
        Returns:
            Object path if successful
        """
        return self.upload_file(
            camera_id=str(camera_id),
            folder="alerts",
            filename=f"{alert_id}{suffix}.jpg",
            data=image_bytes,
            content_type="image/jpeg"
        )
    
    def upload_alert_video(
        self,
        camera_id: str,
        alert_id: str,
        video_bytes: Union[bytes, BinaryIO]
    ) -> Optional[str]:
        """
        Upload alert video recording.
        
        Args:
            camera_id: Camera UUID
            alert_id: Alert/Detection UUID
            video_bytes: MP4 video bytes or file object
            
        Returns:
            Object path if successful
        """
        return self.upload_file(
            camera_id=str(camera_id),
            folder="videos",
            filename=f"{alert_id}.mp4",
            data=video_bytes,
            content_type="video/mp4"
        )
    
    def upload_video(self, camera_id: str, video_id: str, data: Union[bytes, BinaryIO]) -> Optional[str]:
        """Upload video recording (generic)."""
        return self.upload_file(
            camera_id=str(camera_id),
            folder="videos",
            filename=f"{video_id}.mp4",
            data=data,
            content_type="video/mp4"
        )
    
    def delete_file(self, object_path: str) -> bool:
        """
        Delete a file from MinIO.
        
        Args:
            object_path: The full object path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._client.remove_object(self.bucket_name, object_path)
            logger.info(f"Deleted file: {object_path}")
            return True
        except S3Error as e:
            logger.error(f"Failed to delete file {object_path}: {e}")
            return False
    
    def delete_camera_files(self, camera_id: str) -> bool:
        """
        Delete all files for a camera.
        
        Args:
            camera_id: The camera UUID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            prefix = f"cameras/{camera_id}/"
            objects = self._client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            
            for obj in objects:
                self._client.remove_object(self.bucket_name, obj.object_name)
                logger.info(f"Deleted: {obj.object_name}")
            
            logger.info(f"Deleted all files for camera: {camera_id}")
            return True
        except S3Error as e:
            logger.error(f"Failed to delete camera files for {camera_id}: {e}")
            return False
    
    def get_public_url(self, object_path: str) -> str:
        """
        Get the public URL for an object.
        
        Args:
            object_path: The object path in the bucket
            
        Returns:
            The full public URL
        """
        return f"{settings.MINIO_PUBLIC_URL}/{self.bucket_name}/{object_path}"
    
    def get_presigned_url(self, object_path: str, expires: timedelta = timedelta(hours=1)) -> Optional[str]:
        """
        Get a presigned URL for temporary access.
        
        Args:
            object_path: The object path in the bucket
            expires: URL expiration time
            
        Returns:
            The presigned URL if successful, None otherwise
        """
        try:
            url = self._client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_path,
                expires=expires,
            )
            return url
        except S3Error as e:
            logger.error(f"Failed to generate presigned URL for {object_path}: {e}")
            return None
    
    def file_exists(self, object_path: str) -> bool:
        """Check if a file exists in the bucket."""
        try:
            self._client.stat_object(self.bucket_name, object_path)
            return True
        except S3Error:
            return False


# Singleton instance
minio_storage = MinIOStorage()


def initialize_minio():
    """Initialize MinIO bucket on application startup."""
    try:
        success = minio_storage.initialize_bucket()
        if success:
            logger.info("MinIO storage initialized successfully")
        else:
            logger.error("Failed to initialize MinIO storage")
        return success
    except Exception as e:
        logger.error(f"Error initializing MinIO: {e}")
        return False

