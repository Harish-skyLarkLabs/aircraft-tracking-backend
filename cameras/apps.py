from django.apps import AppConfig


class CamerasConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'cameras'

    def ready(self):
        """Initialize MinIO storage when the app is ready."""
        import os
        import sys
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Skip initialization during migrations
        if 'migrate' in sys.argv or 'makemigrations' in sys.argv:
            return
        
        # Initialize MinIO bucket on startup (for both dev and production)
        try:
            from backend.storage import initialize_minio
            logger.info("Initializing MinIO bucket...")
            success = initialize_minio()
            if success:
                logger.info("✓ MinIO bucket initialized successfully")
            else:
                logger.warning("✗ MinIO bucket initialization failed")
        except Exception as e:
            logger.error(f"✗ Error initializing MinIO on startup: {e}", exc_info=True)

