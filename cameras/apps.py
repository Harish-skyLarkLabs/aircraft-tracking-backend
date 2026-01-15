from django.apps import AppConfig


class CamerasConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'cameras'

    def ready(self):
        """Initialize MinIO storage when the app is ready."""
        import os
        # Only run in the main process, not in migrations or shell
        if os.environ.get('RUN_MAIN') == 'true' or os.environ.get('INIT_MINIO') == 'true':
            try:
                from backend.storage import initialize_minio
                initialize_minio()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not initialize MinIO on startup: {e}")

