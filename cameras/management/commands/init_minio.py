"""
Django management command to initialize MinIO bucket.
Usage: python manage.py init_minio
"""
from django.core.management.base import BaseCommand
from backend.storage import initialize_minio


class Command(BaseCommand):
    help = 'Initialize MinIO bucket for the Aircraft Tracking system'

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING('Initializing MinIO bucket...'))
        
        success = initialize_minio()
        
        if success:
            self.stdout.write(self.style.SUCCESS('✓ MinIO bucket initialized successfully!'))
        else:
            self.stdout.write(self.style.ERROR('✗ Failed to initialize MinIO bucket'))
            self.stdout.write(self.style.WARNING('Please check your MinIO configuration in settings.py'))








