"""
Celery configuration for Aircraft Tracking backend
"""
import os
from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

celery = Celery('backend')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
celery.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
celery.autodiscover_tasks()


@celery.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')

