#!/bin/bash

# Exit script on any error
set -e

# Wait for database to be ready
/wait-for-db.sh

# Start Celery worker
celery -A backend worker -l info --concurrency=2

# Optionally start Celery beat for periodic tasks
# celery -A backend beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler



