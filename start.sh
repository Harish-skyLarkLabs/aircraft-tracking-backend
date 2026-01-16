#!/bin/bash

# Exit script on any error
set -e

# Wait for database to be ready
/wait-for-db.sh

python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic --noinput

# Create default admin user if it doesn't exist
python manage.py create_default_user --email=orgadmin@skylarklabs.ai --password=skylark@123 --role=admin || true

# Start the server
daphne -b 0.0.0.0 -p 8000 backend.asgi:application



