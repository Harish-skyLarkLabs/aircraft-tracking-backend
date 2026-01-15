#!/bin/bash

# Exit script on any error
set -e

# Wait for database to be ready
/wait-for-db.sh

python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic --noinput

# Start the server
daphne -b 0.0.0.0 -p 8000 backend.asgi:application



