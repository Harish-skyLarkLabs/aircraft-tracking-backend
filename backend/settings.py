"""
Django settings for Aircraft Tracking backend project.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-aircraft-tracking-change-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

ALLOWED_HOSTS = ["*"]

CORS_ORIGIN_ALLOW_ALL = True
USE_X_FORWARDED_HOST = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Third party apps
    "corsheaders",
    "drf_yasg",
    "rest_framework",
    "rest_framework.authtoken",
    'channels',
    'django_celery_beat',
    'django_celery_results',

    # Local apps
    'accounts',
    'cameras',
    'aircraft_detection',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# ASGI Application
ASGI_APPLICATION = 'backend.asgi.application'

# Channel Layers for WebSocket
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [(os.getenv("REDIS_HOST", "redis"), int(os.getenv("REDIS_PORT", 6379)))],
        },
    },
}

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

if os.getenv('DB_HOST', None) is not None:
    DATABASES = {
        'default': {
            'ENGINE': os.getenv('DB_ENGINE', 'django.db.backends.postgresql'),
            'NAME': os.getenv('DB_NAME', 'aircraft_tracking'),
            'HOST': os.getenv('DB_HOST', 'db'),
            'PASSWORD': os.getenv('DB_PASSWORD', 'postgres'),
            'USER': os.getenv('DB_USER', 'postgres'),
            'PORT': os.getenv('DB_PORT', '5432'),
            'CONN_MAX_AGE': 500,
        }
    }

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# REST Framework configuration
REST_FRAMEWORK = {
    "DEFAULT_FILTER_BACKENDS": [],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 40,
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.TokenAuthentication",
    ),
    "DEFAULT_RENDERER_CLASSES": (
        "rest_framework.renderers.JSONRenderer",
        "rest_framework.renderers.BrowsableAPIRenderer",
    ),
    "NON_FIELD_ERRORS_KEY": "Error",
}

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

# Custom User Model
AUTH_USER_MODEL = "accounts.User"

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'static'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Celery Configuration
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE

# ML Models Configuration
ML_MODELS = {
    'AIRCRAFT_DETECTION': os.path.join(BASE_DIR, 'models', 'aircraft_detection.pt'),
}

# Aircraft Detection Settings
AIRCRAFT_DETECTION_SETTINGS = {
    'MAX_FPS': 15,
    'FRAME_BUFFER_SIZE': 3,
    'DEBUG_VISUALIZATION': True,
    'CONFIDENCE_THRESHOLD': 0.5,
    'IOU_THRESHOLD': 0.45,
}

# Detection Classes for Aircraft
AIRCRAFT_CLASSES = {
    0: 'aircraft',
    1: 'helicopter',
    2: 'drone',
}

# Alert Types
ALERT_TYPES = {
    'LANDING': 'landing',
    'TAKEOFF': 'takeoff',
    # 'TAXIING': 'taxiing',
    # 'PARKED': 'parked',
    # 'INTRUSION': 'intrusion',
}

# MinIO Object Storage Configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_SECURE = os.getenv('MINIO_SECURE', 'False').lower() == 'true'
MINIO_BUCKET_NAME = os.getenv('MINIO_BUCKET_NAME', 'aircraft-tracking')

# MinIO public URL for frontend access (without /bucket-name)
MINIO_PUBLIC_URL = os.getenv('MINIO_PUBLIC_URL', f'http://{MINIO_ENDPOINT}')

# Backend public URL for MJPEG streaming
BACKEND_PUBLIC_URL = os.getenv('BACKEND_PUBLIC_URL', 'http://localhost:8000')

