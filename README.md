# Aircraft Tracking Backend

Django REST Framework backend for AI-powered aircraft tracking system.

## Features

- **User Authentication**: Token-based authentication with role management
- **Camera Management**: CRUD operations for cameras with RTSP stream support
- **AI Detection**: YOLO-based aircraft detection with tracking
- **Real-time Streaming**: WebSocket support for live video feed
- **Alerts System**: Detection alerts for landing, takeoff, and other events
- **Post-processing**: Bounding box drawing and visualization

## Tech Stack

- **Framework**: Django 5.1 + Django REST Framework
- **Database**: PostgreSQL 15
- **Cache/Message Broker**: Redis
- **WebSocket**: Django Channels
- **AI/ML**: PyTorch, Ultralytics YOLO
- **Task Queue**: Celery

## Project Structure

```
aircraft-tracking-backend/
├── backend/                 # Django project settings
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   ├── routing.py          # WebSocket routing
│   └── celery.py
├── accounts/               # User authentication
│   ├── models.py
│   ├── views.py
│   ├── serializers.py
│   └── urls.py
├── cameras/                # Camera management
│   ├── models.py
│   ├── views.py
│   ├── serializers.py
│   └── urls.py
├── aircraft_detection/     # AI detection
│   ├── models.py
│   ├── views.py
│   ├── serializers.py
│   ├── consumers.py        # WebSocket consumers
│   └── utils/
│       ├── aircraft_detector.py
│       ├── camera_manager.py
│       ├── frame_processor.py
│       ├── stream_handler.py
│       ├── tracker.py
│       └── drawing_utils.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── manage.py
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support (optional but recommended)

### Setup

1. **Clone and navigate to the project**:
   ```bash
   cd aircraft-tracking-backend
   ```

2. **Create environment file**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start with Docker**:
   ```bash
   docker-compose up -d
   ```

4. **Create superuser**:
   ```bash
   docker-compose exec aircraft-backend python manage.py createsuperuser
   ```

### Local Development (without Docker)

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```

3. **Start PostgreSQL and Redis** (or use Docker):
   ```bash
   docker-compose up -d db redis
   ```

4. **Run migrations**:
   ```bash
   python manage.py migrate
   ```

5. **Start the server**:
   ```bash
   python manage.py runserver
   # or with ASGI for WebSocket support
   daphne -b 0.0.0.0 -p 8000 backend.asgi:application
   ```

## API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/accounts/register/` | Register new user |
| POST | `/auth/accounts/login/` | Login |
| POST | `/auth/accounts/logout/` | Logout |
| GET | `/auth/accounts/profile/` | Get profile |
| POST | `/auth/accounts/change-password/` | Change password |
| POST | `/get-token/` | Get auth token |

### Cameras

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/cameras/` | List cameras |
| POST | `/cameras/` | Create camera |
| GET | `/cameras/{id}/` | Get camera details |
| PATCH | `/cameras/{id}/` | Update camera |
| DELETE | `/cameras/{id}/` | Delete camera |
| POST | `/cameras/check_rtsp/` | Validate RTSP link |
| POST | `/cameras/{id}/start_processing/` | Start AI processing |
| POST | `/cameras/{id}/stop_processing/` | Stop AI processing |

### Locations & Zones

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/locations/` | List/Create locations |
| GET/PATCH/DELETE | `/locations/{id}/` | Location details |
| GET/POST | `/zones/` | List/Create zones |
| GET/PATCH/DELETE | `/zones/{id}/` | Zone details |

### Detections

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/detections/` | List detections |
| GET | `/detections/{id}/` | Detection details |
| POST | `/detections/{id}/resolve/` | Mark as resolved |
| POST | `/detections/{id}/ignore/` | Mark as ignored |
| GET | `/detections/stats/` | Get statistics |
| GET | `/detections/recent/` | Get recent detections |
| GET | `/detections/alerts/` | Get pending alerts |

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/sessions/` | List sessions |
| GET | `/sessions/{id}/` | Session details |
| POST | `/sessions/{id}/end/` | End session |

## WebSocket Endpoints

### Video Feed
```
ws://localhost:8100/ws/video-feed/{camera_id}
```

**Messages received**:
- `frame`: Processed video frame with detections
- `connection_status`: Connection status
- `new_detections`: New detection alerts

**Commands**:
- `check_status`: Get camera status
- `request_high_quality_frame`: Get HQ snapshot
- `get_detections`: Get recent detections

### Alerts
```
ws://localhost:8100/ws/alerts/
```

**Messages received**:
- `new_alerts`: New alert notifications
- `alerts`: List of alerts

**Commands**:
- `get_alerts`: Get pending alerts

## API Documentation

- **Swagger UI**: `http://localhost:8100/api/docs/swagger/`
- **ReDoc**: `http://localhost:8100/api/docs/redoc/`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | - | Django secret key |
| `DEBUG` | `True` | Debug mode |
| `DB_HOST` | `db` | Database host |
| `DB_PORT` | `5432` | Database port |
| `DB_NAME` | `aircraft_tracking` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | `postgres` | Database password |
| `REDIS_HOST` | `redis` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |

### ML Model Configuration

Place your trained YOLO model in the `models/` directory:
```
models/
└── aircraft_detection.pt
```

## Adding Custom AI Models

1. Place your model file in the `models/` directory
2. Update `ML_MODELS` in `backend/settings.py`
3. Modify `aircraft_detection/utils/aircraft_detector.py` to load your model

## License

BSD License



