# Aircraft Tracking Backend Dockerfile
# Multi-stage build for optimized production image

FROM python:3.10-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Stage 1: Compile dependencies
FROM base AS compile-image
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    cmake \
    libssl-dev libffi-dev libsm6 libxext6 libxrender-dev \
    gcc \
    ffmpeg \
    libsm6 \
    libxext6 libpq-dev \
    graphviz \
    graphviz-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Build final image
FROM base AS build-image

# Install runtime dependencies
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    libpq-dev ffmpeg libsm6 libxext6 graphviz graphviz-dev \
    make curl coreutils netcat-openbsd \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    qdbus-qt5 \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from compile stage
COPY --from=compile-image /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install PyTorch with CUDA support
RUN pip install --user --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Set display environment variables for headless operation
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0

# Copy startup scripts
COPY start.sh /start.sh
COPY celery.sh /celery.sh
COPY wait-for-db.sh /wait-for-db.sh
RUN chmod +x /start.sh /celery.sh /wait-for-db.sh

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app/

# Environment variables
ENV PORT=8000
ENV APP_MODULE=backend.asgi:application
ENV DB_ENGINE=django.db.backends.postgresql
ENV DB_HOST=db
ENV DB_PORT=5432
ENV DB_NAME=aircraft_tracking
ENV DB_USER=postgres

# Expose port
EXPOSE $PORT

# Default command - start the application
CMD ["/start.sh"]







