FROM python:3.10-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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
    graphviz-dev

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM base AS build-image

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    libpq-dev ffmpeg libsm6 libxext6 graphviz graphviz-dev make curl coreutils netcat-openbsd

COPY --from=compile-image /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

RUN pip install --user --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

RUN apt-get update && apt-get install -y \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    qdbus-qt5 \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5

# Install X11 dependencies
RUN apt-get update && apt-get install -y \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0

# Set display environment variable
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0

COPY start.sh /start.sh
COPY celery.sh /celery.sh
COPY wait-for-db.sh /wait-for-db.sh
RUN chmod +x /start.sh /celery.sh /wait-for-db.sh

WORKDIR /app
COPY . /app/

ENV PORT=8000
ENV APP_MODULE=backend.asgi:application

EXPOSE $PORT

ENV DB_ENGINE=django.db.backends.postgresql
ENV DB_HOST=db
ENV DB_PORT=5432
ENV DB_NAME=aircraft_tracking
ENV DB_USER=postgres
ENV DB_PASSWORD=postgres