FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    HEADLESS=1

WORKDIR /app

# Install system dependencies for OpenCV and Audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    portaudio19-dev \
    libsndfile1 \
    espeak \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose standard port
EXPOSE 8000

# Start D-SAAT web server
CMD ["sh", "-c", "python web_app.py --host 0.0.0.0 --port ${PORT:-8000} --no-browser"]
