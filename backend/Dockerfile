FROM python:3.11-slim

LABEL maintainer="PlantDoc Team"
LABEL description="PlantDoc Backend — FastAPI + TensorFlow + YOLOv8"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY classes.txt .
COPY utils/ ./utils/
COPY model/ ./model/

# Create non-root user for security
RUN useradd -m -u 1000 plantdoc && chown -R plantdoc:plantdoc /app
USER plantdoc

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]