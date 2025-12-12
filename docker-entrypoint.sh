#!/bin/bash
set -e

# Service selector for the combined Atlas image
# Usage: docker run -e SERVICE=backend atlas-app:latest
# Or: docker run atlas-app:latest backend

SERVICE=${SERVICE:-${1:-backend}}

echo "Starting AtlasEye service: $SERVICE"

case "$SERVICE" in
  backend)
    echo "Starting Backend (FastAPI)..."
    cd /app/backend
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
    ;;
    
  celery|worker)
    echo "Starting Celery Worker..."
    cd /app/backend
    exec celery -A app.core.celery_app worker --loglevel=info
    ;;
    
  frontend)
    echo "Starting Frontend (Next.js)..."
    cd /app/frontend
    exec npm run start -- -p 3000
    ;;
    
  *)
    echo "Unknown service: $SERVICE"
    echo "Available services: backend, celery, frontend"
    exit 1
    ;;
esac
