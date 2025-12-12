# AtlasEye - Single Image Deployment Guide

## Overview

This guide explains how to deploy and use the AtlasEye application using a single Docker image that contains all three services:
- **Backend** (FastAPI application)
- **Celery Worker** (Background task processor)
- **Frontend** (Next.js web application)

## Quick Start

### 1. Pull or Build the Image

If the image is published to a registry:
```bash
docker pull exprays/atlas-app:latest
```

Or build it locally:
```bash
docker build -t atlas-app:latest .
```

### 2. Start the Complete Application

```bash
docker-compose -f docker-compose.single.yml up -d
```

This will start:
- PostgreSQL database with PostGIS
- Redis message broker
- Atlas Backend (port 8000)
- Atlas Celery Worker
- Atlas Frontend (port 3000)

### 3. Access the Application

- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Backend API**: http://localhost:8000/api/v1

## Image Architecture

### Multi-Stage Build

The `atlas-app:latest` image is built using a multi-stage Docker build:

1. **Stage 1: Backend Builder**
   - Base: `python:3.11-slim`
   - Installs geospatial libraries
   - Installs Python dependencies
   - Copies backend application code

2. **Stage 2: Frontend Builder**
   - Base: `node:23-alpine`
   - Installs npm dependencies
   - Builds Next.js production bundle

3. **Stage 3: Final Runtime Image**
   - Base: `python:3.11-slim`
   - Installs both Node.js and Python runtime
   - Copies backend from Stage 1
   - Copies frontend from Stage 2
   - Single unified image: **~4.5GB**

### Entrypoint Script

The image uses a smart entrypoint script (`/app/docker-entrypoint.sh`) that can start different services based on the command:

```bash
# Start backend
docker run atlas-app:latest backend

# Start celery worker  
docker run atlas-app:latest celery

# Start frontend
docker run atlas-app:latest frontend
```

## Manual Deployment

### Using Docker Run

```bash
# 1. Start PostgreSQL
docker run -d --name atlas-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=satellite_change \
  -p 5432:5432 \
  postgis/postgis:14-3.2

# 2. Start Redis
docker run -d --name atlas-redis \
  -p 6379:6379 \
  redis:7-alpine

# 3. Start Backend
docker run -d --name atlas-backend \
  --link atlas-postgres:postgres \
  --link atlas-redis:redis \
  -e POSTGRES_SERVER=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=satellite_change \
  -e CELERY_BROKER_URL=redis://redis:6379/0 \
  -e CELERY_RESULT_BACKEND=redis://redis:6379/0 \
  -e MODEL_PATH=/app/data/models/final_model.pth \
  -e LOCAL_STORAGE_PATH=/app/data/images \
  -v $(pwd)/data:/app/data \
  -p 8000:8000 \
  atlas-app:latest backend

# 4. Start Celery Worker
docker run -d --name atlas-celery \
  --link atlas-postgres:postgres \
  --link atlas-redis:redis \
  -e POSTGRES_SERVER=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=satellite_change \
  -e CELERY_BROKER_URL=redis://redis:6379/0 \
  -e CELERY_RESULT_BACKEND=redis://redis:6379/0 \
  -e MODEL_PATH=/app/data/models/final_model.pth \
  -e LOCAL_STORAGE_PATH=/app/data/images \
  -v $(pwd)/data:/app/data \
  atlas-app:latest celery

# 5. Start Frontend
docker run -d --name atlas-frontend \
  -e NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1 \
  -e NEXT_PUBLIC_MAPBOX_TOKEN=your_token_here \
  -p 3000:3000 \
  atlas-app:latest frontend
```

## Using Docker Compose

### Basic Setup (Recommended)

Use the provided `docker-compose.single.yml`:

```bash
# Start all services
docker-compose -f docker-compose.single.yml up -d

# View logs
docker-compose -f docker-compose.single.yml logs -f

# Stop all services
docker-compose -f docker-compose.single.yml down

# Stop and remove volumes
docker-compose -f docker-compose.single.yml down -v
```

### Custom Configuration

Create your own `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgis/postgis:14-3.2
    environment:
      - POSTGRES_USER=your_user
      - POSTGRES_PASSWORD=your_secure_password
      - POSTGRES_DB=your_database
    volumes:
      - postgres_data:/var/lib/postgresql/data/

  redis:
    image: redis:7-alpine

  backend:
    image: atlas-app:latest
    command: ["backend"]
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=your_user
      - POSTGRES_PASSWORD=your_secure_password
      - POSTGRES_DB=your_database
      - CELERY_BROKER_URL=redis://redis:6379/0
      - MODEL_PATH=/app/data/models/final_model.pth
    volumes:
      - ./data:/app/data
    ports:
      - "8000:8000"

  celery_worker:
    image: atlas-app:latest
    command: ["celery"]
    environment:
      - POSTGRES_SERVER=postgres
      - CELERY_BROKER_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data

  frontend:
    image: atlas-app:latest
    command: ["frontend"]
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
    ports:
      - "3000:3000"

volumes:
  postgres_data:
```

## Data Management

### Directory Structure

Create the required data directories:

```bash
mkdir -p data/{models,images,training/{before,after,mask},test_data}
```

### Placing Your Model

Place your trained model in:
```
data/models/final_model.pth
```

If you don't have a trained model yet, you can train one after starting the services:

```bash
docker-compose -f docker-compose.single.yml exec backend \
  python -m app.ml.training.train \
  --data_dir=/app/data/training \
  --checkpoint_dir=/app/data/models \
  --num_epochs=50
```

### Uploaded Images Location

User-uploaded images and processing results are stored in:
```
data/images/{job_id}/
├── before_{filename}
├── after_{filename}
├── change_visualization.png
├── change_detection.geojson
└── evaluation_metrics.json
```

## Environment Variables

### Backend & Celery Worker

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_SERVER` | PostgreSQL hostname | `postgres` |
| `POSTGRES_USER` | Database user | `postgres` |
| `POSTGRES_PASSWORD` | Database password | `postgres` |
| `POSTGRES_DB` | Database name | `satellite_change` |
| `CELERY_BROKER_URL` | Redis URL for Celery | `redis://redis:6379/0` |
| `CELERY_RESULT_BACKEND` | Results backend | `redis://redis:6379/0` |
| `MODEL_PATH` | Path to trained model | `/app/data/models/final_model.pth` |
| `LOCAL_STORAGE_PATH` | Image storage path | `/app/data/images` |

### Frontend

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000/api/v1` |
| `NEXT_PUBLIC_MAPBOX_TOKEN` | Mapbox access token | (required for maps) |

## Scaling

### Horizontal Scaling

You can scale the number of workers:

```bash
# Scale celery workers to 3 instances
docker-compose -f docker-compose.single.yml up -d --scale celery_worker=3

# Scale backend to 2 instances (requires load balancer)
docker-compose -f docker-compose.single.yml up -d --scale backend=2
```

### With Load Balancer

Add Nginx for load balancing multiple backend instances:

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
```

## Publishing the Image

### To Docker Hub

```bash
# Tag the image
docker tag atlas-app:latest exprays/atlas-app:latest
docker tag atlas-app:latest exprays/atlas-app:v1.0.0

# Push to Docker Hub
docker push exprays/atlas-app:latest
docker push exprays/atlas-app:v1.0.0
```

### To GitHub Container Registry

```bash
# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag
docker tag atlas-app:latest ghcr.io/exprays/atlas-app:latest

# Push
docker push ghcr.io/exprays/atlas-app:latest
```

## Health Checks

The image includes health checks for the backend:

```bash
# Check backend health
curl http://localhost:8000/health

# Response should be:
# {"status": "healthy"}
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs atlas-backend
docker logs atlas-celery-worker
docker logs atlas-frontend

# Or with docker-compose
docker-compose -f docker-compose.single.yml logs backend
```

### Frontend Can't Reach Backend

1. Check if backend is running:
   ```bash
   curl http://localhost:8000/docs
   ```

2. Verify `NEXT_PUBLIC_API_URL` in frontend:
   ```bash
   docker-compose -f docker-compose.single.yml exec frontend env | grep NEXT_PUBLIC
   ```

3. For production, update to public URL:
   ```
   NEXT_PUBLIC_API_URL=https://your-domain.com/api/v1
   ```

### Celery Worker Not Processing

```bash
# Check Celery worker logs
docker-compose -f docker-compose.single.yml logs celery_worker

# Check Redis connection
docker-compose -f docker-compose.single.yml exec redis redis-cli ping
# Should return: PONG

# Restart worker
docker-compose -f docker-compose.single.yml restart celery_worker
```

### Out of Disk Space

```bash
# Clean up Docker
docker system prune -af --volumes

# Remove old images
docker images | grep atlas-app | awk '{print $3}' | xargs docker rmi
```

## Image Size Optimization

Current image size: **~4.5GB**

The size is large due to:
- PyTorch and CUDA libraries: ~3GB
- Geospatial libraries: ~500MB
- Node.js runtime: ~200MB
- Next.js build: ~200MB

To reduce size:
1. Use CPU-only PyTorch build (reduces by ~1.5GB)
2. Remove unnecessary dependencies
3. Use multi-stage builds more efficiently

## Performance Tips

1. **Mount volumes for data**: Don't copy large datasets into the image
2. **Use bind mounts in development**: For live code changes
3. **Use named volumes in production**: For data persistence
4. **Enable GPU support**: Add NVIDIA runtime for faster processing
5. **Scale workers**: Add more Celery workers for parallel processing

## Security Considerations

For production:
1. **Change default passwords**
2. **Use secrets management** (Docker Secrets, Kubernetes Secrets)
3. **Enable HTTPS** with reverse proxy
4. **Implement authentication** on API endpoints
5. **Limit container resources** (memory, CPU)
6. **Run as non-root user** (update Dockerfile)
7. **Scan images for vulnerabilities**: `docker scan atlas-app:latest`

## Support

For issues or questions:
- GitHub Issues: https://github.com/exprays/Atlas/issues
- Documentation: /DEPLOYMENT_GUIDE.md
- Architecture: /Architecture.md

---

**Version**: 1.0.0  
**Last Updated**: December 2025  
**Maintained by**: Exprays
