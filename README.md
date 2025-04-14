
# AtlasEye Project Documentation

## Table of Contents

- Overview
- System Requirements
- Project Setup
- Docker Configuration
- Training the ML Model
- Testing the System
- Running the Complete Application
- End-to-End Testing Workflow
- Troubleshooting
- Production Deployment
- Advanced Configuration

## Overview

AtlasEye is a satellite change detection system that uses deep learning to identify changes between satellite images captured at different times. The system can detect and visualize urban development, deforestation, natural disasters, and other environmental changes.

**Key Features:**
- Change detection between satellite image pairs
- Geospatial analysis with GeoJSON export
- Interactive visualization with metrics
- REST API for integration
- Asynchronous processing for large images

**Tech Stack:**
- **Backend**: Python, PyTorch, FastAPI, PostgreSQL with PostGIS
- **Frontend**: Next.js, TypeScript, TailwindCSS, Mapbox GL
- **Infrastructure**: Docker, Celery, Redis

## System Requirements

- Docker and Docker Compose
- Git
- 8GB+ RAM
- NVIDIA GPU (optional, for faster training)
- 20GB+ free disk space

## Project Setup

### Clone Repository

```bash
git clone https://github.com/yourusername/atlaseye.git
cd atlaseye
```

### Create Directory Structure

```bash
mkdir -p data/models data/images data/training/before data/training/after data/training/mask data/test_data
```

### Fix Configuration Files

**1. PostCSS Configuration**

```bash
cat > frontend/postcss.config.mjs << EOL
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
EOL
```

**2. Backend Dockerfile**

```bash
cat > backend/Dockerfile << EOL
FROM python:3.9-slim

WORKDIR /app

# Install GDAL and other geospatial dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Command is specified in docker-compose.yml
EOL
```

**3. Frontend Dockerfile**

```bash
cat > frontend/Dockerfile << EOL
FROM node:18-alpine

WORKDIR /app

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the application
COPY . .

# Command is specified in docker-compose.yml
EOL
```

### Prepare Training Data

1. Place before images in `data/training/before/`
2. Place after images in `data/training/after/`
3. Place ground truth masks in `data/training/mask/` (if available)
4. Place test images in `data/test_data/`

## Docker Configuration

The system uses Docker Compose to orchestrate multiple services. The key services are:

- **postgres**: PostgreSQL database with PostGIS extension
- **redis**: Message broker for Celery
- **backend**: Python FastAPI application with ML capabilities
- **celery_worker**: Background task processor
- **frontend**: Next.js web application

### Building Docker Images

```bash
docker-compose build
```

### Environment Variables

The system uses environment variables for configuration:

**Backend Environment Variables:**
- `POSTGRES_SERVER`: Database hostname
- `POSTGRES_USER`: Database username
- `POSTGRES_PASSWORD`: Database password
- `POSTGRES_DB`: Database name
- `CELERY_BROKER_URL`: Redis connection string
- `CELERY_RESULT_BACKEND`: Result storage backend
- `MODEL_PATH`: Path to trained model
- `LOCAL_STORAGE_PATH`: Path for storing uploaded images

**Frontend Environment Variables:**
- `NEXT_PUBLIC_API_URL`: Backend API URL
- `NEXT_PUBLIC_MAPBOX_TOKEN`: Mapbox API token for maps

## Training the ML Model

### Starting Training Process

```bash
docker-compose run --rm backend python -m app.ml.training.train \
    --data_dir=/app/data/training \
    --checkpoint_dir=/app/data/models \
    --batch_size=8 \
    --num_epochs=50 \
    --learning_rate=0.001 \
    --image_size=256 \
    --device=cpu  # Use 'cuda' if GPU available
```

### Training Parameters

| Parameter | Description |
|-----------|-------------|
| `--data_dir` | Directory containing training data |
| `--checkpoint_dir` | Directory to save model checkpoints |
| `--batch_size` | Number of samples per training batch |
| `--num_epochs` | Total number of training epochs |
| `--learning_rate` | Learning rate for optimizer |
| `--image_size` | Size to resize images (e.g., 256 for 256x256) |
| `--device` | 'cuda' for GPU or 'cpu' for CPU-only training |

### Monitoring Training Progress

View training logs in real-time:

```bash
docker-compose logs -f backend
```

The training history plot is saved to `data/models/training_history.png`.

## Testing the System

### Backend Unit Tests

Run all backend tests:

```bash
docker-compose run --rm backend python -m unittest discover tests
```

Run specific test modules:

```bash
# ML module tests
docker-compose run --rm backend python -m unittest backend/tests/test_ml.py

# API tests
docker-compose run --rm backend python -m unittest backend/tests/test_api.py
```

### Testing ML Model Inference

```bash
docker-compose run --rm backend python -m app.ml.inferencer.test_predictor \
    --model_path=/app/data/models/final_model.pth \
    --before=/app/data/test_data/before.tif \
    --after=/app/data/test_data/after.tif \
    --ground_truth=/app/data/test_data/mask.tif \  # Optional
    --output_dir=/app/data/test_results
```

### Frontend Tests

Run lint checks:

```bash
docker-compose run --rm frontend npm run lint
```

### Database Connectivity Test

```bash
docker-compose run --rm backend python -m app.db.test_connection
```

## Running the Complete Application

### Start All Services

```bash
docker-compose up -d
```

### Access Points

- **Backend API**: http://localhost:8000/docs
- **Frontend UI**: http://localhost:3000

### Checking Service Status

```bash
docker-compose ps
```

### Viewing Logs

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs -f backend
```

## End-to-End Testing Workflow

### 1. Upload Images

```bash
curl -X POST http://localhost:8000/api/v1/detection/upload-images/ \
  -F "before_image=@/path/to/before.tif" \
  -F "after_image=@/path/to/after.tif"
```

You should receive a job_id in response.

### 2. Process Images

```bash
curl -X POST http://localhost:8000/api/v1/detection/process/{job_id}
```

Replace `{job_id}` with the actual ID received.

### 3. Get Results

```bash
curl http://localhost:8000/api/v1/detection/results/{job_id}
```

### 4. View in UI

1. Open http://localhost:3000/results/{job_id} in your browser
2. Verify the results display correctly:
   - Change percentage
   - Before/after images
   - Change detection overlay
   - Interactive map
   - Metrics charts

## Troubleshooting

### Database Issues

If PostgreSQL connection fails:

```bash
docker-compose down
docker volume rm atlaseye_postgres_data
docker-compose up -d postgres
# Wait 10 seconds for initialization
docker-compose run --rm backend python -m app.db.test_connection
```

### ML Issues

If model training fails:

```bash
# Check GPU availability
docker-compose run --rm backend python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Verify data paths
docker-compose run --rm backend ls -la /app/data/training/before
docker-compose run --rm backend ls -la /app/data/training/after
docker-compose run --rm backend ls -la /app/data/training/mask
```

### Frontend Issues

For frontend rendering problems:

```bash
# Check if static assets are being served
docker-compose run --rm frontend ls -la /app/public

# Verify environment variables
docker-compose run --rm frontend env | grep NEXT_PUBLIC

# Fix Mapbox token if maps don't render
echo "NEXT_PUBLIC_MAPBOX_TOKEN=your_mapbox_token_here" > frontend/.env.local
docker-compose restart frontend
```

### Checking Logs

Detailed logs can help diagnose issues:

```bash
docker-compose logs -f
```

## Production Deployment

For production deployment, consider the following:

### Security Enhancements

1. **Secure Passwords**: Update environment variables with strong passwords
2. **SSL Configuration**: Add a reverse proxy (Nginx/Traefik) with SSL
3. **Authentication**: Implement proper user authentication and authorization

### Performance Optimization

1. **Database Tuning**: Optimize PostgreSQL for geospatial queries
2. **Caching**: Add Redis caching for API responses
3. **CDN**: Use a CDN for static assets

### High Availability

1. **Load Balancing**: Deploy multiple backend instances behind a load balancer
2. **Database Replication**: Set up PostgreSQL replication
3. **Monitoring**: Add monitoring and alerting

### Deployment Example (AWS)

```yaml
version: '3.8'

services:
  # ... services configuration

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - backend
      - frontend
```

## Advanced Configuration

### GPU Support for Training

To enable GPU acceleration:

1. Install NVIDIA Container Toolkit on host
2. Update docker-compose.yml:

```yaml
services:
  backend:
    # ... other settings
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

3. Update training command:

```bash
docker-compose run --rm backend python -m app.ml.training.train --device=cuda
```

### Custom Model Architecture

To use a custom model architecture:

1. Create a new model class in models
2. Update the model initialization in train.py
3. Update the predictor to use your custom model

### Database Migration

For database schema changes:

```bash
# Generate migration
docker-compose run --rm backend alembic revision --autogenerate -m "Description"

# Apply migration
docker-compose run --rm backend alembic upgrade head
```

---

**Documentation Version**: 1.0.0  
**Last Updated**: April 14, 2025  
**Authors**: Exprays