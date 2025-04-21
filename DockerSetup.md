# Running AtlasEye with Docker

This guide covers how to set up and run the AtlasEye satellite change detection system using Docker and Docker Compose.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (version 20.10.0+)
- [Docker Compose](https://docs.docker.com/compose/install/) (version 2.0.0+)
- Git
- At least 8GB RAM
- 20GB+ free disk space
- NVIDIA GPU (optional, but recommended for faster training)
  - [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) (if using GPU)

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/atlaseye.git
cd atlaseye
```

## Step 2: Create Required Directories

```bash
mkdir -p data/models data/images data/training/before data/training/after data/training/mask data/test_data
```

## Step 3: Configure Environment Variables

Create a `.env` file in the root directory:

```bash
# Backend settings
POSTGRES_SERVER=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=satellite_change
MODEL_PATH=/app/data/models/unet_model.pth
LOCAL_STORAGE_PATH=/app/data/images

# Frontend settings
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_MAPBOX_TOKEN=your_mapbox_token_here
```

> **Note:** Obtain a Mapbox token from [mapbox.com](https://www.mapbox.com/) if you want to use the interactive map features.

## Step 4: Build the Docker Images

```bash
docker-compose build
```

This command builds all the services defined in the `docker-compose.yml` file.

## Step 5: Start the Services

### Start all services:

```bash
docker-compose up -d
```

### Verify services are running:

```bash
docker-compose ps
```

You should see all services (postgres, redis, backend, celery_worker, frontend) in the "Up" state.

## Step 6: Test Database Connection

```bash
docker-compose run --rm backend python -m app.db.test_connection
```

If successful, you'll see PostgreSQL and PostGIS version information.

## Step 7: Training the ML Model

If you have training data prepared in the respective directories, you can train the model:

```bash
docker-compose run --rm backend python -m app.ml.training.train \
    --data_dir=/app/data/training \
    --checkpoint_dir=/app/data/models \
    --batch_size=8 \
    --num_epochs=50 \
    --learning_rate=0.001 \
    --image_size=256 \
    --device=cpu  # Use 'cuda' for GPU acceleration
```

For GPU acceleration:

1. Uncomment the GPU section in `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```
2. Use `--device=cuda` in the training command

## Step 8: Testing the Model

Test the trained model with sample images:

```bash
docker-compose run --rm backend python -m app.ml.inferencer.test_predictor \
    --model_path=/app/data/models/final_model.pth \
    --before=/app/data/test_data/before.tif \
    --after=/app/data/test_data/after.tif \
    --output_dir=/app/data/test_results
```

## Step 9: Accessing the Application

- **Frontend UI**: [http://localhost:3000](http://localhost:3000)
- **Backend API docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

## Step 10: End-to-End Testing

### Upload Test Images

Using the web UI:
1. Go to [http://localhost:3000/upload](http://localhost:3000/upload)
2. Upload before and after satellite images
3. Submit the form

Or using curl:

```bash
curl -X POST http://localhost:8000/api/v1/detection/upload-images/ \
  -F "before_image=@/path/to/before.tif" \
  -F "after_image=@/path/to/after.tif"
```

Note the `job_id` from the response.

### Process Images

Using the API:

```bash
curl -X POST http://localhost:8000/api/v1/detection/process/{job_id}
```

Replace `{job_id}` with the actual ID received from the upload step.

### View Results

In the web UI, navigate to: [http://localhost:3000/results/{job_id}](http://localhost:3000/results/{job_id})

Or via API:

```bash
curl http://localhost:8000/api/v1/detection/results/{job_id}
```

## Step 11: Running Tests

### Backend Unit Tests

Run all backend tests:

```bash
docker-compose run --rm backend python -m unittest discover tests
```

Run specific test modules:

```bash
# ML module tests
docker-compose run --rm backend python -m unittest tests.test_ml

# API tests
docker-compose run --rm backend python -m unittest tests.test_api
```

### Frontend Tests

```bash
docker-compose run --rm frontend npm run lint
```

## Step 12: Monitoring and Troubleshooting

### View service logs:

View all logs:
```bash
docker-compose logs
```

View specific service logs:
```bash
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f postgres
```

### Check container status:

```bash
docker-compose ps
```

### Restart individual services:

```bash
docker-compose restart backend
docker-compose restart frontend
```

## Step 13: Stopping the Application

Stop all services but preserve volumes and containers:

```bash
docker-compose stop
```

Stop and remove containers (preserves volumes):

```bash
docker-compose down
```

Complete cleanup (removes volumes too, WILL DELETE DATABASE DATA):

```bash
docker-compose down -v
```

## Advanced Docker Configuration

### Scaling Services

```bash
docker-compose up -d --scale celery_worker=3
```

### Resource Limits

Edit `docker-compose.yml` to add resource constraints:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

### Custom Networks

For complex deployments, you can create custom networks:

```yaml
networks:
  frontend_network:
  backend_network:

services:
  frontend:
    networks:
      - frontend_network
  backend:
    networks:
      - frontend_network
      - backend_network
```

### Production Deployment

For production, consider adding a reverse proxy like Nginx:

```yaml
services:
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