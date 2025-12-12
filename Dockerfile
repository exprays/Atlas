# Multi-stage build for Atlas application
# Stage 1: Backend base
FROM python:3.11-slim as backend-base

WORKDIR /app/backend

# Install system dependencies for geospatial libraries and OpenCV runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libproj-dev \
    libgeos-dev \
    proj-bin \
    libspatialindex-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy backend application
COPY backend/ .

# Stage 2: Frontend builder
FROM node:23-alpine as frontend-builder

WORKDIR /app/frontend

# Copy frontend files
COPY frontend/package*.json ./
RUN npm install

COPY frontend/ .

# Build Next.js application
RUN npm run build

# Stage 3: Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies for both backend and frontend
RUN apt-get update && apt-get install -y --no-install-recommends \
    libproj-dev \
    libgeos-dev \
    proj-bin \
    libspatialindex-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from backend-base stage
COPY --from=backend-base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-base /usr/local/bin /usr/local/bin

# Copy backend application
COPY --from=backend-base /app/backend /app/backend

# Copy frontend built application
COPY --from=frontend-builder /app/frontend /app/frontend

WORKDIR /app

# Copy entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Set Python path
ENV PYTHONPATH=/app/backend
ENV PATH="/app/backend:$PATH"

# Expose ports
EXPOSE 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["backend"]
