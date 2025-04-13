FROM python:3.13-alpine

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