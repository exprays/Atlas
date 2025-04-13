# configurations for Atlas
# This file contains the configuration settings for the FastAPI application.
# Using Pydantic for settings management and allows for environment variable overrides.
# Made with ❤️ by exprays

import os
from pydantic import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Atlas"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Database
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "satellite_change")
    SQLALCHEMY_DATABASE_URI: Optional[str] = None
    
    # ML model settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "app/ml/models/unet_model.pth")
    INPUT_IMAGE_SIZE: int = 256  # Default size for input to the model
    
    # Storage
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")  # 'local', 's3', or 'gcs'
    STORAGE_BUCKET: Optional[str] = os.getenv("STORAGE_BUCKET")
    LOCAL_STORAGE_PATH: str = os.getenv("LOCAL_STORAGE_PATH", "data/images")
    
    # Celery
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    
    class Config:
        case_sensitive = True
        
    def __init__(self, **data):
        super().__init__(**data)
        self.SQLALCHEMY_DATABASE_URI = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"

settings = Settings()