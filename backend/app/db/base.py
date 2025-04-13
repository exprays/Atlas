# database for ATLAS
# May use Redis in future

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Function to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# backend/app/db/models.py
import datetime
from geoalchemy2 import Geometry
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from .base import Base

class SatelliteImage(Base):
    __tablename__ = "satellite_images"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    capture_date = Column(DateTime)
    cloud_cover_percentage = Column(Float, nullable=True)
    source = Column(String)  # e.g., "Sentinel-2", "Landsat-8"
    resolution = Column(Float)  # in meters/pixel
    storage_path = Column(String)
    
    # Geospatial data
    coverage_area = Column(Geometry('POLYGON'))
    center_point = Column(Geometry('POINT'))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    analyses = relationship("ChangeAnalysis", back_populates="image_pair", 
                           foreign_keys="[ChangeAnalysis.before_image_id, ChangeAnalysis.after_image_id]")

class ChangeAnalysis(Base):
    __tablename__ = "change_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    
    # Image references
    before_image_id = Column(Integer, ForeignKey("satellite_images.id"))
    after_image_id = Column(Integer, ForeignKey("satellite_images.id"))
    
    # Analysis results
    result_path = Column(String)  # Path to the result visualization
    change_percentage = Column(Float)  # Overall percentage of changed area
    
    # Accuracy metrics
    kappa_coefficient = Column(Float, nullable=True)
    overall_accuracy = Column(Float, nullable=True)
    fi_error = Column(Float, nullable=True)  # Fragmentation Index error
    
    # Status
    is_completed = Column(Boolean, default=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Processing metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    image_pair = relationship("SatelliteImage", back_populates="analyses")
    
    # Detailed changes
    detected_changes = relationship("DetectedChange", back_populates="analysis")

class DetectedChange(Base):
    __tablename__ = "detected_changes"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("change_analyses.id"))
    
    # Change type and description
    change_type = Column(String)  # e.g., "deforestation", "urban_development", "water_change"
    confidence_score = Column(Float)  # Model confidence in this change
    area_size = Column(Float)  # Size of the change in square meters/kilometers
    
    # Geospatial data
    geometry = Column(Geometry('POLYGON'))  # The actual polygon of the change
    
    # Metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship
    analysis = relationship("ChangeAnalysis", back_populates="detected_changes")