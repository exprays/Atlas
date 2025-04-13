from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException, Path
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import os
import shutil
from uuid import uuid4

from app.ml.inferencer.predictor import ChangeDetectionPredictor
from app.core.config import settings
from app.db.models import ChangeAnalysis, SatelliteImage
from app.db.base import get_db

router = APIRouter()
predictor = ChangeDetectionPredictor()

@router.post("/upload-images/")
async def upload_images(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    job_name: str = None,
    description: str = None,
    db = Depends(get_db)
):
    """Upload before and after satellite images for change detection"""
    job_id = str(uuid4())
    job_dir = os.path.join(settings.LOCAL_STORAGE_PATH, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Save uploaded files
    before_path = os.path.join(job_dir, f"before_{before_image.filename}")
    after_path = os.path.join(job_dir, f"after_{after_image.filename}")
    
    try:
        with open(before_path, "wb") as f:
            shutil.copyfileobj(before_image.file, f)
            
        with open(after_path, "wb") as f:
            shutil.copyfileobj(after_image.file, f)
            
        # Create database entries
        # ... implement database operations
        
        return {"job_id": job_id, "status": "uploaded"}
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")

@router.post("/process/{job_id}")
async def process_images(
    job_id: str = Path(...),
    background_tasks: BackgroundTasks = None,
    db = Depends(get_db)
):
    """Process previously uploaded images for change detection"""
    job_dir = os.path.join(settings.LOCAL_STORAGE_PATH, job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Find image files
    before_files = [f for f in os.listdir(job_dir) if f.startswith("before_")]
    after_files = [f for f in os.listdir(job_dir) if f.startswith("after_")]
    
    if not before_files or not after_files:
        raise HTTPException(status_code=400, detail="Missing before or after images")
    
    before_path = os.path.join(job_dir, before_files[0])
    after_path = os.path.join(job_dir, after_files[0])
    
    # Start processing (either background or synchronous)
    if background_tasks:
        background_tasks.add_task(process_images_task, job_id, before_path, after_path, job_dir)
        return {"job_id": job_id, "status": "processing"}
    else:
        result = process_images_task(job_id, before_path, after_path, job_dir)
        return result

def process_images_task(job_id, before_path, after_path, output_dir):
    """Background task to process images"""
    try:
        # Process the image pair
        result = predictor.process_image_pair(before_path, after_path, output_dir)
        
        # Update database with results
        # ... implement database updates
        
        return {
            "job_id": job_id,
            "status": "completed",
            "change_percentage": result["analysis"]["change_percentage"],
            "num_regions": result["analysis"]["num_change_regions"],
            "visualization_path": result["visualization_path"],
            "geojson_path": result["geojson_path"]
        }
    except Exception as e:
        # Update job status to failed in database
        # ... implement error handling
        return {"job_id": job_id, "status": "failed", "error": str(e)}

@router.get("/results/{job_id}")
async def get_results(job_id: str = Path(...), db = Depends(get_db)):
    """Get results of a processed job"""
    # Implement database query to get job results
    # ... fetch from database
    
    job_dir = os.path.join(settings.LOCAL_STORAGE_PATH, job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")
    
    viz_path = os.path.join(job_dir, "change_visualization.png")
    geojson_path = os.path.join(job_dir, "change_detection.geojson")
    
    if not os.path.exists(viz_path):
        raise HTTPException(status_code=404, detail="Results not found")
    
    # Return results metadata
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    return {
        "job_id": job_id,
        "status": "completed",
        "visualization_url": f"/api/v1/detection/visualization/{job_id}",
        "geojson_url": f"/api/v1/detection/geojson/{job_id}",
        "change_percentage": geojson_data["properties"]["change_percentage"],
        "num_regions": geojson_data["properties"]["num_change_regions"]
    }

@router.get("/visualization/{job_id}")
async def get_visualization(job_id: str = Path(...)):
    """Get the visualization image for a completed job"""
    viz_path = os.path.join(settings.LOCAL_STORAGE_PATH, job_id, "change_visualization.png")
    if not os.path.exists(viz_path):
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(viz_path)

@router.get("/geojson/{job_id}")
async def get_geojson(job_id: str = Path(...)):
    """Get the GeoJSON for a completed job"""
    geojson_path = os.path.join(settings.LOCAL_STORAGE_PATH, job_id, "change_detection.geojson")
    if not os.path.exists(geojson_path):
        raise HTTPException(status_code=404, detail="GeoJSON not found")
    
    return FileResponse(geojson_path)