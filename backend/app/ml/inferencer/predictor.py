# backend/app/ml/inference/predictor.py
import os
import torch
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import geopandas as gpd
from skimage import measure
import logging
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Dict, List, Tuple, Optional, Union
import json

from app.ml.models.unet import initialize_model
from app.ml.data.preprocessing import prepare_image_pair
from app.core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChangeDetectionPredictor:
    """
    Class for predicting changes between satellite images
    """
    def __init__(self, model_path=None, device=None, threshold=0.5):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained model
            device: Device to use (cpu or cuda)
            threshold: Threshold for binary change detection
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        if model_path is None:
            model_path = settings.MODEL_PATH
            
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        
        logger.info(f"ChangeDetectionPredictor initialized with model: {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the model"""
        if self.model is not None:
            return
            
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = initialize_model(n_channels=6, n_classes=1)
            
            if torch.cuda.is_available() and self.device.type == 'cuda':
                state_dict = torch.load(self.model_path)
            else:
                state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # If model was saved as a checkpoint
            if 'model_state_dict' in state_dict:
                self.model.load_state_dict(state_dict['model_state_dict'])
            else:
                self.model.load_state_dict(state_dict)
                
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, before_image_path, after_image_path, target_size=(256, 256)):
        """
        Predict changes between two satellite images
        
        Args:
            before_image_path: Path to the "before" image
            after_image_path: Path to the "after" image
            target_size: Size to resize images to
            
        Returns:
            Dictionary containing prediction results
        """
        logger.info(f"Predicting changes: {before_image_path} -> {after_image_path}")
        start_time = time.time()
        
        # Load model if not loaded
        if self.model is None:
            self.load_model()
        
        # Prepare the image pair
        try:
            image_tensor, metadata = prepare_image_pair(before_image_path, after_image_path, target_size)
            image_tensor = image_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Error preparing images: {str(e)}")
            raise
        
        # Perform prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
            # Apply sigmoid to get probability map
            prob_map = torch.sigmoid(outputs).cpu().numpy().squeeze()
            
            # Apply threshold to get binary change mask
            binary_mask = (prob_map > self.threshold).astype(np.uint8)
        
        prediction_time = time.time() - start_time
        logger.info(f"Prediction completed in {prediction_time:.2f} seconds")
        
        # Create result dictionary
        result = {
            'probability_map': prob_map,
            'binary_mask': binary_mask,
            'metadata': metadata,
            'prediction_time': prediction_time
        }
        
        return result
    
    def analyze_changes(self, prediction_result, min_area=10):
        """
        Analyze the predicted changes
        
        Args:
            prediction_result: Result from the predict method
            min_area: Minimum area (in pixels) to consider as a valid change
            
        Returns:
            Dictionary with analysis results
        """
        binary_mask = prediction_result['binary_mask']
        
        # Label connected components to identify distinct change regions
        labeled_mask, num_features = measure.label(binary_mask, return_num=True, connectivity=2)
        
        # Calculate region properties
        regions = measure.regionprops(labeled_mask)
        
        # Filter small regions
        valid_regions = [r for r in regions if r.area >= min_area]
        
        # Calculate overall change statistics
        total_pixels = binary_mask.size
        changed_pixels = np.sum(binary_mask)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        # Extract geospatial features (polygon geometries)
        geoms = []
        for region in valid_regions:
            # Create a mask for the current region
            region_mask = labeled_mask == region.label
            
            # Get the contours of the region
            contours = measure.find_contours(region_mask, 0.5)
            
            # Convert contours to polygons
            for contour in contours:
                # Convert from (row, col) to (x, y) coordinates
                contour = np.fliplr(contour)
                
                # Skip if contour has less than 3 points (not a valid polygon)
                if len(contour) < 3:
                    continue
                    
                # Create a polygon
                geoms.append({
                    'type': 'Polygon',
                    'coordinates': [contour.tolist()]
                })
        
        analysis_result = {
            'change_percentage': change_percentage,
            'num_change_regions': len(valid_regions),
            'geometries': geoms,
            'region_stats': [
                {
                    'area': region.area,
                    'centroid': region.centroid,
                    'bbox': region.bbox,
                    'label': region.label
                }
                for region in valid_regions
            ]
        }
        
        return analysis_result
    
    def create_visualization(self, before_image_path, after_image_path, prediction_result, 
                            output_path=None, show_probability=True):
        """
        Create a visualization of the change detection result
        
        Args:
            before_image_path: Path to the "before" image
            after_image_path: Path to the "after" image
            prediction_result: Result from the predict method
            output_path: Path to save the visualization image
            show_probability: Whether to show the probability map or binary mask
            
        Returns:
            Path to the saved visualization image
        """
        # Open the images
        with rasterio.open(before_image_path) as src:
            before_img = src.read([1, 2, 3])  # RGB channels
            before_img = np.transpose(before_img, (1, 2, 0))  # (H, W, C)
            
            # Normalize for visualization
            before_img = (before_img / before_img.max() * 255).astype(np.uint8)
        
        with rasterio.open(after_image_path) as src:
            after_img = src.read([1, 2, 3])  # RGB channels
            after_img = np.transpose(after_img, (1, 2, 0))  # (H, W, C)
            
            # Normalize for visualization
            after_img = (after_img / after_img.max() * 255).astype(np.uint8)
        
        # Get prediction overlay
        if show_probability:
            # Use color map for probability values
            change_overlay = prediction_result['probability_map']
            cmap = plt.cm.jet
            norm = colors.Normalize(vmin=0, vmax=1)
            change_overlay = cmap(norm(change_overlay))[:, :, :3]  # RGB only, no alpha
            change_overlay = (change_overlay * 255).astype(np.uint8)
        else:
            # Binary mask as red overlay
            change_overlay = np.zeros((
                prediction_result['binary_mask'].shape[0],
                prediction_result['binary_mask'].shape[1],
                3
            ), dtype=np.uint8)
            change_overlay[prediction_result['binary_mask'] > 0] = [255, 0, 0]  # Red for changes
        
        # Resize images if needed
        target_h, target_w = prediction_result['binary_mask'].shape
        if before_img.shape[0] != target_h or before_img.shape[1] != target_w:
            before_img = cv2.resize(before_img, (target_w, target_h))
        if after_img.shape[0] != target_h or after_img.shape[1] != target_w:
            after_img = cv2.resize(after_img, (target_w, target_h))
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Before image
        axes[0, 0].imshow(before_img)
        axes[0, 0].set_title('Before Image')
        axes[0, 0].axis('off')
        
        # After image
        axes[0, 1].imshow(after_img)
        axes[0, 1].set_title('After Image')
        axes[0, 1].axis('off')
        
        # Change probability/binary map
        if show_probability:
            axes[1, 0].imshow(prediction_result['probability_map'], cmap='jet')
            axes[1, 0].set_title('Change Probability')
        else:
            axes[1, 0].imshow(prediction_result['binary_mask'], cmap='gray')
            axes[1, 0].set_title('Binary Change Mask')
        axes[1, 0].axis('off')
        
        # Overlay on after image
        # Blend the after image with the change overlay
        alpha = 0.5  # Transparency
        blended = cv2.addWeighted(after_img, 1-alpha, change_overlay, alpha, 0)
        axes[1, 1].imshow(blended)
        axes[1, 1].set_title('Change Overlay')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
                # Save visualization
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        
        plt.close(fig)  # Close the figure to free up resources
        
        return output_path

    def export_geojson(self, prediction_result, analysis_result, output_path):
        """
        Export change detection results as GeoJSON
        
        Args:
            prediction_result: Result from the predict method
            analysis_result: Result from the analyze_changes method
            output_path: Path to save the GeoJSON file
            
        Returns:
            Path to the saved GeoJSON file
        """
        try:
            # Create GeoJSON feature collection
            feature_collection = {
                "type": "FeatureCollection",
                "features": [],
                "properties": {
                    "change_percentage": analysis_result['change_percentage'],
                    "num_change_regions": analysis_result['num_change_regions'],
                    "prediction_time": prediction_result['prediction_time']
                }
            }
            
            # Add each detected change region as a feature
            for idx, geom in enumerate(analysis_result['geometries']):
                region_stats = next((r for r in analysis_result['region_stats'] if r['label'] == idx + 1), None)
                
                feature = {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "id": idx + 1,
                        "area": region_stats['area'] if region_stats else None,
                        "centroid": region_stats['centroid'] if region_stats else None
                    }
                }
                
                feature_collection["features"].append(feature)
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(feature_collection, f)
            
            logger.info(f"GeoJSON exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting GeoJSON: {str(e)}")
            raise
    
    def process_image_pair(self, before_image_path, after_image_path, output_dir, 
                          min_area=10, target_size=(256, 256), export_geojson=True):
        """
        Complete processing pipeline for a pair of images
        
        Args:
            before_image_path: Path to the "before" image
            after_image_path: Path to the "after" image
            output_dir: Directory to save outputs
            min_area: Minimum area for change regions
            target_size: Size to resize images to
            export_geojson: Whether to export results as GeoJSON
            
        Returns:
            Dictionary with all processing results and output paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output paths
        viz_path = os.path.join(output_dir, 'change_visualization.png')
        geojson_path = os.path.join(output_dir, 'change_detection.geojson')
        
        # Run prediction
        prediction_result = self.predict(before_image_path, after_image_path, target_size)
        
        # Analyze changes
        analysis_result = self.analyze_changes(prediction_result, min_area)
        
        # Create visualization
        self.create_visualization(
            before_image_path, 
            after_image_path, 
            prediction_result,
            output_path=viz_path
        )
        
        # Export GeoJSON if requested
        if export_geojson:
            self.export_geojson(prediction_result, analysis_result, geojson_path)
        
        return {
            'prediction': prediction_result,
            'analysis': analysis_result,
            'visualization_path': viz_path,
            'geojson_path': geojson_path if export_geojson else None
        }