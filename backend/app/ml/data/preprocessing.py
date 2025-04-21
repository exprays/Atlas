# Atlas Preprocessor (AtlasEye)
# This script preprocesses satellite images for change detection tasks.
# It includes functions for normalizing images, resizing, and preparing image pairs.


import os
import numpy as np
import torch
import rasterio
import pyproj
import cv2
from torchvision import transforms

def normalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalize image using ImageNet mean and std values
    
    Args:
        image: Input image array (H, W, C)
        mean: Channel means
        std: Channel standard deviations
        
    Returns:
        Normalized image
    """
    # Clone the image to avoid modifying the original
    norm_image = image.copy().astype(np.float32)
    
    # Apply normalization for each channel
    for i in range(min(3, image.shape[2])):
        norm_image[:, :, i] = (norm_image[:, :, i] / 255.0 - mean[i]) / std[i]
    
    return norm_image

def preprocess_satellite_image(image_path, target_size=(256, 256)):
    """
    Preprocess a satellite image for the model
    
    Args:
        image_path: Path to the satellite image
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image tensor
    """
    with rasterio.open(image_path) as src:
        # Read the first 3 bands (typically RGB)
        image = src.read([1, 2, 3])
        
        # Convert from (C, H, W) to (H, W, C)
        image = np.transpose(image, (1, 2, 0))
        
        # Resize image to target size
        if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        image = normalize_image(image)
        
        # Convert to PyTorch tensor: (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        
        # Get geospatial metadata
        metadata = {
            'crs': src.crs.to_string() if src.crs else None,
            'transform': src.transform.to_gdal() if src.transform else None,
            'bounds': src.bounds._asdict() if src.bounds else None
        }
        
        return image_tensor, metadata

def prepare_image_pair(before_image_path, after_image_path, target_size=(256, 256)):
    """
    Prepare a pair of images (before/after) for change detection
    
    Args:
        before_image_path: Path to the "before" image
        after_image_path: Path to the "after" image
        target_size: Target size for resizing
        
    Returns:
        Combined image tensor and metadata
    """
    # Preprocess both images
    before_tensor, before_metadata = preprocess_satellite_image(before_image_path, target_size)
    after_tensor, after_metadata = preprocess_satellite_image(after_image_path, target_size)
    
    # Combine into a single tensor with 6 channels (3 from each image)
    combined_tensor = torch.cat([before_tensor, after_tensor], dim=0)
    
    # Add batch dimension
    combined_tensor = combined_tensor.unsqueeze(0)
    
    return combined_tensor, {'before': before_metadata, 'after': after_metadata}

