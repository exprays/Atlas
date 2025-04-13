# Atlas Preprocessor (AtlasEye)
# This script preprocesses satellite images for change detection tasks.
# It includes functions for normalizing images, resizing, and preparing image pairs.


import os
import numpy as np
import torch
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.features import bounds
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
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds
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

# backend/app/ml/data/augmentations.py
import numpy as np
import torch
import cv2
import random

class SatelliteImageAugmentation:
    """
    Class for satellite image augmentations during training
    """
    def __init__(self, 
                 flip_probability=0.5,
                 rotate_probability=0.5,
                 brightness_contrast_probability=0.3,
                 noise_probability=0.2):
        self.flip_probability = flip_probability
        self.rotate_probability = rotate_probability
        self.brightness_contrast_probability = brightness_contrast_probability
        self.noise_probability = noise_probability
        
    def apply(self, before_image, after_image, mask=None):
        """
        Apply consistent augmentations to image pairs and mask
        
        Args:
            before_image: The "before" image (H, W, C)
            after_image: The "after" image (H, W, C)
            mask: Optional binary mask for change detection
            
        Returns:
            Augmented before image, after image, and mask
        """
        # Clone images to avoid modifying originals
        before = before_image.copy()
        after = after_image.copy()
        mask_out = mask.copy() if mask is not None else None
        
        # Random horizontal flip
        if random.random() < self.flip_probability:
            before = np.fliplr(before)
            after = np.fliplr(after)
            if mask_out is not None:
                mask_out = np.fliplr(mask_out)
        
        # Random vertical flip
        if random.random() < self.flip_probability:
            before = np.flipud(before)
            after = np.flipud(after)
            if mask_out is not None:
                mask_out = np.flipud(mask_out)
        
        # Random rotation (90, 180, or 270 degrees)
        if random.random() < self.rotate_probability:
            k = random.randint(1, 3)  # Number of 90-degree rotations
            before = np.rot90(before, k)
            after = np.rot90(after, k)
            if mask_out is not None:
                mask_out = np.rot90(mask_out, k)
        
        # Random brightness and contrast
        if random.random() < self.brightness_contrast_probability:
            # Apply the same transformation to both images for consistency
            alpha = 1.0 + random.uniform(-0.2, 0.2)  # Contrast factor
            beta = random.uniform(-20, 20)  # Brightness factor
            
            before = cv2.convertScaleAbs(before, alpha=alpha, beta=beta)
            after = cv2.convertScaleAbs(after, alpha=alpha, beta=beta)
        
        # Random noise
        if random.random() < self.noise_probability:
            # Gaussian noise
            noise = np.random.normal(0, 15, before.shape).astype(np.uint8)
            before = cv2.add(before, noise)
            
            noise = np.random.normal(0, 15, after.shape).astype(np.uint8)
            after = cv2.add(after, noise)
        
        return before, after, mask_out