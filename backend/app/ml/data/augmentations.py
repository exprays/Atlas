
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