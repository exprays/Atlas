# Atlas Trainer
# This script is part of the Atlas project
# The script implements a trainer class for training a U-Net model on satellite images.
# The U-Net model is used for semantic segmentation tasks, specifically for detecting changes in satellite images.


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
from datetime import datetime
import logging
import cv2
import rasterio
from tqdm import tqdm

from app.ml.models.unet import UNet
from app.ml.data.preprocessing import preprocess_satellite_image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SatelliteChangeDataset(Dataset):
    """
    Dataset for satellite image change detection
    """
    def __init__(self, 
                 before_image_paths, 
                 after_image_paths, 
                 mask_paths=None, 
                 transform=None, 
                 target_size=(256, 256)):
        """
        Initialize the dataset
        
        Args:
            before_image_paths: List of paths to "before" images
            after_image_paths: List of paths to "after" images
            mask_paths: List of paths to change mask images (optional)
            transform: Image transformations to apply
            target_size: Target size for images
        """
        assert len(before_image_paths) == len(after_image_paths), "Number of before and after images must match"
        if mask_paths:
            assert len(before_image_paths) == len(mask_paths), "Number of images and masks must match"
            
        self.before_image_paths = before_image_paths
        self.after_image_paths = after_image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.before_image_paths)
    
    def __getitem__(self, idx):
        # Load before and after images
        before_tensor, _ = preprocess_satellite_image(self.before_image_paths[idx], self.target_size)
        after_tensor, _ = preprocess_satellite_image(self.after_image_paths[idx], self.target_size)
        
        # Load mask if available
        if self.mask_paths:
            # Load and process mask (assuming it's a binary mask)
            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1)  # Read the first band
                
                # Resize mask to target size
                if mask.shape[0] != self.target_size[0] or mask.shape[1] != self.target_size[1]:
                    mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                
                # Ensure binary values
                mask = (mask > 0).astype(np.float32)
                mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dimension
        else:
            mask_tensor = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
        
        # Apply transformations if specified
        if self.transform:
            # Convert tensors to numpy for transformations
            before_np = before_tensor.numpy().transpose(1, 2, 0)  # C, H, W -> H, W, C
            after_np = after_tensor.numpy().transpose(1, 2, 0)
            mask_np = mask_tensor.numpy().transpose(1, 2, 0)
            
            # Apply the same transformations to all images
            before_np, after_np, mask_np = self.transform.apply(before_np, after_np, mask_np)
            
            # Convert back to tensors - with .copy() to fix the negative strides issue
            before_tensor = torch.from_numpy(before_np.transpose(2, 0, 1).copy()).float()
            after_tensor = torch.from_numpy(after_np.transpose(2, 0, 1).copy()).float()
            mask_tensor = torch.from_numpy(mask_np.transpose(2, 0, 1).copy()).float()
        
        # Combine before and after tensors
        combined_tensor = torch.cat([before_tensor, after_tensor], dim=0)
        
        return {
            'input': combined_tensor,
            'mask': mask_tensor,
            'before_path': self.before_image_paths[idx],
            'after_path': self.after_image_paths[idx]
        }

def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss for segmentation
    """
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance
    """
    pred = torch.sigmoid(pred)
    
    # Binary cross entropy
    bce = nn.BCELoss(reduction='none')(pred, target)
    
    # Focal loss modification
    pt = torch.exp(-bce)
    focal_loss = alpha * (1 - pt) ** gamma * bce
    
    return focal_loss.mean()

def combined_loss(pred, target, dice_weight=0.5, focal_weight=0.5):
    """
    Combined loss function using both Dice and Focal loss
    """
    d_loss = dice_loss(pred, target)
    f_loss = focal_loss(pred, target)
    
    return dice_weight * d_loss + focal_weight * f_loss

class ChangeDetectionTrainer:
    """
    Trainer class for satellite image change detection
    """
    def __init__(self, 
                 model,
                 device,
                 train_loader,
                 val_loader=None,
                 learning_rate=0.001,
                 checkpoint_dir='checkpoints'):
        """
        Initialize the trainer
        
        Args:
            model: The U-Net model
            device: Device to use (cpu or cuda)
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            learning_rate: Learning rate for optimizer
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Create optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch in progress_bar:
            # Move data to device
            inputs = batch['input'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = combined_loss(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update progress
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(self.train_loader)
        self.train_losses.append(epoch_loss)
        
        return epoch_loss
    
    def validate(self):
        """Validate the model"""
        if not self.val_loader:
            return None
        
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                inputs = batch['input'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(self.val_loader)
        self.val_losses.append(val_loss)
        
        # Update learning rate scheduler
        self.scheduler.step(val_loss)
        
        return val_loss
    
    def save_checkpoint(self, epoch, val_loss=None):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if val_loss is not None and val_loss < self.best_val_loss:
            # Save best model
            self.best_val_loss = val_loss
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{epoch}_{timestamp}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint for epoch {epoch}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training history
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        return True
    
    def train(self, num_epochs, validate_every=1, save_every=5):
        """
        Train the model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            validate_every: Validate every N epochs
            save_every: Save checkpoint every N epochs
            
        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train one epoch
            epoch_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")
            
            # Validate if needed
            val_loss = None
            if self.val_loader and (epoch + 1) % validate_every == 0:
                val_loss = self.validate()
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
            
            # Save checkpoint if needed
            if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch + 1, val_loss)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes")
        
        # Final validation
        if self.val_loader:
            final_val_loss = self.validate()
            logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
        
        # Save final model
        final_checkpoint_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, final_checkpoint_path)
        logger.info(f"Saved final model at {final_checkpoint_path}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'training_time': total_time,
            'final_checkpoint': final_checkpoint_path
        }