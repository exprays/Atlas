# This script trains a change detection model using satellite images.
# It includes data loading, augmentation, model initialization, training, and validation.
# The script also handles command-line arguments for configuration and logging.


import os
import argparse
import glob
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from app.ml.models.unet import initialize_model
from app.ml.data.augmentations import SatelliteImageAugmentation
from app.ml.training.trainer import SatelliteChangeDataset, ChangeDetectionTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train change detection model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def plot_training_history(history, output_path='training_history.png'):
    """Plot training and validation loss history"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_losses'], label='Training Loss')
    
    if history['val_losses']:
        plt.plot(history['val_losses'], label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    logger.info(f"Training history plot saved to {output_path}")

def find_data_files(data_dir):
    """Find before/after/mask image files in the data directory"""
    before_images = sorted(glob.glob(os.path.join(data_dir, 'before/*.tif')))
    after_images = sorted(glob.glob(os.path.join(data_dir, 'after/*.tif')))
    mask_images = sorted(glob.glob(os.path.join(data_dir, 'mask/*.tif')))
    
    if not before_images or not after_images:
        raise ValueError("No before/after images found in data directory")
    
    # Check if masks are available
    if not mask_images:
        logger.warning("No mask images found. Will train without ground truth.")
        mask_images = None
        
    return before_images, after_images, mask_images

def main():
    args = parse_args()
    
    # Check if CUDA is available when requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Find data files
    before_images, after_images, mask_images = find_data_files(args.data_dir)
    logger.info(f"Found {len(before_images)} before images, {len(after_images)} after images")
    if mask_images:
        logger.info(f"Found {len(mask_images)} mask images")
    
    # Split into training and validation sets
    indices = np.arange(len(before_images))
    train_indices, val_indices = train_test_split(indices, test_size=args.val_split, random_state=42)
    
    # Create datasets
    augmentation = SatelliteImageAugmentation()
    
    train_dataset = SatelliteChangeDataset(
        [before_images[i] for i in train_indices],
        [after_images[i] for i in train_indices],
        [mask_images[i] for i in train_indices] if mask_images else None,
        transform=augmentation,
        target_size=(args.image_size, args.image_size)
    )
    
    val_dataset = SatelliteChangeDataset(
        [before_images[i] for i in val_indices],
        [after_images[i] for i in val_indices],
        [mask_images[i] for i in val_indices] if mask_images else None,
        transform=None,  # No augmentation for validation
        target_size=(args.image_size, args.image_size)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    logger.info(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Create model
    model = initialize_model(n_channels=6, n_classes=1)
    
    # Create trainer
    trainer = ChangeDetectionTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from checkpoint: {args.resume}")
    
    # Train the model
    history = trainer.train(
        num_epochs=args.num_epochs,
        validate_every=1,
        save_every=5
    )
    
    # Plot training history
    plot_training_history(
        history, 
        output_path=os.path.join(args.checkpoint_dir, 'training_history.png')
    )
    
    logger.info("Training completed.")

if __name__ == "__main__":
    main()