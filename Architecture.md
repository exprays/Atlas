# AtlasEye: CNN Architecture and Methods Documentation

## Table of Contents
- [Overview](#overview)
- [CNN Architecture](#cnn-architecture)
- [Training Methodology](#training-methodology)
- [Loss Functions](#loss-functions)
- [Data Preprocessing and Augmentation](#data-preprocessing-and-augmentation)
- [Evaluation Metrics](#evaluation-metrics)
- [Inference Process](#inference-process)
- [Implementation Details](#implementation-details)
- [Performance Considerations](#performance-considerations)

## Overview

AtlasEye employs a specialized convolutional neural network designed for satellite image change detection. The system analyzes pairs of satellite images (before and after) to accurately detect and segment areas of change. The implementation is based on a Siamese U-Net architecture, which is particularly effective for semantic segmentation tasks.

## CNN Architecture

### Siamese U-Net

The core architecture is a modified U-Net designed specifically for change detection:

- **Input**: 6-channel tensor (3 channels from "before" image + 3 channels from "after" image)
- **Output**: Single-channel change probability mask
- **Architecture Type**: Encoder-decoder with skip connections

### Network Structure

The network consists of the following components:

#### Encoder Path (Downsampling)
- Initial Convolution: 6-channel input → 64 feature maps
- Downsampling Blocks: 4 consecutive blocks with:
  - MaxPooling (2×2)
  - Double Convolution (3×3 kernel, BatchNorm, ReLU)
- Feature Map Progression: 64 → 128 → 256 → 512 → 1024/2 (if bilinear upsampling)

#### Decoder Path (Upsampling)
- Upsampling Blocks: 4 consecutive blocks with:
  - Upsampling (bilinear) or Transposed Convolution
  - Skip connection concatenation from encoder
  - Double Convolution (3×3 kernel, BatchNorm, ReLU)
- Feature Map Progression: 1024 → 512 → 256 → 128 → 64

#### Output Layer
- 1×1 convolution for final prediction map

### Key Components

#### Double Convolution Block

The double convolution block consists of two sequential convolutional layers, each followed by batch normalization and ReLU activation. This block is used throughout both the encoder and decoder paths to extract and refine features.

```python
def double_conv(in_channels, out_channels):
    """
    Double convolution block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        
    Returns:
        nn.Sequential: Sequential container of the double convolution operations
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
```

**Key Components:**

1. **Convolutional Layers**: 3×3 kernels with padding=1 to maintain spatial dimensions
   - First layer: Transforms `in_channels` to `out_channels`
   - Second layer: Maintains `out_channels` while further refining features

2. **Batch Normalization**: Normalizes activations to improve training stability and convergence
   - Normalizes each channel independently
   - Helps address internal covariate shift problem

3. **ReLU Activation**: Introduces non-linearity with Rectified Linear Unit
   - `inplace=True` modifies input directly to save memory
   - Helps with gradient propagation during backpropagation

The double convolution block is implemented as a class in the actual U-Net:

```python
class DoubleConv(nn.Module):
    """Double Convolution block with batch normalization and ReLU activation"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
```

This implementation also allows for an intermediate number of channels through the `mid_channels` parameter, offering greater flexibility in feature map dimensionality between the two convolutions.

#### Downsampling Block
```python
self.maxpool_conv = nn.Sequential(
    nn.MaxPool2d(2),
    DoubleConv(in_channels, out_channels)
)
```

#### Upsampling Block
```python
# Method 1: Bilinear upsampling
self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
# or
# Method 2: Transposed convolution
self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
```

## Training Methodology

### Dataset Preparation
- **Dataset Class**: SatelliteChangeDataset handles loading and preparing image pairs
- **Image Types**: "Before" images, "After" images, and binary change masks (ground truth)
- **Input Size**: Images are resized to a fixed size (default 256×256) for consistent processing

### Training Process
- **Optimizer**: Adam optimizer with configurable learning rate (default 0.001)
- **Learning Rate Scheduling**: ReduceLROnPlateau scheduler that reduces learning rate when validation loss plateaus
- **Batch Size**: Configurable batch size (default 8)
- **Epochs**: Configurable number of epochs (default 50)
- **Validation Split**: 20% of data used for validation by default
- **Checkpointing**: Model checkpoints saved regularly during training:
  - Best model saved based on validation loss
  - Regular checkpoints every N epochs
  - Final model saved after training completion

## Loss Functions

The model is trained using a combination of loss functions specifically designed for semantic segmentation and handling class imbalance:

### Dice Loss
Used to measure the overlap between prediction and ground truth:
```python
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice
```

### Focal Loss
Addresses class imbalance by focusing more on difficult-to-classify examples:
```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    pred = torch.sigmoid(pred)
    
    # Binary cross entropy
    bce = nn.BCELoss(reduction='none')(pred, target)
    
    # Focal loss modification
    pt = torch.exp(-bce)
    focal_loss = alpha * (1 - pt) ** gamma * bce
    
    return focal_loss.mean()
```

### Combined Loss
A weighted combination of Dice and Focal losses:
```python
def combined_loss(pred, target, dice_weight=0.5, focal_weight=0.5):
    d_loss = dice_loss(pred, target)
    f_loss = focal_loss(pred, target)
    
    return dice_weight * d_loss + focal_weight * f_loss
```

## Data Preprocessing and Augmentation

### Preprocessing Steps
- Image loading with Rasterio for geospatial data
- Resizing to target dimensions
- Normalization of pixel values
- Conversion to PyTorch tensors
- Channel stacking for "before" and "after" images

### Data Augmentation
A set of augmentation techniques is applied to increase dataset variability and model robustness:

- Random rotation
- Random horizontal and vertical flips
- Random brightness and contrast adjustments
- Random cropping
- Elastic deformation

Importantly, the same transformations are applied to both "before" and "after" images as well as the mask to maintain alignment.

## Evaluation Metrics

The model performance is evaluated using several metrics:

### Accuracy
```python
accuracy = accuracy_score(ground_truth.flatten(), prediction.flatten())
```

### Kappa Coefficient (Cohen's Kappa)
Measures agreement between prediction and ground truth, accounting for chance agreement:
```python
kappa = cohen_kappa_score(ground_truth.flatten(), prediction.flatten())
```

Key interpretation:
- Kappa = 0: Agreement equivalent to chance
- Kappa = 1: Perfect agreement
- Kappa < 0: Agreement worse than chance

### FI Error (False Information Error)
Measures the proportion of false positives relative to all detected changes:
```python
fi_error = false_positives / (false_positives + true_positives + 1e-10)
```

### Confusion Matrix
The evaluation also computes and visualizes the confusion matrix, showing:
- True Positives (TP): Correctly identified change
- True Negatives (TN): Correctly identified no-change
- False Positives (FP): Incorrectly identified as change
- False Negatives (FN): Incorrectly identified as no-change

## Inference Process

### Prediction Pipeline
- **Model Loading**: Load trained U-Net model
- **Image Preparation**: Process and normalize input image pairs
- **Forward Pass**: Run images through model to get raw outputs
- **Post-processing**:
  - Apply sigmoid activation for probability map
  - Threshold to obtain binary change mask (default threshold = 0.5)
  - Filter small regions (noise removal)

### Change Analysis
After prediction, the system performs detailed analysis:

- **Connected Component Analysis**: Identify distinct change regions
- **Region Filtering**: Remove small regions below a minimum area threshold
- **Change Statistics**: Calculate change percentage and number of regions
- **Geospatial Processing**: Convert pixel regions to geospatial polygons for GIS integration

## Visualization

The system generates multiple visualizations:

- Before/after images side-by-side
- Change probability heatmap
- Binary change mask
- Overlay of changes on the "after" image

