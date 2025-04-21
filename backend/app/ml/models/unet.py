# Siamese U-Net for Change Detection
# Built with extensive care and consideration for the task of change detection in remote sensing images.
# This implementation is based on the U-Net architecture, which is widely used for image segmentation tasks.
# The model is designed to take two images as input (e.g., before and after images) and output a change mask.
# The architecture consists of an encoder-decoder structure with skip connections to preserve spatial information.
# The model is initialized with the option to load pretrained weights for transfer learning.
# Made with love by @exprays

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Double Convolution block:
    (Conv2d -> BatchNorm -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # 3 x 3 kernel for convolution operation with padding 1 and stride 1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels) #  Max-pooling layer operation
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Ensure the dimensions match for concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Final convolution to produce output
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Full U-Net architecture for change detection
    Siamese U-Net
    """
    def __init__(self, n_channels=6, n_classes=1, bilinear=True):
        """
        Args:
            n_channels: Number of input channels. For change detection between two RGB images,
                        this would typically be 6 (3 channels for each image)
            n_classes: Number of output classes. For binary change detection (change/no change),
                       this would be 1.
            bilinear: Whether to use bilinear upsampling or transposed convolutions
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Downsampling path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Upsampling path with skip connections
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Final output convolution
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder / Downsampling path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder / Upsampling path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final layer
        logits = self.outc(x)
        return logits

# Initialize model function
def initialize_model(n_channels=6, n_classes=1, pretrained_path=None):
    """
    Initialize a U-Net model for change detection.
    
    Args:
        n_channels: Number of input channels (default 6 for a pair of RGB images)
        n_classes: Number of output classes (default 1 for binary change detection)
        pretrained_path: Path to load pretrained weights (optional)
        
    Returns:
        Initialized U-Net model
    """
    model = UNet(n_channels=n_channels, n_classes=n_classes)
    
    if pretrained_path and torch.cuda.is_available():
        model.load_state_dict(torch.load(pretrained_path))
    elif pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu'))) # use cuda for GPU Acceleration
    
    return model