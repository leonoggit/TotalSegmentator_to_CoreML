"""
Model architectures for TotalSegmentator
Implements U-Net based architectures used by TotalSegmentator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvBlock(nn.Module):
    """Basic convolution block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    """Downsampling block with max pooling"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2, stride=2)
    
    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connection"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatches - always calculate padding for TorchScript compatibility
        # Calculate differences (will be 0 if sizes match)
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        
        # Apply padding if needed (padding by 0 has no effect)
        x = F.pad(x, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2
        ])
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet3D(nn.Module):
    """3D U-Net architecture for medical image segmentation"""
    
    def __init__(self, 
                 in_channels: int = 1,
                 num_classes: int = 104,
                 features: List[int] = None):
        super().__init__()
        
        if features is None:
            features = [32, 64, 128, 256, 512]
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Initial convolution
        self.init_conv = ConvBlock(in_channels, features[0])
        
        # Encoder
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])
        self.down4 = DownBlock(features[3], features[4])
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[4], features[4] * 2)
        
        # Decoder
        self.up1 = UpBlock(features[4] * 2, features[4], features[3])
        self.up2 = UpBlock(features[3], features[3], features[2])
        self.up3 = UpBlock(features[2], features[2], features[1])
        self.up4 = UpBlock(features[1], features[1], features[0])
        
        # Final convolution
        self.final_conv = nn.Conv3d(features[0], num_classes, 1)
        
        # Apply softmax for multi-class segmentation
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        
        # Final output
        x = self.final_conv(x)
        
        # Apply softmax for inference (not during training)
        if not self.training:
            x = self.softmax(x)
        
        return x


class CompactUNet3D(nn.Module):
    """Compact 3D U-Net for smaller models (lung_vessels, cerebral_bleed, etc.)"""
    
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 6,
                 features: List[int] = None):
        super().__init__()
        
        if features is None:
            features = [16, 32, 64, 128]
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Initial convolution
        self.init_conv = ConvBlock(in_channels, features[0])
        
        # Encoder
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3] * 2)
        
        # Decoder
        self.up1 = UpBlock(features[3] * 2, features[3], features[2])
        self.up2 = UpBlock(features[2], features[2], features[1])
        self.up3 = UpBlock(features[1], features[1], features[0])
        
        # Final convolution
        self.final_conv = nn.Conv3d(features[0], num_classes, 1)
        
        # Softmax for multi-class
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        
        # Final output
        x = self.final_conv(x)
        
        # Apply softmax for inference
        if not self.training:
            x = self.softmax(x)
        
        return x


def create_totalsegmentator_model(num_classes: int, 
                                 input_channels: int = 1,
                                 model_size: str = "auto") -> nn.Module:
    """
    Create a TotalSegmentator model based on number of classes
    
    Args:
        num_classes: Number of segmentation classes
        input_channels: Number of input channels (default: 1 for CT)
        model_size: Model size ("large", "compact", or "auto")
    
    Returns:
        PyTorch model instance
    """
    
    # Auto-determine model size based on number of classes
    if model_size == "auto":
        if num_classes > 10:
            model_size = "large"
        else:
            model_size = "compact"
    
    if model_size == "large":
        # Use full U-Net for body model with many classes
        model = UNet3D(
            in_channels=input_channels,
            num_classes=num_classes,
            features=[32, 64, 128, 256, 512]
        )
    else:
        # Use compact U-Net for specialized models
        model = CompactUNet3D(
            in_channels=input_channels,
            num_classes=num_classes,
            features=[16, 32, 64, 128]
        )
    
    return model


def load_pretrained_model(model_name: str, 
                         checkpoint_path: str,
                         device: str = "cpu") -> nn.Module:
    """
    Load a pretrained TotalSegmentator model
    
    Args:
        model_name: Name of the model (e.g., "body", "lung_vessels")
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    
    # Model configurations
    configs = {
        "body": {"num_classes": 104, "model_size": "large"},
        "lung_vessels": {"num_classes": 6, "model_size": "compact"},
        "cerebral_bleed": {"num_classes": 4, "model_size": "compact"},
        "hip_implant": {"num_classes": 2, "model_size": "compact"},
        "coronary_arteries": {"num_classes": 3, "model_size": "compact"}
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = configs[model_name]
    
    # Create model
    model = create_totalsegmentator_model(
        num_classes=config["num_classes"],
        model_size=config["model_size"]
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model