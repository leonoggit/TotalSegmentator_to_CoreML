"""
Utility functions for TotalSegmentator to CoreML converter
"""

import logging
import colorlog
import torch
from pathlib import Path
from typing import Optional, Union, List
import json
import yaml
import numpy as np


def setup_logging(log_file: Optional[Union[str, Path]] = None, 
                  level: str = "INFO") -> logging.Logger:
    """Setup colored logging with optional file output"""
    
    # Create logger
    logger = logging.getLogger("totalsegmentator_coreml")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with color
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    console_format = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def check_gpu_available() -> bool:
    """Check if GPU is available for PyTorch"""
    
    if not torch.cuda.is_available():
        return False
    
    try:
        # Try to allocate a small tensor on GPU
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        return True
    except:
        return False


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string"""
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def load_config(config_path: Union[str, Path]) -> dict:
    """Load configuration from JSON or YAML file"""
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def save_config(config: dict, output_path: Union[str, Path]) -> None:
    """Save configuration to JSON or YAML file"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if output_path.suffix in ['.yaml', '.yml']:
            yaml.safe_dump(config, f, default_flow_style=False)
        else:
            json.dump(config, f, indent=2)


def get_model_info(model_path: Union[str, Path]) -> dict:
    """Extract information from a PyTorch model file"""
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Extract info
    info = {
        "file_size": format_size(model_path.stat().st_size),
        "num_parameters": len(state_dict),
        "total_params": sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)),
    }
    
    # Try to infer architecture
    layer_types = {}
    for key in state_dict.keys():
        layer_type = key.split('.')[0]
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    info["layer_summary"] = layer_types
    
    return info


def create_dummy_ct_volume(shape: tuple = (128, 128, 128),
                          spacing: tuple = (1.5, 1.5, 1.5)) -> np.ndarray:
    """Create a dummy CT volume for testing"""
    
    # Create base volume with noise
    volume = np.random.randn(*shape) * 100
    
    # Add some structure (sphere)
    center = [s // 2 for s in shape]
    radius = min(shape) // 4
    
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2 <= radius**2
    
    # Set sphere to bone density
    volume[mask] = 1000 + np.random.randn(np.sum(mask)) * 50
    
    # Set background to air/soft tissue
    volume[~mask] = -500 + np.random.randn(np.sum(~mask)) * 100
    
    # Clip to valid HU range
    volume = np.clip(volume, -1000, 3000)
    
    return volume.astype(np.float32)


def validate_input_shape(shape: tuple, model_name: str) -> bool:
    """Validate if input shape is compatible with model"""
    
    # Minimum requirements
    min_size = 64
    max_size = 512
    
    # Check dimensions
    if len(shape) not in [3, 5]:  # 3D or 5D (with batch and channel)
        return False
    
    # Get spatial dimensions
    if len(shape) == 5:
        spatial_dims = shape[2:]
    else:
        spatial_dims = shape
    
    # Check size constraints
    for dim in spatial_dims:
        if dim < min_size or dim > max_size:
            return False
    
    return True


def estimate_memory_usage(model_params: int, 
                         batch_size: int = 1,
                         input_shape: tuple = (128, 128, 128),
                         precision: str = "fp32") -> dict:
    """Estimate memory usage for model inference"""
    
    bytes_per_element = 4 if precision == "fp32" else 2
    
    # Model weights
    model_memory = model_params * bytes_per_element
    
    # Input tensor
    input_elements = batch_size * np.prod(input_shape)
    input_memory = input_elements * bytes_per_element
    
    # Rough estimate for activations (2x input size)
    activation_memory = input_memory * 2
    
    # Total
    total_memory = model_memory + input_memory + activation_memory
    
    return {
        "model_memory": format_size(model_memory),
        "input_memory": format_size(input_memory),
        "activation_memory": format_size(activation_memory),
        "total_memory": format_size(total_memory),
        "total_bytes": total_memory
    }


class ProgressTracker:
    """Track conversion progress across multiple models"""
    
    def __init__(self, total_models: int):
        self.total_models = total_models
        self.completed_models = 0
        self.failed_models = 0
        self.current_model = None
        self.start_time = None
        self.model_times = {}
    
    def start_model(self, model_name: str):
        """Start tracking a model"""
        self.current_model = model_name
        self.start_time = time.time()
    
    def complete_model(self, success: bool = True):
        """Mark current model as complete"""
        if self.current_model and self.start_time:
            elapsed = time.time() - self.start_time
            self.model_times[self.current_model] = elapsed
            
            if success:
                self.completed_models += 1
            else:
                self.failed_models += 1
    
    def get_summary(self) -> dict:
        """Get progress summary"""
        return {
            "total": self.total_models,
            "completed": self.completed_models,
            "failed": self.failed_models,
            "success_rate": self.completed_models / self.total_models if self.total_models > 0 else 0,
            "model_times": self.model_times,
            "average_time": np.mean(list(self.model_times.values())) if self.model_times else 0
        }


# Model architecture helpers
def create_totalsegmentator_model(num_classes: int, input_channels: int = 1):
    """Create a placeholder TotalSegmentator model architecture"""
    
    # This is a placeholder - in real implementation, you would load the actual architecture
    # For now, create a simple 3D U-Net-like structure
    
    import torch.nn as nn
    
    class Simple3DUNet(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv3d(in_channels, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.decoder = nn.Sequential(
                nn.Conv3d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, out_channels, 1),
            )
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    
    return Simple3DUNet(input_channels, num_classes)