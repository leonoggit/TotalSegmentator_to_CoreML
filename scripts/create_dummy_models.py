#!/usr/bin/env python3
"""
Create dummy TotalSegmentator models for testing
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path to import from src
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import directly from the module file to avoid __init__.py imports
import importlib.util
spec = importlib.util.spec_from_file_location("models", parent_dir / "src" / "models.py")
models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_module)
create_totalsegmentator_model = models_module.create_totalsegmentator_model


MODEL_CONFIGS = {
    "body": {"num_classes": 104},
    "lung_vessels": {"num_classes": 6},
    "cerebral_bleed": {"num_classes": 4},
    "hip_implant": {"num_classes": 2},
    "coronary_arteries": {"num_classes": 3}
}


def create_dummy_models(output_dir: Path):
    """Create dummy models for testing when real models aren't available"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating dummy TotalSegmentator models for testing...")
    
    for model_name, config in MODEL_CONFIGS.items():
        output_path = output_dir / f"{model_name}.pth"
        
        if output_path.exists():
            print(f"{model_name}.pth already exists, skipping...")
            continue
        
        print(f"Creating {model_name} model...")
        
        # Create model
        model = create_totalsegmentator_model(
            num_classes=config["num_classes"],
            input_channels=1
        )
        
        # Initialize with random weights
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, -0.1, 0.1)
        
        # Ensure model is in CPU mode and not using MKL-DNN
        model = model.cpu()
        model.eval()
        
        # Save state dict with _use_new_zipfile_serialization for compatibility
        torch.save(
            model.state_dict(), 
            output_path, 
            _use_new_zipfile_serialization=False
        )
        
        print(f"Saved {output_path}")
    
    print("Dummy models created successfully!")


if __name__ == "__main__":
    output_dir = Path("models/pytorch")
    create_dummy_models(output_dir)