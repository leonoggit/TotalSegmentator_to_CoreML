#!/usr/bin/env python3
"""
Create dummy TotalSegmentator models for testing
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_totalsegmentator_model


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
        
        # Save state dict
        torch.save(model.state_dict(), output_path)
        
        print(f"Saved {output_path}")
    
    print("Dummy models created successfully!")


if __name__ == "__main__":
    output_dir = Path("models/pytorch")
    create_dummy_models(output_dir)