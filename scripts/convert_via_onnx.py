#!/usr/bin/env python3
"""
Convert models via ONNX to avoid MKL-DNN issues
"""

import os
import sys
import torch
import torch.onnx
import coremltools as ct
from pathlib import Path
import numpy as np

# Disable MKL
os.environ['PYTORCH_DISABLE_MKL'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_totalsegmentator_model

MODEL_CONFIGS = {
    "body": {"num_classes": 104},
    "lung_vessels": {"num_classes": 6},
    "cerebral_bleed": {"num_classes": 4},
    "hip_implant": {"num_classes": 2},
    "coronary_arteries": {"num_classes": 3}
}

def convert_model_via_onnx(model_name: str, pytorch_path: str, output_path: str):
    """Convert a model using ONNX as intermediate format"""
    
    print(f"Converting {model_name} via ONNX...")
    
    # Load model
    config = MODEL_CONFIGS[model_name]
    model = create_totalsegmentator_model(
        num_classes=config["num_classes"],
        input_channels=1
    )
    
    # Load weights
    state_dict = torch.load(pytorch_path, map_location='cpu')
    
    # Handle wrapped model state dict
    if any(k.startswith('model.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Create example input
    dummy_input = torch.randn(1, 1, 128, 128, 128, device='cpu')
    
    # Export to ONNX
    onnx_path = output_path.replace('.mlpackage', '.onnx')
    
    print(f"Exporting to ONNX: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,  # Use older opset to avoid newer operators
            do_constant_folding=True,
            input_names=['volume'],
            output_names=['segmentation'],
            dynamic_axes={
                'volume': {0: 'batch', 2: 'depth', 3: 'height', 4: 'width'},
                'segmentation': {0: 'batch', 2: 'depth', 3: 'height', 4: 'width'}
            },
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )
    
    print(f"Converting ONNX to CoreML...")
    
    # Convert ONNX to CoreML
    mlmodel = ct.convert(
        onnx_path,
        inputs=[ct.TensorType(
            name="volume",
            shape=(1, 1, ct.RangeDim(64, 512), ct.RangeDim(64, 512), ct.RangeDim(64, 512))
        )],
        outputs=[ct.TensorType(name="segmentation")],
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
    )
    
    # Add metadata
    mlmodel.short_description = f"TotalSegmentator {model_name}"
    mlmodel.author = "TotalSegmentator to CoreML Converter"
    mlmodel.version = "1.0.0"
    
    # Save CoreML model
    mlmodel.save(output_path)
    print(f"Saved CoreML model to {output_path}")
    
    # Clean up ONNX file
    Path(onnx_path).unlink(missing_ok=True)

def main():
    """Convert all models via ONNX"""
    
    pytorch_dir = Path("models/pytorch")
    coreml_dir = Path("models/coreml")
    coreml_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name in MODEL_CONFIGS:
        pytorch_path = pytorch_dir / f"{model_name}.pth"
        coreml_path = coreml_dir / f"{model_name}.mlpackage"
        
        if pytorch_path.exists():
            try:
                convert_model_via_onnx(model_name, str(pytorch_path), str(coreml_path))
                print(f"✓ Successfully converted {model_name}")
            except Exception as e:
                print(f"✗ Failed to convert {model_name}: {e}")
        else:
            print(f"- Skipping {model_name} (model not found)")

if __name__ == "__main__":
    main()