#!/usr/bin/env python3
"""
Minimal test for CoreML conversion
"""

import os
import sys
import torch
import torch.nn as nn
import coremltools as ct

# Disable MKL
os.environ['PYTORCH_DISABLE_MKL'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

if hasattr(torch._C, '_set_mkldnn_enabled'):
    torch._C._set_mkldnn_enabled(False)

# Simple 3D Conv model
class Simple3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(8, 1, 3, padding=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)

print("Creating simple 3D model...")
model = Simple3DModel().cpu().eval()

# Test input
x = torch.randn(1, 1, 32, 32, 32).cpu()

print("Testing forward pass...")
with torch.no_grad():
    output = model(x)
print(f"Output shape: {output.shape}")

print("\nTracing model...")
with torch.no_grad():
    traced = torch.jit.trace(model, x, strict=False)

print("\nConverting to CoreML...")
try:
    # Direct conversion
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 1, 32, 32, 32))],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
    )
    print("✓ Direct conversion successful!")
    
except Exception as e:
    print(f"✗ Direct conversion failed: {e}")
    
    # Try ONNX path
    print("\nTrying ONNX path...")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        torch.onnx.export(
            model,
            x,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=11
        )
        
        mlmodel = ct.convert(
            onnx_path,
            inputs=[ct.TensorType(name="input", shape=(1, 1, 32, 32, 32))],
            outputs=[ct.TensorType(name="output")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15,
        )
        print("✓ ONNX conversion successful!")
        
        # Clean up
        os.unlink(onnx_path)
        
    except Exception as e2:
        print(f"✗ ONNX conversion also failed: {e2}")
        sys.exit(1)

print("\nConversion completed successfully!")