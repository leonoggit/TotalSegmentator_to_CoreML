#!/usr/bin/env python3
"""
Test if MKL-DNN is properly disabled
"""

import os
import sys

# Set environment variables before importing torch
os.environ['PYTORCH_DISABLE_MKL'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn

# Check MKL-DNN status
print("Testing MKL-DNN configuration...")
print(f"PyTorch version: {torch.__version__}")

if hasattr(torch._C, '_get_mkldnn_enabled'):
    mkldnn_enabled = torch._C._get_mkldnn_enabled()
    print(f"MKL-DNN enabled: {mkldnn_enabled}")
else:
    print("MKL-DNN status check not available in this PyTorch version")

if hasattr(torch._C, '_set_mkldnn_enabled'):
    torch._C._set_mkldnn_enabled(False)
    print("MKL-DNN disabled via _set_mkldnn_enabled")

# Test with a simple model
print("\nTesting simple model creation and tracing...")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.conv(x))

try:
    # Create model
    model = SimpleModel().cpu()
    model.eval()
    
    # Create input
    x = torch.randn(1, 1, 32, 32, 32).cpu()
    
    # Test forward pass
    with torch.no_grad():
        output = model(x)
    print("✓ Forward pass successful")
    
    # Test tracing
    with torch.no_grad():
        traced = torch.jit.trace(model, x)
    print("✓ Tracing successful")
    
    # Check traced graph for MKL-DNN ops
    graph_str = str(traced.graph)
    if 'mkldnn' in graph_str.lower():
        print("✗ WARNING: Found MKL-DNN operations in traced graph!")
    else:
        print("✓ No MKL-DNN operations found in traced graph")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("\nTest completed successfully!")