#!/usr/bin/env python3
"""
Install PyTorch without MKL-DNN support
"""

import subprocess
import sys
import os

def install_pytorch_without_mkl():
    """Install PyTorch CPU version without MKL-DNN"""
    
    # Set environment variables
    os.environ['PYTORCH_DISABLE_MKL'] = '1'
    os.environ['USE_MKLDNN'] = '0'
    os.environ['USE_MKL'] = '0'
    
    # Uninstall existing torch if present
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision"], 
                   capture_output=True)
    
    # Install CPU-only version without MKL
    print("Installing PyTorch CPU version without MKL-DNN...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.0+cpu", "torchvision==0.16.0+cpu",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error installing PyTorch: {result.stderr}")
        sys.exit(1)
    
    print("PyTorch installed successfully")
    
    # Verify MKL-DNN is disabled
    try:
        import torch
        if hasattr(torch._C, '_get_mkldnn_enabled'):
            mkldnn_enabled = torch._C._get_mkldnn_enabled()
            print(f"MKL-DNN enabled: {mkldnn_enabled}")
            if mkldnn_enabled:
                print("WARNING: MKL-DNN is still enabled!")
        else:
            print("MKL-DNN status check not available")
    except Exception as e:
        print(f"Error checking MKL-DNN status: {e}")

if __name__ == "__main__":
    install_pytorch_without_mkl()