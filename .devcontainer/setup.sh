#!/bin/bash
# Setup script for TotalSegmentator to CoreML development environment

set -e

echo "Setting up TotalSegmentator to CoreML environment..."

# Update system packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0

# Create necessary directories
mkdir -p models/pytorch models/coreml data logs .cache

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install project dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e .

# Download TotalSegmentator models (optional - requires authentication)
if [ -f "scripts/download_models.py" ]; then
    echo "To download TotalSegmentator models, run:"
    echo "python scripts/download_models.py"
fi

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Setup Jupyter kernel
python -m ipykernel install --user --name totalsegmentator --display-name "TotalSegmentator"

echo "Environment setup complete!"
echo ""
echo "Quick start:"
echo "1. Download models: python scripts/download_models.py"
echo "2. Convert all models: python convert_totalsegmentator.py --all --gpu"
echo "3. Run tests: python -m pytest tests/"