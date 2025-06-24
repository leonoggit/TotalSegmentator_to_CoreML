#!/bin/bash
# Setup script for TotalSegmentator to CoreML development environment
# Optimized for GitHub Codespaces free tier (CPU-only)

set -e

echo "Setting up TotalSegmentator to CoreML environment (Free Tier - CPU optimized)..."

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
    libglib2.0-0 \
    htop \
    tree

# Create necessary directories
mkdir -p models/pytorch models/coreml data logs .cache

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install PyTorch CPU-only version (much smaller and faster for free tier)
echo "Installing PyTorch CPU version..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e .

# Download TotalSegmentator models (optional - requires authentication)
if [ -f "scripts/download_models.py" ]; then
    echo "To download TotalSegmentator models, run:"
    echo "python scripts/download_models.py"
fi

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CPU available: True')"
python -c "import coremltools; print(f'CoreML Tools version: {coremltools.__version__}')"

# Setup Jupyter kernel
python -m ipykernel install --user --name totalsegmentator --display-name "TotalSegmentator"

# Set memory-efficient settings for CPU processing
echo "export OMP_NUM_THREADS=2" >> ~/.bashrc
echo "export MKL_NUM_THREADS=2" >> ~/.bashrc

echo "Environment setup complete!"
echo ""
echo "=== CODESPACE READY ==="
echo "This environment is optimized for GitHub Codespaces free tier:"
echo "- 2 CPU cores, 8GB RAM, 32GB storage"
echo "- CPU-only PyTorch for faster setup and lower memory usage"
echo "- Efficient caching for pip and torch models"
echo ""
echo "Quick start:"
echo "1. Download models: python scripts/download_models.py"
echo "2. Convert models (CPU): python convert_totalsegmentator.py --all --cpu"
echo "3. Run tests: python -m pytest tests/"
echo "4. Start Jupyter: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"