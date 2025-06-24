# TotalSegmentator to CoreML Converter

Convert TotalSegmentator PyTorch models to CoreML format for iOS deployment.

## Overview

This project provides tools to convert TotalSegmentator medical image segmentation models from PyTorch to CoreML format, enabling deployment on iOS devices.

## Features

- Convert multiple TotalSegmentator models to CoreML
- Support for FP16 and FP32 precision
- Validation of converted models
- iOS integration examples
- GitHub Actions workflow for automated conversion

## Models Supported

- **body**: Full body segmentation (104 classes)
- **lung_vessels**: Lung vessel segmentation (6 classes)
- **cerebral_bleed**: Cerebral bleed detection (4 classes)
- **hip_implant**: Hip implant segmentation (2 classes)
- **coronary_arteries**: Coronary artery segmentation (3 classes)

## Requirements

- Python 3.9+
- PyTorch 2.1.0
- CoreMLTools 7.1
- macOS (for CoreML conversion)

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Command Line

Convert all models:
```bash
python convert_totalsegmentator.py --all --precision fp16 --validate
```

Convert specific model:
```bash
python convert_totalsegmentator.py --model body --precision fp32
```

### GitHub Actions

The project includes a GitHub Actions workflow for automated conversion:

1. Go to Actions tab
2. Select "Convert TotalSegmentator Models"
3. Click "Run workflow"
4. Choose model and precision
5. Download converted models from artifacts

## Project Structure

```
TotalSegmentator_to_CoreML/
├── src/                    # Source code
│   ├── converter.py       # Main conversion logic
│   ├── models.py          # Model architectures
│   ├── preprocessing.py   # Image preprocessing
│   └── validation.py      # Model validation
├── scripts/               # Utility scripts
├── examples/              # Usage examples
├── .github/workflows/     # GitHub Actions
└── convert_totalsegmentator.py  # Main CLI
```

## License

This project is licensed under the MIT License.