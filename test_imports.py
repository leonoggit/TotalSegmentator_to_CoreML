#!/usr/bin/env python3
"""
Test script to verify module imports are working correctly
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    # Test individual module imports
    print("1. Testing src module...")
    import src
    print("   ✓ src module imported successfully")
    
    print("2. Testing src.models...")
    from src.models import create_totalsegmentator_model
    print("   ✓ src.models imported successfully")
    
    print("3. Testing src.converter...")
    from src.converter import TotalSegmentatorConverter
    print("   ✓ src.converter imported successfully")
    
    print("4. Testing src.preprocessing...")
    from src.preprocessing import MedicalImagePreprocessor
    print("   ✓ src.preprocessing imported successfully")
    
    print("5. Testing src.validation...")
    from src.validation import ModelValidator
    print("   ✓ src.validation imported successfully")
    
    print("6. Testing src.utils...")
    from src.utils import setup_logging, check_gpu_available
    print("   ✓ src.utils imported successfully")
    
    print("\nAll imports successful! The module structure is correct.")
    
    # Test creating a model
    print("\n7. Testing model creation...")
    model = create_totalsegmentator_model(num_classes=104)
    print(f"   ✓ Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    print("\nMake sure you have installed all requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Unexpected error: {e}")
    sys.exit(1)