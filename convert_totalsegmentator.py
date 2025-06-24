#!/usr/bin/env python3
"""
TotalSegmentator to CoreML Converter
Main conversion script with CLI interface
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import torch
import numpy as np
from tqdm import tqdm
import coremltools as ct
from coremltools.models.neural_network import flexible_shape_utils

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.converter import TotalSegmentatorConverter
from src.preprocessing import MedicalImagePreprocessor
from src.validation import ModelValidator
from src.utils import setup_logging, check_gpu_available, format_size


# Model configurations
MODEL_CONFIGS = {
    "body": {
        "input_shape": (1, 1, 128, 128, 128),
        "num_classes": 104,
        "description": "Major organs and structures"
    },
    "lung_vessels": {
        "input_shape": (1, 1, 128, 128, 128),
        "num_classes": 6,
        "description": "Pulmonary vasculature"
    },
    "cerebral_bleed": {
        "input_shape": (1, 1, 128, 128, 128),
        "num_classes": 4,
        "description": "Brain hemorrhage detection"
    },
    "hip_implant": {
        "input_shape": (1, 1, 128, 128, 128),
        "num_classes": 2,
        "description": "Orthopedic implant segmentation"
    },
    "coronary_arteries": {
        "input_shape": (1, 1, 128, 128, 128),
        "num_classes": 3,
        "description": "Cardiac vessel segmentation"
    }
}


class ConversionPipeline:
    """Main pipeline for converting TotalSegmentator models"""
    
    def __init__(self, 
                 use_gpu: bool = True,
                 precision: str = "fp16",
                 batch_size: int = 1,
                 log_dir: str = "logs",
                 cache_dir: str = ".cache",
                 log_level: str = "INFO"):
        
        self.use_gpu = use_gpu and check_gpu_available()
        self.precision = precision
        self.batch_size = batch_size
        self.log_dir = Path(log_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories
        self.log_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.log_dir / "conversion.log", level=log_level)
        
        # Initialize components
        self.converter = TotalSegmentatorConverter(
            use_gpu=self.use_gpu,
            precision=precision
        )
        self.preprocessor = MedicalImagePreprocessor()
        self.validator = ModelValidator()
        
        # Device setup
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.logger.info(f"Using device: {self.device}")
    
    def convert_model(self,
                     model_name: str,
                     pytorch_path: str,
                     output_path: str,
                     validate: bool = True,
                     optimize: bool = True) -> Optional[ct.models.MLModel]:
        """Convert a single TotalSegmentator model"""
        
        self.logger.info(f"Converting model: {model_name}")
        start_time = time.time()
        
        try:
            # Validate model name
            if model_name not in MODEL_CONFIGS:
                raise ValueError(f"Unknown model: {model_name}")
            
            config = MODEL_CONFIGS[model_name]
            
            # Load PyTorch model
            self.logger.info(f"Loading PyTorch model from {pytorch_path}")
            pytorch_model = self._load_pytorch_model(pytorch_path, config)
            
            # Prepare example input
            example_input = self._create_example_input(config["input_shape"])
            
            # Convert to CoreML using the converter (handles tracing, optimization, and saving)
            coreml_model = self.converter.convert(
                model_name=model_name,
                pytorch_model=pytorch_model,
                example_input=example_input,
                output_path=output_path,
                optimize=optimize
            )
            
            # Validate if requested
            if validate:
                self.logger.info("Validating conversion")
                validation_results = self._validate_conversion(
                    pytorch_model,
                    coreml_model,
                    config
                )
                self._log_validation_results(model_name, validation_results)
            
            # Log summary
            elapsed_time = time.time() - start_time
            pytorch_size = os.path.getsize(pytorch_path)
            coreml_size = os.path.getsize(output_path)
            
            self.logger.info(f"Conversion completed in {elapsed_time:.1f}s")
            self.logger.info(f"PyTorch size: {format_size(pytorch_size)}")
            self.logger.info(f"CoreML size: {format_size(coreml_size)}")
            self.logger.info(f"Compression ratio: {pytorch_size/coreml_size:.2f}x")
            
            return coreml_model
            
        except Exception as e:
            self.logger.error(f"Conversion failed for {model_name}: {str(e)}")
            raise
    
    def convert_all_models(self,
                          input_dir: str,
                          output_dir: str,
                          models: Optional[List[str]] = None,
                          validate: bool = True,
                          optimize: bool = True,
                          parallel: bool = False) -> Dict[str, bool]:
        """Convert all TotalSegmentator models"""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine models to convert
        if models is None:
            models = list(MODEL_CONFIGS.keys())
        
        results = {}
        
        self.logger.info(f"Converting {len(models)} models")
        
        # Convert models
        for model_name in tqdm(models, desc="Converting models"):
            pytorch_path = input_path / f"{model_name}.pth"
            coreml_path = output_path / f"{model_name}.mlpackage"
            
            if not pytorch_path.exists():
                self.logger.warning(f"Model file not found: {pytorch_path}")
                results[model_name] = False
                continue
            
            try:
                self.convert_model(
                    model_name=model_name,
                    pytorch_path=str(pytorch_path),
                    output_path=str(coreml_path),
                    validate=validate,
                    optimize=optimize
                )
                results[model_name] = True
            except Exception as e:
                self.logger.error(f"Failed to convert {model_name}: {e}")
                results[model_name] = False
        
        # Summary
        successful = sum(results.values())
        self.logger.info(f"Conversion summary: {successful}/{len(models)} successful")
        
        # Save results
        results_path = output_path / "conversion_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _load_pytorch_model(self, path: str, config: Dict) -> torch.nn.Module:
        """Load PyTorch model with proper configuration"""
        
        # Load state dict
        state_dict = torch.load(path, map_location=self.device)
        
        # Create model architecture (placeholder - implement actual architecture)
        # In real implementation, this would instantiate the correct U-Net variant
        from src.models import create_totalsegmentator_model
        model = create_totalsegmentator_model(
            num_classes=config["num_classes"],
            input_channels=1
        )
        
        # Load weights
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        return model
    
    def _create_example_input(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Create example input for tracing"""
        
        # Create realistic CT data (-1000 to 1000 HU)
        data = torch.randn(shape, device=self.device) * 500
        data = torch.clamp(data, -1000, 1000)
        
        # Normalize
        data = (data + 1000) / 2000  # Scale to [0, 1]
        
        return data
    
    def _convert_to_coreml(self, 
                          traced_model: torch.jit.ScriptModule,
                          config: Dict,
                          model_name: str) -> ct.models.MLModel:
        """Convert traced model to CoreML"""
        
        # Define input shape with flexible dimensions
        input_shape = ct.Shape(
            shape=(1, 1, ct.RangeDim(64, 512), ct.RangeDim(64, 512), ct.RangeDim(64, 512))
        )
        
        # Conversion configuration
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=input_shape, name="volume")],
            outputs=[ct.TensorType(name="segmentation")],
            compute_precision=ct.precision.FLOAT16 if self.precision == "fp16" else ct.precision.FLOAT32,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15,
        )
        
        # Add metadata
        mlmodel.short_description = f"TotalSegmentator {model_name}"
        mlmodel.long_description = MODEL_CONFIGS[model_name]["description"]
        mlmodel.author = "TotalSegmentator to CoreML Converter"
        mlmodel.version = "1.0.0"
        
        # Add preprocessing info
        mlmodel.user_defined_metadata["preprocessing"] = json.dumps({
            "window_center": 0,
            "window_width": 2000,
            "spacing": [1.5, 1.5, 1.5],
            "normalization": "min_max"
        })
        
        return mlmodel
    
    def _optimize_model(self, model: ct.models.MLModel) -> ct.models.MLModel:
        """Optimize CoreML model for deployment"""
        
        # Apply optimizations
        # Note: In real implementation, would apply various CoreML optimizations
        # such as weight pruning, quantization, etc.
        
        return model
    
    def _validate_conversion(self,
                           pytorch_model: torch.nn.Module,
                           coreml_model: ct.models.MLModel,
                           config: Dict) -> Dict[str, float]:
        """Validate conversion accuracy"""
        
        # Create test input
        test_shape = config["input_shape"]
        test_input = self._create_example_input(test_shape)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)
            pytorch_output = pytorch_output.cpu().numpy()
        
        # CoreML inference
        coreml_input = {"volume": test_input.cpu().numpy()}
        coreml_output = coreml_model.predict(coreml_input)["segmentation"]
        
        # Calculate metrics
        dice_score = self.validator.calculate_dice_score(
            pytorch_output, 
            coreml_output
        )
        
        max_diff = np.max(np.abs(pytorch_output - coreml_output))
        mean_diff = np.mean(np.abs(pytorch_output - coreml_output))
        
        return {
            "dice_score": dice_score,
            "max_difference": max_diff,
            "mean_difference": mean_diff
        }
    
    def _log_validation_results(self, model_name: str, results: Dict[str, float]):
        """Log validation results"""
        
        self.logger.info(f"Validation results for {model_name}:")
        self.logger.info(f"  Dice score: {results['dice_score']:.4f}")
        self.logger.info(f"  Max difference: {results['max_difference']:.6f}")
        self.logger.info(f"  Mean difference: {results['mean_difference']:.6f}")
        
        # Warn if accuracy is too low
        if results['dice_score'] < 0.99:
            self.logger.warning(f"Low dice score for {model_name}: {results['dice_score']:.4f}")


def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="Convert TotalSegmentator PyTorch models to CoreML"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Specific model to convert"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all models"
    )
    
    # Paths
    parser.add_argument(
        "--input",
        type=str,
        default="models/pytorch",
        help="Input directory containing PyTorch models"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/coreml",
        help="Output directory for CoreML models"
    )
    
    # Conversion options
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU-only conversion"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="Model precision"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for conversion"
    )
    
    # Features
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate conversion accuracy"
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip validation"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=True,
        help="Optimize CoreML models"
    )
    parser.add_argument(
        "--no-optimize",
        dest="optimize",
        action="store_false",
        help="Skip optimization"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model and not args.all:
        parser.error("Either --model or --all must be specified")
    
    # Create pipeline
    pipeline = ConversionPipeline(
        use_gpu=args.gpu and not args.cpu_only,
        precision=args.precision,
        batch_size=args.batch_size,
        log_level=args.log_level
    )
    
    # Run conversion
    if args.all:
        results = pipeline.convert_all_models(
            input_dir=args.input,
            output_dir=args.output,
            validate=args.validate,
            optimize=args.optimize
        )
        
        # Exit with error if any conversion failed
        if not all(results.values()):
            sys.exit(1)
    else:
        # Single model conversion
        pytorch_path = os.path.join(args.input, f"{args.model}.pth")
        coreml_path = os.path.join(args.output, f"{args.model}.mlpackage")
        
        pipeline.convert_model(
            model_name=args.model,
            pytorch_path=pytorch_path,
            output_path=coreml_path,
            validate=args.validate,
            optimize=args.optimize
        )


if __name__ == "__main__":
    main()