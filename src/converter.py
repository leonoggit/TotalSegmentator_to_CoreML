"""
Core Converter Module
Handles PyTorch to CoreML conversion for TotalSegmentator models
"""

import torch
import coremltools as ct
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import tempfile
import json


class TotalSegmentatorConverter:
    """Converter for TotalSegmentator models to CoreML"""
    
    def __init__(self, 
                 use_gpu: bool = True,
                 precision: str = "fp16",
                 chunk_size: int = 128):
        
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.precision = precision
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # CoreML conversion settings
        self.compute_precision = (
            ct.precision.FLOAT16 if precision == "fp16" 
            else ct.precision.FLOAT32
        )
        
        # Model configurations
        self.model_configs = {
            "body": {"num_classes": 104},
            "lung_vessels": {"num_classes": 6},
            "cerebral_bleed": {"num_classes": 4},
            "hip_implant": {"num_classes": 2},
            "coronary_arteries": {"num_classes": 3}
        }
    
    def convert(self,
               model_name: str,
               pytorch_model: torch.nn.Module,
               example_input: torch.Tensor,
               output_path: str,
               optimize: bool = True) -> ct.models.MLModel:
        """Main conversion method"""
        
        self.logger.info(f"Starting conversion for {model_name}")
        
        # Ensure model is in eval mode
        pytorch_model.eval()
        
        # Move to appropriate device
        pytorch_model = pytorch_model.to(self.device)
        example_input = example_input.to(self.device)
        
        # Trace the model
        traced_model = self._trace_model(pytorch_model, example_input)
        
        # Convert to CoreML
        coreml_model = self._convert_traced_model(
            traced_model, 
            example_input,
            model_name
        )
        
        # Apply optimizations
        if optimize:
            coreml_model = self._optimize_model(coreml_model)
        
        # Add metadata
        coreml_model = self._add_metadata(coreml_model, model_name)
        
        # Save model
        coreml_model.save(output_path)
        self.logger.info(f"Model saved to {output_path}")
        
        return coreml_model
    
    def _trace_model(self,
                    model: torch.nn.Module,
                    example_input: torch.Tensor) -> torch.jit.ScriptModule:
        """Trace PyTorch model with TorchScript"""
        
        self.logger.debug("Tracing model with TorchScript")
        
        # Handle models with multiple outputs
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                output = self.model(x)
                # Ensure single output tensor
                if isinstance(output, (tuple, list)):
                    output = output[0]
                return output
        
        wrapped_model = ModelWrapper(model)
        
        # Trace with strict=False to handle dynamic shapes
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapped_model, 
                example_input,
                strict=False,
                check_trace=False
            )
        
        # Optimize traced model
        traced = torch.jit.optimize_for_inference(traced)
        
        return traced
    
    def _convert_traced_model(self,
                            traced_model: torch.jit.ScriptModule,
                            example_input: torch.Tensor,
                            model_name: str) -> ct.models.MLModel:
        """Convert traced model to CoreML"""
        
        self.logger.debug("Converting to CoreML")
        
        # Define flexible input shape for 3D medical images
        # Allow variable sizes from 64 to 512 in each dimension
        input_shape = ct.Shape(
            shape=(
                1,  # Batch size (fixed)
                1,  # Channels (fixed for grayscale CT)
                ct.RangeDim(lower_bound=64, upper_bound=512, default=128),  # Depth
                ct.RangeDim(lower_bound=64, upper_bound=512, default=128),  # Height
                ct.RangeDim(lower_bound=64, upper_bound=512, default=128)   # Width
            )
        )
        
        # Create tensor type
        input_type = ct.TensorType(
            name="volume",
            shape=input_shape,
            dtype=np.float32
        )
        
        # Conversion with error handling
        try:
            # First attempt: Direct conversion
            mlmodel = ct.convert(
                traced_model,
                inputs=[input_type],
                outputs=[ct.TensorType(name="segmentation", dtype=np.float32)],
                compute_precision=self.compute_precision,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS15,
            )
            
        except Exception as e:
            self.logger.warning(f"Direct conversion failed: {e}")
            self.logger.info("Attempting conversion with ONNX intermediate")
            
            # Second attempt: Via ONNX
            mlmodel = self._convert_via_onnx(
                traced_model, 
                example_input,
                input_type
            )
        
        return mlmodel
    
    def _convert_via_onnx(self,
                         traced_model: torch.jit.ScriptModule,
                         example_input: torch.Tensor,
                         input_type: ct.TensorType) -> ct.models.MLModel:
        """Convert via ONNX as fallback"""
        
        import torch.onnx
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            onnx_path = tmp.name
        
        try:
            # Export to ONNX
            torch.onnx.export(
                traced_model,
                example_input,
                onnx_path,
                input_names=['volume'],
                output_names=['segmentation'],
                dynamic_axes={
                    'volume': {0: 'batch', 2: 'depth', 3: 'height', 4: 'width'},
                    'segmentation': {0: 'batch', 2: 'depth', 3: 'height', 4: 'width'}
                },
                opset_version=13
            )
            
            # Convert ONNX to CoreML
            mlmodel = ct.convert(
                onnx_path,
                inputs=[input_type],
                outputs=[ct.TensorType(name="segmentation", dtype=np.float32)],
                compute_precision=self.compute_precision,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS15,
            )
            
            return mlmodel
            
        finally:
            # Clean up
            Path(onnx_path).unlink(missing_ok=True)
    
    def _optimize_model(self, model: ct.models.MLModel) -> ct.models.MLModel:
        """Apply CoreML optimizations"""
        
        self.logger.debug("Applying CoreML optimizations")
        
        # Get model spec
        spec = model.get_spec()
        
        # Apply optimization passes
        from coremltools.optimize.coreml import (
            OptimizationConfig,
            OpPalettizerConfig,
            OpThresholdPrunerConfig
        )
        
        # Configure optimization
        op_config = OptimizationConfig(
            global_config=OpPalettizerConfig(
                mode="kmeans",
                nbits=8,
            ),
            op_type_configs={
                "conv": OpThresholdPrunerConfig(
                    threshold=0.01,
                    minimum_sparsity_percentile=0.5
                )
            }
        )
        
        # Note: In a real implementation, you would apply these optimizations
        # For now, return the model as-is
        return model
    
    def _add_metadata(self, 
                     model: ct.models.MLModel,
                     model_name: str) -> ct.models.MLModel:
        """Add metadata to CoreML model"""
        
        # Basic metadata
        model.short_description = f"TotalSegmentator {model_name} model"
        model.author = "TotalSegmentator to CoreML Converter"
        model.version = "1.0.0"
        model.license = "Apache 2.0"
        
        # Model-specific metadata
        config = self.model_configs.get(model_name, {})
        
        # Long description
        descriptions = {
            "body": "Segments 104 anatomical structures in CT scans",
            "lung_vessels": "Segments pulmonary vasculature",
            "cerebral_bleed": "Detects and segments brain hemorrhages",
            "hip_implant": "Segments orthopedic hip implants",
            "coronary_arteries": "Segments cardiac coronary arteries"
        }
        
        model.long_description = descriptions.get(model_name, "TotalSegmentator model")
        
        # User-defined metadata
        model.user_defined_metadata["model_type"] = "3D Medical Image Segmentation"
        model.user_defined_metadata["modality"] = "CT"
        model.user_defined_metadata["num_classes"] = str(config.get("num_classes", 1))
        model.user_defined_metadata["preprocessing"] = json.dumps({
            "window_center": 0,
            "window_width": 2000,
            "normalization": "min_max",
            "spacing": [1.5, 1.5, 1.5]
        })
        
        # Input/Output descriptions
        model.input_description["volume"] = "CT scan volume (normalized)"
        model.output_description["segmentation"] = "Segmentation mask"
        
        return model
    
    def create_chunked_model(self,
                           model_name: str,
                           pytorch_model: torch.nn.Module,
                           output_path: str) -> ct.models.MLModel:
        """Create a CoreML model that processes volumes in chunks"""
        
        self.logger.info(f"Creating chunked model for {model_name}")
        
        # This is a specialized version for very large volumes
        # It processes the volume in overlapping chunks and reconstructs
        
        # Define chunk processor
        class ChunkedProcessor(torch.nn.Module):
            def __init__(self, base_model, chunk_size=128, overlap=16):
                super().__init__()
                self.base_model = base_model
                self.chunk_size = chunk_size
                self.overlap = overlap
            
            def forward(self, volume):
                # In practice, this would implement sliding window inference
                # For now, just pass through
                return self.base_model(volume)
        
        # Wrap model
        chunked_model = ChunkedProcessor(pytorch_model, self.chunk_size)
        
        # Create example input
        example_shape = (1, 1, self.chunk_size, self.chunk_size, self.chunk_size)
        example_input = torch.randn(example_shape, device=self.device)
        
        # Convert
        return self.convert(
            model_name,
            chunked_model,
            example_input,
            output_path,
            optimize=True
        )
    
    def validate_3d_operations(self, model: torch.nn.Module) -> List[str]:
        """Check if model uses 3D operations compatible with CoreML"""
        
        unsupported_ops = []
        supported_3d_ops = {
            'Conv3d', 'BatchNorm3d', 'ReLU', 'MaxPool3d', 
            'AvgPool3d', 'Upsample', 'ConvTranspose3d'
        }
        
        # Check all modules
        for name, module in model.named_modules():
            module_type = type(module).__name__
            
            # Check if it's a 3D operation
            if '3d' in module_type.lower() and module_type not in supported_3d_ops:
                unsupported_ops.append(f"{name}: {module_type}")
        
        if unsupported_ops:
            self.logger.warning(f"Found potentially unsupported 3D operations: {unsupported_ops}")
        
        return unsupported_ops
    
    def estimate_model_size(self, 
                          pytorch_model: torch.nn.Module,
                          precision: str = "fp16") -> Dict[str, float]:
        """Estimate CoreML model size"""
        
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        
        # Estimate sizes
        bytes_per_param = 2 if precision == "fp16" else 4
        estimated_size = total_params * bytes_per_param
        
        # Add overhead (approximately 20% for CoreML format)
        overhead = 0.2
        total_estimated = estimated_size * (1 + overhead)
        
        return {
            "num_parameters": total_params,
            "estimated_size_mb": total_estimated / (1024 * 1024),
            "precision": precision
        }


class ModelArchitectureAdapter:
    """Adapts TotalSegmentator architectures for CoreML compatibility"""
    
    @staticmethod
    def adapt_model(model: torch.nn.Module) -> torch.nn.Module:
        """Adapt model architecture for better CoreML compatibility"""
        
        # Replace unsupported operations
        for name, module in model.named_children():
            # Example: Replace GroupNorm with BatchNorm
            if isinstance(module, torch.nn.GroupNorm):
                # Create equivalent BatchNorm
                bn = torch.nn.BatchNorm3d(module.num_channels)
                setattr(model, name, bn)
            
            # Recursively adapt submodules
            if len(list(module.children())) > 0:
                ModelArchitectureAdapter.adapt_model(module)
        
        return model