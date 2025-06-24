"""
Model Validation Module
Validates conversion accuracy between PyTorch and CoreML models
"""

import numpy as np
import torch
import coremltools as ct
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time


class ModelValidator:
    """Validates model conversion accuracy"""
    
    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
    
    def calculate_dice_score(self, 
                           pred1: np.ndarray, 
                           pred2: np.ndarray,
                           threshold: float = 0.5) -> float:
        """Calculate Dice similarity coefficient"""
        
        # Binarize if continuous
        if pred1.dtype == np.float32 or pred1.dtype == np.float64:
            pred1 = (pred1 > threshold).astype(np.uint8)
        if pred2.dtype == np.float32 or pred2.dtype == np.float64:
            pred2 = (pred2 > threshold).astype(np.uint8)
        
        # Calculate intersection and union
        intersection = np.sum(pred1 * pred2)
        sum_pred1 = np.sum(pred1)
        sum_pred2 = np.sum(pred2)
        
        # Handle empty masks
        if sum_pred1 == 0 and sum_pred2 == 0:
            return 1.0
        
        # Calculate Dice score
        dice = 2.0 * intersection / (sum_pred1 + sum_pred2 + 1e-8)
        
        return float(dice)
    
    def calculate_iou(self,
                     pred1: np.ndarray,
                     pred2: np.ndarray,
                     threshold: float = 0.5) -> float:
        """Calculate Intersection over Union (IoU)"""
        
        # Binarize if continuous
        if pred1.dtype == np.float32 or pred1.dtype == np.float64:
            pred1 = (pred1 > threshold).astype(np.uint8)
        if pred2.dtype == np.float32 or pred2.dtype == np.float64:
            pred2 = (pred2 > threshold).astype(np.uint8)
        
        # Calculate intersection and union
        intersection = np.sum(pred1 * pred2)
        union = np.sum(np.logical_or(pred1, pred2))
        
        # Handle empty masks
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        # Calculate IoU
        iou = intersection / (union + 1e-8)
        
        return float(iou)
    
    def calculate_hausdorff_distance(self,
                                   pred1: np.ndarray,
                                   pred2: np.ndarray,
                                   threshold: float = 0.5) -> Dict[str, float]:
        """Calculate Hausdorff distance metrics"""
        
        # Binarize if continuous
        if pred1.dtype == np.float32 or pred1.dtype == np.float64:
            pred1 = (pred1 > threshold).astype(np.uint8)
        if pred2.dtype == np.float32 or pred2.dtype == np.float64:
            pred2 = (pred2 > threshold).astype(np.uint8)
        
        # Get surface points
        points1 = np.column_stack(np.where(pred1))
        points2 = np.column_stack(np.where(pred2))
        
        # Handle empty masks
        if len(points1) == 0 or len(points2) == 0:
            return {
                "hausdorff": 0.0,
                "hausdorff_95": 0.0,
                "average_surface_distance": 0.0
            }
        
        # Calculate directed Hausdorff distances
        d1_to_2 = directed_hausdorff(points1, points2)[0]
        d2_to_1 = directed_hausdorff(points2, points1)[0]
        
        # Calculate symmetric Hausdorff distance
        hausdorff = max(d1_to_2, d2_to_1)
        
        # Calculate 95th percentile Hausdorff distance
        from scipy.spatial import distance_matrix
        dist_matrix_1_to_2 = distance_matrix(points1, points2)
        dist_matrix_2_to_1 = distance_matrix(points2, points1)
        
        min_dists_1_to_2 = np.min(dist_matrix_1_to_2, axis=1)
        min_dists_2_to_1 = np.min(dist_matrix_2_to_1, axis=1)
        
        all_min_dists = np.concatenate([min_dists_1_to_2, min_dists_2_to_1])
        hausdorff_95 = np.percentile(all_min_dists, 95)
        
        # Calculate average surface distance
        avg_surface_dist = np.mean(all_min_dists)
        
        return {
            "hausdorff": float(hausdorff),
            "hausdorff_95": float(hausdorff_95),
            "average_surface_distance": float(avg_surface_dist)
        }
    
    def calculate_volume_difference(self,
                                  pred1: np.ndarray,
                                  pred2: np.ndarray,
                                  spacing: Optional[Tuple[float, float, float]] = None) -> Dict[str, float]:
        """Calculate volume-based metrics"""
        
        # Binarize predictions
        if pred1.dtype == np.float32 or pred1.dtype == np.float64:
            pred1 = (pred1 > 0.5).astype(np.uint8)
        if pred2.dtype == np.float32 or pred2.dtype == np.float64:
            pred2 = (pred2 > 0.5).astype(np.uint8)
        
        # Calculate volumes
        volume1 = np.sum(pred1)
        volume2 = np.sum(pred2)
        
        # Apply spacing if provided
        if spacing is not None:
            voxel_volume = np.prod(spacing)
            volume1 *= voxel_volume
            volume2 *= voxel_volume
        
        # Calculate metrics
        volume_diff = abs(volume1 - volume2)
        volume_diff_percent = (volume_diff / max(volume1, volume2, 1)) * 100
        
        return {
            "volume1": float(volume1),
            "volume2": float(volume2),
            "volume_difference": float(volume_diff),
            "volume_difference_percent": float(volume_diff_percent)
        }
    
    def compare_outputs(self,
                       pytorch_output: np.ndarray,
                       coreml_output: np.ndarray,
                       metrics: List[str] = None) -> Dict[str, Any]:
        """Comprehensive comparison of model outputs"""
        
        if metrics is None:
            metrics = ["dice", "iou", "hausdorff", "volume", "numerical"]
        
        results = {}
        
        # Ensure same shape
        if pytorch_output.shape != coreml_output.shape:
            self.logger.warning(
                f"Shape mismatch: PyTorch {pytorch_output.shape} vs CoreML {coreml_output.shape}"
            )
            return {"error": "Shape mismatch"}
        
        # Remove batch and channel dimensions if present
        if pytorch_output.ndim == 5:
            pytorch_output = pytorch_output[0, 0]
            coreml_output = coreml_output[0, 0]
        
        # Calculate requested metrics
        if "dice" in metrics:
            results["dice_score"] = self.calculate_dice_score(pytorch_output, coreml_output)
        
        if "iou" in metrics:
            results["iou_score"] = self.calculate_iou(pytorch_output, coreml_output)
        
        if "hausdorff" in metrics:
            results.update(self.calculate_hausdorff_distance(pytorch_output, coreml_output))
        
        if "volume" in metrics:
            results.update(self.calculate_volume_difference(pytorch_output, coreml_output))
        
        if "numerical" in metrics:
            results.update({
                "max_absolute_error": float(np.max(np.abs(pytorch_output - coreml_output))),
                "mean_absolute_error": float(np.mean(np.abs(pytorch_output - coreml_output))),
                "rmse": float(np.sqrt(np.mean((pytorch_output - coreml_output) ** 2))),
                "correlation": float(np.corrcoef(pytorch_output.flatten(), coreml_output.flatten())[0, 1])
            })
        
        return results
    
    def validate_model_outputs(self,
                              pytorch_model: torch.nn.Module,
                              coreml_model: ct.models.MLModel,
                              test_inputs: List[np.ndarray],
                              device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
        """Validate model outputs on multiple test inputs"""
        
        all_results = []
        inference_times = {"pytorch": [], "coreml": []}
        
        for i, test_input in enumerate(test_inputs):
            self.logger.info(f"Validating input {i+1}/{len(test_inputs)}")
            
            # PyTorch inference
            torch_input = torch.from_numpy(test_input).to(device)
            
            start_time = time.time()
            with torch.no_grad():
                pytorch_output = pytorch_model(torch_input)
            pytorch_time = time.time() - start_time
            inference_times["pytorch"].append(pytorch_time)
            
            pytorch_output = pytorch_output.cpu().numpy()
            
            # CoreML inference
            coreml_input = {"volume": test_input}
            
            start_time = time.time()
            coreml_output = coreml_model.predict(coreml_input)["segmentation"]
            coreml_time = time.time() - start_time
            inference_times["coreml"].append(coreml_time)
            
            # Compare outputs
            comparison = self.compare_outputs(pytorch_output, coreml_output)
            comparison["input_index"] = i
            all_results.append(comparison)
        
        # Aggregate results
        aggregated = self._aggregate_validation_results(all_results)
        
        # Add timing information
        aggregated["inference_times"] = {
            "pytorch_mean": np.mean(inference_times["pytorch"]),
            "pytorch_std": np.std(inference_times["pytorch"]),
            "coreml_mean": np.mean(inference_times["coreml"]),
            "coreml_std": np.std(inference_times["coreml"]),
            "speedup": np.mean(inference_times["pytorch"]) / np.mean(inference_times["coreml"])
        }
        
        return aggregated
    
    def _aggregate_validation_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate validation results across multiple inputs"""
        
        # Filter out any errors
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid results"}
        
        # Aggregate metrics
        aggregated = {}
        
        # Get all metric keys
        metric_keys = set()
        for result in valid_results:
            metric_keys.update(result.keys())
        metric_keys.discard("input_index")
        
        # Calculate statistics for each metric
        for key in metric_keys:
            values = [r[key] for r in valid_results if key in r]
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
                aggregated[f"{key}_min"] = float(np.min(values))
                aggregated[f"{key}_max"] = float(np.max(values))
        
        aggregated["num_samples"] = len(valid_results)
        
        return aggregated
    
    def generate_validation_report(self,
                                 validation_results: Dict[str, Any],
                                 output_path: str,
                                 model_name: str) -> None:
        """Generate detailed validation report"""
        
        report_path = Path(output_path)
        report_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = report_path / f"{model_name}_validation.json"
        with open(json_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # Generate visualizations
        self._create_validation_plots(validation_results, report_path, model_name)
        
        # Generate text report
        text_path = report_path / f"{model_name}_validation.txt"
        with open(text_path, 'w') as f:
            f.write(f"Validation Report for {model_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Accuracy metrics
            f.write("Accuracy Metrics:\n")
            f.write(f"  Dice Score: {validation_results.get('dice_score_mean', 'N/A'):.4f} "
                   f"(±{validation_results.get('dice_score_std', 0):.4f})\n")
            f.write(f"  IoU Score: {validation_results.get('iou_score_mean', 'N/A'):.4f} "
                   f"(±{validation_results.get('iou_score_std', 0):.4f})\n")
            
            # Numerical accuracy
            f.write("\nNumerical Accuracy:\n")
            f.write(f"  Max Absolute Error: {validation_results.get('max_absolute_error_mean', 'N/A'):.6f}\n")
            f.write(f"  Mean Absolute Error: {validation_results.get('mean_absolute_error_mean', 'N/A'):.6f}\n")
            f.write(f"  RMSE: {validation_results.get('rmse_mean', 'N/A'):.6f}\n")
            
            # Performance
            if "inference_times" in validation_results:
                times = validation_results["inference_times"]
                f.write("\nInference Performance:\n")
                f.write(f"  PyTorch: {times['pytorch_mean']:.3f}s (±{times['pytorch_std']:.3f}s)\n")
                f.write(f"  CoreML: {times['coreml_mean']:.3f}s (±{times['coreml_std']:.3f}s)\n")
                f.write(f"  Speedup: {times['speedup']:.2f}x\n")
            
            # Validation status
            f.write("\nValidation Status: ")
            if validation_results.get('dice_score_mean', 0) > 0.99:
                f.write("PASSED ✓\n")
            else:
                f.write("FAILED ✗\n")
        
        self.logger.info(f"Validation report saved to {report_path}")
    
    def _create_validation_plots(self,
                               results: Dict[str, Any],
                               output_path: Path,
                               model_name: str) -> None:
        """Create validation visualization plots"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Validation Results for {model_name}", fontsize=16)
        
        # Plot 1: Metric comparison bar chart
        ax = axes[0, 0]
        metrics = ["dice_score_mean", "iou_score_mean"]
        values = [results.get(m, 0) for m in metrics]
        labels = ["Dice Score", "IoU Score"]
        
        bars = ax.bar(labels, values)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Segmentation Metrics")
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom')
        
        # Plot 2: Error distribution
        ax = axes[0, 1]
        error_metrics = ["max_absolute_error_mean", "mean_absolute_error_mean", "rmse_mean"]
        error_values = [results.get(m, 0) for m in error_metrics]
        error_labels = ["Max Error", "Mean Error", "RMSE"]
        
        ax.bar(error_labels, error_values)
        ax.set_ylabel("Error")
        ax.set_title("Numerical Errors")
        ax.set_yscale('log')
        
        # Plot 3: Inference time comparison
        ax = axes[1, 0]
        if "inference_times" in results:
            times = results["inference_times"]
            models = ["PyTorch", "CoreML"]
            mean_times = [times["pytorch_mean"], times["coreml_mean"]]
            std_times = [times["pytorch_std"], times["coreml_std"]]
            
            ax.bar(models, mean_times, yerr=std_times, capsize=10)
            ax.set_ylabel("Time (seconds)")
            ax.set_title("Inference Time Comparison")
        
        # Plot 4: Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"Model: {model_name}\n"
        summary_text += f"Samples Validated: {results.get('num_samples', 'N/A')}\n"
        summary_text += f"\nKey Metrics:\n"
        summary_text += f"Dice Score: {results.get('dice_score_mean', 0):.4f}\n"
        summary_text += f"Max Error: {results.get('max_absolute_error_mean', 0):.6f}\n"
        
        if "inference_times" in results:
            summary_text += f"\nSpeedup: {results['inference_times']['speedup']:.2f}x"
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_path / f"{model_name}_validation_plots.png", dpi=150)
        plt.close()
    
    def validate_preprocessing(self,
                             pytorch_preprocess: Any,
                             coreml_preprocess: Any,
                             test_volume: np.ndarray) -> bool:
        """Validate that preprocessing is consistent"""
        
        # Apply both preprocessing pipelines
        pytorch_processed = pytorch_preprocess(test_volume)
        coreml_processed = coreml_preprocess(test_volume)
        
        # Compare outputs
        max_diff = np.max(np.abs(pytorch_processed - coreml_processed))
        
        if max_diff > self.tolerance:
            self.logger.warning(f"Preprocessing mismatch: max difference = {max_diff}")
            return False
        
        return True