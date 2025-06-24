"""
Medical Image Preprocessing Module
Handles CT image preprocessing for TotalSegmentator models
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import nibabel as nib
from scipy import ndimage
import torch
import logging


class MedicalImagePreprocessor:
    """Preprocessor for medical images compatible with CoreML"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Standard preprocessing parameters
        self.target_spacing = [1.5, 1.5, 1.5]  # mm
        self.window_center = 0
        self.window_width = 2000
        self.min_hu = -1000
        self.max_hu = 1000
    
    def load_nifti(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load NIfTI file and extract volume data"""
        
        self.logger.debug(f"Loading NIfTI file: {file_path}")
        
        # Load NIfTI
        nifti = nib.load(file_path)
        volume = nifti.get_fdata()
        affine = nifti.affine
        
        # Get voxel spacing
        spacing = np.abs(np.diag(affine)[:3])
        
        return volume, affine, spacing
    
    def apply_windowing(self, 
                       volume: np.ndarray,
                       window_center: Optional[float] = None,
                       window_width: Optional[float] = None) -> np.ndarray:
        """Apply HU windowing to CT volume"""
        
        if window_center is None:
            window_center = self.window_center
        if window_width is None:
            window_width = self.window_width
        
        # Calculate window bounds
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        
        # Apply windowing
        windowed = np.clip(volume, min_val, max_val)
        
        return windowed
    
    def resample_volume(self,
                       volume: np.ndarray,
                       current_spacing: np.ndarray,
                       target_spacing: Optional[List[float]] = None,
                       order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Resample volume to target spacing"""
        
        if target_spacing is None:
            target_spacing = self.target_spacing
        
        # Calculate resize factors
        resize_factor = current_spacing / np.array(target_spacing)
        new_shape = np.round(volume.shape * resize_factor).astype(int)
        
        # Resample
        resampled = ndimage.zoom(volume, resize_factor, order=order)
        
        self.logger.debug(f"Resampled from {volume.shape} to {resampled.shape}")
        
        return resampled, np.array(target_spacing)
    
    def normalize_volume(self, 
                        volume: np.ndarray,
                        method: str = "min_max") -> np.ndarray:
        """Normalize volume for neural network input"""
        
        if method == "min_max":
            # Normalize to [0, 1]
            min_val = self.min_hu
            max_val = self.max_hu
            normalized = (volume - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0, 1)
            
        elif method == "z_score":
            # Z-score normalization
            mean = np.mean(volume)
            std = np.std(volume)
            normalized = (volume - mean) / (std + 1e-8)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def pad_volume(self,
                  volume: np.ndarray,
                  target_shape: Tuple[int, int, int],
                  mode: str = "constant",
                  constant_value: float = 0) -> np.ndarray:
        """Pad volume to target shape"""
        
        current_shape = volume.shape
        
        # Calculate padding
        pad_width = []
        for current, target in zip(current_shape, target_shape):
            total_pad = max(0, target - current)
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width.append((pad_before, pad_after))
        
        # Apply padding
        if any(p[0] > 0 or p[1] > 0 for p in pad_width):
            volume = np.pad(volume, pad_width, mode=mode, constant_values=constant_value)
        
        return volume
    
    def crop_volume(self,
                   volume: np.ndarray,
                   target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Center crop volume to target shape"""
        
        current_shape = volume.shape
        
        # Calculate crop indices
        crop_indices = []
        for current, target in zip(current_shape, target_shape):
            if current > target:
                start = (current - target) // 2
                end = start + target
                crop_indices.append(slice(start, end))
            else:
                crop_indices.append(slice(None))
        
        # Apply crop
        cropped = volume[tuple(crop_indices)]
        
        return cropped
    
    def preprocess_for_inference(self,
                                volume: np.ndarray,
                                spacing: np.ndarray,
                                target_shape: Optional[Tuple[int, int, int]] = None) -> Dict:
        """Complete preprocessing pipeline for inference"""
        
        # Apply windowing
        volume = self.apply_windowing(volume)
        
        # Resample to standard spacing
        volume, new_spacing = self.resample_volume(volume, spacing)
        
        # Normalize
        volume = self.normalize_volume(volume)
        
        # Pad or crop to target shape if specified
        if target_shape is not None:
            if any(s < t for s, t in zip(volume.shape, target_shape)):
                volume = self.pad_volume(volume, target_shape)
            else:
                volume = self.crop_volume(volume, target_shape)
        
        # Add batch and channel dimensions
        volume = volume[np.newaxis, np.newaxis, ...]
        
        return {
            "volume": volume,
            "spacing": new_spacing,
            "shape": volume.shape
        }
    
    def create_patches(self,
                      volume: np.ndarray,
                      patch_size: Tuple[int, int, int],
                      overlap: float = 0.25) -> List[Dict]:
        """Create overlapping patches for large volumes"""
        
        patches = []
        volume_shape = volume.shape
        
        # Calculate stride
        stride = tuple(int(p * (1 - overlap)) for p in patch_size)
        
        # Generate patch coordinates
        for z in range(0, volume_shape[0] - patch_size[0] + 1, stride[0]):
            for y in range(0, volume_shape[1] - patch_size[1] + 1, stride[1]):
                for x in range(0, volume_shape[2] - patch_size[2] + 1, stride[2]):
                    # Extract patch
                    patch = volume[
                        z:z+patch_size[0],
                        y:y+patch_size[1],
                        x:x+patch_size[2]
                    ]
                    
                    patches.append({
                        "data": patch,
                        "position": (z, y, x),
                        "size": patch_size
                    })
        
        self.logger.debug(f"Created {len(patches)} patches from volume {volume_shape}")
        
        return patches
    
    def reconstruct_from_patches(self,
                                patches: List[Dict],
                                volume_shape: Tuple[int, int, int],
                                aggregation: str = "average") -> np.ndarray:
        """Reconstruct volume from patches"""
        
        # Initialize output volume and weight map
        output = np.zeros(volume_shape, dtype=np.float32)
        weights = np.zeros(volume_shape, dtype=np.float32)
        
        # Aggregate patches
        for patch_info in patches:
            patch = patch_info["data"]
            z, y, x = patch_info["position"]
            pz, py, px = patch_info["size"]
            
            if aggregation == "average":
                output[z:z+pz, y:y+py, x:x+px] += patch
                weights[z:z+pz, y:y+py, x:x+px] += 1
            elif aggregation == "max":
                output[z:z+pz, y:y+py, x:x+px] = np.maximum(
                    output[z:z+pz, y:y+py, x:x+px], 
                    patch
                )
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Normalize by weights for averaging
        if aggregation == "average":
            output = np.divide(output, weights, where=weights > 0)
        
        return output
    
    def prepare_for_coreml(self, volume: np.ndarray) -> np.ndarray:
        """Prepare volume for CoreML input"""
        
        # Ensure float32
        volume = volume.astype(np.float32)
        
        # Ensure 5D shape (batch, channel, depth, height, width)
        if volume.ndim == 3:
            volume = volume[np.newaxis, np.newaxis, ...]
        elif volume.ndim == 4:
            volume = volume[np.newaxis, ...]
        
        return volume
    
    def postprocess_segmentation(self,
                                segmentation: np.ndarray,
                                threshold: float = 0.5,
                                largest_component: bool = False) -> np.ndarray:
        """Postprocess segmentation output"""
        
        # Remove batch and channel dimensions if present
        if segmentation.ndim == 5:
            segmentation = segmentation[0, 0]
        elif segmentation.ndim == 4:
            segmentation = segmentation[0]
        
        # Apply threshold for binary segmentation
        if segmentation.dtype == np.float32:
            segmentation = (segmentation > threshold).astype(np.uint8)
        
        # Keep only largest connected component if requested
        if largest_component and np.any(segmentation):
            from scipy import ndimage
            labeled, num_features = ndimage.label(segmentation)
            if num_features > 1:
                sizes = ndimage.sum(segmentation, labeled, range(1, num_features + 1))
                largest_label = np.argmax(sizes) + 1
                segmentation = (labeled == largest_label).astype(np.uint8)
        
        return segmentation


class CTAugmentation:
    """Data augmentation for CT volumes (training only)"""
    
    def __init__(self, 
                 rotation_range: float = 10,
                 scale_range: float = 0.1,
                 noise_std: float = 0.01,
                 brightness_range: float = 0.1):
        
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.brightness_range = brightness_range
    
    def random_rotation(self, volume: np.ndarray) -> np.ndarray:
        """Apply random rotation"""
        
        angles = np.random.uniform(-self.rotation_range, self.rotation_range, 3)
        
        # Rotate around each axis
        volume = ndimage.rotate(volume, angles[0], axes=(1, 2), reshape=False)
        volume = ndimage.rotate(volume, angles[1], axes=(0, 2), reshape=False)
        volume = ndimage.rotate(volume, angles[2], axes=(0, 1), reshape=False)
        
        return volume
    
    def random_scale(self, volume: np.ndarray) -> np.ndarray:
        """Apply random scaling"""
        
        scale = 1 + np.random.uniform(-self.scale_range, self.scale_range)
        zoomed = ndimage.zoom(volume, scale, order=1)
        
        # Crop or pad to original shape
        if scale > 1:
            # Center crop
            start = [(z - o) // 2 for z, o in zip(zoomed.shape, volume.shape)]
            zoomed = zoomed[
                start[0]:start[0]+volume.shape[0],
                start[1]:start[1]+volume.shape[1],
                start[2]:start[2]+volume.shape[2]
            ]
        else:
            # Center pad
            pad_width = [((o - z) // 2, (o - z + 1) // 2) 
                        for z, o in zip(zoomed.shape, volume.shape)]
            zoomed = np.pad(zoomed, pad_width, mode='constant')
        
        return zoomed
    
    def random_noise(self, volume: np.ndarray) -> np.ndarray:
        """Add random Gaussian noise"""
        
        noise = np.random.normal(0, self.noise_std, volume.shape)
        return volume + noise
    
    def random_brightness(self, volume: np.ndarray) -> np.ndarray:
        """Apply random brightness adjustment"""
        
        brightness = 1 + np.random.uniform(-self.brightness_range, self.brightness_range)
        return volume * brightness
    
    def augment(self, volume: np.ndarray) -> np.ndarray:
        """Apply all augmentations"""
        
        # Apply augmentations with probability
        if np.random.rand() > 0.5:
            volume = self.random_rotation(volume)
        
        if np.random.rand() > 0.5:
            volume = self.random_scale(volume)
        
        if np.random.rand() > 0.5:
            volume = self.random_noise(volume)
        
        if np.random.rand() > 0.5:
            volume = self.random_brightness(volume)
        
        return volume