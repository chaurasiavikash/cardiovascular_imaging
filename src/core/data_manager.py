"""
Data Management Module for Cardiovascular Validation Pipeline
Handles data loading, preprocessing, and validation dataset management
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import ndimage
import json


class DataManager:
    """
    Comprehensive data manager for cardiovascular imaging validation.
    
    Handles:
    - Multi-format data loading (DICOM, NIFTI, NumPy)
    - Data preprocessing and normalization
    - Validation dataset organization
    - Phantom data generation
    - Data quality assessment
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize data manager.
        
        Args:
            config: Data management configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Data processing settings
        self.supported_formats = self.config.get('input_formats', ['dicom', 'nifti', 'numpy', 'json'])
        self.preprocessing_config = self.config.get('preprocessing', {})
        
    def load_validation_datasets(self, data_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load complete validation datasets from directory.
        
        Args:
            data_path: Path to validation data directory
            
        Returns:
            Dictionary containing all validation datasets
        """
        self.logger.info(f"Loading validation datasets from: {data_path}")
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        datasets = {}
        
        # Look for standard validation data files
        expected_files = {
            'ground_truth': ['ground_truth.npy', 'gt.npy', 'labels.npy'],
            'predictions': ['predictions.npy', 'pred.npy', 'outputs.npy'],
            'demographics': ['demographics.json', 'demographics.csv', 'patient_data.json'],
            'clinical_data': ['clinical.json', 'clinical.csv', 'measurements.json']
        }
        
        for data_type, possible_files in expected_files.items():
            file_found = False
            for filename in possible_files:
                file_path = data_path / filename
                if file_path.exists():
                    datasets[data_type] = self._load_data_file(file_path)
                    file_found = True
                    self.logger.info(f"Loaded {data_type} from {filename}")
                    break
            
            if not file_found and data_type in ['ground_truth', 'predictions']:
                # Generate synthetic data if core files not found
                self.logger.warning(f"Required file {data_type} not found, generating synthetic data")
                datasets[data_type] = self._generate_synthetic_data(data_type)
        
        # Validate dataset consistency
        datasets = self._validate_dataset_consistency(datasets)
        
        return datasets
    
    def _load_data_file(self, file_path: Path) -> Any:
        """Load data from various file formats."""
        try:
            if file_path.suffix == '.npy':
                return np.load(file_path, allow_pickle=True)
            elif file_path.suffix == '.npz':
                return dict(np.load(file_path, allow_pickle=True))
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif file_path.suffix == '.csv':
                return pd.read_csv(file_path).to_dict('list')
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {str(e)}")
            raise
    
    def _generate_synthetic_data(self, data_type: str, n_samples: int = 100) -> Any:
        """Generate synthetic validation data for testing."""
        self.logger.info(f"Generating synthetic {data_type} data")
        
        np.random.seed(42)  # For reproducibility
        
        if data_type == 'ground_truth':
            # Generate synthetic segmentation masks
            masks = []
            for i in range(n_samples):
                mask = self._create_synthetic_vessel_mask((128, 128))
                masks.append(mask)
            return np.array(masks)
        
        elif data_type == 'predictions':
            # Generate predictions with some noise
            gt_data = self._generate_synthetic_data('ground_truth', n_samples)
            predictions = []
            
            for gt_mask in gt_data:
                pred_mask = gt_mask.copy()
                # Add random noise
                noise_level = np.random.uniform(0.05, 0.15)
                noise = np.random.random(gt_mask.shape) < noise_level
                pred_mask = np.logical_xor(pred_mask, noise).astype(np.uint8)
                predictions.append(pred_mask)
            
            return np.array(predictions)
        
        elif data_type == 'demographics':
            return {
                'age_group': np.random.choice(['young', 'middle', 'elderly'], n_samples).tolist(),
                'sex': np.random.choice(['male', 'female'], n_samples).tolist(),
                'race': np.random.choice(['white', 'black', 'asian', 'hispanic'], n_samples).tolist(),
                'institution': np.random.choice(['hospital_a', 'hospital_b', 'hospital_c'], n_samples).tolist()
            }
        
        elif data_type == 'clinical_data':
            return {
                'vessel_diameter': (np.random.normal(3.5, 0.8, n_samples)).tolist(),
                'stenosis_severity': (np.random.uniform(0, 100, n_samples)).tolist(),
                'ejection_fraction': (np.random.normal(60, 10, n_samples)).tolist()
            }
        
        else:
            return None
    
    def _create_synthetic_vessel_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create a synthetic vessel-like binary mask."""
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Create curved vessel structure
        center_x = shape[1] // 2
        for y in range(20, shape[0] - 20):
            # Create sinusoidal curve
            x_offset = int(10 * np.sin(y * 0.1))
            width = max(1, int(3 + np.random.normal(0, 0.5)))
            
            x_start = max(0, center_x + x_offset - width)
            x_end = min(shape[1] - 1, center_x + x_offset + width)
            mask[y, x_start:x_end] = 1
        
        # Add some branching
        if np.random.random() > 0.5:
            branch_start_y = shape[0] // 2
            for y in range(branch_start_y, min(branch_start_y + 30, shape[0] - 10)):
                x_offset = center_x + int((y - branch_start_y) * 0.5)
                if 0 <= x_offset < shape[1]:
                    mask[y, x_offset] = 1
        
        return mask
    
    def _validate_dataset_consistency(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consistency across datasets."""
        self.logger.info("Validating dataset consistency")
        
        # Check if ground truth and predictions have consistent shapes
        if 'ground_truth' in datasets and 'predictions' in datasets:
            gt_shape = np.array(datasets['ground_truth']).shape
            pred_shape = np.array(datasets['predictions']).shape
            
            if gt_shape != pred_shape:
                self.logger.warning(f"Shape mismatch: GT {gt_shape} vs Pred {pred_shape}")
                # Try to fix by cropping to minimum size
                min_samples = min(gt_shape[0], pred_shape[0])
                datasets['ground_truth'] = np.array(datasets['ground_truth'])[:min_samples]
                datasets['predictions'] = np.array(datasets['predictions'])[:min_samples]
        
        # Check demographic data consistency
        if 'demographics' in datasets:
            demo_data = datasets['demographics']
            if isinstance(demo_data, dict):
                demo_lengths = [len(v) for v in demo_data.values() if isinstance(v, list)]
                if demo_lengths and len(set(demo_lengths)) > 1:
                    self.logger.warning("Inconsistent demographic data lengths")
                    # Crop to minimum length
                    min_length = min(demo_lengths)
                    for key, values in demo_data.items():
                        if isinstance(values, list) and len(values) > min_length:
                            demo_data[key] = values[:min_length]
        
        # Add metadata
        datasets['metadata'] = {
            'n_samples': self._get_dataset_size(datasets),
            'data_types': list(datasets.keys()),
            'validation_ready': self._check_validation_readiness(datasets)
        }
        
        return datasets
    
    def _get_dataset_size(self, datasets: Dict[str, Any]) -> int:
        """Get the number of samples in the dataset."""
        if 'ground_truth' in datasets:
            return len(datasets['ground_truth'])
        elif 'predictions' in datasets:
            return len(datasets['predictions'])
        elif 'demographics' in datasets:
            demo_data = datasets['demographics']
            if isinstance(demo_data, dict) and demo_data:
                first_key = list(demo_data.keys())[0]
                return len(demo_data[first_key])
        return 0
    
    def _check_validation_readiness(self, datasets: Dict[str, Any]) -> bool:
        """Check if datasets are ready for validation."""
        required_data = ['ground_truth', 'predictions']
        return all(data_type in datasets for data_type in required_data)
    
    def generate_phantom_data(self, phantom_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate phantom validation data.
        
        Args:
            phantom_config: Phantom configuration parameters
            
        Returns:
            Phantom data dictionary
        """
        self.logger.info("Generating phantom validation data")
        
        phantom_type = phantom_config.get('phantom_type', 'coronary_vessel')
        image_size = phantom_config.get('image_size', (128, 128, 64))
        vessel_diameter = phantom_config.get('vessel_diameter', 3.0)  # mm
        spacing = phantom_config.get('spacing', (0.5, 0.5, 0.5))  # mm
        
        # Generate phantom image
        phantom_image = self._generate_phantom_image(image_size, vessel_diameter, spacing)
        
        # Generate ground truth segmentation
        ground_truth = self._generate_phantom_ground_truth(image_size, vessel_diameter, spacing)
        
        # Generate predictions (with controlled noise)
        predictions = self._generate_phantom_predictions(ground_truth, phantom_config)
        
        phantom_data = {
            'phantom_type': phantom_type,
            'image': phantom_image,
            'ground_truth': ground_truth,
            'predictions': predictions,
            'spacing': spacing,
            'vessel_diameter_mm': vessel_diameter,
            'geometric_truth': self._calculate_phantom_geometric_truth(ground_truth, spacing),
            'noise_parameters': phantom_config.get('noise', {})
        }
        
        return phantom_data
    
    def _generate_phantom_image(self, image_size: Tuple[int, int, int], 
                               vessel_diameter: float, spacing: Tuple[float, float, float]) -> np.ndarray:
        """Generate phantom image with known vessel geometry."""
        image = np.zeros(image_size, dtype=np.float32)
        
        # Background intensity
        background_intensity = 100
        vessel_intensity = 300
        
        # Fill background
        image.fill(background_intensity)
        
        # Create vessel structure
        center_x, center_y = image_size[1] // 2, image_size[0] // 2
        vessel_radius_voxels = vessel_diameter / (2 * min(spacing[:2]))
        
        for z in range(image_size[2]):
            # Create curved vessel
            for y in range(image_size[0]):
                for x in range(image_size[1]):
                    # Distance from vessel centerline
                    centerline_x = center_x + int(5 * np.sin(z * 0.1))
                    distance = np.sqrt((x - centerline_x)**2 + (y - center_y)**2)
                    
                    if distance <= vessel_radius_voxels:
                        # Smooth transition at vessel boundary
                        intensity_factor = max(0, 1 - (distance / vessel_radius_voxels))
                        image[y, x, z] = background_intensity + intensity_factor * (vessel_intensity - background_intensity)
        
        # Add realistic noise
        noise_std = 20
        noise = np.random.normal(0, noise_std, image_size)
        image = image + noise
        
        return image
    
    def _generate_phantom_ground_truth(self, image_size: Tuple[int, int, int],
                                     vessel_diameter: float, spacing: Tuple[float, float, float]) -> np.ndarray:
        """Generate perfect ground truth segmentation for phantom."""
        ground_truth = np.zeros(image_size, dtype=np.uint8)
        
        center_x, center_y = image_size[1] // 2, image_size[0] // 2
        vessel_radius_voxels = vessel_diameter / (2 * min(spacing[:2]))
        
        for z in range(image_size[2]):
            for y in range(image_size[0]):
                for x in range(image_size[1]):
                    # Distance from vessel centerline
                    centerline_x = center_x + int(5 * np.sin(z * 0.1))
                    distance = np.sqrt((x - centerline_x)**2 + (y - center_y)**2)
                    
                    if distance <= vessel_radius_voxels:
                        ground_truth[y, x, z] = 1
        
        return ground_truth
    
    def _generate_phantom_predictions(self, ground_truth: np.ndarray, 
                                    phantom_config: Dict[str, Any]) -> np.ndarray:
        """Generate predictions with controlled errors for phantom validation."""
        predictions = ground_truth.copy().astype(np.uint8)
        
        # Get noise parameters
        noise_config = phantom_config.get('noise', {})
        boundary_error_std = noise_config.get('boundary_error_std', 1.0)  # voxels
        false_positive_rate = noise_config.get('false_positive_rate', 0.02)
        false_negative_rate = noise_config.get('false_negative_rate', 0.02)
        
        # Add boundary errors
        if boundary_error_std > 0:
            # Find boundaries
            from scipy import ndimage
            boundary = ground_truth - ndimage.binary_erosion(ground_truth)
            boundary_indices = np.where(boundary)
            
            # Add random displacement to boundary points
            for i in range(len(boundary_indices[0])):
                y, x, z = boundary_indices[0][i], boundary_indices[1][i], boundary_indices[2][i]
                
                # Random displacement
                dy = int(np.random.normal(0, boundary_error_std))
                dx = int(np.random.normal(0, boundary_error_std))
                
                new_y = np.clip(y + dy, 0, ground_truth.shape[0] - 1)
                new_x = np.clip(x + dx, 0, ground_truth.shape[1] - 1)
                
                # Flip the voxel at new location
                predictions[new_y, new_x, z] = 1 - predictions[new_y, new_x, z]
        
        # Add false positives
        if false_positive_rate > 0:
            false_positive_mask = (ground_truth == 0) & (np.random.random(ground_truth.shape) < false_positive_rate)
            predictions[false_positive_mask] = 1
        
        # Add false negatives
        if false_negative_rate > 0:
            false_negative_mask = (ground_truth == 1) & (np.random.random(ground_truth.shape) < false_negative_rate)
            predictions[false_negative_mask] = 0
        
        return predictions
    
    def _calculate_phantom_geometric_truth(self, ground_truth: np.ndarray, 
                                         spacing: Tuple[float, float, float]) -> Dict[str, float]:
        """Calculate true geometric properties of phantom."""
        # Volume calculation
        voxel_volume = np.prod(spacing)
        true_volume = np.sum(ground_truth) * voxel_volume
        
        # Surface area calculation (approximate)
        from scipy import ndimage
        boundary = ground_truth - ndimage.binary_erosion(ground_truth)
        boundary_voxels = np.sum(boundary)
        voxel_face_area = min(spacing[0] * spacing[1], spacing[0] * spacing[2], spacing[1] * spacing[2])
        true_surface_area = boundary_voxels * voxel_face_area
        
        # Centerline length calculation
        from skimage import morphology
        if np.any(ground_truth):
            skeleton = morphology.skeletonize_3d(ground_truth.astype(bool))
            centerline_voxels = np.sum(skeleton)
            min_spacing = min(spacing)
            true_centerline_length = centerline_voxels * min_spacing
        else:
            true_centerline_length = 0.0
        
        return {
            'true_volume_mm3': float(true_volume),
            'true_surface_area_mm2': float(true_surface_area),
            'true_centerline_length_mm': float(true_centerline_length)
        }
    
    def preprocess_data(self, data: np.ndarray, data_type: str = 'image') -> np.ndarray:
        """
        Preprocess data according to configuration.
        
        Args:
            data: Input data array
            data_type: Type of data ('image', 'segmentation', etc.)
            
        Returns:
            Preprocessed data array
        """
        processed_data = data.copy()
        
        if data_type == 'image':
            # Intensity normalization
            if self.preprocessing_config.get('normalize_intensity', False):
                processed_data = self._normalize_intensity(processed_data)
            
            # Clipping (for CT data)
            clip_range = self.preprocessing_config.get('clip_range')
            if clip_range:
                processed_data = np.clip(processed_data, clip_range[0], clip_range[1])
        
        elif data_type == 'segmentation':
            # Ensure binary segmentation
            processed_data = (processed_data > 0.5).astype(np.uint8)
        
        return processed_data
    
    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity to [0, 1] range."""
        image_min = np.min(image)
        image_max = np.max(image)
        
        if image_max > image_min:
            normalized = (image - image_min) / (image_max - image_min)
        else:
            normalized = np.zeros_like(image)
        
        return normalized
    
    def export_validation_results(self, results: Dict[str, Any], 
                                output_path: Union[str, Path]) -> None:
        """
        Export validation results to file.
        
        Args:
            results: Validation results dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        elif output_path.suffix == '.npz':
            np.savez_compressed(output_path, **results)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        self.logger.info(f"Validation results exported to: {output_path}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def load_dicom_data(self, dicom_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load DICOM data using the DICOM processor.
        
        Args:
            dicom_path: Path to DICOM files
            
        Returns:
            Processed DICOM data
        """
        from .dicom_processor import DICOMProcessor
        
        dicom_processor = DICOMProcessor(anonymize=True)
        dicom_data = dicom_processor.load_dicom_series(dicom_path)
        
        # Validate for cardiovascular analysis
        validation_results = dicom_processor.validate_dicom_for_cardiovascular_analysis(dicom_data)
        
        if not validation_results['is_valid']:
            self.logger.warning("DICOM data validation issues detected")
            for error in validation_results.get('errors', []):
                self.logger.error(f"DICOM Error: {error}")
            for warning in validation_results.get('warnings', []):
                self.logger.warning(f"DICOM Warning: {warning}")
        
        return dicom_data
    
    def create_validation_split(self, data: Dict[str, Any], 
                              train_ratio: float = 0.7, 
                              val_ratio: float = 0.15,
                              test_ratio: float = 0.15,
                              stratify_by: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Create train/validation/test splits for validation studies.
        
        Args:
            data: Complete dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio  
            test_ratio: Test set ratio
            stratify_by: Demographic attribute to stratify by
            
        Returns:
            Dictionary with train/val/test splits
        """
        n_samples = self._get_dataset_size(data)
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Calculate split indices
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        # Generate indices
        indices = np.arange(n_samples)
        
        if stratify_by and stratify_by in data.get('demographics', {}):
            # Stratified split
            from sklearn.model_selection import train_test_split
            
            stratify_values = data['demographics'][stratify_by]
            
            # First split: train vs (val + test)
            train_idx, temp_idx = train_test_split(
                indices, test_size=(val_ratio + test_ratio), 
                stratify=stratify_values, random_state=42
            )
            
            # Second split: val vs test
            temp_stratify = [stratify_values[i] for i in temp_idx]
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=test_ratio/(val_ratio + test_ratio),
                stratify=temp_stratify, random_state=42
            )
        else:
            # Random split
            np.random.seed(42)
            np.random.shuffle(indices)
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
        
        # Create split datasets
        splits = {}
        for split_name, split_indices in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            split_data = {}
            
            for key, values in data.items():
                if key == 'metadata':
                    continue
                    
                if isinstance(values, np.ndarray):
                    split_data[key] = values[split_indices]
                elif isinstance(values, dict):
                    split_data[key] = {k: [v[i] for i in split_indices] if isinstance(v, list) else v 
                                     for k, v in values.items()}
                elif isinstance(values, list):
                    split_data[key] = [values[i] for i in split_indices]
            
            split_data['metadata'] = {
                'n_samples': len(split_indices),
                'split_type': split_name,
                'indices': split_indices.tolist()
            }
            
            splits[split_name] = split_data
        
        self.logger.info(f"Created data splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        return splits
    
    def get_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary statistics of the dataset."""
        summary = {
            'n_samples': self._get_dataset_size(data),
            'data_types': list(data.keys()),
            'validation_ready': self._check_validation_readiness(data)
        }
        
        # Demographic summary
        if 'demographics' in data:
            demo_data = data['demographics']
            summary['demographics'] = {}
            
            for attr, values in demo_data.items():
                if isinstance(values, list):
                    unique_values, counts = np.unique(values, return_counts=True)
                    summary['demographics'][attr] = {
                        'unique_values': unique_values.tolist(),
                        'counts': counts.tolist(),
                        'distribution': dict(zip(unique_values, counts))
                    }
        
        # Data shape information
        for key, values in data.items():
            if isinstance(values, np.ndarray):
                summary[f'{key}_shape'] = values.shape
                summary[f'{key}_dtype'] = str(values.dtype)
        
        return summary