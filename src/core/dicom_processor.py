"""
DICOM Processing Module for Cardiovascular Imaging
Handles DICOM file processing, anonymization, and metadata extraction
"""

import logging
import numpy as np
import pydicom
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import SimpleITK as sitk
from datetime import datetime
import hashlib


class DICOMProcessor:
    """
    DICOM processor for cardiovascular imaging data.
    
    Provides comprehensive DICOM handling including:
    - Loading and parsing DICOM files
    - Anonymization for validation studies
    - Metadata extraction and validation
    - Image preprocessing for validation
    """
    
    def __init__(self, anonymize: bool = True):
        """
        Initialize DICOM processor.
        
        Args:
            anonymize: Whether to anonymize DICOM data by default
        """
        self.logger = logging.getLogger(__name__)
        self.anonymize = anonymize
        self._anonymization_map = {}
        
    def load_dicom_series(self, dicom_path: Union[str, Path]) -> Dict[str, any]:
        """
        Load DICOM series from directory or file.
        
        Args:
            dicom_path: Path to DICOM directory or file
            
        Returns:
            Dictionary containing image data and metadata
        """
        dicom_path = Path(dicom_path)
        
        try:
            if dicom_path.is_file():
                # Single DICOM file
                return self._load_single_dicom(dicom_path)
            elif dicom_path.is_dir():
                # DICOM series
                return self._load_dicom_directory(dicom_path)
            else:
                raise FileNotFoundError(f"DICOM path not found: {dicom_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load DICOM from {dicom_path}: {str(e)}")
            raise
    
    def _load_single_dicom(self, dicom_file: Path) -> Dict[str, any]:
        """Load single DICOM file."""
        self.logger.info(f"Loading DICOM file: {dicom_file}")
        
        # Read DICOM file
        dicom_data = pydicom.dcmread(str(dicom_file))
        
        # Extract image data
        image_array = dicom_data.pixel_array.astype(np.float64)
        
        # Apply rescale slope and intercept if present
        if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
            image_array = image_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
        
        # Extract metadata
        metadata = self._extract_metadata(dicom_data)
        
        # Anonymize if required
        if self.anonymize:
            metadata = self._anonymize_metadata(metadata)
        
        return {
            'image': image_array,
            'metadata': metadata,
            'spacing': self._get_pixel_spacing(dicom_data),
            'orientation': self._get_image_orientation(dicom_data),
            'modality': dicom_data.get('Modality', 'Unknown'),
            'series_uid': dicom_data.get('SeriesInstanceUID', ''),
            'study_uid': dicom_data.get('StudyInstanceUID', ''),
            'acquisition_datetime': self._get_acquisition_datetime(dicom_data)
        }
    
    def _load_dicom_directory(self, dicom_dir: Path) -> Dict[str, any]:
        """Load DICOM series from directory."""
        self.logger.info(f"Loading DICOM series from: {dicom_dir}")
        
        # Get all DICOM files
        dicom_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*"))
        dicom_files = [f for f in dicom_files if f.is_file()]
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_dir}")
        
        # Use SimpleITK for series reading
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        
        if not series_ids:
            # Fallback to manual loading
            return self._load_dicom_files_manually(dicom_files)
        
        # Load the first series (or could iterate through all)
        series_file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
        reader.SetFileNames(series_file_names)
        
        # Read the series
        image = reader.Execute()
        image_array = sitk.GetArrayFromImage(image)
        
        # Get metadata from first file
        first_dicom = pydicom.dcmread(series_file_names[0])
        metadata = self._extract_metadata(first_dicom)
        
        if self.anonymize:
            metadata = self._anonymize_metadata(metadata)
        
        return {
            'image': image_array,
            'metadata': metadata,
            'spacing': image.GetSpacing(),
            'origin': image.GetOrigin(),
            'direction': image.GetDirection(),
            'modality': first_dicom.get('Modality', 'Unknown'),
            'series_uid': first_dicom.get('SeriesInstanceUID', ''),
            'study_uid': first_dicom.get('StudyInstanceUID', ''),
            'series_files': series_file_names,
            'acquisition_datetime': self._get_acquisition_datetime(first_dicom)
        }
    
    def _load_dicom_files_manually(self, dicom_files: List[Path]) -> Dict[str, any]:
        """Manually load and sort DICOM files."""
        dicom_data_list = []
        
        for file_path in dicom_files:
            try:
                dicom_data = pydicom.dcmread(str(file_path))
                if hasattr(dicom_data, 'pixel_array'):
                    dicom_data_list.append((dicom_data, file_path))
            except Exception as e:
                self.logger.warning(f"Could not read {file_path}: {str(e)}")
                continue
        
        if not dicom_data_list:
            raise ValueError("No valid DICOM files with image data found")
        
        # Sort by instance number or slice location
        dicom_data_list.sort(key=lambda x: self._get_sort_key(x[0]))
        
        # Stack images
        images = []
        for dicom_data, _ in dicom_data_list:
            image_array = dicom_data.pixel_array.astype(np.float64)
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                image_array = image_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
            images.append(image_array)
        
        stacked_image = np.stack(images, axis=0)
        
        # Get metadata from first file
        first_dicom = dicom_data_list[0][0]
        metadata = self._extract_metadata(first_dicom)
        
        if self.anonymize:
            metadata = self._anonymize_metadata(metadata)
        
        return {
            'image': stacked_image,
            'metadata': metadata,
            'spacing': self._get_pixel_spacing(first_dicom),
            'orientation': self._get_image_orientation(first_dicom),
            'modality': first_dicom.get('Modality', 'Unknown'),
            'series_uid': first_dicom.get('SeriesInstanceUID', ''),
            'study_uid': first_dicom.get('StudyInstanceUID', ''),
            'num_slices': len(images)
        }
    
    def _extract_metadata(self, dicom_data: pydicom.Dataset) -> Dict[str, any]:
        """Extract relevant metadata from DICOM dataset."""
        metadata = {}
        
        # Patient information
        metadata['patient_id'] = dicom_data.get('PatientID', '')
        metadata['patient_age'] = dicom_data.get('PatientAge', '')
        metadata['patient_sex'] = dicom_data.get('PatientSex', '')
        metadata['patient_weight'] = dicom_data.get('PatientWeight', '')
        
        # Study information
        metadata['study_date'] = dicom_data.get('StudyDate', '')
        metadata['study_time'] = dicom_data.get('StudyTime', '')
        metadata['study_description'] = dicom_data.get('StudyDescription', '')
        
        # Series information
        metadata['series_date'] = dicom_data.get('SeriesDate', '')
        metadata['series_time'] = dicom_data.get('SeriesTime', '')
        metadata['series_description'] = dicom_data.get('SeriesDescription', '')
        metadata['protocol_name'] = dicom_data.get('ProtocolName', '')
        
        # Image acquisition parameters
        metadata['slice_thickness'] = dicom_data.get('SliceThickness', '')
        metadata['repetition_time'] = dicom_data.get('RepetitionTime', '')
        metadata['echo_time'] = dicom_data.get('EchoTime', '')
        metadata['flip_angle'] = dicom_data.get('FlipAngle', '')
        
        # Equipment information
        metadata['manufacturer'] = dicom_data.get('Manufacturer', '')
        metadata['manufacturer_model'] = dicom_data.get('ManufacturerModelName', '')
        metadata['software_version'] = dicom_data.get('SoftwareVersions', '')
        metadata['magnetic_field_strength'] = dicom_data.get('MagneticFieldStrength', '')
        
        # Contrast information (if applicable)
        metadata['contrast_agent'] = dicom_data.get('ContrastBolusAgent', '')
        metadata['contrast_volume'] = dicom_data.get('ContrastBolusVolume', '')
        
        return metadata
    
    def _anonymize_metadata(self, metadata: Dict[str, any]) -> Dict[str, any]:
        """Anonymize patient metadata for validation studies."""
        anonymized = metadata.copy()
        
        # Generate consistent anonymous ID
        if metadata.get('patient_id'):
            anonymous_id = self._generate_anonymous_id(metadata['patient_id'])
            anonymized['patient_id'] = anonymous_id
            anonymized['original_patient_id_hash'] = hashlib.sha256(
                metadata['patient_id'].encode()).hexdigest()[:8]
        
        # Remove or anonymize sensitive fields
        sensitive_fields = [
            'patient_name', 'patient_birth_date', 'referring_physician_name',
            'operator_name', 'performing_physician_name'
        ]
        
        for field in sensitive_fields:
            if field in anonymized:
                del anonymized[field]
        
        # Add anonymization timestamp
        anonymized['anonymized_datetime'] = datetime.now().isoformat()
        
        return anonymized
    
    def _generate_anonymous_id(self, patient_id: str) -> str:
        """Generate consistent anonymous ID for patient."""
        if patient_id not in self._anonymization_map:
            # Generate anonymous ID based on hash
            hash_object = hashlib.md5(patient_id.encode())
            anonymous_id = f"ANON_{hash_object.hexdigest()[:8].upper()}"
            self._anonymization_map[patient_id] = anonymous_id
        
        return self._anonymization_map[patient_id]
    
    def _get_pixel_spacing(self, dicom_data: pydicom.Dataset) -> Tuple[float, float, float]:
        """Get pixel spacing from DICOM data."""
        try:
            pixel_spacing = dicom_data.get('PixelSpacing', [1.0, 1.0])
            slice_thickness = dicom_data.get('SliceThickness', 1.0)
            return (float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness))
        except:
            return (1.0, 1.0, 1.0)
    
    def _get_image_orientation(self, dicom_data: pydicom.Dataset) -> Optional[List[float]]:
        """Get image orientation from DICOM data."""
        try:
            orientation = dicom_data.get('ImageOrientationPatient', None)
            if orientation:
                return [float(x) for x in orientation]
            return None
        except:
            return None
    
    def _get_acquisition_datetime(self, dicom_data: pydicom.Dataset) -> Optional[datetime]:
        """Get acquisition datetime from DICOM data."""
        try:
            date = dicom_data.get('AcquisitionDate', dicom_data.get('StudyDate', ''))
            time = dicom_data.get('AcquisitionTime', dicom_data.get('StudyTime', ''))
            
            if date and time:
                # Parse DICOM date/time format
                date_str = f"{date} {time}"
                return datetime.strptime(date_str[:14], '%Y%m%d %H%M%S')
            elif date:
                return datetime.strptime(date, '%Y%m%d')
            
            return None
        except:
            return None
    
    def _get_sort_key(self, dicom_data: pydicom.Dataset) -> float:
        """Get sorting key for DICOM slices."""
        # Try instance number first
        if hasattr(dicom_data, 'InstanceNumber'):
            return float(dicom_data.InstanceNumber)
        
        # Try slice location
        if hasattr(dicom_data, 'SliceLocation'):
            return float(dicom_data.SliceLocation)
        
        # Try image position patient (Z coordinate)
        if hasattr(dicom_data, 'ImagePositionPatient'):
            return float(dicom_data.ImagePositionPatient[2])
        
        # Default
        return 0.0
    
    def validate_dicom_for_cardiovascular_analysis(self, dicom_data: Dict[str, any]) -> Dict[str, any]:
        """
        Validate DICOM data for cardiovascular analysis.
        
        Args:
            dicom_data: Loaded DICOM data dictionary
            
        Returns:
            Validation results and recommendations
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check modality
        modality = dicom_data.get('modality', '').upper()
        if modality not in ['CT', 'MR', 'XA', 'RF', 'US']:
            validation_results['warnings'].append(
                f"Modality '{modality}' may not be suitable for cardiovascular analysis"
            )
        
        # Check image dimensions
        image = dicom_data.get('image')
        if image is not None:
            if image.ndim < 2:
                validation_results['errors'].append("Image must be at least 2D")
                validation_results['is_valid'] = False
            
            if image.shape[0] < 128 or image.shape[1] < 128:
                validation_results['warnings'].append(
                    "Low resolution image may affect analysis accuracy"
                )
        
        # Check pixel spacing
        spacing = dicom_data.get('spacing')
        if spacing:
            if any(s > 2.0 for s in spacing[:2]):  # Check in-plane spacing
                validation_results['warnings'].append(
                    "Large pixel spacing may reduce measurement accuracy"
                )
        
        # Check for contrast enhancement (for angiography)
        metadata = dicom_data.get('metadata', {})
        if modality in ['CT', 'XA'] and not metadata.get('contrast_agent'):
            validation_results['recommendations'].append(
                "Contrast enhancement recommended for optimal vessel visualization"
            )
        
        # Check acquisition parameters for cardiac imaging
        if modality == 'MR':
            if not metadata.get('repetition_time'):
                validation_results['warnings'].append("Missing TR information")
            if not metadata.get('echo_time'):
                validation_results['warnings'].append("Missing TE information")
        
        return validation_results
    
    def preprocess_for_validation(self, dicom_data: Dict[str, any], 
                                target_spacing: Optional[Tuple[float, float, float]] = None) -> Dict[str, any]:
        """
        Preprocess DICOM data for validation analysis.
        
        Args:
            dicom_data: Loaded DICOM data
            target_spacing: Target pixel spacing for resampling
            
        Returns:
            Preprocessed DICOM data
        """
        processed_data = dicom_data.copy()
        image = dicom_data['image'].copy()
        
        # Normalize intensity values
        if image.dtype != np.float64:
            image = image.astype(np.float64)
        
        # Handle different modalities
        modality = dicom_data.get('modality', '').upper()
        
        if modality == 'CT':
            # CT: Clip to reasonable HU range for cardiovascular imaging
            image = np.clip(image, -200, 800)
        elif modality == 'MR':
            # MR: Normalize to 0-1 range
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Resample if target spacing is provided
        if target_spacing and 'spacing' in dicom_data:
            current_spacing = dicom_data['spacing']
            if current_spacing != target_spacing:
                image = self._resample_image(image, current_spacing, target_spacing)
                processed_data['spacing'] = target_spacing
        
        processed_data['image'] = image
        processed_data['preprocessing_applied'] = {
            'normalization': True,
            'resampling': target_spacing is not None,
            'target_spacing': target_spacing,
            'timestamp': datetime.now().isoformat()
        }
        
        return processed_data
    
    def _resample_image(self, image: np.ndarray, current_spacing: Tuple[float, float, float], 
                       target_spacing: Tuple[float, float, float]) -> np.ndarray:
        """Resample image to target spacing using SimpleITK."""
        try:
            # Convert to SimpleITK image
            sitk_image = sitk.GetImageFromArray(image)
            sitk_image.SetSpacing(current_spacing)
            
            # Calculate new size
            current_size = sitk_image.GetSize()
            new_size = [
                int(current_size[i] * current_spacing[i] / target_spacing[i])
                for i in range(len(current_spacing))
            ]
            
            # Resample
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(new_size)
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetInterpolator(sitk.sitkLinear)
            
            resampled_image = resampler.Execute(sitk_image)
            
            return sitk.GetArrayFromImage(resampled_image)
            
        except Exception as e:
            self.logger.error(f"Resampling failed: {str(e)}")
            return image
    
    def extract_cardiovascular_roi(self, dicom_data: Dict[str, any], 
                                  roi_type: str = "cardiac") -> Dict[str, any]:
        """
        Extract region of interest for cardiovascular analysis.
        
        Args:
            dicom_data: Processed DICOM data
            roi_type: Type of ROI ('cardiac', 'coronary', 'aortic')
            
        Returns:
            DICOM data with ROI extracted
        """
        image = dicom_data['image']
        
        if roi_type == "cardiac":
            # Simple cardiac ROI based on image center
            h, w = image.shape[-2:]
            center_y, center_x = h // 2, w // 2
            roi_size = min(h, w) // 2
            
            y1, y2 = max(0, center_y - roi_size), min(h, center_y + roi_size)
            x1, x2 = max(0, center_x - roi_size), min(w, center_x + roi_size)
            
            if image.ndim == 3:
                roi_image = image[:, y1:y2, x1:x2]
            else:
                roi_image = image[y1:y2, x1:x2]
            
            roi_data = dicom_data.copy()
            roi_data['image'] = roi_image
            roi_data['roi_bounds'] = (y1, y2, x1, x2)
            roi_data['roi_type'] = roi_type
            
            return roi_data
        
        else:
            # For other ROI types, return original data
            self.logger.warning(f"ROI type '{roi_type}' not implemented, returning original data")
            return dicom_data