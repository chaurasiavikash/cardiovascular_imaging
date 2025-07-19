#!/usr/bin/env python3
"""
Basic Validation Example for Cardiovascular Image Validation Pipeline
ARCADE Dataset Only - Simplified Version

Author: Vikash Chaurasia
For: Medis Imaging Scientific Validation Specialist Interview
"""

import logging
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any

# Get the absolute path to the root directory (parent of examples)
root_dir = Path(__file__).parent.parent
# Add the src directory to sys.path
sys.path.insert(0, str(root_dir / "src"))

from core.data_manager import DataManager
from validation.vessel_segmentation_validator import VesselSegmentationValidator
from validation.statistical_validator import StatisticalValidator
from validation.bias_assessor import BiasAssessor
from validation.uncertainty_quantifier import UncertaintyQuantifier
from visualization.report_generator import ValidationReportGenerator
from utils.logger import setup_logging
from utils.arcade_dataset_loader import ARCADEDatasetLoader


def load_arcade_data(data_path: str, task_type: str = "syntax") -> Dict[str, Any]:
    """
    Load ARCADE dataset with fallback to synthetic data.
    
    Args:
        data_path: Path to ARCADE dataset directory
        task_type: "syntax" for vessel segmentation or "stenosis" for stenosis detection
        
    Returns:
        Validation data dictionary
    """
    logger = logging.getLogger(__name__)
    
    # Try to load ARCADE dataset
    try:
        logger.info(f"🏥 Loading ARCADE {task_type} dataset from: {data_path}")
        
        arcade_loader = ARCADEDatasetLoader(task_type)
        validation_data = arcade_loader.load_arcade_dataset(data_path, split="train")
        
        logger.info(f"✅ Successfully loaded ARCADE {task_type}: {validation_data['metadata']['n_samples']} samples")
        
        # Create reports directory
        Path("reports").mkdir(exist_ok=True)
        
        # Visualize first sample for verification (optional)
        if validation_data['metadata']['n_samples'] > 0:
            try:
                arcade_loader.visualize_sample(validation_data, 0, f"reports/arcade_sample_{task_type}.png")
                logger.info(f"📊 Sample visualization saved to: reports/arcade_sample_{task_type}.png")
            except Exception as e:
                logger.warning(f"Could not create visualization: {e}")
        
        return validation_data
        
    except FileNotFoundError as e:
        logger.warning(f"⚠️  ARCADE dataset not found: {e}")
        logger.info("⚠️  Falling back to synthetic data")
        
    except Exception as e:
        logger.error(f"⚠️  Failed to load ARCADE dataset: {e}")
        logger.info("⚠️  Falling back to synthetic data")
    
    # Fallback to synthetic data
    logger.info("🔬 Generating synthetic validation data...")
    return generate_synthetic_validation_data()


def generate_synthetic_validation_data():
    """Generate synthetic data for demonstration (fallback when no ARCADE data)."""
    logger = logging.getLogger(__name__)
    logger.info("Generating synthetic cardiovascular validation data")
    
    np.random.seed(42)
    n_samples = 50
    
    # Generate synthetic vessel segmentation masks
    ground_truth_masks = []
    predicted_masks = []
    
    for i in range(n_samples):
        # Create synthetic vessel-like structure
        gt_mask = create_synthetic_vessel_mask((64, 64))  # 2D for simplicity
        
        # Create prediction with realistic errors
        pred_mask = add_realistic_segmentation_errors(gt_mask)
        
        ground_truth_masks.append(gt_mask)
        predicted_masks.append(pred_mask)
    
    # Generate synthetic demographics
    demographics = {
        'age_group': np.random.choice(['young', 'middle', 'elderly'], n_samples, p=[0.2, 0.5, 0.3]).tolist(),
        'sex': np.random.choice(['male', 'female'], n_samples).tolist(),
        'race': np.random.choice(['white', 'black', 'asian', 'hispanic'], n_samples).tolist(),
        'institution': np.random.choice(['hospital_a', 'hospital_b', 'hospital_c'], n_samples).tolist()
    }
    
    # Generate synthetic clinical measurements
    clinical_data = {
        'vessel_area_mm2': np.random.normal(25, 5, n_samples).tolist(),
        'stenosis_severity_percent': np.random.uniform(0, 85, n_samples).tolist(),
        'vessel_complexity': np.random.normal(1.2, 0.3, n_samples).tolist(),
        'calcium_score': np.random.uniform(0, 300, n_samples).tolist()
    }
    
    return {
        'ground_truth': ground_truth_masks,
        'predictions': predicted_masks,
        'demographics': demographics,
        'clinical_data': clinical_data,
        'metadata': {
            'n_samples': n_samples,
            'dataset_name': 'Synthetic_Vessel_Segmentation',
            'data_source': 'synthetic_generation',
            'task_type': 'vessel_segmentation'
        }
    }


def create_synthetic_vessel_mask(shape):
    """Create synthetic vessel-like binary mask (2D)."""
    mask = np.zeros(shape, dtype=np.uint8)
    
    # Create main vessel branch
    center_x, center_y = shape[1] // 2, shape[0] // 2
    
    # Create curved vessel path
    for y in range(10, shape[0] - 10):
        # Create sinusoidal curve
        x_offset = int(8 * np.sin(y * 0.1))
        width = max(1, int(2 + np.sin(y * 0.2)))
        
        x_center = center_x + x_offset
        
        for dx in range(-width, width + 1):
            x = x_center + dx
            if 0 <= x < shape[1]:
                if dx * dx <= width * width:
                    mask[y, x] = 1
    
    return mask


def add_realistic_segmentation_errors(ground_truth):
    """Add realistic segmentation errors to ground truth."""
    import cv2
    
    prediction = ground_truth.copy()
    
    # Add boundary errors
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.morphologyEx(ground_truth, cv2.MORPH_GRADIENT, kernel)
    boundary_noise = np.random.random(ground_truth.shape) < 0.1
    prediction[boundary & boundary_noise] = 1 - prediction[boundary & boundary_noise]
    
    # Add false positives
    false_positives = (ground_truth == 0) & (np.random.random(ground_truth.shape) < 0.02)
    prediction[false_positives] = 1
    
    # Add false negatives
    false_negatives = (ground_truth == 1) & (np.random.random(ground_truth.shape) < 0.05)
    prediction[false_negatives] = 0
    
    return prediction


def analyze_arcade_performance(validation_data: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance specifically for ARCADE dataset characteristics."""
    
    task_type = validation_data['metadata'].get('task_type', 'vessel_segmentation')
    n_samples = validation_data['metadata']['n_samples']
    
    # Calculate ARCADE-specific metrics
    if task_type == 'stenosis_detection':
        # Stenosis detection analysis
        gt_values = validation_data['ground_truth']
        pred_values = validation_data['predictions']
        
        # Calculate stenosis-specific performance
        mae = np.mean(np.abs(np.array(gt_values) - np.array(pred_values)))
        stenosis_accuracy = 1.0 - (mae / 100.0)  # Convert MAE to accuracy
        
        overall_score = stenosis_accuracy
        data_quality = "excellent" if n_samples > 500 else "good" if n_samples > 100 else "limited"
        clinical_relevance = "high" if mae < 10 else "moderate" if mae < 20 else "low"
        
    else:
        # Vessel segmentation analysis
        vessel_results = results.get('vessel_segmentation', {})
        geometric_metrics = vessel_results.get('geometric_metrics', {})
        
        dice = geometric_metrics.get('dice_coefficient', 0)
        sensitivity = geometric_metrics.get('sensitivity', 0)
        specificity = geometric_metrics.get('specificity', 0)
        
        # ARCADE performance score (weighted combination)
        overall_score = 0.4 * dice + 0.3 * sensitivity + 0.3 * specificity
        
        data_quality = "excellent" if n_samples > 500 else "good" if n_samples > 100 else "limited"
        
        if dice >= 0.8:
            clinical_relevance = "high"
        elif dice >= 0.6:
            clinical_relevance = "moderate"
        else:
            clinical_relevance = "low"
    
    return {
        'overall_score': overall_score,
        'data_quality': data_quality,
        'clinical_relevance': clinical_relevance,
        'task_type': task_type,
        'sample_size': n_samples
    }


def main():
    """Run comprehensive cardiovascular validation with ARCADE dataset."""
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    print("🏥 Cardiovascular Image Validation Pipeline")
    print("=" * 50)
    print("🔬 FDA-Compliant Validation for Medical Imaging Systems")
    print("🎯 Using ARCADE Coronary Artery Dataset")
    print()
    
    # CONFIGURATION: ARCADE Dataset Settings
    ARCADE_DATA_PATH = "data/arcade"  # Path to your ARCADE dataset
    TASK_TYPE = "syntax"  # "syntax" for vessel segmentation, "stenosis" for stenosis detection
    
    # Alternative paths to try
    arcade_paths = [
        "data/arcade",
        "../data/arcade", 
        "arcade",
        str(Path.cwd() / "data" / "arcade"),
        str(Path.cwd() / "arcade")
    ]
    
    print(f"📊 Task Type: {TASK_TYPE}")
    print(f"📁 Looking for ARCADE data...")
    
    # Try to find and load ARCADE dataset
    validation_data = None
    
    for data_path in arcade_paths:
        if Path(data_path).exists():
            print(f"   ✅ Found ARCADE data at: {data_path}")
            try:
                validation_data = load_arcade_data(data_path, TASK_TYPE)
                break
            except Exception as e:
                logger.warning(f"Failed to load from {data_path}: {str(e)}")
                continue
    
    # If no ARCADE data found, use synthetic
    if validation_data is None:
        print("   ⚠️  No ARCADE data found in expected locations")
        print("📊 Using synthetic data for demonstration")
        validation_data = load_arcade_data("", TASK_TYPE)  # Will fallback to synthetic
    else:
        print(f"🏥 Successfully loaded: {validation_data['metadata']['dataset_name']}")
        print(f"👥 {validation_data['metadata']['n_samples']} samples loaded")
        print(f"🔬 Data source: {validation_data['metadata']['data_source']}")
    
    print()
    
    # Initialize validation components
    logger.info("Initializing validation components")
    
    vessel_validator = VesselSegmentationValidator()
    statistical_validator = StatisticalValidator()
    bias_assessor = BiasAssessor()
    uncertainty_quantifier = UncertaintyQuantifier()
    report_generator = ValidationReportGenerator()
    
    # Extract data
    ground_truth = validation_data['ground_truth']
    predictions = validation_data['predictions']
    demographics = validation_data['demographics']
    clinical_data = validation_data['clinical_data']
    
    print("🔬 Running Comprehensive Validation Analysis...")
    print()
    
    # Handle different data types based on task
    vessel_results = {}
    
    if validation_data['metadata'].get('task_type') == 'stenosis_detection':
        # For stenosis detection, we have scalar values
        print("🩺 Stenosis Detection Validation")
        
        # Convert to measurements for validation
        gt_measurements = np.array(ground_truth)
        pred_measurements = np.array(predictions)
        
    else:
        # For vessel segmentation, we have binary masks
        print("🔬 Vessel Segmentation Validation")
        
        # 1. Vessel Segmentation Validation
        print("1️⃣  Geometric Validation (Vessel Segmentation)")
        vessel_results = vessel_validator.validate_segmentation(
            np.array(ground_truth), 
            np.array(predictions)
        )
        
        dice_score = vessel_results['geometric_metrics']['dice_coefficient']
        hausdorff_dist = vessel_results['geometric_metrics']['hausdorff_distance']
        print(f"   📊 Dice Coefficient: {dice_score:.3f}")
        print(f"   📏 Hausdorff Distance: {hausdorff_dist:.2f}")
        print(f"   ✅ Quality: {vessel_results['quality_assessment']['overall_quality']}")
        print()
        
        # Convert masks to measurements for statistical analysis
        gt_measurements = [np.sum(mask) for mask in ground_truth]  # Volume proxy
        pred_measurements = [np.sum(mask) for mask in predictions]
    
    # 2. Statistical Validation
    print("2️⃣  Statistical Validation (FDA Methods)")
    
    statistical_results = statistical_validator.comprehensive_statistical_validation(
        np.array(gt_measurements),
        np.array(pred_measurements),
        clinical_data
    )
    
    icc_value = statistical_results['agreement_analysis']['icc']['icc_value']
    bias = statistical_results['bland_altman_analysis']['bias']
    correlation = statistical_results['correlation_analysis']['pearson']['correlation']
    
    print(f"   📈 ICC: {icc_value:.3f} ({statistical_results['agreement_analysis']['icc']['interpretation']})")
    print(f"   ⚖️  Bias: {bias:.3f}")
    print(f"   🔗 Correlation: {correlation:.3f}")
    print()
    
    # 3. Bias Assessment
    print("3️⃣  Bias Assessment (Algorithmic Fairness)")
    bias_results = bias_assessor.assess_algorithmic_bias(
        np.array(pred_measurements),
        demographics,
        np.array(gt_measurements)
    )
    
    bias_status = bias_results['bias_summary']['overall_bias_status']
    critical_issues = len(bias_results['bias_summary']['critical_bias_issues'])
    
    print(f"   ⚖️  Bias Status: {bias_status}")
    print(f"   ⚠️  Critical Issues: {critical_issues}")
    print()
    
    # 4. Uncertainty Quantification
    print("4️⃣  Uncertainty Quantification")
    uncertainty_results = uncertainty_quantifier.quantify_uncertainty(
        np.array(pred_measurements),
        ground_truth=np.array(gt_measurements)
    )
    
    uncertainty_status = uncertainty_results['uncertainty_summary']['uncertainty_status']
    
    print(f"   🎲 Uncertainty Status: {uncertainty_status}")
    print()
    
    # 5. Generate Comprehensive Report
    print("5️⃣  Generating FDA-Compliant Validation Report")
    
    all_results = {
        'vessel_segmentation': vessel_results,
        'statistical_analysis': statistical_results,
        'bias_assessment': bias_results,
        'uncertainty_quantification': uncertainty_results
    }
    
    # Create reports directory
    Path("reports/arcade_validation").mkdir(parents=True, exist_ok=True)
    
    report_path = report_generator.generate_comprehensive_report(
        all_results,
        output_dir="reports/arcade_validation"
    )
    
    print(f"   📋 Report Generated: {report_path}")
    print()
    
    # 6. ARCADE-Specific Analysis
    print("6️⃣  ARCADE Dataset-Specific Analysis")
    arcade_analysis = analyze_arcade_performance(validation_data, all_results)
    print(f"   🎯 ARCADE Performance Score: {arcade_analysis['overall_score']:.3f}")
    print(f"   📊 Dataset Quality: {arcade_analysis['data_quality']}")
    print(f"   🏥 Clinical Relevance: {arcade_analysis['clinical_relevance']}")
    print()
    
    # Summary
    print("🎯 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"📊 Dataset: {validation_data['metadata']['dataset_name']}")
    print(f"👥 Samples: {validation_data['metadata']['n_samples']}")
    
    if validation_data['metadata'].get('task_type') != 'stenosis_detection':
        print(f"🎯 Dice Score: {dice_score:.3f}")
    
    print(f"📈 ICC: {icc_value:.3f}")
    print(f"⚖️  Bias: {bias_status}")
    print(f"🎲 Uncertainty: {uncertainty_status}")
    print()
    
    # Determine overall validation status
    if validation_data['metadata'].get('task_type') == 'stenosis_detection':
        # For stenosis detection
        if icc_value >= 0.75 and bias_status in ['minimal_bias', 'acceptable_bias']:
            print("✅ STENOSIS DETECTION VALIDATION PASSED")
        elif icc_value >= 0.5:
            print("⚠️  CONDITIONAL PASS - Additional validation recommended")
        else:
            print("❌ VALIDATION FAILED - Algorithm requires improvement")
    else:
        # For vessel segmentation
        if dice_score >= 0.7 and icc_value >= 0.75 and bias_status in ['minimal_bias', 'acceptable_bias']:
            print("✅ VESSEL SEGMENTATION VALIDATION PASSED")
        elif dice_score >= 0.5 and icc_value >= 0.5:
            print("⚠️  CONDITIONAL PASS - Additional validation recommended")
        else:
            print("❌ VALIDATION FAILED - Algorithm requires improvement")
    
    print()
    print("🏥 ARCADE cardiovascular validation pipeline completed successfully!")
    print(f"📋 Detailed results available in: {report_path}")
    print("🎯 Ready for Medis Imaging interview demonstration!")
    
    return all_results


if __name__ == "__main__":
    results = main()