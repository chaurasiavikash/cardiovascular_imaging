#!/usr/bin/env python3
"""
Basic Validation Example for Cardiovascular Image Validation Pipeline
Demonstrates end-to-end validation workflow for cardiovascular imaging systems

Author: Vikash Chaurasia
For: Medis Imaging Scientific Validation Specialist Interview
"""

import logging
import numpy as np
import sys
from pathlib import Path
 

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
from utils.config_loader import ConfigLoader


def main():
    """Run basic validation example."""
    
    # Setup logging
    setup_logging("INFO", "validation_example.log")
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Cardiovascular Image Validation Pipeline Demo")
    logger.info("=" * 60)
    
    try:
        # Step 1: Initialize components
        logger.info("📋 Step 1: Initializing validation components")
        
        # Load configuration (create default if not exists)
        config_path = "config/validation_config.yaml"
        try:
            config = ConfigLoader.load_config(config_path)
            logger.info(f"✅ Loaded configuration from {config_path}")
        except:
            config = create_default_config()
            logger.info("✅ Using default configuration")
        
        # Initialize components
        data_manager = DataManager(config.get("data", {}))
        vessel_validator = VesselSegmentationValidator(config.get("vessel_validation", {}))
        statistical_validator = StatisticalValidator(config.get("statistical_validation", {}))
        bias_assessor = BiasAssessor(config.get("bias_assessment", {}))
        uncertainty_quantifier = UncertaintyQuantifier(config.get("uncertainty", {}))
        report_generator = ValidationReportGenerator(config.get("reporting", {}))
        
        logger.info("✅ All validation components initialized successfully")
        
        # Step 2: Generate or load validation data
        logger.info("📊 Step 2: Preparing validation datasets")
        
        # Create synthetic data for demonstration
        validation_data = generate_synthetic_validation_data()
        logger.info(f"✅ Generated synthetic validation data: {validation_data['metadata']['n_samples']} samples")
        
        # Step 3: Vessel segmentation validation
        logger.info("🔬 Step 3: Running vessel segmentation validation")
        
        vessel_results = vessel_validator.validate_segmentation(
            validation_data["ground_truth"], 
            validation_data["predictions"],
            spacing=(0.5, 0.5, 0.5)  # Sample spacing in mm
        )
        
        dice_score = vessel_results['geometric_metrics']['dice_coefficient']
        logger.info(f"✅ Vessel validation completed - Dice Score: {dice_score:.3f}")
        
        # Step 4: Statistical validation
        logger.info("📈 Step 4: Performing statistical validation analysis")
        
        # Use flattened data for statistical validation
        gt_flat = validation_data["ground_truth"].flatten()
        pred_flat = validation_data["predictions"].flatten()
        
        statistical_results = statistical_validator.comprehensive_statistical_validation(
            gt_flat,
            pred_flat,
            validation_data.get("clinical_data", {})
        )
        
        icc_value = statistical_results.get('agreement_analysis', {}).get('icc', {}).get('icc_value', 0.0)
        logger.info(f"✅ Statistical validation completed - ICC: {icc_value:.3f}")
        
        # Step 5: Bias assessment
        logger.info("⚖️ Step 5: Conducting algorithmic bias assessment")
        
        # Fix data dimensions for bias assessment
        pred_flat = validation_data["predictions"].flatten()
        gt_flat = validation_data["ground_truth"].flatten()
        demographics = validation_data.get("demographics", {})
        
        # Ensure demographics match the number of samples, not flattened data
        n_samples = len(validation_data["predictions"])
        demographics_fixed = {}
        for key, values in demographics.items():
            if len(values) == n_samples:
                # Expand demographics to match flattened data
                demographics_fixed[key] = []
                for i, sample_value in enumerate(values):
                    # Repeat each demographic value for all pixels in that sample
                    pixels_per_sample = validation_data["predictions"][i].size
                    demographics_fixed[key].extend([sample_value] * pixels_per_sample)
            else:
                demographics_fixed[key] = values
        
        bias_results = bias_assessor.assess_algorithmic_bias(
            pred_flat,
            demographics_fixed,
            gt_flat
        )
        
        bias_status = bias_results.get('bias_summary', {}).get('overall_bias_status', 'unknown')
        logger.info(f"✅ Bias assessment completed - Status: {bias_status}")
        
        # Step 6: Uncertainty quantification
        logger.info("🎲 Step 6: Quantifying prediction uncertainty")
        
        # Create mock model outputs for uncertainty analysis
        pred_flat = validation_data["predictions"].flatten()
        mock_model_outputs = {
            'prediction_variance': np.random.uniform(0.01, 0.1, len(pred_flat)),
            'ensemble_predictions': np.random.random((5, len(pred_flat)))
        }
        
        uncertainty_results = uncertainty_quantifier.quantify_uncertainty(
            pred_flat,
            mock_model_outputs,
            validation_data["ground_truth"].flatten()
        )
        
        uncertainty_status = uncertainty_results.get('uncertainty_summary', {}).get('uncertainty_status', 'unknown')
        logger.info(f"✅ Uncertainty quantification completed - Status: {uncertainty_status}")
        
        # Step 7: Generate comprehensive reports
        logger.info("📋 Step 7: Generating validation reports")
        
        # Compile all results
        complete_results = {
            'vessel_segmentation': vessel_results,
            'statistical_analysis': statistical_results,
            'bias_assessment': bias_results,
            'uncertainty_quantification': uncertainty_results,
            'metadata': {
                'validation_timestamp': np.datetime64('now').item().isoformat(),
                'algorithm_version': '1.0.0',
                'validation_framework_version': '1.0.0'
            }
        }
        
        # Generate reports
        report_path = report_generator.generate_comprehensive_report(
            complete_results, 
            "reports/demo"
        )
        
        logger.info(f"✅ Comprehensive report generated: {report_path}")
        
        # Step 8: Generate regulatory package
        logger.info("📄 Step 8: Generating FDA regulatory package")
        
        regulatory_path = report_generator.generate_regulatory_package(
            complete_results,
            "reports/regulatory_demo"
        )
        
        logger.info(f"✅ Regulatory package generated: {regulatory_path}")
        
        # Step 9: Display summary results
        logger.info("📊 Step 9: Validation Summary")
        logger.info("=" * 60)
        
        print_validation_summary(complete_results)
        
        logger.info("🎉 Cardiovascular validation pipeline demo completed successfully!")
        logger.info(f"📁 Check the 'reports' directory for generated validation reports")
        logger.info("=" * 60)
        
        return complete_results
        
    except Exception as e:
        logger.error(f"❌ Validation pipeline failed: {str(e)}")
        raise


def create_default_config():
    """Create default configuration for demonstration."""
    return {
        "data": {
            "input_formats": ["numpy", "dicom", "nifti"],
            "preprocessing": {
                "normalize_intensity": True,
                "clip_range": [-200, 800]
            }
        },
        "vessel_validation": {
            "dice_threshold": 0.7,
            "hausdorff_threshold": 5.0,
            "connectivity_threshold": 0.95
        },
        "statistical_validation": {
            "alpha": 0.05,
            "icc_threshold": 0.75,
            "agreement_threshold": 0.8,
            "power_threshold": 0.8
        },
        "bias_assessment": {
            "fairness_threshold": 0.8,
            "statistical_alpha": 0.05,
            "min_group_size": 30,
            "protected_attributes": ["age_group", "sex", "race", "institution"]
        },
        "uncertainty": {
            "calibration_threshold": 0.1,
            "confidence_level": 0.95,
            "bootstrap_samples": 1000,
            "uncertainty_threshold": 0.2
        },
        "reporting": {
            "report_title": "Cardiovascular Validation Report - Demo",
            "organization": "Medical Imaging Validation Lab",
            "template_style": "professional"
        }
    }


def generate_synthetic_validation_data():
    """Generate synthetic validation data for demonstration."""
    np.random.seed(42)  # For reproducible results
    
    n_samples = 150
    image_shape = (32, 32)  # Simplified 2D images for demo
    
    # Generate ground truth segmentations (vessel-like structures)
    ground_truth = []
    predictions = []
    
    for i in range(n_samples):
        # Create synthetic vessel mask
        gt_mask = create_synthetic_vessel_mask(image_shape)
        ground_truth.append(gt_mask)
        
        # Create prediction with some noise
        noise_level = np.random.uniform(0.05, 0.15)
        pred_mask = add_segmentation_noise(gt_mask, noise_level)
        predictions.append(pred_mask)
    
    # Generate demographic data
    demographics = {
        'age_group': np.random.choice(['young', 'middle', 'elderly'], n_samples).tolist(),
        'sex': np.random.choice(['male', 'female'], n_samples).tolist(),
        'race': np.random.choice(['white', 'black', 'asian', 'hispanic'], n_samples).tolist(),
        'institution': np.random.choice(['hospital_a', 'hospital_b', 'hospital_c'], n_samples).tolist()
    }
    
    # Generate clinical data
    clinical_data = {
        'vessel_diameter': (np.random.normal(3.5, 0.8, n_samples)).tolist(),
        'stenosis_severity': (np.random.uniform(0, 100, n_samples)).tolist(),
        'ejection_fraction': (np.random.normal(60, 10, n_samples)).tolist(),
        'age': (np.random.randint(18, 85, n_samples)).tolist()
    }
    
    return {
        'ground_truth': np.array(ground_truth),
        'predictions': np.array(predictions),
        'demographics': demographics,
        'clinical_data': clinical_data,
        'metadata': {
            'n_samples': n_samples,
            'image_shape': image_shape,
            'data_type': 'synthetic_cardiovascular',
            'generation_seed': 42
        }
    }


def create_synthetic_vessel_mask(shape):
    """Create a synthetic vessel-like binary mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    
    # Create curved vessel structure
    center_x = shape[1] // 2
    vessel_width = np.random.randint(2, 5)
    
    for y in range(5, shape[0] - 5):
        # Create sinusoidal curve with some randomness
        x_offset = int(8 * np.sin(y * 0.2) + np.random.normal(0, 1))
        x_center = center_x + x_offset
        
        # Draw vessel with varying width
        current_width = vessel_width + int(np.random.normal(0, 0.5))
        current_width = max(1, min(current_width, 6))
        
        x_start = max(0, x_center - current_width)
        x_end = min(shape[1], x_center + current_width)
        
        mask[y, x_start:x_end] = 1
    
    # Add some branching occasionally
    if np.random.random() > 0.7:
        branch_start_y = shape[0] // 2
        for y in range(branch_start_y, min(branch_start_y + 15, shape[0] - 5)):
            x_offset = center_x + int((y - branch_start_y) * 0.8)
            if 0 <= x_offset < shape[1]:
                mask[y, x_offset:x_offset+2] = 1
    
    return mask


def add_segmentation_noise(gt_mask, noise_level):
    """Add realistic segmentation noise to ground truth mask."""
    pred_mask = gt_mask.copy()
    
    # Add boundary errors (most common type of segmentation error)
    from scipy import ndimage
    
    # Find boundaries
    boundary = gt_mask - ndimage.binary_erosion(gt_mask)
    boundary_indices = np.where(boundary)
    
    # Add noise to boundary pixels
    for i in range(len(boundary_indices[0])):
        if np.random.random() < noise_level:
            y, x = boundary_indices[0][i], boundary_indices[1][i]
            
            # Random displacement
            dy = np.random.randint(-1, 2)
            dx = np.random.randint(-1, 2)
            
            new_y = np.clip(y + dy, 0, pred_mask.shape[0] - 1)
            new_x = np.clip(x + dx, 0, pred_mask.shape[1] - 1)
            
            # Flip the pixel at new location
            pred_mask[new_y, new_x] = 1 - pred_mask[new_y, new_x]
    
    # Add small false positives
    false_positive_rate = noise_level * 0.5
    false_positive_mask = (gt_mask == 0) & (np.random.random(gt_mask.shape) < false_positive_rate)
    pred_mask[false_positive_mask] = 1
    
    # Add small false negatives
    false_negative_rate = noise_level * 0.3
    false_negative_mask = (gt_mask == 1) & (np.random.random(gt_mask.shape) < false_negative_rate)
    pred_mask[false_negative_mask] = 0
    
    return pred_mask.astype(np.uint8)


def print_validation_summary(results):
    """Print a formatted validation summary."""
    print("\n🏥 CARDIOVASCULAR VALIDATION SUMMARY")
    print("=" * 50)
    
    # Vessel Segmentation Results
    vessel_results = results.get('vessel_segmentation', {})
    geometric = vessel_results.get('geometric_metrics', {})
    quality = vessel_results.get('quality_assessment', {})
    
    print("\n🔬 VESSEL SEGMENTATION VALIDATION")
    print(f"   Dice Coefficient:     {geometric.get('dice_coefficient', 0.0):.3f}")
    print(f"   Jaccard Index:        {geometric.get('jaccard_index', 0.0):.3f}")
    print(f"   Sensitivity:          {geometric.get('sensitivity', 0.0):.3f}")
    print(f"   Specificity:          {geometric.get('specificity', 0.0):.3f}")
    print(f"   Overall Quality:      {quality.get('overall_quality', 'unknown')}")
    
    # Statistical Validation Results
    statistical_results = results.get('statistical_analysis', {})
    agreement = statistical_results.get('agreement_analysis', {})
    icc_data = agreement.get('icc', {})
    
    print("\n📈 STATISTICAL VALIDATION")
    print(f"   ICC Value:            {icc_data.get('icc_value', 0.0):.3f}")
    print(f"   ICC Interpretation:   {icc_data.get('interpretation', 'unknown')}")
    print(f"   Mean Absolute Error:  {agreement.get('mean_absolute_error', 0.0):.3f}")
    print(f"   RMSE:                 {agreement.get('root_mean_square_error', 0.0):.3f}")
    
    # Bias Assessment Results
    bias_results = results.get('bias_assessment', {})
    bias_summary = bias_results.get('bias_summary', {})
    
    print("\n⚖️ BIAS ASSESSMENT")
    print(f"   Overall Bias Status:  {bias_summary.get('overall_bias_status', 'unknown')}")
    print(f"   Fairness Scores:      {len(bias_summary.get('fairness_scores', {}))} demographic groups analyzed")
    print(f"   Critical Issues:      {len(bias_summary.get('critical_bias_issues', []))} issues detected")
    
    # Uncertainty Quantification Results
    uncertainty_results = results.get('uncertainty_quantification', {})
    uncertainty_summary = uncertainty_results.get('uncertainty_summary', {})
    calibration = uncertainty_results.get('calibration_analysis', {})
    
    print("\n🎲 UNCERTAINTY QUANTIFICATION")
    print(f"   Uncertainty Status:   {uncertainty_summary.get('uncertainty_status', 'unknown')}")
    print(f"   Calibration Error:    {calibration.get('expected_calibration_error', 0.0):.3f}")
    print(f"   Well Calibrated:      {calibration.get('is_well_calibrated', False)}")
    
    # Overall Assessment
    print("\n🎯 OVERALL ASSESSMENT")
    
    # Determine overall validation status with safe access
    dice_pass = geometric.get('dice_coefficient', 0.0) >= 0.7
    icc_pass = icc_data.get('icc_value', 0.0) >= 0.75
    bias_pass = bias_summary.get('overall_bias_status', '') in ['minimal_bias', 'acceptable_bias']
    uncertainty_pass = uncertainty_summary.get('uncertainty_status', '') in ['excellent', 'good']
    
    validation_status = "PASS" if all([dice_pass, icc_pass, bias_pass, uncertainty_pass]) else "CONDITIONAL PASS" if sum([dice_pass, icc_pass, bias_pass, uncertainty_pass]) >= 3 else "NEEDS IMPROVEMENT"
    
    status_emoji = "✅" if validation_status == "PASS" else "⚠️" if validation_status == "CONDITIONAL PASS" else "❌"
    
    print(f"   {status_emoji} Validation Status: {validation_status}")
    print(f"   📊 Dice Coefficient:   {'✅' if dice_pass else '❌'} {geometric.get('dice_coefficient', 0.0):.3f}")
    print(f"   📈 ICC Agreement:      {'✅' if icc_pass else '❌'} {icc_data.get('icc_value', 0.0):.3f}")
    print(f"   ⚖️ Bias Assessment:    {'✅' if bias_pass else '❌'} {bias_summary.get('overall_bias_status', 'unknown')}")
    print(f"   🎲 Uncertainty:        {'✅' if uncertainty_pass else '❌'} {uncertainty_summary.get('uncertainty_status', 'unknown')}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    if validation_status == "PASS":
        print("   • Algorithm ready for clinical validation studies")
        print("   • Proceed with regulatory submission preparation")
        print("   • Implement monitoring for production deployment")
    elif validation_status == "CONDITIONAL PASS":
        print("   • Address failing validation criteria before clinical use")
        print("   • Consider additional validation studies")
        print("   • Implement enhanced monitoring and safety measures")
    else:
        print("   • Significant improvements needed before clinical consideration")
        print("   • Review algorithm architecture and training data")
        print("   • Conduct comprehensive failure analysis")
    
    print("\n📋 REGULATORY READINESS")
    regulatory_ready = validation_status == "PASS"
    print(f"   FDA 510(k) Ready:     {'✅' if regulatory_ready else '❌'}")
    print(f"   Clinical Study Ready: {'✅' if validation_status in ['PASS', 'CONDITIONAL PASS'] else '❌'}")
    print(f"   Production Ready:     {'✅' if regulatory_ready else '❌'}")
    
    print("\n" + "=" * 50)
    print("🎉 Validation pipeline demonstration completed!")
    print("📁 Check 'reports/demo' for detailed validation reports")
    print("📄 Check 'reports/regulatory_demo' for FDA documentation")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        # Run the validation example
        results = main()
        
        # Additional demo features
        print("\n🚀 ADDITIONAL DEMO FEATURES")
        print("=" * 50)
        print("To explore more features, try:")
        print("• python main.py --help                    # View CLI options")
        print("• python main.py --validation-type phantom # Run phantom validation")
        print("• python main.py --validation-type regulatory # Generate regulatory package")
        print("• Check config/validation_config.yaml      # Customize validation parameters")
        print("• Review src/validation/ modules           # Explore validation algorithms")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("Please check the logs and ensure all dependencies are installed.")
        sys.exit(1)