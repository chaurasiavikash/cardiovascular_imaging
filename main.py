"""
Cardiovascular Image Validation Pipeline - Main Entry Point
FDA-compliant validation pipeline for cardiovascular imaging systems

Author: Vikash Chaurasia
For: Medis Imaging Scientific Validation Specialist Interview
"""

import click
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from src.core.data_manager import DataManager
from src.validation.vessel_segmentation_validator import VesselSegmentationValidator
from src.validation.statistical_validator import StatisticalValidator
from src.validation.bias_assessor import BiasAssessor
from src.validation.uncertainty_quantifier import UncertaintyQuantifier
from src.visualization.report_generator import ValidationReportGenerator
from src.utils.logger import setup_logging
from src.utils.config_loader import ConfigLoader


class CardiovascularValidationPipeline:
    """
    Main validation pipeline for cardiovascular imaging systems.
    
    Implements FDA-compliant validation protocols for medical imaging devices,
    focusing on statistical validation, bias assessment, and uncertainty quantification.
    """
    
    def __init__(self, config_path: str = "config/validation_config.yaml"):
        """Initialize the validation pipeline."""
        self.config = ConfigLoader.load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_manager = DataManager(self.config.get("data", {}))
        self.vessel_validator = VesselSegmentationValidator(self.config.get("vessel_validation", {}))
        self.statistical_validator = StatisticalValidator(self.config.get("statistical_validation", {}))
        self.bias_assessor = BiasAssessor(self.config.get("bias_assessment", {}))
        self.uncertainty_quantifier = UncertaintyQuantifier(self.config.get("uncertainty", {}))
        self.report_generator = ValidationReportGenerator(self.config.get("reporting", {}))
        
        self.validation_results = {}
        
    def run_comprehensive_validation(self, data_path: str, output_dir: str = "reports/validation") -> Dict[str, Any]:
        """
        Run complete validation pipeline including all FDA-required components.
        
        Args:
            data_path: Path to input data directory
            output_dir: Path to output directory for reports
            
        Returns:
            Dictionary containing all validation results
        """
        self.logger.info("Starting comprehensive cardiovascular validation pipeline")
        
        try:
            # Step 1: Load and prepare data
            self.logger.info("Loading and preparing validation data")
            datasets = self.data_manager.load_validation_datasets(data_path)
            
            # Step 2: Vessel segmentation validation
            self.logger.info("Running vessel segmentation validation")
            vessel_results = self.vessel_validator.validate_segmentation(
                datasets["ground_truth"], 
                datasets["predictions"]
            )
            self.validation_results["vessel_segmentation"] = vessel_results
            
            # Step 3: Statistical validation
            self.logger.info("Performing statistical validation analysis")
            statistical_results = self.statistical_validator.comprehensive_statistical_validation(
                datasets["ground_truth"], 
                datasets["predictions"],
                datasets.get("clinical_data", {})
            )
            self.validation_results["statistical_analysis"] = statistical_results
            
            # Step 4: Bias assessment
            self.logger.info("Conducting bias assessment across patient demographics")
            bias_results = self.bias_assessor.assess_algorithmic_bias(
                datasets["predictions"],
                datasets.get("demographics", {}),
                datasets["ground_truth"]
            )
            self.validation_results["bias_assessment"] = bias_results
            
            # Step 5: Uncertainty quantification
            self.logger.info("Quantifying prediction uncertainty")
            uncertainty_results = self.uncertainty_quantifier.quantify_uncertainty(
                datasets["predictions"],
                datasets.get("model_outputs", {})
            )
            self.validation_results["uncertainty_quantification"] = uncertainty_results
            
            # Step 6: Generate comprehensive reports
            self.logger.info("Generating validation reports")
            report_path = self.report_generator.generate_comprehensive_report(
                self.validation_results,
                output_dir
            )
            
            self.logger.info(f"Validation pipeline completed successfully. Report saved to: {report_path}")
            
            return self.validation_results
            
        except Exception as e:
            self.logger.error(f"Validation pipeline failed: {str(e)}")
            raise
    
    def run_phantom_validation(self, phantom_config: Dict[str, Any], output_dir: str = "reports/phantom") -> Dict[str, Any]:
        """
        Run phantom-based validation for accuracy assessment.
        
        Args:
            phantom_config: Configuration for phantom validation
            output_dir: Output directory for phantom reports
            
        Returns:
            Phantom validation results
        """
        self.logger.info("Running phantom validation")
        
        # Generate synthetic phantom data
        phantom_data = self.data_manager.generate_phantom_data(phantom_config)
        
        # Run validation on phantom
        phantom_results = self.vessel_validator.validate_phantom_accuracy(phantom_data)
        
        # Generate phantom report
        phantom_report_path = self.report_generator.generate_phantom_report(
            phantom_results, 
            output_dir
        )
        
        self.logger.info(f"Phantom validation completed. Report: {phantom_report_path}")
        
        return phantom_results
    
    def generate_regulatory_package(self, output_dir: str = "reports/regulatory") -> str:
        """
        Generate FDA-compliant regulatory validation package.
        
        Args:
            output_dir: Output directory for regulatory package
            
        Returns:
            Path to generated regulatory package
        """
        self.logger.info("Generating FDA-compliant regulatory validation package")
        
        regulatory_package_path = self.report_generator.generate_regulatory_package(
            self.validation_results,
            output_dir
        )
        
        self.logger.info(f"Regulatory package generated: {regulatory_package_path}")
        
        return regulatory_package_path


@click.command()
@click.option("--data-path", "-d", required=True, help="Path to validation data directory")
@click.option("--config", "-c", default="config/validation_config.yaml", help="Path to configuration file")
@click.option("--output-dir", "-o", default="reports", help="Output directory for reports")
@click.option("--validation-type", "-t", 
              type=click.Choice(["comprehensive", "phantom", "regulatory"]), 
              default="comprehensive",
              help="Type of validation to run")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--phantom-config", help="Path to phantom configuration file")
def main(data_path: str, config: str, output_dir: str, validation_type: str, 
         log_level: str, phantom_config: Optional[str]):
    """
    Cardiovascular Image Validation Pipeline - Main CLI Interface
    
    FDA-compliant validation pipeline for cardiovascular imaging systems.
    Designed for medical device validation according to FDA guidance.
    """
    
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Cardiovascular Image Validation Pipeline")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Config: {config}")
    logger.info(f"Validation type: {validation_type}")
    
    try:
        # Initialize pipeline
        pipeline = CardiovascularValidationPipeline(config)
        
        # Run specified validation type
        if validation_type == "comprehensive":
            results = pipeline.run_comprehensive_validation(data_path, output_dir)
            logger.info("Comprehensive validation completed successfully")
            
        elif validation_type == "phantom":
            if not phantom_config:
                raise ValueError("Phantom configuration file required for phantom validation")
            
            with open(phantom_config, 'r') as f:
                phantom_cfg = yaml.safe_load(f)
            
            results = pipeline.run_phantom_validation(phantom_cfg, output_dir)
            logger.info("Phantom validation completed successfully")
            
        elif validation_type == "regulatory":
            # Run comprehensive validation first if not already done
            if not hasattr(pipeline, 'validation_results') or not pipeline.validation_results:
                logger.info("Running comprehensive validation for regulatory package")
                pipeline.run_comprehensive_validation(data_path, output_dir)
            
            regulatory_path = pipeline.generate_regulatory_package(output_dir)
            logger.info(f"Regulatory package generated: {regulatory_path}")
        
        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise click.ClickException(f"Validation failed: {str(e)}")


if __name__ == "__main__":
    main()