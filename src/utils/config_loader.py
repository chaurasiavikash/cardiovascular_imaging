"""
Configuration Loader for Cardiovascular Validation Pipeline
Handles loading and validation of configuration files
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Configuration loader for the cardiovascular validation pipeline.
    
    Handles loading YAML configuration files and provides defaults
    for missing configuration sections.
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        logger = logging.getLogger(__name__)
        
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            logger.info("Using default configuration")
            return ConfigLoader.get_default_config()
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                logger.warning("Configuration file is empty, using defaults")
                return ConfigLoader.get_default_config()
            
            # Validate and fill missing sections
            config = ConfigLoader.validate_and_fill_config(config)
            
            logger.info(f"Configuration loaded successfully from: {config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {str(e)}")
            logger.info("Using default configuration")
            return ConfigLoader.get_default_config()
        
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            logger.info("Using default configuration")
            return ConfigLoader.get_default_config()
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default configuration for the validation pipeline.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "data": {
                "input_formats": ["numpy", "dicom", "nifti", "json"],
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
                "report_title": "Cardiovascular Validation Report",
                "organization": "Medical Imaging Validation Lab",
                "template_style": "professional"
            }
        }
    
    @staticmethod
    def validate_and_fill_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration and fill missing sections with defaults.
        
        Args:
            config: Input configuration dictionary
            
        Returns:
            Validated and completed configuration
        """
        logger = logging.getLogger(__name__)
        default_config = ConfigLoader.get_default_config()
        
        # Fill missing top-level sections
        for section_name, default_section in default_config.items():
            if section_name not in config:
                config[section_name] = default_section
                logger.info(f"Added missing configuration section: {section_name}")
            elif isinstance(default_section, dict):
                # Fill missing subsection items
                for key, default_value in default_section.items():
                    if key not in config[section_name]:
                        config[section_name][key] = default_value
                        logger.debug(f"Added missing config item: {section_name}.{key}")
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary to save
            config_path: Path where to save the configuration
        """
        logger = logging.getLogger(__name__)
        
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    @staticmethod
    def get_section(config: Dict[str, Any], section_name: str) -> Dict[str, Any]:
        """
        Get a specific configuration section.
        
        Args:
            config: Configuration dictionary
            section_name: Name of the section to retrieve
            
        Returns:
            Configuration section or empty dict if not found
        """
        return config.get(section_name, {})
    
    @staticmethod
    def get_value(config: Dict[str, Any], section_name: str, 
                  key: str, default_value: Any = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            config: Configuration dictionary
            section_name: Name of the configuration section
            key: Configuration key
            default_value: Default value if not found
            
        Returns:
            Configuration value or default
        """
        section = config.get(section_name, {})
        return section.get(key, default_value)
    
    @staticmethod
    def validate_required_sections(config: Dict[str, Any], 
                                 required_sections: list) -> bool:
        """
        Validate that required configuration sections are present.
        
        Args:
            config: Configuration dictionary
            required_sections: List of required section names
            
        Returns:
            True if all required sections are present
        """
        logger = logging.getLogger(__name__)
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            logger.error(f"Missing required configuration sections: {missing_sections}")
            return False
        
        return True
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base with
            
        Returns:
            Merged configuration dictionary
        """
        merged_config = base_config.copy()
        
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged_config:
                if isinstance(merged_config[key], dict):
                    merged_config[key] = ConfigLoader.merge_configs(
                        merged_config[key], value
                    )
                else:
                    merged_config[key] = value
            else:
                merged_config[key] = value
        
        return merged_config