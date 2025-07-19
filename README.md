# Cardiovascular Image Validation Pipeline

A comprehensive FDA-compliant validation framework for cardiovascular imaging algorithms, specifically designed for medical device regulatory submissions and clinical validation studies.

## Overview

This pipeline implements rigorous validation methodologies required for cardiovascular imaging systems, including vessel segmentation algorithms, quantitative flow ratio (QFR) technologies, and other cardiac imaging applications. The framework follows FDA guidance for Software as Medical Device (SaMD) validation and supports regulatory submissions such as 510(k) premarket notifications.

### Key Features

- **Statistical Validation**: Intraclass correlation coefficient (ICC) analysis, Bland-Altman agreement assessment, and comprehensive hypothesis testing
- **Bias Assessment**: Algorithmic fairness evaluation across demographic groups with demographic parity and equalized odds analysis
- **Uncertainty Quantification**: Calibration analysis, bootstrap confidence intervals, and predictive uncertainty estimation
- **Vessel Segmentation Validation**: Geometric accuracy metrics including Dice coefficient, Hausdorff distance, and topological validation
- **Regulatory Documentation**: Automated generation of FDA-compliant validation reports and 510(k) submission materials

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Quick Setup

1. Clone the repository and navigate to the project directory
2. Make the setup script executable:
   ```bash
   chmod +x setup_project.sh
   ```
3. Run the automated setup:
   ```bash
   ./setup_project.sh
   ```

### Manual Installation

If you prefer manual installation:

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```bash
   mkdir -p reports data/input data/output logs
   ```

## Quick Start

### Running the Demo

To see the complete validation pipeline in action:

```bash
python examples/basic_validation_example.py
```

This demonstration will:
- Generate synthetic cardiovascular validation data
- Perform comprehensive validation analysis
- Create detailed HTML and PDF reports
- Generate FDA-compliant regulatory documentation

### Using the Command Line Interface

The pipeline includes a CLI for production use:

```bash
# View available options
python main.py --help

# Run comprehensive validation
python main.py --data-path data/example --validation-type comprehensive

# Generate regulatory package
python main.py --data-path data/example --validation-type regulatory

# Run phantom validation
python main.py --data-path data/example --validation-type phantom
```

## Project Structure

```
cardiovascular-validation-pipeline/
├── setup_project.sh                 # Automated setup script
├── setup.py                         # Package installation configuration
├── requirements.txt                 # Python dependencies
├── main.py                          # Command line interface
├── config/
│   └── validation_config.yaml       # Validation parameters and thresholds
├── src/
│   ├── core/
│   │   ├── dicom_processor.py        # Medical imaging data handling
│   │   └── data_manager.py           # Data loading and preprocessing
│   ├── validation/
│   │   ├── vessel_segmentation_validator.py  # Geometric validation metrics
│   │   ├── statistical_validator.py          # Statistical analysis methods
│   │   ├── bias_assessor.py                  # Algorithmic bias evaluation
│   │   └── uncertainty_quantifier.py         # Uncertainty analysis
│   ├── visualization/
│   │   └── report_generator.py       # Report and documentation generation
│   └── utils/
│       ├── logger.py                 # Logging configuration
│       └── config_loader.py          # Configuration management
├── examples/
│   └── basic_validation_example.py  # Complete demonstration workflow
├── tests/                           # Unit and integration tests
└── docs/                           # Documentation
```

## Configuration

The validation pipeline is highly configurable through `config/validation_config.yaml`. Key configuration sections include:

### Validation Thresholds
- **Dice Coefficient**: Minimum threshold for geometric accuracy (default: 0.7)
- **ICC Threshold**: Minimum value for statistical agreement (default: 0.75)
- **Calibration Error**: Maximum expected calibration error (default: 0.1)

### Bias Assessment Parameters
- **Fairness Threshold**: Minimum demographic parity ratio (default: 0.8)
- **Protected Attributes**: Demographic groups to analyze (age, sex, race, institution)
- **Minimum Group Size**: Minimum samples per demographic group (default: 30)

### Statistical Analysis Settings
- **Confidence Level**: Statistical confidence for intervals (default: 95%)
- **Bootstrap Samples**: Number of bootstrap iterations (default: 1000)
- **Significance Level**: Alpha for hypothesis testing (default: 0.05)

## Validation Methodology

### Statistical Validation

The pipeline implements comprehensive statistical validation following FDA guidance:

- **Agreement Analysis**: Intraclass correlation coefficient (ICC) with confidence intervals
- **Bias Assessment**: Bland-Altman analysis with limits of agreement
- **Hypothesis Testing**: Paired t-tests and non-parametric alternatives
- **Power Analysis**: Sample size adequacy assessment

### Bias Assessment

Algorithmic bias is evaluated across multiple dimensions:

- **Demographic Parity**: Equal positive prediction rates across groups
- **Equalized Odds**: Equal true positive and false positive rates
- **Calibration Fairness**: Consistent prediction calibration across demographics
- **Intersectional Analysis**: Multi-attribute bias detection

### Uncertainty Quantification

The framework provides comprehensive uncertainty analysis:

- **Predictive Uncertainty**: Entropy-based measures for classification and regression
- **Calibration Assessment**: Expected calibration error and reliability diagrams
- **Bootstrap Confidence Intervals**: Non-parametric uncertainty bounds
- **Model Uncertainty**: Ensemble and dropout-based uncertainty estimation

### Performance Validation

Vessel segmentation validation includes:

- **Geometric Metrics**: Dice coefficient, Jaccard index, Hausdorff distance
- **Topological Analysis**: Connectivity preservation and centerline accuracy
- **Clinical Metrics**: Vessel diameter accuracy, stenosis detection capability
- **Quality Assessment**: Overall segmentation quality scoring

## Output and Reports

The pipeline generates several types of output:

### Validation Reports
- **HTML Report**: Interactive validation dashboard with plots and metrics
- **PDF Summary**: Executive summary suitable for regulatory review
- **JSON Export**: Machine-readable results for further analysis

### Regulatory Documentation
- **510(k) Summary**: Premarket submission documentation
- **Validation Protocol**: Detailed methodology description
- **Statistical Analysis Plan**: Comprehensive statistical methodology
- **Risk Analysis**: ISO 14971 compliant risk assessment

## Medical Device Compliance

This validation framework is designed to support medical device regulatory submissions:

### FDA Compliance
- Follows FDA guidance for Software as Medical Device (SaMD)
- Implements statistical methods recommended for medical device validation
- Generates documentation suitable for 510(k) premarket submissions

### Quality Standards
- **ISO 13485**: Quality management system requirements
- **ISO 14971**: Risk management for medical devices
- **IEC 62304**: Medical device software lifecycle processes

## Use Cases

### Cardiovascular Imaging Applications
- Quantitative Flow Ratio (QFR) algorithm validation
- Coronary artery segmentation assessment
- Cardiac chamber quantification validation
- Stenosis detection algorithm evaluation

### Multi-Center Studies
- Cross-site validation with site effect analysis
- Demographic bias assessment across institutions
- Performance consistency evaluation

### Regulatory Submissions
- FDA 510(k) premarket notification support
- CE marking technical documentation
- Clinical evaluation report generation

## Data Requirements

### Input Data Format
The pipeline supports multiple data formats:
- DICOM medical imaging files
- NIfTI neuroimaging format
- NumPy arrays for processed data
- JSON for metadata and demographics

### Ground Truth Requirements
- Binary or multi-class segmentation masks
- Quantitative measurements (diameters, volumes)
- Clinical annotations and expert assessments

### Demographic Data
For bias assessment, provide:
- Age groups (young, middle-aged, elderly)
- Sex/gender information
- Race and ethnicity data
- Institution or site identifiers

## Performance Considerations

The pipeline is optimized for large-scale validation studies:
- Efficient memory usage for large medical imaging datasets
- Parallel processing for statistical computations
- Sampling strategies for very large datasets
- Configurable precision vs. speed trade-offs

## Troubleshooting

### Common Issues

**Import Errors**: Ensure virtual environment is activated and all dependencies are installed
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Memory Issues**: For large datasets, consider reducing bootstrap samples or enabling data sampling in configuration

**Missing Dependencies**: Some medical imaging packages may require system-level libraries:
```bash
# On Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx

# On macOS with Homebrew
brew install vtk
```

### Data Privacy

This framework is designed with medical data privacy in mind:
- No patient data is stored in the repository
- Automatic anonymization of DICOM metadata
- Secure handling of protected health information (PHI)
- Compliance with HIPAA and GDPR requirements

## Contributing

This project follows standard software development practices:
- Unit tests for all validation methods
- Comprehensive documentation
- Code review process
- Continuous integration testing

## License

This project is intended for medical device validation and regulatory compliance. Please ensure appropriate licensing for commercial use in medical applications.

## Support

For questions about medical device validation or regulatory compliance, please refer to:
- FDA Software as Medical Device guidance documents
- ISO 13485 and ISO 14971 standards
- Relevant literature on medical imaging validation methodologies

---

**Note**: This validation pipeline is designed for research and regulatory purposes. Always consult with regulatory experts and quality assurance professionals when preparing medical device submissions.