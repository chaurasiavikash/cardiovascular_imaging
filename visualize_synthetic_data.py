#!/usr/bin/env python3
"""
Synthetic Cardiovascular Data Generation Visualizer
Shows exactly how fake vessel images and demographics are created

Run this to see what the validation pipeline is actually working with!
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
import seaborn as sns
from typing import Tuple, Dict, List
import json


def create_synthetic_vessel_mask(shape: Tuple[int, int], seed: int = None) -> np.ndarray:
    """
    Create a synthetic vessel-like binary mask that resembles coronary arteries.
    
    Args:
        shape: Image dimensions (height, width)
        seed: Random seed for reproducible results
        
    Returns:
        Binary mask with vessel-like structures
    """
    if seed is not None:
        np.random.seed(seed)
    
    mask = np.zeros(shape, dtype=np.uint8)
    
    # Main vessel parameters
    center_x = shape[1] // 2
    vessel_base_width = np.random.randint(2, 4)
    
    # Create main curved vessel structure (like LAD artery)
    for y in range(5, shape[0] - 5):
        # Create sinusoidal curve with some randomness (like natural vessel path)
        curve_factor = 8 * np.sin(y * 0.15) + np.random.normal(0, 0.8)
        x_center = int(center_x + curve_factor)
        
        # Vary vessel width along length (stenosis simulation)
        width_variation = np.sin(y * 0.3) * 0.8 + np.random.normal(0, 0.3)
        current_width = max(1, int(vessel_base_width + width_variation))
        
        # Draw vessel with tapering towards edges
        x_start = max(0, x_center - current_width)
        x_end = min(shape[1], x_center + current_width + 1)
        
        if 0 <= x_center < shape[1]:
            mask[y, x_start:x_end] = 1
    
    # Add branching vessels (like diagonal or circumflex branches)
    if np.random.random() > 0.4:  # 60% chance of branch
        branch_start_y = np.random.randint(shape[0] // 3, 2 * shape[0] // 3)
        branch_length = np.random.randint(8, 18)
        branch_angle = np.random.choice([-1, 1]) * np.random.uniform(0.3, 0.8)
        
        for i in range(branch_length):
            y = branch_start_y + i
            if y >= shape[0] - 2:
                break
                
            x_offset = int(i * branch_angle)
            branch_x = center_x + x_offset
            branch_width = max(1, vessel_base_width - 1)
            
            x_start = max(0, branch_x - branch_width)
            x_end = min(shape[1], branch_x + branch_width + 1)
            
            if 0 <= y < shape[0]:
                mask[y, x_start:x_end] = 1
    
    # Add small secondary branches occasionally
    if np.random.random() > 0.7:  # 30% chance
        secondary_y = np.random.randint(shape[0] // 4, 3 * shape[0] // 4)
        secondary_length = np.random.randint(4, 8)
        
        for i in range(secondary_length):
            y = secondary_y + i
            if y >= shape[0] - 1:
                break
            x = center_x + np.random.randint(-2, 3)
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                mask[y, x] = 1
    
    return mask


def add_segmentation_noise(gt_mask: np.ndarray, noise_level: float, seed: int = None) -> np.ndarray:
    """
    Add realistic AI segmentation errors to ground truth mask.
    
    Args:
        gt_mask: Perfect ground truth segmentation
        noise_level: Amount of noise to add (0.0 to 1.0)
        seed: Random seed for reproducible results
        
    Returns:
        Noisy prediction mask simulating AI algorithm output
    """
    if seed is not None:
        np.random.seed(seed)
    
    pred_mask = gt_mask.copy()
    
    # 1. Boundary errors (most common AI mistake)
    # Find vessel boundaries
    boundary = gt_mask - ndimage.binary_erosion(gt_mask)
    boundary_indices = np.where(boundary)
    
    # Add noise to boundary pixels
    n_boundary_errors = int(len(boundary_indices[0]) * noise_level)
    error_indices = np.random.choice(len(boundary_indices[0]), 
                                   min(n_boundary_errors, len(boundary_indices[0])), 
                                   replace=False)
    
    for idx in error_indices:
        y, x = boundary_indices[0][idx], boundary_indices[1][idx]
        
        # Random displacement
        dy = np.random.randint(-1, 2)
        dx = np.random.randint(-1, 2)
        
        new_y = np.clip(y + dy, 0, pred_mask.shape[0] - 1)
        new_x = np.clip(x + dx, 0, pred_mask.shape[1] - 1)
        
        # Flip the pixel at new location
        pred_mask[new_y, new_x] = 1 - pred_mask[new_y, new_x]
    
    # 2. False positives (algorithm sees vessel where there isn't one)
    false_positive_rate = noise_level * 0.4
    background_mask = (gt_mask == 0)
    false_positive_pixels = np.random.random(gt_mask.shape) < false_positive_rate
    pred_mask[background_mask & false_positive_pixels] = 1
    
    # 3. False negatives (algorithm misses real vessel)
    false_negative_rate = noise_level * 0.2
    vessel_mask = (gt_mask == 1)
    false_negative_pixels = np.random.random(gt_mask.shape) < false_negative_rate
    pred_mask[vessel_mask & false_negative_pixels] = 0
    
    return pred_mask.astype(np.uint8)


def generate_patient_demographics(n_patients: int, seed: int = 42) -> Dict[str, List]:
    """
    Generate realistic patient demographics for validation study.
    
    Args:
        n_patients: Number of patients to generate
        seed: Random seed for reproducible results
        
    Returns:
        Dictionary with demographic information
    """
    np.random.seed(seed)
    
    # Age distribution (realistic for cardiac patients)
    age_weights = [0.15, 0.35, 0.50]  # young, middle, elderly
    age_groups = np.random.choice(['young', 'middle', 'elderly'], 
                                 n_patients, p=age_weights)
    
    # Sex distribution (slightly more males in cardiac studies)
    sex_weights = [0.58, 0.42]  # male, female
    sex = np.random.choice(['male', 'female'], n_patients, p=sex_weights)
    
    # Race/ethnicity distribution (US demographics)
    race_weights = [0.60, 0.18, 0.13, 0.09]  # white, hispanic, black, asian
    race = np.random.choice(['white', 'hispanic', 'black', 'asian'], 
                           n_patients, p=race_weights)
    
    # Institution distribution (multi-center study)
    institution_weights = [0.45, 0.35, 0.20]  # hospital_a, hospital_b, hospital_c
    institution = np.random.choice(['hospital_a', 'hospital_b', 'hospital_c'], 
                                  n_patients, p=institution_weights)
    
    return {
        'age_group': age_groups.tolist(),
        'sex': sex.tolist(),
        'race': race.tolist(),
        'institution': institution.tolist()
    }


def generate_clinical_measurements(n_patients: int, seed: int = 42) -> Dict[str, List]:
    """
    Generate realistic clinical measurements for cardiac patients.
    
    Args:
        n_patients: Number of patients
        seed: Random seed for reproducible results
        
    Returns:
        Dictionary with clinical measurements
    """
    np.random.seed(seed)
    
    # Vessel diameter (mm) - realistic coronary artery range
    vessel_diameter = np.random.normal(3.2, 0.7, n_patients)
    vessel_diameter = np.clip(vessel_diameter, 1.5, 6.0)  # Physiological range
    
    # Stenosis severity (%) - range from normal to severe
    stenosis_severity = np.random.beta(1.2, 3.0, n_patients) * 100  # Skewed toward lower values
    
    # Ejection fraction (%) - cardiac function measure
    ejection_fraction = np.random.normal(58, 12, n_patients)
    ejection_fraction = np.clip(ejection_fraction, 20, 80)  # Physiological range
    
    # Age (actual age in years)
    age = np.random.normal(65, 15, n_patients)
    age = np.clip(age, 25, 90).astype(int)
    
    return {
        'vessel_diameter_mm': vessel_diameter.tolist(),
        'stenosis_severity_percent': stenosis_severity.tolist(),
        'ejection_fraction_percent': ejection_fraction.tolist(),
        'age_years': age.tolist()
    }


def visualize_synthetic_data_generation():
    """
    Create comprehensive visualization of synthetic data generation process.
    """
    print("🏥 Cardiovascular Synthetic Data Generation Visualizer")
    print("=" * 60)
    
    # Parameters
    n_patients = 150
    image_shape = (32, 32)
    
    # Generate data
    print(f"📊 Generating data for {n_patients} patients...")
    
    # 1. Create vessel images
    print("   Creating synthetic vessel images...")
    ground_truth_images = []
    prediction_images = []
    
    for i in range(min(n_patients, 20)):  # Generate subset for visualization
        # Create ground truth vessel
        gt_mask = create_synthetic_vessel_mask(image_shape, seed=i)
        ground_truth_images.append(gt_mask)
        
        # Create algorithm prediction with errors
        noise_level = np.random.uniform(0.05, 0.15)
        pred_mask = add_segmentation_noise(gt_mask, noise_level, seed=i+1000)
        prediction_images.append(pred_mask)
    
    # 2. Generate demographics
    print("   Generating patient demographics...")
    demographics = generate_patient_demographics(n_patients)
    
    # 3. Generate clinical data
    print("   Generating clinical measurements...")
    clinical_data = generate_clinical_measurements(n_patients)
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Sample vessel images
    print("   Creating vessel image visualizations...")
    for i in range(8):
        # Ground truth
        plt.subplot(6, 8, i + 1)
        plt.imshow(ground_truth_images[i], cmap='Reds', interpolation='nearest')
        plt.title(f'GT Patient {i+1}', fontsize=8)
        plt.axis('off')
        
        # Prediction
        plt.subplot(6, 8, i + 9)
        plt.imshow(prediction_images[i], cmap='Blues', interpolation='nearest')
        plt.title(f'Pred Patient {i+1}', fontsize=8)
        plt.axis('off')
    
    # Plot 2: Difference maps
    print("   Creating error visualizations...")
    for i in range(8):
        difference = prediction_images[i].astype(float) - ground_truth_images[i].astype(float)
        plt.subplot(6, 8, i + 17)
        plt.imshow(difference, cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
        plt.title(f'Error Map {i+1}', fontsize=8)
        plt.axis('off')
    
    # Plot 3: Demographics distribution
    print("   Creating demographic visualizations...")
    
    # Age distribution
    plt.subplot(6, 4, 13)
    age_counts = pd.Series(demographics['age_group']).value_counts()
    plt.pie(age_counts.values, labels=age_counts.index, autopct='%1.1f%%')
    plt.title('Age Distribution')
    
    # Sex distribution
    plt.subplot(6, 4, 14)
    sex_counts = pd.Series(demographics['sex']).value_counts()
    plt.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%')
    plt.title('Sex Distribution')
    
    # Race distribution
    plt.subplot(6, 4, 15)
    race_counts = pd.Series(demographics['race']).value_counts()
    plt.pie(race_counts.values, labels=race_counts.index, autopct='%1.1f%%')
    plt.title('Race Distribution')
    
    # Institution distribution
    plt.subplot(6, 4, 16)
    inst_counts = pd.Series(demographics['institution']).value_counts()
    plt.pie(inst_counts.values, labels=inst_counts.index, autopct='%1.1f%%')
    plt.title('Institution Distribution')
    
    # Plot 4: Clinical measurements
    print("   Creating clinical measurement visualizations...")
    
    # Vessel diameter histogram
    plt.subplot(6, 4, 17)
    plt.hist(clinical_data['vessel_diameter_mm'], bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('Vessel Diameter (mm)')
    plt.ylabel('Frequency')
    plt.title('Vessel Diameter Distribution')
    
    # Stenosis severity histogram
    plt.subplot(6, 4, 18)
    plt.hist(clinical_data['stenosis_severity_percent'], bins=20, alpha=0.7, color='lightcoral')
    plt.xlabel('Stenosis Severity (%)')
    plt.ylabel('Frequency')
    plt.title('Stenosis Distribution')
    
    # Ejection fraction histogram
    plt.subplot(6, 4, 19)
    plt.hist(clinical_data['ejection_fraction_percent'], bins=20, alpha=0.7, color='lightgreen')
    plt.xlabel('Ejection Fraction (%)')
    plt.ylabel('Frequency')
    plt.title('Ejection Fraction Distribution')
    
    # Age histogram
    plt.subplot(6, 4, 20)
    plt.hist(clinical_data['age_years'], bins=20, alpha=0.7, color='gold')
    plt.xlabel('Age (years)')
    plt.ylabel('Frequency')
    plt.title('Age Distribution')
    
    plt.tight_layout()
    plt.savefig('synthetic_data_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n📊 Data Generation Summary:")
    print("=" * 40)
    print(f"Total patients generated: {n_patients}")
    print(f"Image dimensions: {image_shape}")
    print(f"Total pixels per patient: {image_shape[0] * image_shape[1]}")
    print(f"Total validation pixels: {n_patients * image_shape[0] * image_shape[1]:,}")
    
    print("\n🏥 Demographic Breakdown:")
    for key, values in demographics.items():
        unique_values, counts = np.unique(values, return_counts=True)
        print(f"  {key}:")
        for val, count in zip(unique_values, counts):
            percentage = (count / len(values)) * 100
            print(f"    {val}: {count} ({percentage:.1f}%)")
    
    print("\n📈 Clinical Measurements (Mean ± Std):")
    for key, values in clinical_data.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {key}: {mean_val:.2f} ± {std_val:.2f}")
    
    # Calculate validation metrics preview
    print("\n🎯 Sample Validation Metrics:")
    dice_scores = []
    for i in range(min(20, len(ground_truth_images))):
        gt = ground_truth_images[i]
        pred = prediction_images[i]
        
        intersection = np.sum(gt * pred)
        union = np.sum(gt) + np.sum(pred)
        dice = 2.0 * intersection / (union + 1e-8)
        dice_scores.append(dice)
    
    print(f"  Average Dice Coefficient: {np.mean(dice_scores):.3f}")
    print(f"  Dice Std Dev: {np.std(dice_scores):.3f}")
    print(f"  Dice Range: {np.min(dice_scores):.3f} - {np.max(dice_scores):.3f}")
    
    # Save sample data
    print("\n💾 Saving sample data...")
    sample_data = {
        'ground_truth_sample': [img.tolist() for img in ground_truth_images[:5]],
        'predictions_sample': [img.tolist() for img in prediction_images[:5]],
        'demographics_sample': {k: v[:5] for k, v in demographics.items()},
        'clinical_data_sample': {k: v[:5] for k, v in clinical_data.items()},
        'generation_parameters': {
            'n_patients': n_patients,
            'image_shape': image_shape,
            'noise_levels': [0.05, 0.15],
            'random_seed': 42
        }
    }
    
    with open('synthetic_data_sample.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Sample data saved to 'synthetic_data_sample.json'")
    print("✅ Visualization saved to 'synthetic_data_visualization.png'")
    
    return {
        'ground_truth': ground_truth_images,
        'predictions': prediction_images,
        'demographics': demographics,
        'clinical_data': clinical_data
    }


def demonstrate_data_flow():
    """Show the exact data flow in the validation pipeline."""
    print("\n🔄 Data Flow Demonstration:")
    print("=" * 50)
    
    # Show data shapes and transformations
    n_patients = 150
    image_shape = (32, 32)
    
    print("1. Raw Generation:")
    print(f"   - {n_patients} patients")
    print(f"   - Each has {image_shape} image")
    print(f"   - Ground truth shape: ({n_patients}, {image_shape[0]}, {image_shape[1]})")
    print(f"   - Predictions shape: ({n_patients}, {image_shape[0]}, {image_shape[1]})")
    
    print("\n2. Statistical Validation Transformation:")
    total_pixels = n_patients * image_shape[0] * image_shape[1]
    print(f"   - Flatten images: {n_patients} × {image_shape[0]} × {image_shape[1]} = {total_pixels:,} pixels")
    print(f"   - gt_flat shape: ({total_pixels:,},)")
    print(f"   - pred_flat shape: ({total_pixels:,},)")
    
    print("\n3. Demographics Expansion for Bias Assessment:")
    pixels_per_patient = image_shape[0] * image_shape[1]
    print(f"   - Original demographics: {n_patients} values")
    print(f"   - Expand to pixel level: {n_patients} × {pixels_per_patient} = {total_pixels:,} values")
    print(f"   - Each pixel gets its patient's demographic label")
    
    print("\n4. Validation Metrics Calculation:")
    print(f"   - Dice coefficient: Compare {n_patients} image pairs")
    print(f"   - ICC analysis: Analyze {total_pixels:,} pixel pairs")
    print(f"   - Bias assessment: Check fairness across {total_pixels:,} labeled pixels")
    print(f"   - Uncertainty: Bootstrap {total_pixels:,} predictions")


if __name__ == "__main__":
    # Run the complete visualization
    synthetic_data = visualize_synthetic_data_generation()
    
    # Demonstrate data flow
    demonstrate_data_flow()
    
    print("\n🎉 Synthetic Data Generation Complete!")
    print("📊 Check the generated visualization and sample data files.")
    print("🔍 This shows exactly what your validation pipeline analyzes!")