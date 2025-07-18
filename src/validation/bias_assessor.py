"""
Bias Assessment Module for Medical AI Systems
Implements comprehensive bias detection and fairness evaluation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


class BiasAssessor:
    """
    Comprehensive bias assessment for medical AI systems.
    
    Implements FDA-recommended bias evaluation methods including:
    - Demographic parity assessment
    - Equalized odds evaluation
    - Calibration fairness
    - Performance disparity analysis
    - Statistical significance testing
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize bias assessor.
        
        Args:
            config: Configuration dictionary with bias assessment parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Bias thresholds
        self.fairness_threshold = self.config.get('fairness_threshold', 0.8)  # 80% parity
        self.statistical_alpha = self.config.get('statistical_alpha', 0.05)
        self.min_group_size = self.config.get('min_group_size', 30)
        
        # Protected attributes to analyze
        self.protected_attributes = self.config.get('protected_attributes', [
            'age_group', 'sex', 'race', 'ethnicity', 'institution'
        ])
    
    def assess_algorithmic_bias(self, predictions: np.ndarray,
                              demographics: Dict[str, np.ndarray],
                              ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive algorithmic bias assessment.
        
        Args:
            predictions: Algorithm predictions
            demographics: Dictionary of demographic variables
            ground_truth: Ground truth labels (if available)
            
        Returns:
            Comprehensive bias assessment results
        """
        self.logger.info("Starting comprehensive bias assessment")
        
        # Validate inputs
        pred_array = np.asarray(predictions)
        n_samples = len(pred_array)
        
        # Validate demographics
        validated_demographics = self._validate_demographics(demographics, n_samples)
        
        if not validated_demographics:
            return {'error': 'No valid demographic data for bias assessment'}
        
        bias_results = {
            'demographic_analysis': self._analyze_demographic_distribution(validated_demographics),
            'performance_disparity': self._assess_performance_disparity(
                pred_array, validated_demographics, ground_truth
            ),
            'fairness_metrics': self._calculate_fairness_metrics(
                pred_array, validated_demographics, ground_truth
            ),
            'statistical_tests': self._perform_bias_statistical_tests(
                pred_array, validated_demographics, ground_truth
            ),
            'calibration_fairness': self._assess_calibration_fairness(
                pred_array, validated_demographics, ground_truth
            ),
            'intersectional_analysis': self._perform_intersectional_analysis(
                pred_array, validated_demographics, ground_truth
            )
        }
        
        # Overall bias summary
        bias_results['bias_summary'] = self._generate_bias_summary(bias_results)
        
        self.logger.info("Bias assessment completed")
        return bias_results
    
    def _validate_demographics(self, demographics: Dict[str, np.ndarray], 
                             n_samples: int) -> Dict[str, np.ndarray]:
        """Validate and clean demographic data."""
        validated = {}
        
        for attr_name, attr_values in demographics.items():
            attr_array = np.asarray(attr_values)
            
            # Check length
            if len(attr_array) != n_samples:
                self.logger.warning(f"Demographic '{attr_name}' length mismatch: "
                                  f"{len(attr_array)} vs {n_samples}")
                continue
            
            # Remove NaN values by creating a mask
            valid_mask = ~pd.isna(attr_array)
            
            if np.sum(valid_mask) < self.min_group_size:
                self.logger.warning(f"Insufficient valid data for '{attr_name}': "
                                  f"{np.sum(valid_mask)} samples")
                continue
            
            # Check if we have at least 2 groups with minimum size
            unique_values, counts = np.unique(attr_array[valid_mask], return_counts=True)
            valid_groups = np.sum(counts >= self.min_group_size)
            
            if valid_groups < 2:
                self.logger.warning(f"Insufficient groups for '{attr_name}': "
                                  f"only {valid_groups} groups with ≥{self.min_group_size} samples")
                continue
            
            validated[attr_name] = attr_array
            self.logger.info(f"Validated demographic '{attr_name}': "
                           f"{len(unique_values)} groups, {np.sum(valid_mask)} valid samples")
        
        return validated
    
    def _analyze_demographic_distribution(self, demographics: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze demographic distribution in the dataset."""
        distribution_analysis = {}
        
        for attr_name, attr_values in demographics.items():
            # Remove NaN values
            valid_values = attr_values[~pd.isna(attr_values)]
            
            unique_values, counts = np.unique(valid_values, return_counts=True)
            proportions = counts / len(valid_values)
            
            # Calculate representation balance
            # Ideal would be equal representation
            expected_proportion = 1.0 / len(unique_values)
            balance_score = 1.0 - np.std(proportions) / expected_proportion
            
            distribution_analysis[attr_name] = {
                'groups': unique_values.tolist(),
                'counts': counts.tolist(),
                'proportions': proportions.tolist(),
                'balance_score': float(balance_score),
                'is_balanced': bool(balance_score > 0.7),  # Threshold for balance
                'min_group_size': int(np.min(counts)),
                'max_group_size': int(np.max(counts)),
                'representation_ratio': float(np.min(counts) / np.max(counts))
            }
        
        return distribution_analysis
    
    def _assess_performance_disparity(self, predictions: np.ndarray,
                                    demographics: Dict[str, np.ndarray],
                                    ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Assess performance disparities across demographic groups."""
        disparity_results = {}
        
        if ground_truth is None:
            # Can only assess prediction disparities without ground truth
            for attr_name, attr_values in demographics.items():
                disparity_results[attr_name] = self._assess_prediction_disparity(
                    predictions, attr_values
                )
        else:
            # Full performance disparity analysis with ground truth
            gt_array = np.asarray(ground_truth)
            
            for attr_name, attr_values in demographics.items():
                disparity_results[attr_name] = self._assess_full_performance_disparity(
                    predictions, gt_array, attr_values
                )
        
        return disparity_results
    
    def _assess_prediction_disparity(self, predictions: np.ndarray, 
                                   demographic_attr: np.ndarray) -> Dict[str, Any]:
        """Assess disparities in predictions across demographic groups."""
        # Remove NaN values
        valid_mask = ~pd.isna(demographic_attr)
        pred_valid = predictions[valid_mask]
        attr_valid = demographic_attr[valid_mask]
        
        unique_groups = np.unique(attr_valid)
        group_stats = {}
        
        for group in unique_groups:
            group_mask = attr_valid == group
            group_predictions = pred_valid[group_mask]
            
            if len(group_predictions) >= self.min_group_size:
                group_stats[str(group)] = {
                    'n_samples': int(len(group_predictions)),
                    'mean_prediction': float(np.mean(group_predictions)),
                    'std_prediction': float(np.std(group_predictions)),
                    'median_prediction': float(np.median(group_predictions)),
                    'prediction_range': [float(np.min(group_predictions)), 
                                       float(np.max(group_predictions))]
                }
        
        # Calculate disparity metrics
        disparity_metrics = self._calculate_prediction_disparity_metrics(group_stats)
        
        return {
            'group_statistics': group_stats,
            'disparity_metrics': disparity_metrics
        }
    
    def _calculate_prediction_disparity_metrics(self, group_stats: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate disparity metrics for predictions."""
        if len(group_stats) < 2:
            return {'error': 'Insufficient groups for disparity calculation'}
        
        # Extract mean predictions for each group
        means = [stats['mean_prediction'] for stats in group_stats.values()]
        
        # Calculate disparity metrics
        max_mean = max(means)
        min_mean = min(means)
        
        # Ratio-based disparity (should be close to 1.0 for fairness)
        disparity_ratio = min_mean / (max_mean + 1e-8)
        
        # Difference-based disparity
        disparity_difference = max_mean - min_mean
        
        # Coefficient of variation across groups
        cv = np.std(means) / (np.mean(means) + 1e-8)
        
        return {
            'disparity_ratio': float(disparity_ratio),
            'disparity_difference': float(disparity_difference),
            'coefficient_of_variation': float(cv),
            'is_fair_ratio': bool(disparity_ratio >= self.fairness_threshold),
            'is_fair_cv': bool(cv <= 0.1)  # 10% threshold
        }
    
    def _assess_full_performance_disparity(self, predictions: np.ndarray,
                                         ground_truth: np.ndarray,
                                         demographic_attr: np.ndarray) -> Dict[str, Any]:
        """Assess full performance disparities with ground truth."""
        # Remove NaN values
        valid_mask = ~(pd.isna(demographic_attr) | pd.isna(ground_truth) | pd.isna(predictions))
        pred_valid = predictions[valid_mask]
        gt_valid = ground_truth[valid_mask]
        attr_valid = demographic_attr[valid_mask]
        
        unique_groups = np.unique(attr_valid)
        group_performance = {}
        
        for group in unique_groups:
            group_mask = attr_valid == group
            group_pred = pred_valid[group_mask]
            group_gt = gt_valid[group_mask]
            
            if len(group_pred) >= self.min_group_size:
                # Calculate performance metrics
                performance = self._calculate_group_performance(group_pred, group_gt)
                group_performance[str(group)] = performance
        
        # Calculate performance disparities
        disparity_metrics = self._calculate_performance_disparity_metrics(group_performance)
        
        return {
            'group_performance': group_performance,
            'disparity_metrics': disparity_metrics
        }
    
    def _calculate_group_performance(self, predictions: np.ndarray, 
                                   ground_truth: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics for a specific group."""
        # Determine if this is classification or regression
        is_classification = self._is_classification_task(ground_truth)
        
        performance = {
            'n_samples': int(len(predictions))
        }
        
        if is_classification:
            # Classification metrics
            pred_binary = (predictions > 0.5).astype(int) if predictions.dtype == float else predictions.astype(int)
            gt_binary = ground_truth.astype(int)
            
            performance.update({
                'accuracy': float(accuracy_score(gt_binary, pred_binary)),
                'precision': float(precision_score(gt_binary, pred_binary, average='weighted', zero_division=0)),
                'recall': float(recall_score(gt_binary, pred_binary, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(gt_binary, pred_binary, average='weighted', zero_division=0))
            })
            
            # Additional binary classification metrics
            if len(np.unique(gt_binary)) == 2:
                from sklearn.metrics import roc_auc_score, confusion_matrix
                
                try:
                    if predictions.dtype == float:
                        auc = roc_auc_score(gt_binary, predictions)
                    else:
                        auc = roc_auc_score(gt_binary, pred_binary)
                    performance['auc'] = float(auc)
                except:
                    performance['auc'] = 0.5
                
                # Confusion matrix metrics
                tn, fp, fn, tp = confusion_matrix(gt_binary, pred_binary).ravel()
                performance.update({
                    'sensitivity': float(tp / (tp + fn + 1e-8)),
                    'specificity': float(tn / (tn + fp + 1e-8)),
                    'positive_predictive_value': float(tp / (tp + fp + 1e-8)),
                    'negative_predictive_value': float(tn / (tn + fn + 1e-8))
                })
        
        else:
            # Regression metrics
            mae = np.mean(np.abs(predictions - ground_truth))
            mse = np.mean((predictions - ground_truth) ** 2)
            rmse = np.sqrt(mse)
            
            # R-squared
            ss_res = np.sum((ground_truth - predictions) ** 2)
            ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            performance.update({
                'mean_absolute_error': float(mae),
                'mean_squared_error': float(mse),
                'root_mean_squared_error': float(rmse),
                'r_squared': float(r2)
            })
        
        return performance
    
    def _is_classification_task(self, ground_truth: np.ndarray) -> bool:
        """Determine if this is a classification task."""
        unique_values = np.unique(ground_truth)
        
        # If all values are integers and there are few unique values, likely classification
        if len(unique_values) <= 10 and np.all(ground_truth == np.round(ground_truth)):
            return True
        
        return False
    
    def _calculate_performance_disparity_metrics(self, group_performance: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate disparity metrics across group performance."""
        if len(group_performance) < 2:
            return {'error': 'Insufficient groups for disparity calculation'}
        
        disparity_metrics = {}
        
        # Get common metrics across all groups
        common_metrics = set.intersection(*[set(perf.keys()) for perf in group_performance.values()])
        common_metrics.discard('n_samples')  # Exclude sample count
        
        for metric in common_metrics:
            values = [group_performance[group][metric] for group in group_performance.keys()]
            
            if all(isinstance(v, (int, float)) for v in values):
                max_val = max(values)
                min_val = min(values)
                
                # Disparity ratio (min/max for metrics where higher is better)
                if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'r_squared',
                             'sensitivity', 'specificity', 'positive_predictive_value', 'negative_predictive_value']:
                    ratio = min_val / (max_val + 1e-8)
                    is_fair = ratio >= self.fairness_threshold
                else:
                    # For error metrics, lower is better, so use max/min
                    ratio = max_val / (min_val + 1e-8)
                    is_fair = ratio <= (1.0 / self.fairness_threshold)
                
                disparity_metrics[metric] = {
                    'min_value': float(min_val),
                    'max_value': float(max_val),
                    'disparity_ratio': float(ratio),
                    'is_fair': bool(is_fair),
                    'coefficient_of_variation': float(np.std(values) / (np.mean(values) + 1e-8))
                }
        
        return disparity_metrics
    
    def _calculate_fairness_metrics(self, predictions: np.ndarray,
                                  demographics: Dict[str, np.ndarray],
                                  ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate comprehensive fairness metrics."""
        fairness_results = {}
        
        for attr_name, attr_values in demographics.items():
            if ground_truth is not None:
                # Full fairness metrics with ground truth
                fairness_results[attr_name] = self._calculate_group_fairness_metrics(
                    predictions, ground_truth, attr_values
                )
            else:
                # Limited fairness metrics without ground truth
                fairness_results[attr_name] = self._calculate_demographic_parity(
                    predictions, attr_values
                )
        
        return fairness_results
    
    def _calculate_demographic_parity(self, predictions: np.ndarray, 
                                    demographic_attr: np.ndarray) -> Dict[str, Any]:
        """Calculate demographic parity (statistical parity)."""
        # Remove NaN values
        valid_mask = ~pd.isna(demographic_attr)
        pred_valid = predictions[valid_mask]
        attr_valid = demographic_attr[valid_mask]
        
        unique_groups = np.unique(attr_valid)
        
        if len(unique_groups) < 2:
            return {'error': 'Insufficient groups for demographic parity calculation'}
        
        # For binary predictions, calculate positive prediction rates
        if self._is_binary_predictions(pred_valid):
            group_rates = {}
            for group in unique_groups:
                group_mask = attr_valid == group
                group_pred = pred_valid[group_mask]
                
                if len(group_pred) >= self.min_group_size:
                    positive_rate = np.mean(group_pred)
                    group_rates[str(group)] = {
                        'positive_rate': float(positive_rate),
                        'n_samples': int(len(group_pred))
                    }
            
            # Calculate demographic parity metrics
            if len(group_rates) >= 2:
                rates = [rate_info['positive_rate'] for rate_info in group_rates.values()]
                max_rate = max(rates)
                min_rate = min(rates)
                
                parity_ratio = min_rate / (max_rate + 1e-8)
                parity_difference = max_rate - min_rate
                
                return {
                    'group_rates': group_rates,
                    'demographic_parity_ratio': float(parity_ratio),
                    'demographic_parity_difference': float(parity_difference),
                    'satisfies_demographic_parity': bool(parity_ratio >= self.fairness_threshold)
                }
        
        return {'note': 'Demographic parity calculation requires binary predictions'}
    
    def _is_binary_predictions(self, predictions: np.ndarray) -> bool:
        """Check if predictions are binary."""
        unique_pred = np.unique(predictions)
        return len(unique_pred) <= 2 or np.all((predictions >= 0) & (predictions <= 1))
    
    def _calculate_group_fairness_metrics(self, predictions: np.ndarray,
                                        ground_truth: np.ndarray,
                                        demographic_attr: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive group fairness metrics."""
        # Remove NaN values
        valid_mask = ~(pd.isna(demographic_attr) | pd.isna(ground_truth) | pd.isna(predictions))
        pred_valid = predictions[valid_mask]
        gt_valid = ground_truth[valid_mask]
        attr_valid = demographic_attr[valid_mask]
        
        unique_groups = np.unique(attr_valid)
        
        if len(unique_groups) < 2:
            return {'error': 'Insufficient groups for fairness calculation'}
        
        fairness_metrics = {}
        
        # Check if this is a classification task
        if self._is_classification_task(gt_valid):
            fairness_metrics.update(self._calculate_classification_fairness(
                pred_valid, gt_valid, attr_valid, unique_groups
            ))
        else:
            fairness_metrics.update(self._calculate_regression_fairness(
                pred_valid, gt_valid, attr_valid, unique_groups
            ))
        
        return fairness_metrics
    
    def _calculate_classification_fairness(self, predictions: np.ndarray,
                                         ground_truth: np.ndarray,
                                         demographic_attr: np.ndarray,
                                         unique_groups: np.ndarray) -> Dict[str, Any]:
        """Calculate fairness metrics for classification tasks."""
        # Convert to binary if needed
        pred_binary = (predictions > 0.5).astype(int) if predictions.dtype == float else predictions.astype(int)
        gt_binary = ground_truth.astype(int)
        
        group_metrics = {}
        
        for group in unique_groups:
            group_mask = demographic_attr == group
            group_pred = pred_binary[group_mask]
            group_gt = gt_binary[group_mask]
            
            if len(group_pred) >= self.min_group_size:
                from sklearn.metrics import confusion_matrix
                
                # Calculate confusion matrix
                cm = confusion_matrix(group_gt, group_pred)
                
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    
                    # True positive rate (sensitivity, recall)
                    tpr = tp / (tp + fn + 1e-8)
                    # False positive rate
                    fpr = fp / (fp + tn + 1e-8)
                    # Positive predictive value (precision)
                    ppv = tp / (tp + fp + 1e-8)
                    # True negative rate (specificity)
                    tnr = tn / (tn + fp + 1e-8)
                    
                    group_metrics[str(group)] = {
                        'true_positive_rate': float(tpr),
                        'false_positive_rate': float(fpr),
                        'positive_predictive_value': float(ppv),
                        'true_negative_rate': float(tnr),
                        'n_samples': int(len(group_pred))
                    }
        
        # Calculate fairness disparities
        fairness_disparities = {}
        
        if len(group_metrics) >= 2:
            for metric in ['true_positive_rate', 'false_positive_rate', 'positive_predictive_value']:
                values = [group_metrics[group][metric] for group in group_metrics.keys()]
                max_val = max(values)
                min_val = min(values)
                
                fairness_disparities[f'{metric}_disparity'] = {
                    'min_value': float(min_val),
                    'max_value': float(max_val),
                    'ratio': float(min_val / (max_val + 1e-8)),
                    'difference': float(max_val - min_val)
                }
            
            # Equalized odds (TPR and FPR should be similar across groups)
            tpr_values = [group_metrics[group]['true_positive_rate'] for group in group_metrics.keys()]
            fpr_values = [group_metrics[group]['false_positive_rate'] for group in group_metrics.keys()]
            
            tpr_disparity = max(tpr_values) - min(tpr_values)
            fpr_disparity = max(fpr_values) - min(fpr_values)
            
            fairness_disparities['equalized_odds'] = {
                'tpr_disparity': float(tpr_disparity),
                'fpr_disparity': float(fpr_disparity),
                'satisfies_equalized_odds': bool(tpr_disparity <= 0.1 and fpr_disparity <= 0.1)
            }
        
        return {
            'group_metrics': group_metrics,
            'fairness_disparities': fairness_disparities
        }
    
    def _calculate_regression_fairness(self, predictions: np.ndarray,
                                     ground_truth: np.ndarray,
                                     demographic_attr: np.ndarray,
                                     unique_groups: np.ndarray) -> Dict[str, Any]:
        """Calculate fairness metrics for regression tasks."""
        group_metrics = {}
        
        for group in unique_groups:
            group_mask = demographic_attr == group
            group_pred = predictions[group_mask]
            group_gt = ground_truth[group_mask]
            
            if len(group_pred) >= self.min_group_size:
                # Calculate error metrics
                mae = np.mean(np.abs(group_pred - group_gt))
                mse = np.mean((group_pred - group_gt) ** 2)
                bias = np.mean(group_pred - group_gt)
                
                group_metrics[str(group)] = {
                    'mean_absolute_error': float(mae),
                    'mean_squared_error': float(mse),
                    'bias': float(bias),
                    'n_samples': int(len(group_pred))
                }
        
        # Calculate fairness disparities
        fairness_disparities = {}
        
        if len(group_metrics) >= 2:
            for metric in ['mean_absolute_error', 'mean_squared_error', 'bias']:
                values = [abs(group_metrics[group][metric]) for group in group_metrics.keys()]
                max_val = max(values)
                min_val = min(values)
                
                fairness_disparities[f'{metric}_disparity'] = {
                    'min_value': float(min_val),
                    'max_value': float(max_val),
                    'ratio': float(max_val / (min_val + 1e-8)),  # For errors, higher ratio means worse fairness
                    'difference': float(max_val - min_val)
                }
        
        return {
            'group_metrics': group_metrics,
            'fairness_disparities': fairness_disparities
        }
    
    def _perform_bias_statistical_tests(self, predictions: np.ndarray,
                                      demographics: Dict[str, np.ndarray],
                                      ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform statistical tests for bias detection."""
        statistical_results = {}
        
        for attr_name, attr_values in demographics.items():
            if ground_truth is not None:
                statistical_results[attr_name] = self._test_performance_differences(
                    predictions, ground_truth, attr_values
                )
            else:
                statistical_results[attr_name] = self._test_prediction_differences(
                    predictions, attr_values
                )
        
        return statistical_results
    
    def _test_prediction_differences(self, predictions: np.ndarray, 
                                   demographic_attr: np.ndarray) -> Dict[str, Any]:
        """Test for statistically significant differences in predictions."""
        # Remove NaN values
        valid_mask = ~pd.isna(demographic_attr)
        pred_valid = predictions[valid_mask]
        attr_valid = demographic_attr[valid_mask]
        
        unique_groups = np.unique(attr_valid)
        
        if len(unique_groups) < 2:
            return {'error': 'Insufficient groups for statistical testing'}
        
        # Prepare group data
        group_predictions = []
        group_names = []
        
        for group in unique_groups:
            group_mask = attr_valid == group
            group_pred = pred_valid[group_mask]
            
            if len(group_pred) >= self.min_group_size:
                group_predictions.append(group_pred)
                group_names.append(str(group))
        
        if len(group_predictions) < 2:
            return {'error': 'Insufficient groups with adequate sample size'}
        
        test_results = {}
        
        # ANOVA test for multiple groups
        if len(group_predictions) > 2:
            try:
                f_stat, p_value = stats.f_oneway(*group_predictions)
                test_results['anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant_difference': bool(p_value < self.statistical_alpha)
                }
            except Exception as e:
                test_results['anova'] = {'error': str(e)}
        
        # Pairwise t-tests
        pairwise_results = {}
        for i in range(len(group_predictions)):
            for j in range(i + 1, len(group_predictions)):
                group1_name = group_names[i]
                group2_name = group_names[j]
                
                try:
                    t_stat, p_value = stats.ttest_ind(group_predictions[i], group_predictions[j])
                    pairwise_results[f'{group1_name}_vs_{group2_name}'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant_difference': bool(p_value < self.statistical_alpha)
                    }
                except Exception as e:
                    pairwise_results[f'{group1_name}_vs_{group2_name}'] = {'error': str(e)}
        
        test_results['pairwise_tests'] = pairwise_results
        
        return test_results
    
    def _test_performance_differences(self, predictions: np.ndarray,
                                    ground_truth: np.ndarray,
                                    demographic_attr: np.ndarray) -> Dict[str, Any]:
        """Test for statistically significant differences in performance."""
        # This would implement more sophisticated performance difference testing
        # For now, use the same approach as prediction differences but on errors
        errors = predictions - ground_truth
        return self._test_prediction_differences(errors, demographic_attr)
    
    def _assess_calibration_fairness(self, predictions: np.ndarray,
                                   demographics: Dict[str, np.ndarray],
                                   ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Assess calibration fairness across demographic groups."""
        if ground_truth is None:
            return {'note': 'Calibration fairness requires ground truth labels'}
        
        # Only applicable for classification with probability scores
        if not (np.all((predictions >= 0) & (predictions <= 1)) and 
                self._is_classification_task(ground_truth)):
            return {'note': 'Calibration fairness requires probability scores for classification'}
        
        calibration_results = {}
        
        for attr_name, attr_values in demographics.items():
            # Remove NaN values
            valid_mask = ~(pd.isna(attr_values) | pd.isna(ground_truth) | pd.isna(predictions))
            pred_valid = predictions[valid_mask]
            gt_valid = ground_truth[valid_mask].astype(int)
            attr_valid = attr_values[valid_mask]
            
            unique_groups = np.unique(attr_valid)
            group_calibration = {}
            
            for group in unique_groups:
                group_mask = attr_valid == group
                group_pred = pred_valid[group_mask]
                group_gt = gt_valid[group_mask]
                
                if len(group_pred) >= self.min_group_size:
                    try:
                        # Calculate calibration curve
                        fraction_of_positives, mean_predicted_value = calibration_curve(
                            group_gt, group_pred, n_bins=5
                        )
                        
                        # Calculate calibration error
                        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                        
                        group_calibration[str(group)] = {
                            'calibration_error': float(calibration_error),
                            'n_samples': int(len(group_pred))
                        }
                        
                    except Exception as e:
                        group_calibration[str(group)] = {'error': str(e)}
            
            # Calculate calibration fairness
            if len(group_calibration) >= 2:
                errors = [cal['calibration_error'] for cal in group_calibration.values() 
                         if 'calibration_error' in cal]
                
                if errors:
                    max_error = max(errors)
                    min_error = min(errors)
                    calibration_disparity = max_error - min_error
                    
                    calibration_results[attr_name] = {
                        'group_calibration': group_calibration,
                        'calibration_disparity': float(calibration_disparity),
                        'is_fairly_calibrated': bool(calibration_disparity <= 0.1)
                    }
        
        return calibration_results
    
    def _perform_intersectional_analysis(self, predictions: np.ndarray,
                                       demographics: Dict[str, np.ndarray],
                                       ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform intersectional bias analysis."""
        if len(demographics) < 2:
            return {'note': 'Intersectional analysis requires at least 2 demographic attributes'}
        
        intersectional_results = {}
        
        # Analyze pairs of demographic attributes
        attr_names = list(demographics.keys())
        
        for i in range(len(attr_names)):
            for j in range(i + 1, len(attr_names)):
                attr1_name = attr_names[i]
                attr2_name = attr_names[j]
                
                attr1_values = demographics[attr1_name]
                attr2_values = demographics[attr2_name]
                
                # Create intersectional groups
                intersectional_groups = self._create_intersectional_groups(
                    attr1_values, attr2_values, predictions, ground_truth
                )
                
                if intersectional_groups:
                    pair_name = f'{attr1_name}_x_{attr2_name}'
                    intersectional_results[pair_name] = intersectional_groups
        
        return intersectional_results
    
    def _create_intersectional_groups(self, attr1: np.ndarray, attr2: np.ndarray,
                                    predictions: np.ndarray,
                                    ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Create and analyze intersectional demographic groups."""
        # Remove NaN values
        if ground_truth is not None:
            valid_mask = ~(pd.isna(attr1) | pd.isna(attr2) | pd.isna(predictions) | pd.isna(ground_truth))
        else:
            valid_mask = ~(pd.isna(attr1) | pd.isna(attr2) | pd.isna(predictions))
        
        attr1_valid = attr1[valid_mask]
        attr2_valid = attr2[valid_mask]
        pred_valid = predictions[valid_mask]
        
        if ground_truth is not None:
            gt_valid = ground_truth[valid_mask]
        else:
            gt_valid = None
        
        # Create intersectional groups
        intersectional_groups = {}
        
        for val1 in np.unique(attr1_valid):
            for val2 in np.unique(attr2_valid):
                group_mask = (attr1_valid == val1) & (attr2_valid == val2)
                group_pred = pred_valid[group_mask]
                
                if len(group_pred) >= self.min_group_size:
                    group_name = f'{val1}_{val2}'
                    
                    if gt_valid is not None:
                        group_gt = gt_valid[group_mask]
                        group_performance = self._calculate_group_performance(group_pred, group_gt)
                    else:
                        group_performance = {
                            'mean_prediction': float(np.mean(group_pred)),
                            'std_prediction': float(np.std(group_pred))
                        }
                    
                    group_performance['n_samples'] = int(len(group_pred))
                    intersectional_groups[group_name] = group_performance
        
        return intersectional_groups
    
    def _generate_bias_summary(self, bias_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall bias assessment summary."""
        summary = {
            'overall_bias_status': 'unknown',
            'critical_bias_issues': [],
            'fairness_scores': {},
            'recommendations': []
        }
        
        # Analyze performance disparities
        performance_disparity = bias_results.get('performance_disparity', {})
        fairness_metrics = bias_results.get('fairness_metrics', {})
        
        bias_detected = False
        fairness_scores = {}
        
        for attr_name, disparity_data in performance_disparity.items():
            if isinstance(disparity_data, dict) and 'disparity_metrics' in disparity_data:
                disparity_metrics = disparity_data['disparity_metrics']
                
                # Check fairness ratios
                unfair_metrics = []
                for metric, metric_data in disparity_metrics.items():
                    if isinstance(metric_data, dict) and 'is_fair' in metric_data:
                        if not metric_data['is_fair']:
                            unfair_metrics.append(metric)
                            bias_detected = True
                
                if unfair_metrics:
                    summary['critical_bias_issues'].append(
                        f"Unfair performance for {attr_name}: {', '.join(unfair_metrics)}"
                    )
                
                # Calculate overall fairness score for this attribute
                fair_count = sum(1 for metric, data in disparity_metrics.items() 
                               if isinstance(data, dict) and data.get('is_fair', False))
                total_count = len([metric for metric, data in disparity_metrics.items() 
                                 if isinstance(data, dict) and 'is_fair' in data])
                
                if total_count > 0:
                    fairness_scores[attr_name] = fair_count / total_count
        
        # Analyze statistical significance
        statistical_tests = bias_results.get('statistical_tests', {})
        for attr_name, test_data in statistical_tests.items():
            if isinstance(test_data, dict):
                # Check ANOVA results
                if 'anova' in test_data and isinstance(test_data['anova'], dict):
                    if test_data['anova'].get('significant_difference', False):
                        summary['critical_bias_issues'].append(
                            f"Statistically significant performance differences for {attr_name}"
                        )
                        bias_detected = True
                
                # Check pairwise tests
                pairwise = test_data.get('pairwise_tests', {})
                significant_pairs = [pair for pair, data in pairwise.items()
                                   if isinstance(data, dict) and data.get('significant_difference', False)]
                
                if significant_pairs:
                    summary['critical_bias_issues'].append(
                        f"Significant pairwise differences for {attr_name}: {len(significant_pairs)} pairs"
                    )
        
        # Determine overall bias status
        if not bias_detected and fairness_scores:
            avg_fairness = np.mean(list(fairness_scores.values()))
            if avg_fairness >= 0.9:
                summary['overall_bias_status'] = 'minimal_bias'
            elif avg_fairness >= 0.7:
                summary['overall_bias_status'] = 'acceptable_bias'
            else:
                summary['overall_bias_status'] = 'concerning_bias'
        elif bias_detected:
            summary['overall_bias_status'] = 'significant_bias'
        else:
            summary['overall_bias_status'] = 'insufficient_data'
        
        summary['fairness_scores'] = fairness_scores
        
        # Generate recommendations
        if summary['overall_bias_status'] == 'significant_bias':
            summary['recommendations'].extend([
                "Conduct thorough bias mitigation before clinical deployment",
                "Consider algorithm retraining with balanced datasets",
                "Implement bias monitoring in production environment",
                "Perform additional validation studies on underrepresented groups"
            ])
        elif summary['overall_bias_status'] == 'concerning_bias':
            summary['recommendations'].extend([
                "Address identified bias issues before deployment",
                "Increase representation of affected demographic groups in training",
                "Implement bias monitoring and alerting systems"
            ])
        elif summary['overall_bias_status'] == 'acceptable_bias':
            summary['recommendations'].extend([
                "Continue monitoring for bias in clinical deployment",
                "Document bias assessment results for regulatory review"
            ])
        else:
            summary['recommendations'].append(
                "Maintain current bias monitoring and assessment protocols"
            )
        
        # Add regulatory compliance assessment
        summary['regulatory_compliance'] = {
            'fda_bias_assessment_complete': len(fairness_scores) > 0,
            'demographic_groups_analyzed': len(fairness_scores),
            'requires_bias_mitigation': summary['overall_bias_status'] in ['significant_bias', 'concerning_bias'],
            'suitable_for_clinical_deployment': summary['overall_bias_status'] in ['minimal_bias', 'acceptable_bias']
        }
        
        return summary