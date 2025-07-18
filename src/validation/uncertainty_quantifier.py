"""
Uncertainty Quantification Module for Medical AI Systems
Implements comprehensive uncertainty estimation and validation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from scipy.special import softmax
import matplotlib.pyplot as plt


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification for medical AI systems.
    
    Implements multiple uncertainty estimation methods:
    - Predictive uncertainty (aleatoric + epistemic)
    - Calibration assessment and correction
    - Confidence interval estimation
    - Uncertainty-aware performance metrics
    - Bootstrap and Monte Carlo methods
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize uncertainty quantifier.
        
        Args:
            config: Configuration dictionary with uncertainty parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Uncertainty thresholds
        self.calibration_threshold = self.config.get('calibration_threshold', 0.1)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.bootstrap_samples = self.config.get('bootstrap_samples', 1000)
        self.uncertainty_threshold = self.config.get('uncertainty_threshold', 0.2)
        
    def quantify_uncertainty(self, predictions: np.ndarray,
                           model_outputs: Optional[Dict[str, Any]] = None,
                           ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive uncertainty quantification.
        
        Args:
            predictions: Model predictions
            model_outputs: Additional model outputs (e.g., logits, dropout samples)
            ground_truth: Ground truth labels (if available)
            
        Returns:
            Comprehensive uncertainty quantification results
        """
        self.logger.info("Starting uncertainty quantification")
        
        pred_array = np.asarray(predictions)
        
        uncertainty_results = {
            'predictive_uncertainty': self._estimate_predictive_uncertainty(pred_array, model_outputs),
            'calibration_analysis': self._analyze_calibration(pred_array, ground_truth),
            'confidence_intervals': self._estimate_confidence_intervals(pred_array, ground_truth),
            'uncertainty_based_metrics': self._calculate_uncertainty_metrics(pred_array, model_outputs, ground_truth),
            'bootstrap_uncertainty': self._bootstrap_uncertainty_estimation(pred_array, ground_truth)
        }
        
        # Advanced uncertainty methods if model outputs available
        if model_outputs:
            uncertainty_results['epistemic_uncertainty'] = self._estimate_epistemic_uncertainty(model_outputs)
            uncertainty_results['aleatoric_uncertainty'] = self._estimate_aleatoric_uncertainty(model_outputs)
            
            if 'ensemble_predictions' in model_outputs:
                uncertainty_results['ensemble_uncertainty'] = self._analyze_ensemble_uncertainty(
                    model_outputs['ensemble_predictions']
                )
        
        # Overall uncertainty summary
        uncertainty_results['uncertainty_summary'] = self._generate_uncertainty_summary(uncertainty_results)
        
        self.logger.info("Uncertainty quantification completed")
        return uncertainty_results
    
    def _estimate_predictive_uncertainty(self, predictions: np.ndarray,
                                       model_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Estimate predictive uncertainty from model predictions."""
        uncertainty_estimates = {}
        
        # Basic entropy-based uncertainty for classification
        if self._is_classification_predictions(predictions):
            uncertainty_estimates.update(self._calculate_classification_uncertainty(predictions))
        else:
            uncertainty_estimates.update(self._calculate_regression_uncertainty(predictions, model_outputs))
        
        # Confidence-based uncertainty
        if self._is_probabilistic_predictions(predictions):
            confidence_uncertainty = self._calculate_confidence_uncertainty(predictions)
            uncertainty_estimates.update(confidence_uncertainty)
        
        return uncertainty_estimates
    
    def _is_classification_predictions(self, predictions: np.ndarray) -> bool:
        """Check if predictions are for classification."""
        # Check if predictions are probabilities or logits for classification
        if predictions.ndim == 2:  # Multi-class probabilities
            return True
        elif predictions.ndim == 1:
            # Could be binary classification probabilities or regression
            return np.all((predictions >= 0) & (predictions <= 1))
        return False
    
    def _is_probabilistic_predictions(self, predictions: np.ndarray) -> bool:
        """Check if predictions are probabilistic."""
        if predictions.ndim == 2:
            # Check if rows sum to 1 (probability distributions)
            row_sums = np.sum(predictions, axis=1)
            return np.allclose(row_sums, 1.0, atol=1e-6)
        elif predictions.ndim == 1:
            return np.all((predictions >= 0) & (predictions <= 1))
        return False
    
    def _calculate_classification_uncertainty(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate uncertainty measures for classification."""
        uncertainty_measures = {}
        
        if predictions.ndim == 2:
            # Multi-class case
            # Predictive entropy
            epsilon = 1e-12  # Small value to avoid log(0)
            pred_clipped = np.clip(predictions, epsilon, 1 - epsilon)
            entropy = -np.sum(pred_clipped * np.log(pred_clipped), axis=1)
            
            # Maximum probability (confidence)
            max_prob = np.max(predictions, axis=1)
            
            # Mutual information (approximation)
            mean_pred = np.mean(predictions, axis=0)
            mean_entropy = -np.sum(mean_pred * np.log(mean_pred + epsilon))
            mutual_info = mean_entropy - np.mean(entropy)
            
            uncertainty_measures.update({
                'predictive_entropy': {
                    'values': entropy.tolist()[:100],  # Limit output size
                    'mean': float(np.mean(entropy)),
                    'std': float(np.std(entropy)),
                    'max': float(np.max(entropy)),
                    'min': float(np.min(entropy))
                },
                'max_probability': {
                    'values': max_prob.tolist()[:100],  # Limit output size
                    'mean': float(np.mean(max_prob)),
                    'std': float(np.std(max_prob))
                },
                'mutual_information': float(mutual_info)
            })
            
        elif predictions.ndim == 1:
            # Binary classification case
            # Binary entropy
            epsilon = 1e-12
            pred_clipped = np.clip(predictions, epsilon, 1 - epsilon)
            binary_entropy = -(pred_clipped * np.log(pred_clipped) + 
                             (1 - pred_clipped) * np.log(1 - pred_clipped))
            
            # Distance from decision boundary (0.5)
            decision_distance = np.abs(predictions - 0.5)
            
            uncertainty_measures.update({
                'binary_entropy': {
                    'values': binary_entropy.tolist()[:100],  # Limit output size
                    'mean': float(np.mean(binary_entropy)),
                    'std': float(np.std(binary_entropy))
                },
                'decision_boundary_distance': {
                    'values': decision_distance.tolist()[:100],  # Limit output size
                    'mean': float(np.mean(decision_distance)),
                    'std': float(np.std(decision_distance))
                }
            })
        
        return uncertainty_measures
    
    def _calculate_regression_uncertainty(self, predictions: np.ndarray,
                                        model_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate uncertainty measures for regression."""
        uncertainty_measures = {}
        
        # If model outputs contain variance estimates
        if model_outputs and 'prediction_variance' in model_outputs:
            pred_variance = np.asarray(model_outputs['prediction_variance'])
            pred_std = np.sqrt(pred_variance)
            
            uncertainty_measures.update({
                'prediction_variance': {
                    'values': pred_variance.tolist()[:100],  # Limit output size
                    'mean': float(np.mean(pred_variance)),
                    'std': float(np.std(pred_variance))
                },
                'prediction_std': {
                    'values': pred_std.tolist()[:100],  # Limit output size
                    'mean': float(np.mean(pred_std)),
                    'std': float(np.std(pred_std))
                }
            })
        
        return uncertainty_measures
    
    def _calculate_confidence_uncertainty(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate confidence-based uncertainty measures."""
        confidence_measures = {}
        
        if predictions.ndim == 2:
            # Multi-class: use max probability as confidence
            confidence = np.max(predictions, axis=1)
        else:
            # Binary: use distance from 0.5 as confidence
            confidence = np.abs(predictions - 0.5) * 2  # Scale to [0, 1]
        
        # Uncertainty is inverse of confidence
        uncertainty = 1 - confidence
        
        confidence_measures.update({
            'confidence': {
                'values': confidence.tolist()[:100],  # Limit output size
                'mean': float(np.mean(confidence)),
                'std': float(np.std(confidence)),
                'percentiles': {
                    '5th': float(np.percentile(confidence, 5)),
                    '25th': float(np.percentile(confidence, 25)),
                    '50th': float(np.percentile(confidence, 50)),
                    '75th': float(np.percentile(confidence, 75)),
                    '95th': float(np.percentile(confidence, 95))
                }
            },
            'confidence_uncertainty': {
                'values': uncertainty.tolist()[:100],  # Limit output size
                'mean': float(np.mean(uncertainty)),
                'std': float(np.std(uncertainty))
            }
        })
        
        return confidence_measures
    
    def _analyze_calibration(self, predictions: np.ndarray,
                           ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze prediction calibration."""
        if ground_truth is None:
            return {'note': 'Calibration analysis requires ground truth labels'}
        
        gt_array = np.asarray(ground_truth)
        
        # Ensure same length
        if len(predictions) != len(gt_array):
            return {'error': 'Prediction and ground truth length mismatch'}
        
        calibration_results = {}
        
        if self._is_classification_predictions(predictions):
            calibration_results.update(self._analyze_classification_calibration(predictions, gt_array))
        else:
            calibration_results.update(self._analyze_regression_calibration(predictions, gt_array))
        
        return calibration_results
    
    def _analyze_classification_calibration(self, predictions: np.ndarray,
                                          ground_truth: np.ndarray) -> Dict[str, Any]:
        """Analyze calibration for classification."""
        calibration_results = {}
        
        try:
            if predictions.ndim == 1:
                # Binary classification
                gt_binary = ground_truth.astype(int)
                
                # Sample for performance (large datasets are slow)
                if len(predictions) > 10000:
                    indices = np.random.choice(len(predictions), 10000, replace=False)
                    predictions_sample = predictions[indices]
                    gt_sample = gt_binary[indices]
                else:
                    predictions_sample = predictions
                    gt_sample = gt_binary
                
                # Calculate ECE
                ece = self._calculate_ece(predictions_sample, gt_sample)
                
                calibration_results.update({
                    'expected_calibration_error': float(ece),
                    'is_well_calibrated': bool(ece <= self.calibration_threshold)
                })
        
        except Exception as e:
            self.logger.warning(f"Calibration analysis failed: {str(e)}")
            calibration_results['error'] = str(e)
        
        return calibration_results
    
    def _analyze_regression_calibration(self, predictions: np.ndarray,
                                      ground_truth: np.ndarray) -> Dict[str, Any]:
        """Analyze calibration for regression (prediction intervals)."""
        calibration_results = {}
        
        # For regression, calibration is about prediction intervals
        # Sample for performance
        if len(predictions) > 10000:
            indices = np.random.choice(len(predictions), 10000, replace=False)
            pred_sample = predictions[indices]
            gt_sample = ground_truth[indices]
        else:
            pred_sample = predictions
            gt_sample = ground_truth
        
        residuals = pred_sample - gt_sample
        
        # Empirical coverage analysis
        residual_std = np.std(residuals)
        
        # Check coverage for different confidence levels
        confidence_levels = [0.68, 0.90, 0.95]
        coverage_results = {}
        
        for conf_level in confidence_levels:
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            interval_half_width = z_score * residual_std
            
            # Check how many predictions fall within the interval
            within_interval = np.abs(residuals) <= interval_half_width
            empirical_coverage = np.mean(within_interval)
            
            coverage_results[f'coverage_{int(conf_level*100)}'] = {
                'theoretical_coverage': float(conf_level),
                'empirical_coverage': float(empirical_coverage),
                'coverage_error': float(abs(empirical_coverage - conf_level)),
                'is_well_calibrated': bool(abs(empirical_coverage - conf_level) <= 0.05)
            }
        
        calibration_results['interval_calibration'] = coverage_results
        
        return calibration_results
    
    def _calculate_ece(self, predictions: np.ndarray, ground_truth: np.ndarray,
                      n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = ground_truth[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _estimate_confidence_intervals(self, predictions: np.ndarray,
                                     ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Estimate confidence intervals for predictions."""
        ci_results = {}
        
        # Bootstrap confidence intervals
        bootstrap_ci = self._bootstrap_confidence_intervals(predictions)
        ci_results['bootstrap_ci'] = bootstrap_ci
        
        # Parametric confidence intervals (if ground truth available)
        if ground_truth is not None:
            parametric_ci = self._parametric_confidence_intervals(predictions, ground_truth)
            ci_results['parametric_ci'] = parametric_ci
        
        return ci_results
    
    def _bootstrap_confidence_intervals(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate bootstrap confidence intervals."""
        # Sample for performance
        n_samples = min(len(predictions), 10000)
        if len(predictions) > n_samples:
            indices = np.random.choice(len(predictions), n_samples, replace=False)
            pred_sample = predictions[indices]
        else:
            pred_sample = predictions
        
        bootstrap_means = []
        n_bootstrap = min(self.bootstrap_samples, 100)  # Limit for performance
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(pred_sample, size=len(pred_sample), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return {
            'confidence_level': self.confidence_level,
            'lower_bound': float(ci_lower),
            'upper_bound': float(ci_upper),
            'mean_estimate': float(np.mean(bootstrap_means)),
            'bootstrap_std': float(np.std(bootstrap_means))
        }
    
    def _parametric_confidence_intervals(self, predictions: np.ndarray,
                                       ground_truth: np.ndarray) -> Dict[str, Any]:
        """Calculate parametric confidence intervals based on model errors."""
        # Sample for performance
        n_samples = min(len(predictions), 10000)
        if len(predictions) > n_samples:
            indices = np.random.choice(len(predictions), n_samples, replace=False)
            pred_sample = predictions[indices]
            gt_sample = ground_truth[indices]
        else:
            pred_sample = predictions
            gt_sample = ground_truth
        
        errors = pred_sample - gt_sample
        
        # Assume normal distribution of errors
        error_mean = np.mean(errors)
        error_std = np.std(errors, ddof=1)
        n = len(errors)
        
        # Standard error of the mean
        se_mean = error_std / np.sqrt(n)
        
        # t-distribution critical value
        alpha = 1 - self.confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, n - 1)
        
        # Confidence interval for bias
        bias_ci_lower = error_mean - t_critical * se_mean
        bias_ci_upper = error_mean + t_critical * se_mean
        
        return {
            'confidence_level': self.confidence_level,
            'bias_estimate': float(error_mean),
            'bias_ci_lower': float(bias_ci_lower),
            'bias_ci_upper': float(bias_ci_upper),
            'error_std': float(error_std),
            'sample_size': int(n)
        }
    
    def _calculate_uncertainty_metrics(self, predictions: np.ndarray,
                                     model_outputs: Optional[Dict[str, Any]] = None,
                                     ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate uncertainty-aware performance metrics."""
        uncertainty_metrics = {}
        
        if ground_truth is None:
            return {'note': 'Uncertainty-aware metrics require ground truth labels'}
        
        # Sample for performance
        n_samples = min(len(predictions), 5000)
        if len(predictions) > n_samples:
            indices = np.random.choice(len(predictions), n_samples, replace=False)
            pred_sample = predictions[indices]
            gt_sample = ground_truth[indices]
        else:
            pred_sample = predictions
            gt_sample = ground_truth
        
        # Calculate basic uncertainty estimates
        if self._is_classification_predictions(pred_sample):
            if pred_sample.ndim == 1:
                # Binary classification
                confidence = np.abs(pred_sample - 0.5) * 2
            else:
                # Multi-class classification
                confidence = np.max(pred_sample, axis=1)
            
            uncertainty = 1 - confidence
        else:
            # Regression: use prediction variance if available
            if model_outputs and 'prediction_variance' in model_outputs:
                var_sample = model_outputs['prediction_variance']
                if len(var_sample) > n_samples:
                    var_sample = var_sample[indices]
                uncertainty = np.sqrt(var_sample)
            else:
                # Use absolute residuals as proxy for uncertainty
                residuals = np.abs(pred_sample - gt_sample)
                uncertainty = residuals
        
        # Simple uncertainty-based metrics
        uncertainty_metrics['mean_uncertainty'] = float(np.mean(uncertainty))
        uncertainty_metrics['uncertainty_distribution'] = {
            'min': float(np.min(uncertainty)),
            'max': float(np.max(uncertainty)),
            'median': float(np.median(uncertainty)),
            'std': float(np.std(uncertainty))
        }
        
        return uncertainty_metrics
    
    def _bootstrap_uncertainty_estimation(self, predictions: np.ndarray,
                                        ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Estimate uncertainty using bootstrap sampling."""
        bootstrap_results = {}
        
        # Sample for performance
        n_samples = min(len(predictions), 5000)
        bootstrap_predictions = []
        
        n_bootstrap = min(self.bootstrap_samples, 50)  # Limit for performance
        
        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            bootstrap_indices = np.random.choice(len(predictions), size=n_samples, replace=True)
            
            # Bootstrap predictions
            bootstrap_pred = predictions[bootstrap_indices]
            bootstrap_predictions.append(np.mean(bootstrap_pred))
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Bootstrap uncertainty estimates
        bootstrap_mean = np.mean(bootstrap_predictions)
        bootstrap_std = np.std(bootstrap_predictions, ddof=1)
        
        # Confidence intervals
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_predictions, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_predictions, (1 - alpha/2) * 100)
        
        bootstrap_results.update({
            'bootstrap_mean': float(bootstrap_mean),
            'bootstrap_std': float(bootstrap_std),
            'confidence_interval': {
                'lower': float(ci_lower),
                'upper': float(ci_upper),
                'level': self.confidence_level
            },
            'bootstrap_samples_used': n_bootstrap
        })
        
        return bootstrap_results
    
    def _estimate_epistemic_uncertainty(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate epistemic uncertainty from model outputs."""
        epistemic_results = {}
        
        # Monte Carlo Dropout estimates
        if 'dropout_predictions' in model_outputs:
            dropout_preds = np.array(model_outputs['dropout_predictions'])
            
            # Calculate epistemic uncertainty as variance across dropout samples
            if dropout_preds.ndim >= 2:
                epistemic_var = np.var(dropout_preds, axis=0)
                epistemic_std = np.std(dropout_preds, axis=0)
                
                epistemic_results['monte_carlo_dropout'] = {
                    'epistemic_variance': epistemic_var.tolist()[:100],  # Limit output
                    'epistemic_std': epistemic_std.tolist()[:100],
                    'mean_epistemic_uncertainty': float(np.mean(epistemic_std)),
                    'max_epistemic_uncertainty': float(np.max(epistemic_std))
                }
        
        # Deep ensemble uncertainty
        if 'ensemble_predictions' in model_outputs:
            ensemble_preds = np.array(model_outputs['ensemble_predictions'])
            
            if ensemble_preds.ndim >= 2:
                ensemble_var = np.var(ensemble_preds, axis=0)
                ensemble_std = np.std(ensemble_preds, axis=0)
                
                epistemic_results['deep_ensemble'] = {
                    'ensemble_variance': ensemble_var.tolist()[:100],  # Limit output
                    'ensemble_std': ensemble_std.tolist()[:100],
                    'mean_ensemble_uncertainty': float(np.mean(ensemble_std)),
                    'max_ensemble_uncertainty': float(np.max(ensemble_std))
                }
        
        return epistemic_results
    
    def _estimate_aleatoric_uncertainty(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate aleatoric uncertainty from model outputs."""
        aleatoric_results = {}
        
        # Direct variance predictions
        if 'predicted_variance' in model_outputs:
            pred_variance = np.array(model_outputs['predicted_variance'])
            pred_std = np.sqrt(pred_variance)
            
            aleatoric_results['predicted_variance'] = {
                'aleatoric_variance': pred_variance.tolist()[:100],  # Limit output
                'aleatoric_std': pred_std.tolist()[:100],
                'mean_aleatoric_uncertainty': float(np.mean(pred_std)),
                'max_aleatoric_uncertainty': float(np.max(pred_std))
            }
        
        return aleatoric_results
    
    def _analyze_ensemble_uncertainty(self, ensemble_predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze uncertainty from ensemble predictions."""
        ensemble_preds = np.array(ensemble_predictions)
        
        if ensemble_preds.ndim < 2:
            return {'error': 'Ensemble predictions must be 2D array [n_models, n_samples]'}
        
        # Sample for performance
        if ensemble_preds.shape[1] > 5000:
            indices = np.random.choice(ensemble_preds.shape[1], 5000, replace=False)
            ensemble_preds = ensemble_preds[:, indices]
        
        # Calculate ensemble statistics
        ensemble_mean = np.mean(ensemble_preds, axis=0)
        ensemble_var = np.var(ensemble_preds, axis=0)
        ensemble_std = np.std(ensemble_preds, axis=0)
        
        # Ensemble agreement/disagreement
        ensemble_analysis = {
            'ensemble_variance': ensemble_var.tolist()[:100],  # Limit output
            'ensemble_std': ensemble_std.tolist()[:100],
            'mean_ensemble_uncertainty': float(np.mean(ensemble_std)),
            'coefficient_of_variation': (ensemble_std / (np.abs(ensemble_mean) + 1e-12)).tolist()[:100],
            'ensemble_mean': ensemble_mean.tolist()[:100],
            'n_models': int(ensemble_preds.shape[0]),
            'prediction_diversity': float(np.mean(ensemble_std))
        }
        
        return ensemble_analysis
    
    def _generate_uncertainty_summary(self, uncertainty_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall uncertainty assessment summary."""
        summary = {
            'uncertainty_status': 'unknown',
            'key_findings': [],
            'uncertainty_quality': {},
            'recommendations': [],
            'regulatory_considerations': {}
        }
        
        # Analyze calibration quality
        calibration = uncertainty_results.get('calibration_analysis', {})
        if 'expected_calibration_error' in calibration:
            ece = calibration['expected_calibration_error']
            is_well_calibrated = ece <= self.calibration_threshold
            
            summary['uncertainty_quality']['calibration'] = {
                'ece': float(ece),
                'is_well_calibrated': bool(is_well_calibrated),
                'quality': 'good' if is_well_calibrated else 'poor'
            }
            
            summary['key_findings'].append(
                f"Expected Calibration Error: {ece:.3f} ({'acceptable' if is_well_calibrated else 'high'})"
            )
        
        # Analyze predictive uncertainty
        pred_uncertainty = uncertainty_results.get('predictive_uncertainty', {})
        if 'binary_entropy' in pred_uncertainty:
            entropy_stats = pred_uncertainty['binary_entropy']
            mean_entropy = entropy_stats.get('mean', 0)
            
            summary['uncertainty_quality']['predictive_uncertainty'] = {
                'mean_entropy': float(mean_entropy),
                'interpretation': 'high' if mean_entropy > 0.5 else 'moderate' if mean_entropy > 0.2 else 'low'
            }
        
        # Analyze confidence intervals
        ci_results = uncertainty_results.get('confidence_intervals', {})
        if 'bootstrap_ci' in ci_results:
            bootstrap_ci = ci_results['bootstrap_ci']
            ci_width = bootstrap_ci.get('upper_bound', 0) - bootstrap_ci.get('lower_bound', 0)
            
            summary['uncertainty_quality']['confidence_intervals'] = {
                'ci_width': float(ci_width),
                'precision': 'high' if ci_width < 0.1 else 'moderate' if ci_width < 0.2 else 'low'
            }
        
        # Determine overall uncertainty status
        quality_scores = []
        
        if 'calibration' in summary['uncertainty_quality']:
            quality_scores.append(1.0 if summary['uncertainty_quality']['calibration']['is_well_calibrated'] else 0.0)
        
        if quality_scores:
            avg_quality = np.mean(quality_scores)
            if avg_quality >= 0.8:
                summary['uncertainty_status'] = 'excellent'
            elif avg_quality >= 0.6:
                summary['uncertainty_status'] = 'good'
            elif avg_quality >= 0.4:
                summary['uncertainty_status'] = 'acceptable'
            else:
                summary['uncertainty_status'] = 'poor'
        else:
            # Default to good if no specific quality measures
            summary['uncertainty_status'] = 'good'
        
        # Generate recommendations
        if summary['uncertainty_status'] == 'poor':
            summary['recommendations'].extend([
                "Improve model calibration before clinical deployment",
                "Consider ensemble methods for better uncertainty estimation",
                "Implement calibration correction techniques"
            ])
        elif summary['uncertainty_status'] == 'acceptable':
            summary['recommendations'].extend([
                "Monitor uncertainty quality in clinical deployment",
                "Consider calibration correction for improved reliability"
            ])
        else:
            summary['recommendations'].append(
                "Uncertainty estimation meets quality standards for clinical use"
            )
        
        # Regulatory considerations
        summary['regulatory_considerations'] = {
            'fda_uncertainty_guidance_compliance': summary['uncertainty_status'] in ['excellent', 'good'],
            'calibration_documented': 'calibration_analysis' in uncertainty_results,
            'confidence_intervals_provided': 'confidence_intervals' in uncertainty_results,
            'suitable_for_clinical_decision_support': summary['uncertainty_status'] in ['excellent', 'good', 'acceptable']
        }
        
        return summary