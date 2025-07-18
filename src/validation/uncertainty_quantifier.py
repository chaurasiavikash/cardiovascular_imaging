 
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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
                    'values': entropy.tolist(),
                    'mean': float(np.mean(entropy)),
                    'std': float(np.std(entropy)),
                    'max': float(np.max(entropy)),
                    'min': float(np.min(entropy))
                },
                'max_probability': {
                    'values': max_prob.tolist(),
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
                    'values': binary_entropy.tolist(),
                    'mean': float(np.mean(binary_entropy)),
                    'std': float(np.std(binary_entropy))
                },
                'decision_boundary_distance': {
                    'values': decision_distance.tolist(),
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
                    'values': pred_variance.tolist(),
                    'mean': float(np.mean(pred_variance)),
                    'std': float(np.std(pred_variance))
                },
                'prediction_std': {
                    'values': pred_std.tolist(),
                    'mean': float(np.mean(pred_std)),
                    'std': float(np.std(pred_std))
                }
            })
        
        # If multiple predictions available (e.g., from dropout)
        if model_outputs and 'multiple_predictions' in model_outputs:
            multiple_preds = np.asarray(model_outputs['multiple_predictions'])
            
            if multiple_preds.ndim == 2:  # [n_samples, n_predictions]
                empirical_variance = np.var(multiple_preds, axis=1)
                empirical_std = np.std(multiple_preds, axis=1)
                
                uncertainty_measures.update({
                    'empirical_variance': {
                        'values': empirical_variance.tolist(),
                        'mean': float(np.mean(empirical_variance)),
                        'std': float(np.std(empirical_variance))
                    },
                    'empirical_std': {
                        'values': empirical_std.tolist(),
                        'mean': float(np.mean(empirical_std)),
                        'std': float(np.std(empirical_std))
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
                'values': confidence.tolist(),
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
                'values': uncertainty.tolist(),
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
                
                # Reliability diagram
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    gt_binary, predictions, n_bins=10
                )
                
                # Expected Calibration Error (ECE)
                bin_boundaries = np.linspace(0, 1, 11)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                ece = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = gt_binary[in_bin].mean()
                        avg_confidence_in_bin = predictions[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                # Maximum Calibration Error (MCE)
                mce = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
                    
                    if in_bin.sum() > 0:
                        accuracy_in_bin = gt_binary[in_bin].mean()
                        avg_confidence_in_bin = predictions[in_bin].mean()
                        mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
                
                # Brier Score
                brier_score = brier_score_loss(gt_binary, predictions)
                
                calibration_results.update({
                    'reliability_diagram': {
                        'fraction_of_positives': fraction_of_positives.tolist(),
                        'mean_predicted_value': mean_predicted_value.tolist()
                    },
                    'expected_calibration_error': float(ece),
                    'maximum_calibration_error': float(mce),
                    'brier_score': float(brier_score),
                    'is_well_calibrated': bool(ece <= self.calibration_threshold)
                })
                
                # Calibration correction
                calibration_results['calibration_correction'] = self._perform_calibration_correction(
                    predictions, gt_binary
                )
                
            elif predictions.ndim == 2:
                # Multi-class classification
                gt_int = ground_truth.astype(int)
                n_classes = predictions.shape[1]
                
                # Calculate ECE for multi-class
                confidences = np.max(predictions, axis=1)
                predicted_classes = np.argmax(predictions, axis=1)
                accuracies = (predicted_classes == gt_int).astype(float)
                
                bin_boundaries = np.linspace(0, 1, 11)
                ece = 0
                
                for i in range(10):
                    bin_lower = bin_boundaries[i]
                    bin_upper = bin_boundaries[i + 1]
                    in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = accuracies[in_bin].mean()
                        avg_confidence_in_bin = confidences[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                calibration_results.update({
                    'multiclass_ece': float(ece),
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
        # This requires uncertainty estimates
        residuals = predictions - ground_truth
        
        # Empirical coverage analysis
        residual_std = np.std(residuals)
        
        # Check coverage for different confidence levels
        confidence_levels = [0.68, 0.90, 0.95, 0.99]
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
    
    def _perform_calibration_correction(self, predictions: np.ndarray,
                                      ground_truth: np.ndarray) -> Dict[str, Any]:
        """Perform calibration correction for classification."""
        correction_results = {}
        
        try:
            # Platt scaling (logistic regression)
            platt_calibrator = LogisticRegression()
            platt_calibrator.fit(predictions.reshape(-1, 1), ground_truth)
            platt_corrected = platt_calibrator.predict_proba(predictions.reshape(-1, 1))[:, 1]
            
            # Isotonic regression
            isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
            isotonic_corrected = isotonic_calibrator.fit_transform(predictions, ground_truth)
            
            # Evaluate corrected calibrations
            original_ece = self._calculate_ece(predictions, ground_truth)
            platt_ece = self._calculate_ece(platt_corrected, ground_truth)
            isotonic_ece = self._calculate_ece(isotonic_corrected, ground_truth)
            
            correction_results.update({
                'original_ece': float(original_ece),
                'platt_scaling': {
                    'corrected_predictions': platt_corrected.tolist(),
                    'ece_after_correction': float(platt_ece),
                    'improvement': float(original_ece - platt_ece)
                },
                'isotonic_regression': {
                    'corrected_predictions': isotonic_corrected.tolist(),
                    'ece_after_correction': float(isotonic_ece),
                    'improvement': float(original_ece - isotonic_ece)
                },
                'best_method': 'platt_scaling' if platt_ece < isotonic_ece else 'isotonic_regression'
            })
            
        except Exception as e:
            self.logger.warning(f"Calibration correction failed: {str(e)}")
            correction_results['error'] = str(e)
        
        return correction_results
    
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
        n_samples = len(predictions)
        bootstrap_means = []
        
        for _ in range(self.bootstrap_samples):
            bootstrap_sample = np.random.choice(predictions, size=n_samples, replace=True)
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
        errors = predictions - ground_truth
        
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