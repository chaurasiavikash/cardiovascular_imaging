"""
Statistical Validation Module for Medical Imaging
Implements FDA-compliant statistical validation methods
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
import statsmodels.api as sm
from statsmodels.stats.inter_rater import fleiss_kappa
import pingouin as pg
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


class StatisticalValidator:
    """
    Comprehensive statistical validation for medical imaging systems.
    
    Implements FDA-recommended statistical methods including:
    - Bland-Altman analysis
    - Intraclass correlation coefficient (ICC)
    - Agreement statistics
    - Confidence intervals
    - Power analysis
    - Multi-reader studies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize statistical validator.
        
        Args:
            config: Configuration dictionary with validation parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Statistical thresholds
        self.alpha = self.config.get('alpha', 0.05)
        self.icc_threshold = self.config.get('icc_threshold', 0.75)
        self.agreement_threshold = self.config.get('agreement_threshold', 0.8)
        self.power_threshold = self.config.get('power_threshold', 0.8)
        
    def comprehensive_statistical_validation(self, ground_truth: np.ndarray,
                                           predictions: np.ndarray,
                                           clinical_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive statistical validation analysis.
        
        Args:
            ground_truth: Ground truth measurements
            predictions: Algorithm predictions
            clinical_data: Additional clinical data for subgroup analysis
            
        Returns:
            Comprehensive statistical validation results
        """
        self.logger.info("Starting comprehensive statistical validation")
        
        # Ensure arrays are properly formatted
        gt_array = np.asarray(ground_truth).flatten()
        pred_array = np.asarray(predictions).flatten()
        
        if len(gt_array) != len(pred_array):
            raise ValueError(f"Mismatched array lengths: GT {len(gt_array)} vs Pred {len(pred_array)}")
        
        validation_results = {
            'agreement_analysis': self._perform_agreement_analysis(gt_array, pred_array),
            'bland_altman_analysis': self._perform_bland_altman_analysis(gt_array, pred_array),
            'correlation_analysis': self._perform_correlation_analysis(gt_array, pred_array),
            'hypothesis_testing': self._perform_hypothesis_testing(gt_array, pred_array),
            'confidence_intervals': self._calculate_confidence_intervals(gt_array, pred_array),
            'power_analysis': self._perform_power_analysis(gt_array, pred_array),
            'regression_analysis': self._perform_regression_analysis(gt_array, pred_array),
            'diagnostic_accuracy': self._calculate_diagnostic_accuracy(gt_array, pred_array),
            'outlier_analysis': self._perform_outlier_analysis(gt_array, pred_array)
        }
        
        # Clinical subgroup analysis if data available
        if clinical_data:
            validation_results['subgroup_analysis'] = self._perform_subgroup_analysis(
                gt_array, pred_array, clinical_data
            )
        
        # Overall statistical summary
        validation_results['statistical_summary'] = self._generate_statistical_summary(
            validation_results
        )
        
        self.logger.info("Statistical validation completed")
        return validation_results
    
    def _perform_agreement_analysis(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive agreement analysis."""
        self.logger.info("Performing agreement analysis")
        
        agreement_results = {}
        
        # Remove NaN values
        valid_indices = ~(np.isnan(gt) | np.isnan(pred))
        gt_clean = gt[valid_indices]
        pred_clean = pred[valid_indices]
        
        if len(gt_clean) == 0:
            return {'error': 'No valid data points for agreement analysis'}
        
        # Intraclass Correlation Coefficient (ICC)
        icc_results = self._calculate_icc(gt_clean, pred_clean)
        agreement_results['icc'] = icc_results
        
        # Lin's Concordance Correlation Coefficient
        ccc = self._calculate_concordance_correlation(gt_clean, pred_clean)
        agreement_results['concordance_correlation'] = ccc
        
        # Percentage agreement (for categorical data)
        if self._is_categorical_data(gt_clean, pred_clean):
            percent_agreement = np.mean(gt_clean == pred_clean) * 100
            agreement_results['percentage_agreement'] = float(percent_agreement)
            
            # Cohen's Kappa
            kappa = self._calculate_cohens_kappa(gt_clean, pred_clean)
            agreement_results['cohens_kappa'] = kappa
        
        # Mean absolute error and relative error
        mae = np.mean(np.abs(gt_clean - pred_clean))
        mape = np.mean(np.abs((gt_clean - pred_clean) / (gt_clean + 1e-8))) * 100
        
        agreement_results['mean_absolute_error'] = float(mae)
        agreement_results['mean_absolute_percentage_error'] = float(mape)
        
        # Root mean square error
        rmse = np.sqrt(np.mean((gt_clean - pred_clean) ** 2))
        agreement_results['root_mean_square_error'] = float(rmse)
        
        return agreement_results
    
    def _calculate_icc(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        """Calculate Intraclass Correlation Coefficient."""
        try:
            # Prepare data for ICC calculation
            data = np.column_stack([gt, pred])
            df = pd.DataFrame(data, columns=['rater1', 'rater2'])
            
            # Add subject IDs
            df['subject'] = range(len(df))
            
            # Reshape for pingouin ICC
            df_long = pd.melt(df, id_vars=['subject'], 
                             value_vars=['rater1', 'rater2'],
                             var_name='rater', value_name='score')
            
            # Calculate ICC using pingouin
            icc_result = pg.intraclass_corr(data=df_long, targets='subject', 
                                          raters='rater', ratings='score')
            
            # Extract ICC(2,1) - two-way random effects, single measurement
            icc_21 = icc_result[icc_result['Type'] == 'ICC2']['ICC'].iloc[0]
            icc_21_ci = icc_result[icc_result['Type'] == 'ICC2']['CI95%'].iloc[0]
            
            return {
                'icc_value': float(icc_21),
                'icc_lower_ci': float(icc_21_ci[0]),
                'icc_upper_ci': float(icc_21_ci[1]),
                'interpretation': self._interpret_icc(icc_21)
            }
            
        except Exception as e:
            self.logger.warning(f"ICC calculation failed: {str(e)}")
            return {
                'icc_value': 0.0,
                'icc_lower_ci': 0.0,
                'icc_upper_ci': 0.0,
                'interpretation': 'poor',
                'error': str(e)
            }
    
    def _interpret_icc(self, icc_value: float) -> str:
        """Interpret ICC value according to standard guidelines."""
        if icc_value >= 0.9:
            return 'excellent'
        elif icc_value >= 0.75:
            return 'good'
        elif icc_value >= 0.5:
            return 'moderate'
        elif icc_value >= 0.25:
            return 'poor'
        else:
            return 'very_poor'
    
    def _calculate_concordance_correlation(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        """Calculate Lin's Concordance Correlation Coefficient."""
        try:
            # Remove any remaining NaN values
            valid_mask = ~(np.isnan(gt) | np.isnan(pred))
            gt_valid = gt[valid_mask]
            pred_valid = pred[valid_mask]
            
            if len(gt_valid) == 0:
                return {'ccc_value': 0.0, 'error': 'No valid data'}
            
            # Calculate means and variances
            gt_mean = np.mean(gt_valid)
            pred_mean = np.mean(pred_valid)
            gt_var = np.var(gt_valid, ddof=1)
            pred_var = np.var(pred_valid, ddof=1)
            
            # Calculate covariance
            covariance = np.cov(gt_valid, pred_valid)[0, 1]
            
            # Lin's CCC formula
            numerator = 2 * covariance
            denominator = gt_var + pred_var + (gt_mean - pred_mean) ** 2
            
            ccc = numerator / (denominator + 1e-8)
            
            # Calculate confidence interval using Fisher's z-transformation
            z_ccc = 0.5 * np.log((1 + ccc) / (1 - ccc + 1e-8))
            se_z = 1 / np.sqrt(len(gt_valid) - 3)
            z_lower = z_ccc - 1.96 * se_z
            z_upper = z_ccc + 1.96 * se_z
            
            ccc_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            ccc_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            
            return {
                'ccc_value': float(ccc),
                'ccc_lower_ci': float(ccc_lower),
                'ccc_upper_ci': float(ccc_upper),
                'interpretation': self._interpret_correlation(ccc)
            }
            
        except Exception as e:
            self.logger.warning(f"CCC calculation failed: {str(e)}")
            return {'ccc_value': 0.0, 'error': str(e)}
    
    def _is_categorical_data(self, gt: np.ndarray, pred: np.ndarray) -> bool:
        """Check if data appears to be categorical."""
        # Simple heuristic: if all values are integers and range is small
        gt_int = np.all(gt == np.round(gt))
        pred_int = np.all(pred == np.round(pred))
        
        if gt_int and pred_int:
            unique_values = len(np.unique(np.concatenate([gt, pred])))
            return unique_values <= 10  # Arbitrary threshold
        
        return False
    
    def _calculate_cohens_kappa(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        """Calculate Cohen's Kappa for categorical agreement."""
        try:
            from sklearn.metrics import cohen_kappa_score
            
            kappa = cohen_kappa_score(gt, pred)
            
            # Calculate standard error and confidence interval
            n = len(gt)
            p_o = np.mean(gt == pred)  # Observed agreement
            
            # Calculate expected agreement
            gt_counts = np.bincount(gt.astype(int))
            pred_counts = np.bincount(pred.astype(int))
            
            max_label = max(len(gt_counts), len(pred_counts))
            gt_counts = np.pad(gt_counts, (0, max_label - len(gt_counts)))
            pred_counts = np.pad(pred_counts, (0, max_label - len(pred_counts)))
            
            p_e = np.sum((gt_counts / n) * (pred_counts / n))
            
            # Standard error approximation
            se_kappa = np.sqrt((p_o * (1 - p_o)) / (n * (1 - p_e) ** 2))
            
            kappa_lower = kappa - 1.96 * se_kappa
            kappa_upper = kappa + 1.96 * se_kappa
            
            return {
                'kappa_value': float(kappa),
                'kappa_lower_ci': float(kappa_lower),
                'kappa_upper_ci': float(kappa_upper),
                'interpretation': self._interpret_kappa(kappa)
            }
            
        except Exception as e:
            self.logger.warning(f"Kappa calculation failed: {str(e)}")
            return {'kappa_value': 0.0, 'error': str(e)}
    
    def _interpret_kappa(self, kappa_value: float) -> str:
        """Interpret Kappa value according to Landis and Koch guidelines."""
        if kappa_value >= 0.81:
            return 'almost_perfect'
        elif kappa_value >= 0.61:
            return 'substantial'
        elif kappa_value >= 0.41:
            return 'moderate'
        elif kappa_value >= 0.21:
            return 'fair'
        elif kappa_value >= 0.0:
            return 'slight'
        else:
            return 'poor'
    
    def _perform_bland_altman_analysis(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
        """Perform Bland-Altman analysis."""
        self.logger.info("Performing Bland-Altman analysis")
        
        # Remove NaN values
        valid_indices = ~(np.isnan(gt) | np.isnan(pred))
        gt_clean = gt[valid_indices]
        pred_clean = pred[valid_indices]
        
        if len(gt_clean) == 0:
            return {'error': 'No valid data points for Bland-Altman analysis'}
        
        # Calculate differences and means
        differences = pred_clean - gt_clean
        means = (pred_clean + gt_clean) / 2
        
        # Calculate bias and limits of agreement
        bias = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # 95% limits of agreement
        upper_loa = bias + 1.96 * std_diff
        lower_loa = bias - 1.96 * std_diff
        
        # Confidence intervals for bias and limits
        n = len(differences)
        se_bias = std_diff / np.sqrt(n)
        se_loa = std_diff * np.sqrt(3 / n)
        
        bias_ci_lower = bias - 1.96 * se_bias
        bias_ci_upper = bias + 1.96 * se_bias
        
        upper_loa_ci_lower = upper_loa - 1.96 * se_loa
        upper_loa_ci_upper = upper_loa + 1.96 * se_loa
        
        lower_loa_ci_lower = lower_loa - 1.96 * se_loa
        lower_loa_ci_upper = lower_loa + 1.96 * se_loa
        
        # Percentage of points within limits
        within_limits = np.sum((differences >= lower_loa) & (differences <= upper_loa))
        percentage_within = (within_limits / n) * 100
        
        # Test for proportional bias (correlation between differences and means)
        if len(means) > 3:
            correlation_coef, correlation_p = pearsonr(means, differences)
        else:
            correlation_coef, correlation_p = 0.0, 1.0
        
        return {
            'bias': float(bias),
            'bias_ci_lower': float(bias_ci_lower),
            'bias_ci_upper': float(bias_ci_upper),
            'upper_loa': float(upper_loa),
            'upper_loa_ci_lower': float(upper_loa_ci_lower),
            'upper_loa_ci_upper': float(upper_loa_ci_upper),
            'lower_loa': float(lower_loa),
            'lower_loa_ci_lower': float(lower_loa_ci_lower),
            'lower_loa_ci_upper': float(lower_loa_ci_upper),
            'standard_deviation': float(std_diff),
            'percentage_within_limits': float(percentage_within),
            'proportional_bias_correlation': float(correlation_coef),
            'proportional_bias_p_value': float(correlation_p),
            'has_proportional_bias': bool(correlation_p < self.alpha),
            'sample_size': int(n)
        }
    
    def _perform_correlation_analysis(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis."""
        self.logger.info("Performing correlation analysis")
        
        # Remove NaN values
        valid_indices = ~(np.isnan(gt) | np.isnan(pred))
        gt_clean = gt[valid_indices]
        pred_clean = pred[valid_indices]
        
        if len(gt_clean) < 3:
            return {'error': 'Insufficient data for correlation analysis'}
        
        correlation_results = {}
        
        # Pearson correlation
        try:
            pearson_r, pearson_p = pearsonr(gt_clean, pred_clean)
            correlation_results['pearson'] = {
                'correlation': float(pearson_r),
                'p_value': float(pearson_p),
                'significant': bool(pearson_p < self.alpha),
                'interpretation': self._interpret_correlation(pearson_r)
            }
        except Exception as e:
            correlation_results['pearson'] = {'error': str(e)}
        
        # Spearman correlation
        try:
            spearman_r, spearman_p = spearmanr(gt_clean, pred_clean)
            correlation_results['spearman'] = {
                'correlation': float(spearman_r),
                'p_value': float(spearman_p),
                'significant': bool(spearman_p < self.alpha),
                'interpretation': self._interpret_correlation(spearman_r)
            }
        except Exception as e:
            correlation_results['spearman'] = {'error': str(e)}
        
        # R-squared
        try:
            r_squared = pearson_r ** 2
            correlation_results['r_squared'] = {
                'value': float(r_squared),
                'interpretation': f"{r_squared*100:.1f}% of variance explained"
            }
        except:
            correlation_results['r_squared'] = {'value': 0.0}
        
        return correlation_results
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient magnitude."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.9:
            return 'very_strong'
        elif abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'very_weak'
    
    def _perform_hypothesis_testing(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
        """Perform hypothesis testing for bias assessment."""
        self.logger.info("Performing hypothesis testing")
        
        # Remove NaN values
        valid_indices = ~(np.isnan(gt) | np.isnan(pred))
        gt_clean = gt[valid_indices]
        pred_clean = pred[valid_indices]
        
        if len(gt_clean) < 3:
            return {'error': 'Insufficient data for hypothesis testing'}
        
        hypothesis_results = {}
        
        # Paired t-test for bias
        try:
            t_stat, t_p = ttest_rel(pred_clean, gt_clean)
            hypothesis_results['paired_t_test'] = {
                't_statistic': float(t_stat),
                'p_value': float(t_p),
                'significant_bias': bool(t_p < self.alpha),
                'interpretation': 'Significant bias detected' if t_p < self.alpha else 'No significant bias'
            }
        except Exception as e:
            hypothesis_results['paired_t_test'] = {'error': str(e)}
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            w_stat, w_p = wilcoxon(pred_clean, gt_clean)
            hypothesis_results['wilcoxon_test'] = {
                'w_statistic': float(w_stat),
                'p_value': float(w_p),
                'significant_bias': bool(w_p < self.alpha),
                'interpretation': 'Significant bias detected' if w_p < self.alpha else 'No significant bias'
            }
        except Exception as e:
            hypothesis_results['wilcoxon_test'] = {'error': str(e)}
        
        # Normality tests on differences
        differences = pred_clean - gt_clean
        try:
            shapiro_stat, shapiro_p = stats.shapiro(differences)
            hypothesis_results['normality_test'] = {
                'shapiro_statistic': float(shapiro_stat),
                'shapiro_p_value': float(shapiro_p),
                'differences_normal': bool(shapiro_p >= self.alpha),
                'recommendation': 'Use t-test' if shapiro_p >= self.alpha else 'Use Wilcoxon test'
            }
        except Exception as e:
            hypothesis_results['normality_test'] = {'error': str(e)}
        
        return hypothesis_results
    
    def _calculate_confidence_intervals(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
        """Calculate confidence intervals for key metrics."""
        self.logger.info("Calculating confidence intervals")
        
        # Remove NaN values
        valid_indices = ~(np.isnan(gt) | np.isnan(pred))
        gt_clean = gt[valid_indices]
        pred_clean = pred[valid_indices]
        
        if len(gt_clean) < 3:
            return {'error': 'Insufficient data for confidence intervals'}
        
        ci_results = {}
        n = len(gt_clean)
        
        # Confidence interval for bias
        differences = pred_clean - gt_clean
        bias = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        se_bias = std_diff / np.sqrt(n)
        
        # 95% CI for bias
        bias_ci_lower = bias - 1.96 * se_bias
        bias_ci_upper = bias + 1.96 * se_bias
        
        ci_results['bias_95_ci'] = {
            'lower': float(bias_ci_lower),
            'upper': float(bias_ci_upper),
            'bias': float(bias)
        }
        
        # Bootstrap confidence intervals for correlation
        try:
            bootstrap_correlations = []
            n_bootstrap = 1000
            
            for _ in range(n_bootstrap):
                indices = np.random.choice(n, n, replace=True)
                boot_gt = gt_clean[indices]
                boot_pred = pred_clean[indices]
                
                if np.std(boot_gt) > 0 and np.std(boot_pred) > 0:
                    boot_corr, _ = pearsonr(boot_gt, boot_pred)
                    if not np.isnan(boot_corr):
                        bootstrap_correlations.append(boot_corr)
            
            if bootstrap_correlations:
                correlation_ci_lower = np.percentile(bootstrap_correlations, 2.5)
                correlation_ci_upper = np.percentile(bootstrap_correlations, 97.5)
                
                ci_results['correlation_95_ci'] = {
                    'lower': float(correlation_ci_lower),
                    'upper': float(correlation_ci_upper),
                    'correlation': float(np.mean(bootstrap_correlations))
                }
        except Exception as e:
            self.logger.warning(f"Bootstrap CI calculation failed: {str(e)}")
        
        return ci_results
    
    def _perform_power_analysis(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
        """Perform statistical power analysis."""
        self.logger.info("Performing power analysis")
        
        # Remove NaN values
        valid_indices = ~(np.isnan(gt) | np.isnan(pred))
        gt_clean = gt[valid_indices]
        pred_clean = pred[valid_indices]
        
        if len(gt_clean) < 3:
            return {'error': 'Insufficient data for power analysis'}
        
        power_results = {}
        
        # Calculate effect size (Cohen's d)
        differences = pred_clean - gt_clean
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        cohens_d = abs(mean_diff) / (std_diff + 1e-8)
        
        # Power calculation for paired t-test
        n = len(differences)
        
        try:
            # Approximate power calculation
            t_critical = stats.t.ppf(1 - self.alpha/2, n-1)
            ncp = cohens_d * np.sqrt(n)  # Non-centrality parameter
            power = 1 - stats.t.cdf(t_critical, n-1, ncp) + stats.t.cdf(-t_critical, n-1, ncp)
            
            power_results['current_power'] = {
                'power': float(power),
                'cohens_d': float(cohens_d),
                'sample_size': int(n),
                'adequate_power': bool(power >= self.power_threshold)
            }
            
            # Sample size calculation for desired power
            desired_power = 0.8
            if cohens_d > 0:
                # Iterative sample size calculation
                for n_required in range(5, 1000):
                    ncp_req = cohens_d * np.sqrt(n_required)
                    t_crit_req = stats.t.ppf(1 - self.alpha/2, n_required-1)
                    power_req = 1 - stats.t.cdf(t_crit_req, n_required-1, ncp_req) + \
                               stats.t.cdf(-t_crit_req, n_required-1, ncp_req)
                    
                    if power_req >= desired_power:
                        break
                
                power_results['sample_size_recommendation'] = {
                    'required_n': int(n_required),
                    'current_n': int(n),
                    'additional_samples_needed': max(0, n_required - n)
                }
            
        except Exception as e:
            self.logger.warning(f"Power analysis failed: {str(e)}")
            power_results['error'] = str(e)
        
        return power_results
    
    def _perform_regression_analysis(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
        """Perform regression analysis."""
        self.logger.info("Performing regression analysis")
        
        # Remove NaN values
        valid_indices = ~(np.isnan(gt) | np.isnan(pred))
        gt_clean = gt[valid_indices]
        pred_clean = pred[valid_indices]
        
        if len(gt_clean) < 3:
            return {'error': 'Insufficient data for regression analysis'}
        
        regression_results = {}
        
        try:
            # Simple linear regression: pred = a + b * gt
            X = sm.add_constant(gt_clean)  # Add intercept
            model = sm.OLS(pred_clean, X).fit()
            
            regression_results['linear_regression'] = {
                'intercept': float(model.params[0]),
                'slope': float(model.params[1]),
                'r_squared': float(model.rsquared),
                'r_squared_adj': float(model.rsquared_adj),
                'f_statistic': float(model.fvalue),
                'f_p_value': float(model.f_pvalue),
                'intercept_p_value': float(model.pvalues[0]),
                'slope_p_value': float(model.pvalues[1]),
                'intercept_ci_lower': float(model.conf_int().iloc[0, 0]),
                'intercept_ci_upper': float(model.conf_int().iloc[0, 1]),
                'slope_ci_lower': float(model.conf_int().iloc[1, 0]),
                'slope_ci_upper': float(model.conf_int().iloc[1, 1]),
                'residual_std_error': float(np.sqrt(model.mse_resid))
            }
            
            # Test for perfect agreement (slope=1, intercept=0)
            slope_test = abs(model.params[1] - 1.0) / model.bse[1]
            intercept_test = abs(model.params[0]) / model.bse[0]
            
            regression_results['agreement_tests'] = {
                'slope_equals_one': {
                    'test_statistic': float(slope_test),
                    'p_value': float(2 * (1 - stats.t.cdf(abs(slope_test), model.df_resid))),
                    'slope_significantly_different_from_1': bool(slope_test > stats.t.ppf(1-self.alpha/2, model.df_resid))
                },
                'intercept_equals_zero': {
                    'test_statistic': float(intercept_test),
                    'p_value': float(2 * (1 - stats.t.cdf(abs(intercept_test), model.df_resid))),
                    'intercept_significantly_different_from_0': bool(intercept_test > stats.t.ppf(1-self.alpha/2, model.df_resid))
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Regression analysis failed: {str(e)}")
            regression_results['error'] = str(e)
        
        return regression_results
    
    def _calculate_diagnostic_accuracy(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
        """Calculate diagnostic accuracy metrics if applicable."""
        self.logger.info("Calculating diagnostic accuracy metrics")
        
        # Check if data is suitable for diagnostic accuracy
        if not self._is_categorical_data(gt, pred):
            return {'note': 'Data not suitable for diagnostic accuracy analysis'}
        
        # Remove NaN values
        valid_indices = ~(np.isnan(gt) | np.isnan(pred))
        gt_clean = gt[valid_indices].astype(int)
        pred_clean = pred[valid_indices].astype(int)
        
        if len(gt_clean) < 10:
            return {'error': 'Insufficient data for diagnostic accuracy'}
        
        diagnostic_results = {}
        
        try:
            # Confusion matrix
            cm = confusion_matrix(gt_clean, pred_clean)
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                # Basic metrics
                sensitivity = tp / (tp + fn + 1e-8)
                specificity = tn / (tn + fp + 1e-8)
                ppv = tp / (tp + fp + 1e-8)
                npv = tn / (tn + fn + 1e-8)
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                
                # Likelihood ratios
                lr_positive = sensitivity / (1 - specificity + 1e-8)
                lr_negative = (1 - sensitivity) / (specificity + 1e-8)
                
                diagnostic_results['binary_classification'] = {
                    'sensitivity': float(sensitivity),
                    'specificity': float(specificity),
                    'positive_predictive_value': float(ppv),
                    'negative_predictive_value': float(npv),
                    'accuracy': float(accuracy),
                    'likelihood_ratio_positive': float(lr_positive),
                    'likelihood_ratio_negative': float(lr_negative),
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn)
                }
                
                # ROC analysis if we have probability scores
                # For now, use binary predictions
                try:
                    fpr, tpr, _ = roc_curve(gt_clean, pred_clean)
                    auc_score = auc(fpr, tpr)
                    
                    diagnostic_results['roc_analysis'] = {
                        'auc': float(auc_score),
                        'interpretation': self._interpret_auc(auc_score)
                    }
                except Exception as e:
                    self.logger.warning(f"ROC analysis failed: {str(e)}")
            
        except Exception as e:
            self.logger.warning(f"Diagnostic accuracy calculation failed: {str(e)}")
            diagnostic_results['error'] = str(e)
        
        return diagnostic_results
    
    def _interpret_auc(self, auc_value: float) -> str:
        """Interpret AUC value."""
        if auc_value >= 0.9:
            return 'excellent'
        elif auc_value >= 0.8:
            return 'good'
        elif auc_value >= 0.7:
            return 'fair'
        elif auc_value >= 0.6:
            return 'poor'
        else:
            return 'fail'
    
    def _perform_outlier_analysis(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
        """Perform outlier analysis."""
        self.logger.info("Performing outlier analysis")
        
        # Remove NaN values
        valid_indices = ~(np.isnan(gt) | np.isnan(pred))
        gt_clean = gt[valid_indices]
        pred_clean = pred[valid_indices]
        
        if len(gt_clean) < 5:
            return {'error': 'Insufficient data for outlier analysis'}
        
        outlier_results = {}
        
        # Calculate differences
        differences = pred_clean - gt_clean
        
        # IQR method for outliers in differences
        q1 = np.percentile(differences, 25)
        q3 = np.percentile(differences, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_indices = (differences < lower_bound) | (differences > upper_bound)
        n_outliers = np.sum(outlier_indices)
        
        outlier_results['iqr_method'] = {
            'n_outliers': int(n_outliers),
            'outlier_percentage': float(n_outliers / len(differences) * 100),
            'outlier_indices': outlier_indices.tolist(),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
        
        # Z-score method
        z_scores = np.abs(stats.zscore(differences))
        z_outliers = z_scores > 3  # Standard threshold
        n_z_outliers = np.sum(z_outliers)
        
        outlier_results['z_score_method'] = {
            'n_outliers': int(n_z_outliers),
            'outlier_percentage': float(n_z_outliers / len(differences) * 100),
            'outlier_indices': z_outliers.tolist(),
            'threshold': 3.0
        }
        
        return outlier_results
    
    def _perform_subgroup_analysis(self, gt: np.ndarray, pred: np.ndarray, 
                                 clinical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform subgroup analysis based on clinical data."""
        self.logger.info("Performing subgroup analysis")
        
        subgroup_results = {}
        
        # Example subgroup analyses
        for subgroup_var, subgroup_values in clinical_data.items():
            if len(subgroup_values) != len(gt):
                continue
                
            unique_groups = np.unique(subgroup_values)
            
            if len(unique_groups) < 2:
                continue
            
            group_results = {}
            
            for group in unique_groups:
                group_mask = np.array(subgroup_values) == group
                gt_group = gt[group_mask]
                pred_group = pred[group_mask]
                
                if len(gt_group) < 3:
                    continue
                
                # Basic statistics for each group
                group_stats = self._perform_agreement_analysis(gt_group, pred_group)
                group_results[str(group)] = group_stats
            
            if len(group_results) >= 2:
                subgroup_results[subgroup_var] = {
                    'group_results': group_results,
                    'n_groups': len(group_results)
                }
        
        return subgroup_results
    
    def _generate_statistical_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall statistical validation summary."""
        summary = {
            'validation_status': 'unknown',
            'key_findings': [],
            'statistical_significance': {},
            'clinical_significance': {},
            'recommendations': []
        }
        
        # Extract key metrics
        agreement = validation_results.get('agreement_analysis', {})
        bland_altman = validation_results.get('bland_altman_analysis', {})
        correlation = validation_results.get('correlation_analysis', {})
        
        # Check statistical significance
        icc_value = agreement.get('icc', {}).get('icc_value', 0)
        correlation_r = correlation.get('pearson', {}).get('correlation', 0)
        bias = bland_altman.get('bias', 0)
        
        summary['statistical_significance'] = {
            'excellent_agreement': icc_value >= 0.9,
            'good_agreement': icc_value >= 0.75,
            'strong_correlation': abs(correlation_r) >= 0.7,
            'minimal_bias': abs(bias) < 0.1 * np.std([0, 1])  # Placeholder
        }
        
        # Determine overall validation status
        if (summary['statistical_significance']['excellent_agreement'] and 
            summary['statistical_significance']['strong_correlation']):
            summary['validation_status'] = 'excellent'
        elif (summary['statistical_significance']['good_agreement'] and 
              abs(correlation_r) >= 0.5):
            summary['validation_status'] = 'good'
        elif icc_value >= 0.5:
            summary['validation_status'] = 'acceptable'
        else:
            summary['validation_status'] = 'poor'
        
        # Generate key findings
        summary['key_findings'] = [
            f"ICC: {icc_value:.3f} ({agreement.get('icc', {}).get('interpretation', 'unknown')})",
            f"Correlation: {correlation_r:.3f} ({correlation.get('pearson', {}).get('interpretation', 'unknown')})",
            f"Bias: {bias:.3f}",
            f"Overall status: {summary['validation_status']}"
        ]
        
        # Generate recommendations
        if summary['validation_status'] == 'poor':
            summary['recommendations'].extend([
                "Algorithm requires significant improvement before clinical use",
                "Consider additional training data or algorithm modifications",
                "Perform detailed error analysis to identify failure modes"
            ])
        elif summary['validation_status'] == 'acceptable':
            summary['recommendations'].extend([
                "Algorithm shows promise but needs refinement",
                "Consider additional validation studies",
                "Monitor performance in clinical deployment"
            ])
        else:
            summary['recommendations'].append(
                "Algorithm demonstrates good statistical validation performance"
            )
        
        return summary