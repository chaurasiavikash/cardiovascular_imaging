def _calculate_uncertainty_metrics(self, predictions: np.ndarray,
                                     model_outputs: Optional[Dict[str, Any]] = None,
                                     ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate uncertainty-aware performance metrics."""
        uncertainty_metrics = {}
        
        if ground_truth is None:
            return {'note': 'Uncertainty-aware metrics require ground truth labels'}
        
        gt_array = np.asarray(ground_truth)
        
        # Calculate basic uncertainty estimates
        if self._is_classification_predictions(predictions):
            if predictions.ndim == 1:
                # Binary classification
                confidence = np.abs(predictions - 0.5) * 2
            else:
                # Multi-class classification
                confidence = np.max(predictions, axis=1)
            
            uncertainty = 1 - confidence
            
        else:
            # Regression: use prediction variance if available
            if model_outputs and 'prediction_variance' in model_outputs:
                uncertainty = np.sqrt(model_outputs['prediction_variance'])
            else:
                # Use absolute residuals as proxy for uncertainty
                residuals = np.abs(predictions - gt_array)
                uncertainty = residuals
        
        # Uncertainty-based performance metrics
        uncertainty_metrics.update(self._calculate_selective_performance(
            predictions, gt_array, uncertainty
        ))
        
        # Risk-coverage curves
        uncertainty_metrics['risk_coverage'] = self._calculate_risk_coverage_curve(
            predictions, gt_array, uncertainty
        )
        
        return uncertainty_metrics
    
    def _calculate_selective_performance(self, predictions: np.ndarray,
                                       ground_truth: np.ndarray,
                                       uncertainty: np.ndarray) -> Dict[str, Any]:
        """Calculate performance metrics with selective prediction based on uncertainty."""
        selective_metrics = {}
        
        # Sort by uncertainty (low to high)
        sorted_indices = np.argsort(uncertainty)
        
        # Calculate performance at different coverage levels
        coverage_levels = np.arange(0.1, 1.01, 0.1)
        
        performance_at_coverage = []
        
        for coverage in coverage_levels:
            n_samples_to_keep = int(coverage * len(predictions))
            indices_to_keep = sorted_indices[:n_samples_to_keep]
            
            pred_subset = predictions[indices_to_keep]
            gt_subset = ground_truth[indices_to_keep]
            
            if self._is_classification_predictions(predictions):
                if predictions.ndim == 1:
                    # Binary classification
                    pred_binary = (pred_subset > 0.5).astype(int)
                    accuracy = accuracy_score(gt_subset.astype(int), pred_binary)
                    performance = accuracy
                else:
                    # Multi-class classification
                    pred_classes = np.argmax(pred_subset, axis=1)
                    accuracy = accuracy_score(gt_subset.astype(int), pred_classes)
                    performance = accuracy
            else:
                # Regression: use negative MAE as performance (higher is better)
                mae = np.mean(np.abs(pred_subset - gt_subset))
                performance = -mae
            
            performance_at_coverage.append({
                'coverage': float(coverage),
                'performance': float(performance),
                'n_samples': int(n_samples_to_keep)
            })
        
        selective_metrics['selective_performance'] = performance_at_coverage
        
        # Area under the risk-coverage curve
        coverages = [item['coverage'] for item in performance_at_coverage]
        performances = [item['performance'] for item in performance_at_coverage]
        
        # Calculate AUC using trapezoidal rule
        auc_selective = np.trapz(performances, coverages)
        selective_metrics['auc_selective_performance'] = float(auc_selective)
        
        return selective_metrics
    
    def _calculate_risk_coverage_curve(self, predictions: np.ndarray,
                                     ground_truth: np.ndarray,
                                     uncertainty: np.ndarray) -> Dict[str, Any]:
        """Calculate risk-coverage curve for uncertainty evaluation."""
        # Sort by uncertainty (high to low for risk-coverage)
        sorted_indices = np.argsort(-uncertainty)
        
        coverages = []
        risks = []
        
        for i in range(1, len(predictions) + 1):
            # Keep top i most certain predictions
            indices_to_keep = sorted_indices[len(predictions) - i:]
            
            pred_subset = predictions[indices_to_keep]
            gt_subset = ground_truth[indices_to_keep]
            
            coverage = i / len(predictions)
            
            if self._is_classification_predictions(predictions):
                if predictions.ndim == 1:
                    pred_binary = (pred_subset > 0.5).astype(int)
                    error_rate = 1 - accuracy_score(gt_subset.astype(int), pred_binary)
                    risk = error_rate
                else:
                    pred_classes = np.argmax(pred_subset, axis=1)
                    error_rate = 1 - accuracy_score(gt_subset.astype(int), pred_classes)
                    risk = error_rate
            else:
                # Regression: use MAE as risk
                mae = np.mean(np.abs(pred_subset - gt_subset))
                risk = mae
            
            coverages.append(coverage)
            risks.append(risk)
        
        return {
            'coverages': coverages,
            'risks': risks,
            'area_under_curve': float(np.trapz(risks, coverages))
        }
    
    def _bootstrap_uncertainty_estimation(self, predictions: np.ndarray,
                                        ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Estimate uncertainty using bootstrap sampling."""
        bootstrap_results = {}
        
        n_samples = len(predictions)
        bootstrap_predictions = []
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap sample indices
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
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
            'bootstrap_samples_used': self.bootstrap_samples
        })
        
        # If ground truth available, bootstrap performance metrics
        if ground_truth is not None:
            bootstrap_performance = self._bootstrap_performance_uncertainty(
                predictions, ground_truth
            )
            bootstrap_results['performance_uncertainty'] = bootstrap_performance
        
        return bootstrap_results
    
    def _bootstrap_performance_uncertainty(self, predictions: np.ndarray,
                                         ground_truth: np.ndarray) -> Dict[str, Any]:
        """Bootstrap uncertainty estimation for performance metrics."""
        n_samples = len(predictions)
        bootstrap_performances = []
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            bootstrap_pred = predictions[bootstrap_indices]
            bootstrap_gt = ground_truth[bootstrap_indices]
            
            # Calculate performance metric
            if self._is_classification_predictions(predictions):
                if predictions.ndim == 1:
                    pred_binary = (bootstrap_pred > 0.5).astype(int)
                    performance = accuracy_score(bootstrap_gt.astype(int), pred_binary)
                else:
                    pred_classes = np.argmax(bootstrap_pred, axis=1)
                    performance = accuracy_score(bootstrap_gt.astype(int), pred_classes)
            else:
                # Regression: use negative MAE
                mae = np.mean(np.abs(bootstrap_pred - bootstrap_gt))
                performance = -mae
            
            bootstrap_performances.append(performance)
        
        bootstrap_performances = np.array(bootstrap_performances)
        
        # Performance uncertainty estimates
        performance_mean = np.mean(bootstrap_performances)
        performance_std = np.std(bootstrap_performances, ddof=1)
        
        # Confidence intervals for performance
        alpha = 1 - self.confidence_level
        perf_ci_lower = np.percentile(bootstrap_performances, (alpha/2) * 100)
        perf_ci_upper = np.percentile(bootstrap_performances, (1 - alpha/2) * 100)
        
        return {
            'performance_mean': float(performance_mean),
            'performance_std': float(performance_std),
            'performance_ci_lower': float(perf_ci_lower),
            'performance_ci_upper': float(perf_ci_upper),
            'performance_samples': bootstrap_performances.tolist()
        }
    
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
                    'epistemic_variance': epistemic_var.tolist(),
                    'epistemic_std': epistemic_std.tolist(),
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
                    'ensemble_variance': ensemble_var.tolist(),
                    'ensemble_std': ensemble_std.tolist(),
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
                'aleatoric_variance': pred_variance.tolist(),
                'aleatoric_std': pred_std.tolist(),
                'mean_aleatoric_uncertainty': float(np.mean(pred_std)),
                'max_aleatoric_uncertainty': float(np.max(pred_std))
            }
        
        # Gaussian likelihood parameters
        if 'mean_predictions' in model_outputs and 'variance_predictions' in model_outputs:
            mean_preds = np.array(model_outputs['mean_predictions'])
            var_preds = np.array(model_outputs['variance_predictions'])
            std_preds = np.sqrt(var_preds)
            
            aleatoric_results['gaussian_likelihood'] = {
                'mean_predictions': mean_preds.tolist(),
                'variance_predictions': var_preds.tolist(),
                'std_predictions': std_preds.tolist(),
                'mean_aleatoric_uncertainty': float(np.mean(std_preds))
            }
        
        return aleatoric_results
    
    def _analyze_ensemble_uncertainty(self, ensemble_predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze uncertainty from ensemble predictions."""
        ensemble_preds = np.array(ensemble_predictions)
        
        if ensemble_preds.ndim < 2:
            return {'error': 'Ensemble predictions must be 2D array [n_models, n_samples]'}
        
        # Calculate ensemble statistics
        ensemble_mean = np.mean(ensemble_preds, axis=0)
        ensemble_var = np.var(ensemble_preds, axis=0)
        ensemble_std = np.std(ensemble_preds, axis=0)
        
        # Ensemble agreement/disagreement
        if self._is_classification_predictions(ensemble_preds):
            # For classification, look at prediction disagreement
            if ensemble_preds.ndim == 2 and ensemble_preds.shape[1] > 1:
                # Multi-class: calculate entropy of ensemble mean
                ensemble_entropy = -np.sum(ensemble_mean * np.log(ensemble_mean + 1e-12), axis=1)
                
                # Mutual information (knowledge uncertainty)
                individual_entropies = []
                for i in range(ensemble_preds.shape[0]):
                    pred = ensemble_preds[i]
                    entropy = -np.sum(pred * np.log(pred + 1e-12), axis=1)
                    individual_entropies.append(entropy)
                
                mean_individual_entropy = np.mean(individual_entropies, axis=0)
                mutual_information = mean_individual_entropy - ensemble_entropy
                
                ensemble_analysis = {
                    'ensemble_entropy': ensemble_entropy.tolist(),
                    'mutual_information': mutual_information.tolist(),
                    'mean_ensemble_entropy': float(np.mean(ensemble_entropy)),
                    'mean_mutual_information': float(np.mean(mutual_information))
                }
            else:
                # Binary classification
                ensemble_agreement = np.std(ensemble_preds, axis=0)
                ensemble_analysis = {
                    'ensemble_agreement': ensemble_agreement.tolist(),
                    'mean_disagreement': float(np.mean(ensemble_agreement))
                }
        else:
            # Regression ensemble analysis
            ensemble_analysis = {
                'ensemble_variance': ensemble_var.tolist(),
                'ensemble_std': ensemble_std.tolist(),
                'mean_ensemble_uncertainty': float(np.mean(ensemble_std)),
                'coefficient_of_variation': (ensemble_std / (np.abs(ensemble_mean) + 1e-12)).tolist()
            }
        
        ensemble_analysis.update({
            'ensemble_mean': ensemble_mean.tolist(),
            'n_models': int(ensemble_preds.shape[0]),
            'prediction_diversity': float(np.mean(ensemble_std))
        })
        
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
        if 'predictive_entropy' in pred_uncertainty:
            entropy_stats = pred_uncertainty['predictive_entropy']
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
    
    def _calculate_selective_performance(self, predictions: np.ndarray,
                                       ground_truth: np.ndarray,
                                       uncertainty: np.ndarray) -> Dict[str, Any]:
        """Calculate performance metrics with selective prediction based on uncertainty."""
        selective_metrics = {}
        
        # Sort by uncertainty (low to high)
        sorted_indices = np.argsort(uncertainty)
        
        # Calculate performance at different coverage levels
        coverage_levels = np.arange(0.1, 1.01, 0.1)
        
        performance_at_coverage = []
        
        for coverage in coverage_levels:
            n_samples_to_keep = int(coverage * len(predictions))
            indices_to_keep = sorted_indices[:n_samples_to_keep]
            
            pred_subset = predictions[indices_to_keep]
            gt_subset = ground_truth[indices_to_keep]
            
            if self._is_classification_predictions(predictions):
                if predictions.ndim == 1:
                    # Binary classification
                    pred_binary = (pred_subset > 0.5).astype(int)
                    accuracy = accuracy_score(gt_subset.astype(int), pred_binary)
                    performance = accuracy
                else:
                    # Multi-class classification
                    pred_classes = np.argmax(pred_subset, axis=1)
                    accuracy = accuracy_score(gt_subset.astype(int), pred_classes)
                    performance = accuracy
            else:
                # Regression: use negative MAE as performance (higher is better)
                mae = np.mean(np.abs(pred_subset - gt_subset))
                performance = -mae
            
            performance_at_coverage.append({
                'coverage': float(coverage),
                'performance': float(performance),
                'n_samples': int(n_samples_to_keep)
            })
        
        selective_metrics['selective_performance'] = performance_at_coverage
        
        # Area under the risk-coverage curve
        coverages = [item['coverage'] for item in performance_at_coverage]
        performances = [item['performance'] for item in performance_at_coverage]
        
        # Calculate AUC using trapezoidal rule
        auc_selective = np.trapz(performances, coverages)
        selective_metrics['auc_selective_performance'] = float(auc_selective)
        
        return selective_metrics
    
    def _calculate_risk_coverage_curve(self, predictions: np.ndarray,
                                     ground_truth: np.ndarray,
                                     uncertainty: np.ndarray) -> Dict[str, Any]:
        """Calculate risk-coverage curve for uncertainty evaluation."""
        # Sort by uncertainty (high to low for risk-coverage)
        sorted_indices = np.argsort(-uncertainty)
        
        coverages = []
        risks = []
        
        for i in range(1, len(predictions) + 1):
            # Keep top i most certain predictions
            indices_to_keep = sorted_indices[len(predictions) - i:]
            
            pred_subset = predictions[indices_to_keep]
            gt_subset = ground_truth[indices_to_keep]
            
            coverage = i / len(predictions)
            
            if self._is_classification_predictions(predictions):
                if predictions.ndim == 1:
                    pred_binary = (pred_subset > 0.5).astype(int)
                    error_rate = 1 - accuracy_score(gt_subset.astype(int), pred_binary)
                    risk = error_rate
                else:
                    pred_classes = np.argmax(pred_subset, axis=1)
                    error_rate = 1 - accuracy_score(gt_subset.astype(int), pred_classes)
                    risk = error_rate
            else:
                # Regression: use MAE as risk
                mae = np.mean(np.abs(pred_subset - gt_subset))
                risk = mae
            
            coverages.append(coverage)
            risks.append(risk)
        
        return {
            'coverages': coverages,
            'risks': risks,
            'area_under_curve': float(np.trapz(risks, coverages))
        }
    
    def _bootstrap_uncertainty_estimation(self, predictions: np.ndarray,
                                        ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Estimate uncertainty using bootstrap sampling."""
        bootstrap_results = {}
        
        n_samples = len(predictions)
        bootstrap_predictions = []
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap sample indices
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
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
            'bootstrap_samples_used': self.bootstrap_samples
        })
        
        # If ground truth available, bootstrap performance metrics
        if ground_truth is not None:
            bootstrap_performance = self._bootstrap_performance_uncertainty(
                predictions, ground_truth
            )
            bootstrap_results['performance_uncertainty'] = bootstrap_performance
        
        return bootstrap_results
    
    def _bootstrap_performance_uncertainty(self, predictions: np.ndarray,
                                         ground_truth: np.ndarray) -> Dict[str, Any]:
        """Bootstrap uncertainty estimation for performance metrics."""
        n_samples = len(predictions)
        bootstrap_performances = []
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            bootstrap_pred = predictions[bootstrap_indices]
            bootstrap_gt = ground_truth[bootstrap_indices]
            
            # Calculate performance metric
            if self._is_classification_predictions(predictions):
                if predictions.ndim == 1:
                    pred_binary = (bootstrap_pred > 0.5).astype(int)
                    performance = accuracy_score(bootstrap_gt.astype(int), pred_binary)
                else:
                    pred_classes = np.argmax(bootstrap_pred, axis=1)
                    performance = accuracy_score(bootstrap_gt.astype(int), pred_classes)
            else:
                # Regression: use negative MAE
                mae = np.mean(np.abs(bootstrap_pred - bootstrap_gt))
                performance = -mae
            
            bootstrap_performances.append(performance)
        
        bootstrap_performances = np.array(bootstrap_performances)
        
        # Performance uncertainty estimates
        performance_mean = np.mean(bootstrap_performances)
        performance_std = np.std(bootstrap_performances, ddof=1)
        
        # Confidence intervals for performance
        alpha = 1 - self.confidence_level
        perf_ci_lower = np.percentile(bootstrap_performances, (alpha/2) * 100)
        perf_ci_upper = np.percentile(bootstrap_performances, (1 - alpha/2) * 100)
        
        return {
            'performance_mean': float(performance_mean),
            'performance_std': float(performance_std),
            'performance_ci_lower': float(perf_ci_lower),
            'performance_ci_upper': float(perf_ci_upper),
            'performance_samples': bootstrap_performances.tolist()
        }
    
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
                    'epistemic_variance': epistemic_var.tolist(),
                    'epistemic_std': epistemic_std.tolist(),
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
                    'ensemble_variance': ensemble_var.tolist(),
                    'ensemble_std': ensemble_std.tolist(),
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
                'aleatoric_variance': pred_variance.tolist(),
                'aleatoric_std': pred_std.tolist(),
                'mean_aleatoric_uncertainty': float(np.mean(pred_std)),
                'max_aleatoric_uncertainty': float(np.max(pred_std))
            }
        
        # Gaussian likelihood parameters
        if 'mean_predictions' in model_outputs and 'variance_predictions' in model_outputs:
            mean_preds = np.array(model_outputs['mean_predictions'])
            var_preds = np.array(model_outputs['variance_predictions'])
            std_preds = np.sqrt(var_preds)
            
            aleatoric_results['gaussian_likelihood'] = {
                'mean_predictions': mean_preds.tolist(),
                'variance_predictions': var_preds.tolist(),
                'std_predictions': std_preds.tolist(),
                'mean_aleatoric_uncertainty': float(np.mean(std_preds))
            }
        
        return aleatoric_results
    
    def _analyze_ensemble_uncertainty(self, ensemble_predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze uncertainty from ensemble predictions."""
        ensemble_preds = np.array(ensemble_predictions)
        
        if ensemble_preds.ndim < 2:
            return {'error': 'Ensemble predictions must be 2D array [n_models, n_samples]'}
        
        # Calculate ensemble statistics
        ensemble_mean = np.mean(ensemble_preds, axis=0)
        ensemble_var = np.var(ensemble_preds, axis=0)
        ensemble_std = np.std(ensemble_preds, axis=0)
        
        # Ensemble agreement/disagreement
        if self._is_classification_predictions(ensemble_preds):
            # For classification, look at prediction disagreement
            if ensemble_preds.ndim == 2 and ensemble_preds.shape[1] > 1:
                # Multi-class: calculate entropy of ensemble mean
                ensemble_entropy = -np.sum(ensemble_mean * np.log(ensemble_mean + 1e-12), axis=1)
                
                # Mutual information (knowledge uncertainty)
                individual_entropies = []
                for i in range(ensemble_preds.shape[0]):
                    pred = ensemble_preds[i]
                    entropy = -np.sum(pred * np.log(pred + 1e-12), axis=1)
                    individual_entropies.append(entropy)
                
                mean_individual_entropy = np.mean(individual_entropies, axis=0)
                mutual_information = mean_individual_entropy - ensemble_entropy
                
                ensemble_analysis = {
                    'ensemble_entropy': ensemble_entropy.tolist(),
                    'mutual_information': mutual_information.tolist(),
                    'mean_ensemble_entropy': float(np.mean(ensemble_entropy)),
                    'mean_mutual_information': float(np.mean(mutual_information))
                }
            else:
                # Binary classification
                ensemble_agreement = np.std(ensemble_preds, axis=0)
                ensemble_analysis = {
                    'ensemble_agreement': ensemble_agreement.tolist(),
                    'mean_disagreement': float(np.mean(ensemble_agreement))
                }
        else:
            # Regression ensemble analysis
            ensemble_analysis = {
                'ensemble_variance': ensemble_var.tolist(),
                'ensemble_std': ensemble_std.tolist(),
                'mean_ensemble_uncertainty': float(np.mean(ensemble_std)),
                'coefficient_of_variation': ensemble_std / (np.abs(ensemble_mean) + 1e-12)
            }
        
        ensemble_analysis.update({
            'ensemble_mean': ensemble_mean.tolist(),
            'n_models': int(ensemble_preds.shape[0]),
            'prediction_diversity': float(np.mean(ensemble_std))
        })
        
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
        if 'predictive_entropy' in pred_uncertainty:
            entropy_stats = pred_uncertainty['predictive_entropy']
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
        """
Uncertainty Quantification Module for Medical AI Systems
Implements comprehensive uncertainty estimation and validation
"""

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