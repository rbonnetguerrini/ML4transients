import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from scipy.stats import spearmanr
from typing import Dict, Tuple, Optional
import h5py
from pathlib import Path

class EvaluationMetrics:
    """Compute and summarize binary classification evaluation metrics.

    Supports:
      - Basic metrics: accuracy, precision, recall, F1, specificity
      - Confusion matrix statistics
      - ROC + PR curve (with probability scores)
      - SNR-based metric breakdowns
      - Internal caching to prevent redundant curve recomputation
    """

    def __init__(self, predictions: np.ndarray, labels: np.ndarray,
                 probabilities: Optional[np.ndarray] = None,
                 snr_values: Optional[np.ndarray] = None,
                 uncertainties: Optional[np.ndarray] = None,
                 source_ids: Optional[np.ndarray] = None,
                 dataset_loader = None):
        """
        Parameters
        ----------
        predictions : np.ndarray
            Binary predictions (0/1) of shape (n_samples,)
        labels : np.ndarray
            True binary labels (0/1) of shape (n_samples,)
        probabilities : np.ndarray, optional
            Probabilistic scores (same length). Required for ROC/PR.
        snr_values : np.ndarray, optional
            SNR values for each sample. Required for SNR-based metrics.
        uncertainties : np.ndarray, optional
            Uncertainty values for each prediction (e.g., ensemble std, model disagreement).
            Useful for UQ analysis, particularly with ensemble/coteaching models.
        source_ids : np.ndarray, optional
            diaSourceId values for each sample. Required for lightcurve-level metrics.
        dataset_loader : DatasetLoader, optional
            Dataset loader instance. Required for lightcurve-level metrics.
        """
        # Store inputs
        self.predictions = predictions
        self.labels = labels
        self.probabilities = probabilities
        self.snr_values = snr_values
        self.uncertainties = uncertainties
        self.source_ids = source_ids
        self.dataset_loader = dataset_loader

        # Internal caches to avoid repeated expensive computations
        self._roc_cache: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
        self._pr_cache: Optional[Tuple[np.ndarray, np.ndarray, float]] = None

        # Compute basic metrics eagerly (cheap, used everywhere)
        self._compute_basic_metrics()

    def _compute_basic_metrics(self) -> None:
        """Compute basic scalar metrics + confusion matrix (single pass)."""
        self.accuracy = accuracy_score(self.labels, self.predictions)
        self.precision = precision_score(self.labels, self.predictions, zero_division=0)
        self.recall = recall_score(self.labels, self.predictions, zero_division=0)
        self.f1 = f1_score(self.labels, self.predictions, zero_division=0)
        self.confusion_mat = confusion_matrix(self.labels, self.predictions)
        self.specificity = self._compute_specificity()

    def _compute_specificity(self) -> float:
        """Return specificity = TN / (TN + FP), guarded for zero division."""
        tn, fp, fn, tp = self.confusion_mat.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def get_confusion_matrix_stats(self) -> Dict[str, float]:
        """Return derived confusion matrix statistics.

        Returns
        -------
        dict
            Keys include counts (tp/tn/fp/fn) and rates (TPR/TNR/FPR/FNR/PPV/NPV).
        """
        tn, fp, fn, tp = self.confusion_mat.ravel()
        total = tn + fp + fn + tp
        return {
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'true_negative_rate': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0.0,
            'total_samples': total
        }

    def get_roc_data(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (fpr, tpr, auc) with internal caching."""
        if self.probabilities is None:
            raise ValueError("Probabilities needed for ROC curve")
        if self._roc_cache is None:
            fpr, tpr, _ = roc_curve(self.labels, self.probabilities)
            roc_auc = auc(fpr, tpr)
            self._roc_cache = (fpr, tpr, roc_auc)
        return self._roc_cache

    def get_pr_data(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (precision, recall, average_precision) with caching."""
        if self.probabilities is None:
            raise ValueError("Probabilities needed for PR curve")
        if self._pr_cache is None:
            precision, recall, _ = precision_recall_curve(self.labels, self.probabilities)
            pr_auc = average_precision_score(self.labels, self.probabilities)
            self._pr_cache = (precision, recall, pr_auc)
        return self._pr_cache

    def summary(self) -> Dict[str, float]:
        """Return consolidated metrics dictionary (uses cached curves if present)."""
        summary = {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1,
            'specificity': self.specificity
        }
        if self.probabilities is not None:
            _, _, roc_auc = self.get_roc_data()
            _, _, pr_auc = self.get_pr_data()
            summary.update({'roc_auc': roc_auc, 'pr_auc': pr_auc})
        return summary

    def get_snr_based_metrics(self, snr_threshold: float = 5.0) -> Dict[str, Dict[str, float]]:
        """Compute metrics separately for low and high SNR samples.
        
        Parameters
        ----------
        snr_threshold : float
            SNR threshold to separate low and high SNR samples (default: 5.0)
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary with 'low_snr' and 'high_snr' keys, each containing metrics
        """
        if self.snr_values is None:
            raise ValueError("SNR values needed for SNR-based metrics")
            
        # Create absolute SNR values for threshold comparison
        abs_snr = np.abs(self.snr_values)
        
        # Split into low and high SNR
        low_snr_mask = abs_snr < snr_threshold
        high_snr_mask = abs_snr >= snr_threshold
        
        results = {}
        
        # Compute metrics for low SNR
        if np.any(low_snr_mask):
            low_predictions = self.predictions[low_snr_mask]
            low_labels = self.labels[low_snr_mask]
            low_probabilities = self.probabilities[low_snr_mask] if self.probabilities is not None else None
            
            low_metrics = EvaluationMetrics(low_predictions, low_labels, low_probabilities)
            results['low_snr'] = {
                **low_metrics.summary(),
                'n_samples': np.sum(low_snr_mask)
            }
        else:
            results['low_snr'] = self._empty_metrics_dict()
            
        # Compute metrics for high SNR
        if np.any(high_snr_mask):
            high_predictions = self.predictions[high_snr_mask]
            high_labels = self.labels[high_snr_mask]
            high_probabilities = self.probabilities[high_snr_mask] if self.probabilities is not None else None
            
            high_metrics = EvaluationMetrics(high_predictions, high_labels, high_probabilities)
            results['high_snr'] = {
                **high_metrics.summary(),
                'n_samples': np.sum(high_snr_mask)
            }
        else:
            results['high_snr'] = self._empty_metrics_dict()
            
        return results
    
    def _empty_metrics_dict(self) -> Dict[str, float]:
        """Return empty metrics dictionary for cases with no samples."""
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'specificity': 0.0,
            'n_samples': 0
        }
    
    def get_uncertainty_by_confusion_categories(self, snr_threshold: float = 5.0) -> Dict[str, Dict[str, float]]:
        """Compute uncertainty statistics by confusion matrix categories (TP/FP/TN/FN).
        
        Provides breakdown for:
        - All samples
        - Low SNR samples (|SNR| < threshold)
        - High SNR samples (|SNR| >= threshold)
        
        Parameters
        ----------
        snr_threshold : float
            SNR threshold to separate low and high SNR samples (default: 5.0)
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dictionary with structure:
            {
                'all': {'tp': ..., 'fp': ..., 'tn': ..., 'fn': ..., 'mean': ...},
                'low_snr': {...},
                'high_snr': {...}
            }
            Each inner dict contains mean uncertainty for each category.
        """
        if self.uncertainties is None:
            raise ValueError("Uncertainties needed for UQ analysis")
        
        results = {}
        
        # Create confusion matrix masks
        tp_mask = (self.predictions == 1) & (self.labels == 1)
        fp_mask = (self.predictions == 1) & (self.labels == 0)
        tn_mask = (self.predictions == 0) & (self.labels == 0)
        fn_mask = (self.predictions == 0) & (self.labels == 1)
        
        # Compute for all samples
        results['all'] = {
            'tp': np.mean(self.uncertainties[tp_mask]) if np.any(tp_mask) else np.nan,
            'fp': np.mean(self.uncertainties[fp_mask]) if np.any(fp_mask) else np.nan,
            'tn': np.mean(self.uncertainties[tn_mask]) if np.any(tn_mask) else np.nan,
            'fn': np.mean(self.uncertainties[fn_mask]) if np.any(fn_mask) else np.nan,
            'mean': np.mean(self.uncertainties),
            'n_tp': np.sum(tp_mask),
            'n_fp': np.sum(fp_mask),
            'n_tn': np.sum(tn_mask),
            'n_fn': np.sum(fn_mask)
        }
        
        # Compute for low and high SNR if available
        if self.snr_values is not None:
            abs_snr = np.abs(self.snr_values)
            low_snr_mask = abs_snr < snr_threshold
            high_snr_mask = abs_snr >= snr_threshold
            
            # Low SNR
            results['low_snr'] = {
                'tp': np.mean(self.uncertainties[tp_mask & low_snr_mask]) if np.any(tp_mask & low_snr_mask) else np.nan,
                'fp': np.mean(self.uncertainties[fp_mask & low_snr_mask]) if np.any(fp_mask & low_snr_mask) else np.nan,
                'tn': np.mean(self.uncertainties[tn_mask & low_snr_mask]) if np.any(tn_mask & low_snr_mask) else np.nan,
                'fn': np.mean(self.uncertainties[fn_mask & low_snr_mask]) if np.any(fn_mask & low_snr_mask) else np.nan,
                'mean': np.mean(self.uncertainties[low_snr_mask]) if np.any(low_snr_mask) else np.nan,
                'n_tp': np.sum(tp_mask & low_snr_mask),
                'n_fp': np.sum(fp_mask & low_snr_mask),
                'n_tn': np.sum(tn_mask & low_snr_mask),
                'n_fn': np.sum(fn_mask & low_snr_mask)
            }
            
            # High SNR
            results['high_snr'] = {
                'tp': np.mean(self.uncertainties[tp_mask & high_snr_mask]) if np.any(tp_mask & high_snr_mask) else np.nan,
                'fp': np.mean(self.uncertainties[fp_mask & high_snr_mask]) if np.any(fp_mask & high_snr_mask) else np.nan,
                'tn': np.mean(self.uncertainties[tn_mask & high_snr_mask]) if np.any(tn_mask & high_snr_mask) else np.nan,
                'fn': np.mean(self.uncertainties[fn_mask & high_snr_mask]) if np.any(fn_mask & high_snr_mask) else np.nan,
                'mean': np.mean(self.uncertainties[high_snr_mask]) if np.any(high_snr_mask) else np.nan,
                'n_tp': np.sum(tp_mask & high_snr_mask),
                'n_fp': np.sum(fp_mask & high_snr_mask),
                'n_tn': np.sum(tn_mask & high_snr_mask),
                'n_fn': np.sum(fn_mask & high_snr_mask)
            }
        else:
            # If no SNR data, return empty dicts
            results['low_snr'] = {
                'tp': np.nan, 'fp': np.nan, 'tn': np.nan, 'fn': np.nan,
                'mean': np.nan, 'n_tp': 0, 'n_fp': 0, 'n_tn': 0, 'n_fn': 0
            }
            results['high_snr'] = {
                'tp': np.nan, 'fp': np.nan, 'tn': np.nan, 'fn': np.nan,
                'mean': np.nan, 'n_tp': 0, 'n_fp': 0, 'n_tn': 0, 'n_fn': 0
            }
        
        return results
    
    def get_snr_uncertainty_correlation(self) -> Dict[str, float]:
        """Compute Spearman correlation between SNR and uncertainty.
        
        Useful for understanding if model uncertainty increases with lower SNR.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with:
            - 'correlation': Spearman correlation coefficient
            - 'p_value': Statistical significance p-value
            - 'n_samples': Number of samples used
            
        Notes
        -----
        Uses absolute SNR values for correlation.
        Positive correlation means higher SNR -> higher uncertainty (unexpected).
        Negative correlation means lower SNR -> higher uncertainty (expected).
        """
        if self.uncertainties is None:
            raise ValueError("Uncertainties needed for correlation analysis")
        if self.snr_values is None:
            raise ValueError("SNR values needed for correlation analysis")
        
        # Use absolute SNR for correlation
        abs_snr = np.abs(self.snr_values)
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(abs_snr) & np.isfinite(self.uncertainties)
        valid_snr = abs_snr[valid_mask]
        valid_uncertainty = self.uncertainties[valid_mask]
        
        if len(valid_snr) < 3:
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_samples': len(valid_snr)
            }
        
        # Compute Spearman correlation
        correlation, p_value = spearmanr(valid_snr, valid_uncertainty)
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'n_samples': len(valid_snr)
        }
    
    def get_snr_uncertainty_correlation_by_confusion(self) -> Dict[str, Dict[str, float]]:
        """Compute Spearman correlation between SNR and uncertainty for each confusion category.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary with correlation stats for each category (TP, TN, FP, FN):
            - 'correlation': Spearman correlation coefficient
            - 'p_value': Statistical significance p-value  
            - 'n_samples': Number of samples used
        """
        if self.uncertainties is None or self.snr_values is None:
            return {}
        
        abs_snr = np.abs(self.snr_values)
        
        # Create confusion category masks
        tp_mask = (self.predictions == 1) & (self.labels == 1)
        tn_mask = (self.predictions == 0) & (self.labels == 0)
        fp_mask = (self.predictions == 1) & (self.labels == 0)
        fn_mask = (self.predictions == 0) & (self.labels == 1)
        
        results = {}
        
        for cat_name, mask in [('TP', tp_mask), ('TN', tn_mask), ('FP', fp_mask), ('FN', fn_mask)]:
            # Filter valid values for this category
            cat_snr = abs_snr[mask]
            cat_uncertainty = self.uncertainties[mask]
            
            valid_mask = np.isfinite(cat_snr) & np.isfinite(cat_uncertainty)
            valid_snr = cat_snr[valid_mask]
            valid_uncertainty = cat_uncertainty[valid_mask]
            
            if len(valid_snr) >= 3:  # Need at least 3 points for correlation
                correlation, p_value = spearmanr(valid_snr, valid_uncertainty)
                results[cat_name] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'n_samples': len(valid_snr)
                }
            else:
                results[cat_name] = {
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'n_samples': len(valid_snr)
                }
        
        return results
    
    def get_nll(self) -> float:
        """Compute Negative Log-Likelihood (NLL) for probabilistic predictions.
        
        Lower NLL indicates better calibration. NLL penalizes confident wrong predictions.
        
        Returns
        -------
        float
            Mean negative log-likelihood across all samples
        """
        if self.probabilities is None:
            return np.nan
        
        # Clip probabilities to avoid log(0) - use larger epsilon for numerical stability
        eps = 1e-7
        probs_clipped = np.clip(self.probabilities, eps, 1 - eps)
        
        # NLL for binary classification: -[y*log(p) + (1-y)*log(1-p)]
        # Compute each term separately for better numerical stability
        log_probs = np.log(probs_clipped)
        log_one_minus_probs = np.log(1 - probs_clipped)
        
        nll = -(self.labels * log_probs + (1 - self.labels) * log_one_minus_probs)
        
        # Check for any NaN or inf values and handle them
        valid_mask = np.isfinite(nll)
        if not np.all(valid_mask):
            # If we have invalid values, compute mean only on valid ones
            if np.any(valid_mask):
                return float(np.mean(nll[valid_mask]))
            else:
                return np.nan
        
        return float(np.mean(nll))
    
    def get_brier_score(self) -> float:
        """Compute Brier Score for probabilistic predictions.
        
        Brier Score measures the mean squared difference between predicted probabilities
        and actual outcomes. Lower is better (perfect score = 0).
        
        Returns
        -------
        float
            Brier score (0 = perfect, 1 = worst)
        """
        if self.probabilities is None:
            return np.nan
        
        # Brier score: mean((p - y)^2)
        brier = np.mean((self.probabilities - self.labels) ** 2)
        
        return float(brier)
    
    def get_extended_uq_summary(self, snr_threshold: float = 5.0) -> Dict:
        """Get comprehensive UQ summary including confusion categories and correlation.
        
        Parameters
        ----------
        snr_threshold : float
            SNR threshold for low/high separation
            
        Returns
        -------
        Dict
            Complete UQ analysis including:
            - 'overall': Overall UQ statistics (mean, std, median)
            - 'calibration': NLL and Brier score (if probabilities available)
            - 'by_correctness': UQ for correct vs incorrect predictions
            - 'by_confusion_category': UQ for TP, TN, FP, FN
            - 'by_snr': UQ breakdown by low/high SNR (if SNR available)
            - 'snr_uq_correlation': Overall SNR-UQ correlation (if SNR available)
            - 'snr_uq_correlation_by_category': Per-category SNR-UQ correlation (if SNR available)
        """
        if self.uncertainties is None:
            return {'available': False}
        
        summary = {'available': True}
        
        # Overall statistics
        summary['overall'] = {
            'mean': float(np.mean(self.uncertainties)),
            'std': float(np.std(self.uncertainties)),
            'median': float(np.median(self.uncertainties))
        }
        
        # Calibration metrics (if probabilities available)
        if self.probabilities is not None:
            summary['calibration'] = {
                'nll': self.get_nll(),
                'brier_score': self.get_brier_score()
            }
        
        # By correctness
        correct_mask = self.predictions == self.labels
        incorrect_mask = ~correct_mask
        
        summary['by_correctness'] = {}
        if np.any(correct_mask):
            summary['by_correctness']['correct'] = {
                'mean': float(np.mean(self.uncertainties[correct_mask])),
                'std': float(np.std(self.uncertainties[correct_mask])),
                'count': int(np.sum(correct_mask))
            }
        if np.any(incorrect_mask):
            summary['by_correctness']['incorrect'] = {
                'mean': float(np.mean(self.uncertainties[incorrect_mask])),
                'std': float(np.std(self.uncertainties[incorrect_mask])),
                'count': int(np.sum(incorrect_mask))
            }
        
        # By confusion category (all samples)
        tp_mask = (self.predictions == 1) & (self.labels == 1)
        tn_mask = (self.predictions == 0) & (self.labels == 0)
        fp_mask = (self.predictions == 1) & (self.labels == 0)
        fn_mask = (self.predictions == 0) & (self.labels == 1)
        
        summary['by_confusion_category'] = {}
        for cat_name, mask in [('TP', tp_mask), ('TN', tn_mask), ('FP', fp_mask), ('FN', fn_mask)]:
            if np.any(mask):
                summary['by_confusion_category'][cat_name] = {
                    'mean': float(np.mean(self.uncertainties[mask])),
                    'std': float(np.std(self.uncertainties[mask])),
                    'median': float(np.median(self.uncertainties[mask])),
                    'count': int(np.sum(mask))
                }
        
        # SNR-based analysis if available
        if self.snr_values is not None:
            # Get overall SNR-UQ correlation
            summary['snr_uq_correlation'] = self.get_snr_uncertainty_correlation()
            
            # Get correlation by confusion category
            summary['snr_uq_correlation_by_category'] = self.get_snr_uncertainty_correlation_by_confusion()
            
            # Get SNR-based breakdown with confusion categories
            abs_snr = np.abs(self.snr_values)
            low_snr_mask = abs_snr < snr_threshold
            high_snr_mask = abs_snr >= snr_threshold
            
            summary['by_snr'] = {}
            
            # Low SNR
            if np.any(low_snr_mask):
                summary['by_snr']['low_snr'] = {
                    'mean': float(np.mean(self.uncertainties[low_snr_mask])),
                    'std': float(np.std(self.uncertainties[low_snr_mask])),
                    'median': float(np.median(self.uncertainties[low_snr_mask])),
                    'count': int(np.sum(low_snr_mask)),
                    'by_confusion': {}
                }
                
                # Low SNR by confusion category
                for cat_name, cat_mask in [('TP', tp_mask), ('TN', tn_mask), ('FP', fp_mask), ('FN', fn_mask)]:
                    combined_mask = low_snr_mask & cat_mask
                    if np.any(combined_mask):
                        summary['by_snr']['low_snr']['by_confusion'][cat_name] = {
                            'mean': float(np.mean(self.uncertainties[combined_mask])),
                            'std': float(np.std(self.uncertainties[combined_mask])),
                            'count': int(np.sum(combined_mask))
                        }
            
            # High SNR
            if np.any(high_snr_mask):
                summary['by_snr']['high_snr'] = {
                    'mean': float(np.mean(self.uncertainties[high_snr_mask])),
                    'std': float(np.std(self.uncertainties[high_snr_mask])),
                    'median': float(np.median(self.uncertainties[high_snr_mask])),
                    'count': int(np.sum(high_snr_mask)),
                    'by_confusion': {}
                }
                
                # High SNR by confusion category
                for cat_name, cat_mask in [('TP', tp_mask), ('TN', tn_mask), ('FP', fp_mask), ('FN', fn_mask)]:
                    combined_mask = high_snr_mask & cat_mask
                    if np.any(combined_mask):
                        summary['by_snr']['high_snr']['by_confusion'][cat_name] = {
                            'mean': float(np.mean(self.uncertainties[combined_mask])),
                            'std': float(np.std(self.uncertainties[combined_mask])),
                            'count': int(np.sum(combined_mask))
                        }
        
        return summary
    
    def get_lightcurve_metrics(self, other_transients: np.ndarray = None, lc_threshold: float = 0.9) -> Dict[str, Dict[str, float]]:
        """Compute overall bogus/real ratio per lightcurve, with configurable threshold for classification.
        
        Parameters
        ----------
        other_transients : np.ndarray (optional)
            Array indicating transient type (0=SN, 1=other_transient, -1=missing)
            If provided: uses ground truth (labels==1 are real transients)
            If None: assumes all lightcurves are bogus (ground truth = 0)
        lc_threshold : float (default: 0.9)
            Threshold for classifying a lightcurve as real or bogus.
            - If >=lc_threshold of sources predicted as real -> classified as real
            - If >=lc_threshold of sources predicted as bogus -> classified as bogus
            - Otherwise -> classified as unclassified
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary with lightcurve-level metrics:
            - 'overall': Overall lightcurve statistics
            - 'correct_lcs': Correctly classified lightcurves
            - 'incorrect_lcs': Incorrectly classified lightcurves
            - 'by_true_class': Breakdown by true class (if other_transients provided)
        """
        if self.source_ids is None or self.dataset_loader is None:
            raise ValueError("source_ids and dataset_loader are required for lightcurve metrics")
        
        # Get lightcurve loader to map sources to objects
        lightcurve_path = None
        for data_path in self.dataset_loader.data_paths:
            lc_path = Path(data_path) / 'lightcurves'
            if lc_path.exists():
                lightcurve_path = lc_path
                break
        
        if lightcurve_path is None:
            raise ValueError("Could not find lightcurves directory in dataset paths")
        
        from ML4transients.data_access.data_loaders import LightCurveLoader
        lc_loader = LightCurveLoader(lightcurve_path)
        
        print(f"Computing lightcurve-level metrics for {len(self.source_ids)} sources...")
        
        # Map source IDs to object IDs using diasource_index
        print("  Mapping sources to lightcurves...")
        
        # Ensure diasource_index has proper dtypes
        diasource_index = lc_loader.diasource_index.copy()
        if diasource_index['diaObjectId'].dtype == 'object':
            diasource_index['diaObjectId'] = pd.to_numeric(diasource_index['diaObjectId'], errors='coerce').astype(np.int64)
        
        # Create mapping from source ID to object ID
        source_to_object = {}
        for source_id in self.source_ids:
            source_id_int = int(source_id)
            if source_id_int in diasource_index.index:
                obj_id = diasource_index.loc[source_id_int, 'diaObjectId']
                if pd.notna(obj_id):
                    source_to_object[source_id_int] = int(obj_id)
        
        print(f"  Mapped {len(source_to_object)}/{len(self.source_ids)} sources to objects")
        
        # Group predictions by lightcurve
        from collections import defaultdict
        lc_data = defaultdict(lambda: {'predictions': [], 'labels': [], 'indices': []})
        
        for idx, source_id in enumerate(self.source_ids):
            object_id = source_to_object.get(int(source_id))
            if object_id is not None:
                lc_data[object_id]['predictions'].append(self.predictions[idx])
                lc_data[object_id]['labels'].append(self.labels[idx])
                lc_data[object_id]['indices'].append(idx)
        
        print(f"  Grouped into {len(lc_data)} unique lightcurves")
        
        # Compute lightcurve-level classifications (configurable threshold)
        # Classification logic:
        # - If >=lc_threshold of sources predicted as real (1) -> lightcurve classified as real (1)
        # - If >=lc_threshold of sources predicted as bogus (0) -> lightcurve classified as bogus (0)
        # - Otherwise (neither reaches threshold) -> lightcurve UNCLASSIFIED (-1)
        lc_classifications = {}
        lc_true_labels = {}
        lc_source_accuracy = {}  # Track average correct prediction percentage per lightcurve
        lc_transient_percentages = {}  # Track percentage of sources predicted as transient (real) per lightcurve
        
        print(f"  Using LC classification threshold: {lc_threshold:.1%}")
        
        for obj_id, data in lc_data.items():
            preds = np.array(data['predictions'])
            labels = np.array(data['labels'])
            
            # Lightcurve classification with three categories
            real_ratio = np.mean(preds == 1)
            bogus_ratio = np.mean(preds == 0)
            
            if real_ratio >= lc_threshold:
                lc_pred = 1  # Real
            elif bogus_ratio >= lc_threshold:
                lc_pred = 0  # Bogus
            else:
                lc_pred = -1  # Unclassified
            
            lc_classifications[obj_id] = lc_pred
            
            # Calculate percentage of correctly classified sources within this lightcurve
            correct_preds = np.sum(preds == labels)
            lc_source_accuracy[obj_id] = correct_preds / len(preds)
            
            # Track percentage of sources predicted as transient (for distribution plots)
            lc_transient_percentages[obj_id] = real_ratio
            
            # Ground truth for this lightcurve
            if other_transients is not None:
                # Use majority vote of true labels (should be consistent within lightcurve)
                lc_true = int(np.round(np.mean(labels)))
            else:
                # No ground truth: assume all lightcurves are bogus
                lc_true = 0
            
            lc_true_labels[obj_id] = lc_true
        
        # Compute lightcurve-level metrics
        lc_preds = np.array([lc_classifications[obj_id] for obj_id in lc_data.keys()])
        lc_trues = np.array([lc_true_labels[obj_id] for obj_id in lc_data.keys()])
        
        n_total_lcs = len(lc_preds)
        n_unclassified = np.sum(lc_preds == -1)
        
        # For accuracy, only consider classified lightcurves (exclude unclassified)
        classified_mask = lc_preds != -1
        n_classified = np.sum(classified_mask)
        
        if n_classified > 0:
            n_correct_lcs = np.sum(lc_preds[classified_mask] == lc_trues[classified_mask])
            n_incorrect_lcs = np.sum(lc_preds[classified_mask] != lc_trues[classified_mask])
            accuracy = float(n_correct_lcs / n_classified)
        else:
            n_correct_lcs = 0
            n_incorrect_lcs = 0
            accuracy = 0.0
        
        # Overall statistics
        results = {
            'threshold': lc_threshold,  # Store the threshold used for classification
            'overall': {
                'n_lightcurves': n_total_lcs,
                'n_classified': n_classified,
                'n_unclassified': n_unclassified,
                'n_correct': n_correct_lcs,
                'n_incorrect': n_incorrect_lcs,
                'accuracy': accuracy,  # Accuracy only on classified LCs
                'n_classified_as_real': int(np.sum(lc_preds == 1)),
                'n_classified_as_bogus': int(np.sum(lc_preds == 0))
            }
        }
        
        # Store distribution data for visualization
        all_obj_ids = list(lc_data.keys())
        results['distributions'] = {
            'all_lcs': {
                'object_ids': all_obj_ids,
                'transient_percentages': [lc_transient_percentages[obj_id] for obj_id in all_obj_ids],
                'true_labels': [lc_true_labels[obj_id] for obj_id in all_obj_ids]
            }
        }
        
        # Confusion matrix for lightcurves (only for classified LCs)
        # Unclassified predictions (-1) are not counted in confusion matrix
        lc_tp = np.sum((lc_preds == 1) & (lc_trues == 1))
        lc_tn = np.sum((lc_preds == 0) & (lc_trues == 0))
        lc_fp = np.sum((lc_preds == 1) & (lc_trues == 0))
        lc_fn = np.sum((lc_preds == 0) & (lc_trues == 1))
        
        # Count unclassified by true class
        lc_unclassified_real = np.sum((lc_preds == -1) & (lc_trues == 1))
        lc_unclassified_bogus = np.sum((lc_preds == -1) & (lc_trues == 0))
        
        results['confusion'] = {
            'true_positive': int(lc_tp),
            'true_negative': int(lc_tn),
            'false_positive': int(lc_fp),
            'false_negative': int(lc_fn),
            'unclassified_real': int(lc_unclassified_real),
            'unclassified_bogus': int(lc_unclassified_bogus)
        }
        
        # Breakdown by true class (if ground truth available)
        if other_transients is not None:
            n_real_lcs = np.sum(lc_trues == 1)
            n_bogus_lcs = np.sum(lc_trues == 0)
            
            # Calculate average source-level accuracy for each lightcurve type
            real_lc_ids = [obj_id for obj_id, true_label in lc_true_labels.items() if true_label == 1]
            bogus_lc_ids = [obj_id for obj_id, true_label in lc_true_labels.items() if true_label == 0]
            
            avg_real_source_acc = np.mean([lc_source_accuracy[obj_id] for obj_id in real_lc_ids]) if real_lc_ids else 0.0
            avg_bogus_source_acc = np.mean([lc_source_accuracy[obj_id] for obj_id in bogus_lc_ids]) if bogus_lc_ids else 0.0
            
            results['by_true_class'] = {
                'real_transients': {
                    'n_lightcurves': int(n_real_lcs),
                    'n_correct': int(lc_tp),
                    'n_incorrect': int(lc_fn),
                    'n_unclassified': int(lc_unclassified_real),
                    'recall': float(lc_tp / n_real_lcs) if n_real_lcs > 0 else 0.0,
                    'avg_source_accuracy': float(avg_real_source_acc)  # Average % of correctly classified sources within real transient LCs
                },
                'bogus': {
                    'n_lightcurves': int(n_bogus_lcs),
                    'n_correct': int(lc_tn),
                    'n_incorrect': int(lc_fp),
                    'n_unclassified': int(lc_unclassified_bogus),
                    'specificity': float(lc_tn / n_bogus_lcs) if n_bogus_lcs > 0 else 0.0,
                    'avg_source_accuracy': float(avg_bogus_source_acc)  # Average % of correctly classified sources within bogus LCs
                }
            }
        else:
            # No ground truth mode: all should be classified as bogus
            # Calculate average source-level accuracy for all lightcurves
            avg_source_acc = np.mean(list(lc_source_accuracy.values())) if lc_source_accuracy else 0.0
            
            results['by_true_class'] = {
                'bogus': {
                    'n_lightcurves': int(n_total_lcs),
                    'n_correct': int(lc_tn),
                    'n_incorrect': int(lc_fp),
                    'specificity': float(lc_tn / n_total_lcs) if n_total_lcs > 0 else 0.0,
                    'avg_source_accuracy': float(avg_source_acc)
                }
            }
        
        print(f"  Lightcurve-level accuracy: {results['overall']['accuracy']:.2%}")
        print(f"  Correct: {n_correct_lcs}/{n_total_lcs} lightcurves")
        
        return results

    def get_transient_type_metrics(self, other_transients: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute metrics separately for SN and other_transient subclasses within is_injection=1.
        
        Splits the positive class (is_injection=1) into two subclasses:
        - SN (supernovae): is_injection=1 && other_transients=0
        - Other Transients: is_injection=1 && other_transients=1
        
        Parameters
        ----------
        other_transients : np.ndarray
            Array indicating transient type (0=SN, 1=other_transient, -1=missing)
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary with 'sn' and 'other_transients' keys, each containing:
            - n_samples: number of samples
            - n_correct: correctly classified samples
            - n_incorrect: incorrectly classified samples (FN)
            - recall: recall (sensitivity) for this subclass
            - mean_probability: mean prediction probability (if available)
            - mean_uncertainty: mean uncertainty (if available)
        """
        results = {}
        
        # Get injection samples (positive class)
        injection_mask = self.labels == 1
        
        # SN subclass: is_injection=1 && other_transients=0
        sn_mask = injection_mask & (other_transients == 0)
        if np.any(sn_mask):
            sn_predictions = self.predictions[sn_mask]
            n_sn = np.sum(sn_mask)
            n_sn_correct = np.sum(sn_predictions == 1)  # True Positives for SN
            n_sn_incorrect = np.sum(sn_predictions == 0)  # False Negatives for SN
            
            results['sn'] = {
                'n_samples': int(n_sn),
                'n_correct': int(n_sn_correct),
                'n_incorrect': int(n_sn_incorrect),
                'recall': float(n_sn_correct / n_sn) if n_sn > 0 else 0.0
            }
            
            # Add probability stats if available
            if self.probabilities is not None:
                sn_probs = self.probabilities[sn_mask]
                results['sn']['mean_probability'] = float(np.mean(sn_probs))
                results['sn']['std_probability'] = float(np.std(sn_probs))
            
            # Add uncertainty stats if available
            if self.uncertainties is not None:
                sn_uncertainties = self.uncertainties[sn_mask]
                results['sn']['mean_uncertainty'] = float(np.mean(sn_uncertainties))
                results['sn']['std_uncertainty'] = float(np.std(sn_uncertainties))
        else:
            results['sn'] = self._empty_transient_metrics_dict()
        
        # Other Transients subclass: is_injection=1 && other_transients=1
        other_mask = injection_mask & (other_transients == 1)
        if np.any(other_mask):
            other_predictions = self.predictions[other_mask]
            n_other = np.sum(other_mask)
            n_other_correct = np.sum(other_predictions == 1)  # True Positives for other transients
            n_other_incorrect = np.sum(other_predictions == 0)  # False Negatives for other transients
            
            results['other_transients'] = {
                'n_samples': int(n_other),
                'n_correct': int(n_other_correct),
                'n_incorrect': int(n_other_incorrect),
                'recall': float(n_other_correct / n_other) if n_other > 0 else 0.0
            }
            
            # Add probability stats if available
            if self.probabilities is not None:
                other_probs = self.probabilities[other_mask]
                results['other_transients']['mean_probability'] = float(np.mean(other_probs))
                results['other_transients']['std_probability'] = float(np.std(other_probs))
            
            # Add uncertainty stats if available
            if self.uncertainties is not None:
                other_uncertainties = self.uncertainties[other_mask]
                results['other_transients']['mean_uncertainty'] = float(np.mean(other_uncertainties))
                results['other_transients']['std_uncertainty'] = float(np.std(other_uncertainties))
        else:
            results['other_transients'] = self._empty_transient_metrics_dict()
        
        # Add bogus (real negatives) stats for context
        bogus_mask = self.labels == 0
        if np.any(bogus_mask):
            bogus_predictions = self.predictions[bogus_mask]
            n_bogus = np.sum(bogus_mask)
            n_bogus_correct = np.sum(bogus_predictions == 0)  # True Negatives
            n_bogus_fp = np.sum(bogus_predictions == 1)  # False Positives
            
            results['bogus'] = {
                'n_samples': int(n_bogus),
                'n_correct': int(n_bogus_correct),
                'n_fp': int(n_bogus_fp),
                'specificity': float(n_bogus_correct / n_bogus) if n_bogus > 0 else 0.0,
                'fp_rate': float(n_bogus_fp / n_bogus) if n_bogus > 0 else 0.0
            }
            
            # Add probability stats if available
            if self.probabilities is not None:
                bogus_probs = self.probabilities[bogus_mask]
                results['bogus']['mean_probability'] = float(np.mean(bogus_probs))
                results['bogus']['std_probability'] = float(np.std(bogus_probs))
            
            # Add uncertainty stats if available
            if self.uncertainties is not None:
                bogus_uncertainties = self.uncertainties[bogus_mask]
                results['bogus']['mean_uncertainty'] = float(np.mean(bogus_uncertainties))
                results['bogus']['std_uncertainty'] = float(np.std(bogus_uncertainties))
        else:
            results['bogus'] = {'n_samples': 0, 'n_correct': 0, 'n_fp': 0, 
                               'specificity': 0.0, 'fp_rate': 0.0}
        
        return results
    
    def _empty_transient_metrics_dict(self) -> Dict[str, float]:
        """Return empty transient type metrics dictionary."""
        return {
            'n_samples': 0,
            'n_correct': 0,
            'n_incorrect': 0,
            'recall': 0.0
        }

    def get_sn_phase_analysis(self, other_transients: np.ndarray, 
                              dist_max_bright: np.ndarray) -> Dict[str, any]:
        """Analyze SN detection performance as a function of distance from maximum brightness.
        
        Computes:
        - Spearman correlation between uncertainty and dist_max_bright
        - Accuracy as a function of dist_max_bright (binned analysis)
        - Detection rates at different phases (pre-max, near-max, post-max)
        
        Parameters
        ----------
        other_transients : np.ndarray
            Array indicating transient type (0=SN, 1=other_transient)
        dist_max_bright : np.ndarray
            Distance from maximum brightness in days (positive = before max, negative = after max)
            
        Returns
        -------
        Dict
            Dictionary containing:
            - 'uq_correlation': Spearman correlation between UQ and dist_max_bright
            - 'accuracy_by_phase': Accuracy binned by dist_max_bright
            - 'phase_breakdown': Detection stats for pre-max, near-max, post-max
        """
        results = {'available': False}
        
        # Get SN mask
        sn_mask = other_transients == 0
        if not np.any(sn_mask):
            return results
        
        # Get valid dist_max_bright mask (not NaN and is SN)
        valid_mask = sn_mask & ~np.isnan(dist_max_bright)
        if np.sum(valid_mask) < 10:
            print(f"Warning: Only {np.sum(valid_mask)} valid SN samples with dist_max_bright")
            return results
        
        results['available'] = True
        results['n_samples'] = int(np.sum(valid_mask))
        
        # Extract data for valid SN samples
        sn_dist = dist_max_bright[valid_mask]
        sn_preds = self.predictions[valid_mask]
        sn_labels = self.labels[valid_mask]
        sn_correct = (sn_preds == sn_labels).astype(int)
        
        # 1. Correlation between UQ and dist_max_bright (if UQ available)
        if self.uncertainties is not None:
            sn_uncertainties = self.uncertainties[valid_mask]
            valid_uq_mask = ~np.isnan(sn_uncertainties)
            
            if np.sum(valid_uq_mask) >= 3:
                correlation, p_value = spearmanr(
                    np.abs(sn_dist[valid_uq_mask]), 
                    sn_uncertainties[valid_uq_mask]
                )
                results['uq_dist_correlation'] = {
                    'spearman_rho': float(correlation),
                    'n_samples': int(np.sum(valid_uq_mask))
                }
        
        # 2. Correlation between accuracy and dist_max_bright
        if len(sn_correct) >= 3:
            correlation, p_value = spearmanr(np.abs(sn_dist), sn_correct)
            results['accuracy_dist_correlation'] = {
                'spearman_rho': float(correlation),
                'n_samples': int(len(sn_correct))
            }
        
        # 3. Phase-based breakdown (pre-max, near-max, post-max)
        # Define phases: pre-max (>5 days before), near-max (-5 to +5), post-max (>5 days after)
        pre_max_mask = sn_dist > 5
        near_max_mask = (sn_dist >= -5) & (sn_dist <= 5)
        post_max_mask = sn_dist < -5
        
        phase_breakdown = {}
        
        for phase_name, phase_mask in [('pre_max', pre_max_mask), 
                                        ('near_max', near_max_mask), 
                                        ('post_max', post_max_mask)]:
            n_phase = np.sum(phase_mask)
            if n_phase > 0:
                phase_correct = np.sum(sn_correct[phase_mask])
                phase_accuracy = phase_correct / n_phase
                
                phase_data = {
                    'n_samples': int(n_phase),
                    'n_correct': int(phase_correct),
                    'n_incorrect': int(n_phase - phase_correct),
                    'accuracy': float(phase_accuracy)
                }
                
                # Add UQ stats if available
                if self.uncertainties is not None:
                    phase_uq = self.uncertainties[valid_mask][phase_mask]
                    valid_phase_uq = phase_uq[~np.isnan(phase_uq)]
                    if len(valid_phase_uq) > 0:
                        phase_data['mean_uncertainty'] = float(np.mean(valid_phase_uq))
                        phase_data['std_uncertainty'] = float(np.std(valid_phase_uq))
                
                # Add mean dist_max_bright for this phase
                phase_data['mean_dist'] = float(np.mean(sn_dist[phase_mask]))
                
                phase_breakdown[phase_name] = phase_data
            else:
                phase_breakdown[phase_name] = {
                    'n_samples': 0, 'n_correct': 0, 'n_incorrect': 0, 'accuracy': 0.0
                }
        
        results['phase_breakdown'] = phase_breakdown
        
        # 4. Binned accuracy analysis (for visualization)
        # Create bins based on absolute distance from max
        abs_dist = np.abs(sn_dist)
        bins = [0, 5, 10, 20, 50, np.inf]
        bin_labels = ['0-5', '5-10', '10-20', '20-50', '>50']
        
        binned_accuracy = {}
        for i, label in enumerate(bin_labels):
            bin_mask = (abs_dist >= bins[i]) & (abs_dist < bins[i+1])
            n_bin = np.sum(bin_mask)
            if n_bin > 0:
                bin_correct = np.sum(sn_correct[bin_mask])
                binned_accuracy[label] = {
                    'n_samples': int(n_bin),
                    'accuracy': float(bin_correct / n_bin),
                    'mean_dist': float(np.mean(abs_dist[bin_mask]))
                }
                
                # Add UQ if available
                if self.uncertainties is not None:
                    bin_uq = self.uncertainties[valid_mask][bin_mask]
                    valid_bin_uq = bin_uq[~np.isnan(bin_uq)]
                    if len(valid_bin_uq) > 0:
                        binned_accuracy[label]['mean_uncertainty'] = float(np.mean(valid_bin_uq))
        
        results['binned_by_distance'] = binned_accuracy
        
        return results
    


def load_inference_metrics(inference_file: Path) -> EvaluationMetrics:
    """Load HDF5 file and build EvaluationMetrics (probabilities and SNR optional).

    Parameters
    ----------
    inference_file : Path
        Path to file containing 'predictions' and 'labels' datasets.

    Returns
    -------
    EvaluationMetrics
        Initialized instance.
    """
    with h5py.File(inference_file, 'r') as f:
        predictions = f['predictions'][:]
        labels = f['labels'][:]
        probabilities = f['probabilities'][:] if 'probabilities' in f else None
        snr_values = f['snr_values'][:] if 'snr_values' in f else None
    return EvaluationMetrics(predictions, labels, probabilities, snr_values)