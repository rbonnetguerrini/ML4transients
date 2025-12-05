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
                 uncertainties: Optional[np.ndarray] = None):
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
        """
        # Store inputs
        self.predictions = predictions
        self.labels = labels
        self.probabilities = probabilities
        self.snr_values = snr_values
        self.uncertainties = uncertainties

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