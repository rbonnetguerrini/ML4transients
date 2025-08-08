import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from typing import Dict, Tuple, Optional
import h5py
from pathlib import Path

class EvaluationMetrics:
    """Compute and summarize binary classification evaluation metrics.

    Supports:
      - Basic metrics: accuracy, precision, recall, F1, specificity
      - Confusion matrix statistics
      - ROC + PR curve (with probability scores)
      - Internal caching to prevent redundant curve recomputation
    """

    def __init__(self, predictions: np.ndarray, labels: np.ndarray,
                 probabilities: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        predictions : np.ndarray
            Binary predictions (0/1) of shape (n_samples,)
        labels : np.ndarray
            True binary labels (0/1) of shape (n_samples,)
        probabilities : np.ndarray, optional
            Probabilistic scores (same length). Required for ROC/PR.
        """
        # Store inputs
        self.predictions = predictions
        self.labels = labels
        self.probabilities = probabilities

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

def load_inference_metrics(inference_file: Path) -> EvaluationMetrics:
    """Load HDF5 file and build EvaluationMetrics (probabilities optional).

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
    return EvaluationMetrics(predictions, labels, probabilities)