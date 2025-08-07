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
    """Class for computing standard ML evaluation metrics."""
    
    def __init__(self, predictions: np.ndarray, labels: np.ndarray, 
                 probabilities: Optional[np.ndarray] = None):
        """
        Initialize with predictions and labels.
        
        Args:
            predictions: Binary predictions (0/1)
            labels: True binary labels (0/1) 
            probabilities: Prediction probabilities (optional, for ROC/PR curves)
        """
        self.predictions = predictions
        self.labels = labels
        self.probabilities = probabilities
        
        # Compute basic metrics
        self._compute_basic_metrics()
        
    def _compute_basic_metrics(self):
        """Compute basic classification metrics.
        
        Calculates accuracy, precision, recall, F1-score, confusion matrix,
        and specificity from predictions and labels.
        """
        self.accuracy = accuracy_score(self.labels, self.predictions)
        self.precision = precision_score(self.labels, self.predictions, zero_division=0)
        self.recall = recall_score(self.labels, self.predictions, zero_division=0)
        self.f1 = f1_score(self.labels, self.predictions, zero_division=0)
        self.confusion_mat = confusion_matrix(self.labels, self.predictions)

        self.specificity = self._compute_specificity()
        
    def _compute_specificity(self) -> float:
        """Compute specificity (true negative rate).
        
        Returns
        -------
        float
            Specificity = TN / (TN + FP), or 0 if no negative samples
        """
        tn, fp, fn, tp = self.confusion_mat.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
        
    def get_confusion_matrix_stats(self) -> Dict:
        """Get detailed confusion matrix statistics.
        
        Returns
        -------
        dict
            Dictionary containing:
            - true_positive, true_negative, false_positive, false_negative: int
                Raw counts from confusion matrix
            - true_positive_rate, true_negative_rate: float
                Sensitivity and specificity
            - positive_predictive_value, negative_predictive_value: float
                Precision and negative predictive value
            - false_positive_rate, false_negative_rate: float
                Error rates
            - total_samples: int
                Total number of samples
        """
        tn, fp, fn, tp = self.confusion_mat.ravel()
        total = tn + fp + fn + tp
        
        return {
            'true_positive': tp,
            'true_negative': tn, 
            'false_positive': fp,
            'false_negative': fn,
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Sensitivity/Recall
            'true_negative_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,  # Specificity
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Precision
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'total_samples': total
        }
    
    def get_roc_data(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute ROC curve data.
        
        Returns
        -------
        fpr : np.ndarray
            False positive rates
        tpr : np.ndarray
            True positive rates
        roc_auc : float
            Area under the ROC curve
            
        Raises
        ------
        ValueError
            If probabilities are not available
        """
        if self.probabilities is None:
            raise ValueError("Probabilities needed for ROC curve")
        
        fpr, tpr, thresholds = roc_curve(self.labels, self.probabilities)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    
    def get_pr_data(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute Precision-Recall curve data.
        
        Returns
        -------
        precision : np.ndarray
            Precision values
        recall : np.ndarray
            Recall values
        pr_auc : float
            Area under the Precision-Recall curve
            
        Raises
        ------
        ValueError
            If probabilities are not available
        """
        if self.probabilities is None:
            raise ValueError("Probabilities needed for PR curve")
        
        precision, recall, thresholds = precision_recall_curve(self.labels, self.probabilities)
        pr_auc = average_precision_score(self.labels, self.probabilities)
        return precision, recall, pr_auc
    
    def summary(self) -> Dict:
        """Get summary of all metrics.
        
        Returns
        -------
        dict
            Dictionary containing all computed metrics:
            - accuracy, precision, recall, f1_score, specificity: float
                Basic classification metrics
            - roc_auc, pr_auc: float (if probabilities available)
                Area under curve metrics
        """
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
            summary.update({
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            })
        
        return summary

def load_inference_metrics(inference_file: Path) -> EvaluationMetrics:
    """Load inference results and create EvaluationMetrics object.
    
    Parameters
    ----------
    inference_file : Path
        Path to HDF5 file containing inference results
        
    Returns
    -------
    EvaluationMetrics
        Initialized metrics object with loaded predictions and labels
    """
    with h5py.File(inference_file, 'r') as f:
        predictions = f['predictions'][:]
        labels = f['labels'][:]
        
        # Check if probabilities are saved (for future enhancement)
        probabilities = None
        if 'probabilities' in f:
            probabilities = f['probabilities'][:]
    
    return EvaluationMetrics(predictions, labels, probabilities)