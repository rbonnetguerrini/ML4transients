import numpy as np
import pandas as pd
import torch
from bokeh.plotting import figure, save, output_file
from bokeh.models import (
    HoverTool, ColumnDataSource, ColorBar, LinearColorMapper,
    Title, Div, CustomJS, Tabs, TabPanel, CategoricalColorMapper, Spacer
)
from bokeh.layouts import column, row, gridplot
from bokeh.palettes import Spectral11, Category10, Viridis256
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.io import curdoc, output_file, save
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import colorcet as cc

from .metrics import EvaluationMetrics

class BokehEvaluationPlots:
    """Class for creating standard evaluation visualizations."""
    
    def __init__(self, width: int = 800, height: int = 550):
        """Initialize with plot dimensions."""
        curdoc().theme = 'dark_minimal'
        self.width = width
        self.height = height
    
    def plot_confusion_matrix(self, metrics: EvaluationMetrics, 
                            title: str = "Confusion Matrix") -> figure:
        """Create interactive confusion matrix plot."""
        stats = metrics.get_confusion_matrix_stats()
        
        # Calculate percentages
        total = stats['total_samples']
        tp_pct = f"TP: {(stats['true_positive']/total*100):.1f}%"
        tn_pct = f"TN: {(stats['true_negative']/total*100):.1f}%"
        fp_pct = f"FP: {(stats['false_positive']/total*100):.1f}%"
        fn_pct = f"FN: {(stats['false_negative']/total*100):.1f}%"
        
        # Additional metrics
        sensitivity_pct = f"Sensitivity:\n{(stats['true_positive_rate']*100):.1f}%"
        specificity_pct = f"Specificity:\n{(stats['true_negative_rate']*100):.1f}%"
        precision_pct = f"Precision:\n{(stats['positive_predictive_value']*100):.1f}%"
        npv_pct = f"NPV:\n{(stats['negative_predictive_value']*100):.1f}%"
        accuracy_pct = f"Accuracy:\n{(metrics.accuracy*100):.1f}%"
        
        # Data for visualization
        data = ColumnDataSource(data=dict(
            labels=['True Positive', 'False Negative', 'False Positive', 'True Negative',
                    'Specificity', 'Sensitivity','Precision', 'NPV', 'Accuracy'],
            counts=[stats['true_positive'], stats['false_negative'], 
                   stats['false_positive'], stats['true_negative'],
                   None, None, None, None, None],
            percentages=[tp_pct, fn_pct, fp_pct, tn_pct, 
                        specificity_pct, sensitivity_pct, precision_pct, npv_pct, accuracy_pct],
            x=[0, 1, 0, 1, 2, 2, 0, 1, 2],
            y=[2, 2, 1, 1, 1, 2, 0, 0, 0],
            colors=['#FFC300', '#b3b6b7', '#FF5733', '#C70039', 
                   '#ECEDED', '#ECEDED', '#ECEDED', '#ECEDED', '#ECEDED']
        ))
        
        p = figure(title=title, x_range=(-0.5, 2.5), y_range=(-0.5, 2.5),
                  width=400, height=400, tools="")
        
        p.rect(x='x', y='y', width=1, height=1, source=data,
               color='colors', alpha=0.9, line_color="colors", line_width=2)
        
        p.text(x='x', y='y', text='percentages', source=data,
               text_font_size="14pt", text_align="center", text_baseline="middle")
        
        hover = HoverTool(tooltips=[('Class', '@labels'), ('Count', '@counts')])
        p.add_tools(hover)
        
        # Clean styling
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.axis.visible = False
        
        return p
    
    def plot_roc_curve(self, metrics: EvaluationMetrics,
                      title: str = "ROC Curve") -> figure:
        """Create ROC curve plot with threshold markers at 0.25, 0.5, and 0.75."""
        if metrics.probabilities is None:
            raise ValueError("Probabilities needed for ROC curve")
        
        fpr, tpr, roc_auc = metrics.get_roc_data()
        
        source = ColumnDataSource(data=dict(fpr=fpr, tpr=tpr))
        
        p = figure(title=f"{title} (AUC = {roc_auc:.3f})",
                  x_axis_label="False Positive Rate",
                  y_axis_label="True Positive Rate",
                  width=self.width//2, height=self.height//2)
        
        p.line('fpr', 'tpr', source=source, line_width=2, color='#2F7ED8')
        p.line([0, 1], [0, 1], line_dash="dashed", line_color="gray", alpha=0.5)
        
        # Add threshold markers at 0.25, 0.5, and 0.75
        from sklearn.metrics import roc_curve
        fpr_full, tpr_full, thresholds = roc_curve(metrics.labels, metrics.probabilities)
        
        # Find points closest to desired thresholds
        threshold_points = []
        threshold_values = [0.25, 0.5, 0.75]
        colors = ['#C70039', '#FFE66D', '#FFC300']  # Red, Yellow, Cyan
        
        for thresh, color in zip(threshold_values, colors):
            # Find the index of the threshold closest to our target
            idx = np.argmin(np.abs(thresholds - thresh))
            fpr_point = fpr_full[idx]
            tpr_point = tpr_full[idx]
            actual_thresh = thresholds[idx]
            
            # Add marker
            p.circle([fpr_point], [tpr_point], size=10, color=color, alpha=0.8,
                    legend_label=f'Threshold {thresh}')
            
            # Add text annotation
            from bokeh.models import Label
            label = Label(x=fpr_point, y=tpr_point, 
                         text=f'{thresh}', text_font_size='9pt',
                         x_offset=8, y_offset=5, text_color=color)
            p.add_layout(label)
        
        hover = HoverTool(tooltips=[("FPR", "@fpr{0.000}"), ("TPR", "@tpr{0.000}")])
        p.add_tools(hover)
        
        p.legend.location = "bottom_right"
        p.legend.click_policy = "hide"
        
        return p
    
    def plot_precision_recall_curve(self, metrics: EvaluationMetrics,
                                   title: str = "Precision-Recall Curve") -> figure:
        """Create Precision-Recall curve plot."""
        if metrics.probabilities is None:
            raise ValueError("Probabilities needed for PR curve")
        
        precision, recall, pr_auc = metrics.get_pr_data()
        
        source = ColumnDataSource(data=dict(precision=precision, recall=recall))
        
        p = figure(title=f"{title} (AUC = {pr_auc:.3f})",
                  x_axis_label="Recall",
                  y_axis_label="Precision",
                  width=self.width//2, height=self.height//2)
        
        p.line('recall', 'precision', source=source, line_width=2, color='#FF7F0E')
        
        hover = HoverTool(tooltips=[("Recall", "@recall{0.000}"), ("Precision", "@precision{0.000}")])
        p.add_tools(hover)
        
        return p
    
    def plot_prediction_distribution(self, predictions: np.ndarray, labels: np.ndarray,
                                   probabilities: Optional[np.ndarray] = None,
                                   title: str = "Prediction Distribution") -> figure:
        """Plot distribution of predictions by true class."""
        if probabilities is None:
            # Use predictions as probabilities for visualization
            probabilities = predictions.astype(float)
        
        # Create histograms
        bins = np.linspace(0, 1, 11)
        all_hist, _ = np.histogram(probabilities, bins=bins)
        pos_hist, _ = np.histogram(probabilities[labels == 1], bins=bins)
        
        source_all = ColumnDataSource(data=dict(
            x=bins[:-1], top=all_hist, count=all_hist
        ))
        source_positive = ColumnDataSource(data=dict(
            x=bins[:-1], top=pos_hist, count=pos_hist
        ))
        
        p = figure(title=title,
                  x_axis_label="Prediction Probability",
                  y_axis_label="Count",
                  width=self.width//2, height=self.height//2)
        
        p.vbar(x='x', top='top', width=0.08, color='lightgray', 
               source=source_all, legend_label='All data', alpha=0.7)
        p.vbar(x='x', top='top', width=0.08, color='#5b75cd',
               source=source_positive, legend_label='True positives')
        
        hover_all = HoverTool(renderers=[p.renderers[0]],
                             tooltips=[("Probability", "@x{0.0}"), ("Count", "@count")])
        hover_pos = HoverTool(renderers=[p.renderers[1]],
                             tooltips=[("Probability", "@x{0.0}"), ("True Positive Count", "@count")])
        p.add_tools(hover_all, hover_pos)
        
        p.legend.click_policy = "hide"
        p.legend.location = "top_left"
        
        return p
    
    def create_metrics_summary_div(self, metrics: EvaluationMetrics) -> Div:
        """Create a summary div with key metrics."""
        summary = metrics.summary()
        
        html_content = f"""
        <div style="padding: 10px; background-color: #2F2F2F; border-radius: 5px;">
            <h3 style="color: white; margin-top: 0;">Model Performance Summary</h3>
            <table style="color: white; width: 100%;">
                <tr><td>Accuracy:</td><td>{summary['accuracy']:.3f}</td></tr>
                <tr><td>Precision:</td><td>{summary['precision']:.3f}</td></tr>
                <tr><td>Recall:</td><td>{summary['recall']:.3f}</td></tr>
                <tr><td>F1-Score:</td><td>{summary['f1_score']:.3f}</td></tr>
                <tr><td>Specificity:</td><td>{summary['specificity']:.3f}</td></tr>
        """
        
        if 'roc_auc' in summary:
            html_content += f"""
                <tr><td>ROC AUC:</td><td>{summary['roc_auc']:.3f}</td></tr>
                <tr><td>PR AUC:</td><td>{summary['pr_auc']:.3f}</td></tr>
            """
        
        html_content += """
            </table>
        </div>
        """
        
        return Div(text=html_content, width=300, height=200)
    
    def create_uncertainty_metrics_div(self, metrics: EvaluationMetrics, snr_threshold: float = 5.0) -> Div:
        """Create a summary div with uncertainty quantification metrics.
        
        Provides quantitative UQ analysis for ensemble/coteaching models including
        breakdown by confusion matrix categories and SNR-UQ correlation.
        
        Parameters
        ----------
        metrics : EvaluationMetrics
            Evaluation metrics object with uncertainties
        snr_threshold : float
            SNR threshold to separate low and high SNR samples (default: 5.0)
            
        Returns
        -------
        Div
            Bokeh Div element with UQ statistics
        """
        if not hasattr(metrics, 'uncertainties') or metrics.uncertainties is None:
            html_content = """
            <div style="padding: 10px; background-color: #2F2F2F; border-radius: 5px;">
                <h3 style="color: white; margin-top: 0;">Uncertainty Quantification</h3>
                <p style="color: white;">Not available (requires ensemble or coteaching model)</p>
            </div>
            """
            return Div(text=html_content, width=400, height=200)
        
        # Get extended UQ summary
        uq_summary = metrics.get_extended_uq_summary(snr_threshold)
        
        # Start HTML content
        html_content = f"""
        <div style="padding: 10px; background-color: #2F2F2F; border-radius: 5px;">
            <h3 style="color: white; margin-top: 0;">Uncertainty Quantification</h3>
            <table style="color: white; width: 100%; font-size: 11px;">
        """
        
        # Add calibration metrics if available
        if 'calibration' in uq_summary:
            cal = uq_summary['calibration']
            html_content += f"""
                <tr><td colspan="2"><b>Calibration Metrics:</b></td></tr>
                <tr><td>NLL:</td><td>{cal['nll']:.4f}</td></tr>
                <tr><td>Brier Score:</td><td>{cal['brier_score']:.4f}</td></tr>
                <tr><td colspan="2" style="padding-top: 8px;"><b>Overall Statistics:</b></td></tr>
            """
        else:
            html_content += """
                <tr><td colspan="2"><b>Overall Statistics:</b></td></tr>
            """
        
        html_content += f"""
                <tr><td>Mean:</td><td>{uq_summary['overall']['mean']:.4f}</td></tr>
                <tr><td>Median:</td><td>{uq_summary['overall']['median']:.4f}</td></tr>
                <tr><td>Std Dev:</td><td>{uq_summary['overall']['std']:.4f}</td></tr>
                <tr><td colspan="2" style="padding-top: 8px;"><b>All Samples (By Type):</b></td></tr>
        """
        
        # Add confusion category stats
        for cat in ['TP', 'TN', 'FP', 'FN']:
            if cat in uq_summary['by_confusion_category']:
                stats = uq_summary['by_confusion_category'][cat]
                html_content += f"""<tr><td>{cat}:</td><td>{stats['mean']:.4f} (n={stats['count']})</td></tr>\n        """
        
        # Add overall SNR-UQ correlation if available
        if 'snr_uq_correlation' in uq_summary:
            corr = uq_summary['snr_uq_correlation']
            corr_val = corr['correlation']
            p_val = corr['p_value']
            
            html_content += f"""
                <tr><td colspan="2" style="padding-top: 8px;"><b>SNR-UQ Correlation (Overall):</b></td></tr>
                <tr><td>Spearman ρ:</td><td>{corr_val:.4f} (p={p_val:.4e})</td></tr>
            """
            
            # Add per-category correlations
            if 'snr_uq_correlation_by_category' in uq_summary:
                html_content += """<tr><td colspan="2" style="padding-top: 4px; font-size: 10px;"><em>By category:</em></td></tr>\n        """
                for cat in ['TP', 'TN', 'FP', 'FN']:
                    if cat in uq_summary['snr_uq_correlation_by_category']:
                        cat_corr = uq_summary['snr_uq_correlation_by_category'][cat]
                        if not np.isnan(cat_corr['correlation']):
                            html_content += f"""<tr><td style="padding-left: 10px; font-size: 10px;">{cat}:</td><td style="font-size: 10px;">ρ={cat_corr['correlation']:.3f} (n={cat_corr['n_samples']})</td></tr>\n        """
                    
        html_content += """
            </table>
        </div>
        """
        
        # Adjust height based on content: add extra height if calibration metrics present
        div_height = 400 if 'calibration' in uq_summary else 350
        
        return Div(text=html_content, width=400, height=div_height)
    
    def create_snr_metrics_divs(self, metrics: EvaluationMetrics, snr_threshold: float = 5.0) -> Tuple[Div, Div]:
        """Create separate metric boxes for low and high SNR samples.
        
        Includes uncertainty metrics if available (for ensemble/coteaching models).
        
        Parameters
        ----------
        metrics : EvaluationMetrics
            Evaluation metrics object with SNR values
        snr_threshold : float
            SNR threshold to separate low and high SNR samples
            
        Returns
        -------
        Tuple[Div, Div]
            (low_snr_div, high_snr_div) - Bokeh Div elements
        """
        if metrics.snr_values is None:
            # Return empty divs if SNR not available
            empty_div = Div(text="<div style='padding: 10px; color: white;'>SNR data not available</div>", 
                           width=300, height=200)
            return empty_div, empty_div
            
        snr_metrics = metrics.get_snr_based_metrics(snr_threshold)
        
        # Check if uncertainty data is available
        has_uncertainty = (hasattr(metrics, 'uncertainties') and 
                          metrics.uncertainties is not None and 
                          len(metrics.uncertainties) > 0)
        
        # Low SNR metrics div
        low_snr = snr_metrics['low_snr']
        low_html = f"""
        <div style="padding: 10px; background-color: #3B2F2F; border-radius: 5px; border-left: 4px solid #C70039;">
            <h3 style="color: white; margin-top: 0;">Low SNR Performance (|SNR| < {snr_threshold})</h3>
            <table style="color: white; width: 100%; font-size: 11px;">
                <tr><td>Samples:</td><td>{int(low_snr['n_samples'])}</td></tr>
                <tr><td>Accuracy:</td><td>{low_snr['accuracy']:.3f}</td></tr>
                <tr><td>Precision:</td><td>{low_snr['precision']:.3f}</td></tr>
                <tr><td>Recall:</td><td>{low_snr['recall']:.3f}</td></tr>
                <tr><td>F1-Score:</td><td>{low_snr['f1_score']:.3f}</td></tr>
                <tr><td>Specificity:</td><td>{low_snr['specificity']:.3f}</td></tr>
        """
        
        if 'roc_auc' in low_snr:
            low_html += f"""
                <tr><td>ROC AUC:</td><td>{low_snr['roc_auc']:.3f}</td></tr>
                <tr><td>PR AUC:</td><td>{low_snr['pr_auc']:.3f}</td></tr>
            """
        
        # Add uncertainty metrics if available
        if has_uncertainty:
            uq_summary = metrics.get_extended_uq_summary(snr_threshold)
            if 'by_snr' in uq_summary and 'low_snr' in uq_summary['by_snr']:
                low_uq = uq_summary['by_snr']['low_snr']
                low_html += f"""
                <tr><td colspan="2" style="padding-top: 8px;"><b>Uncertainty:</b></td></tr>
                <tr><td>Mean UQ:</td><td>{low_uq['mean']:.4f} (±{low_uq['std']:.4f})</td></tr>
                """
                if 'by_confusion' in low_uq:
                    low_html += """<tr><td colspan="2" style="font-size: 10px; padding-top: 4px;"><em>By category:</em></td></tr>"""
                    for cat in ['TP', 'FP', 'TN', 'FN']:
                        if cat in low_uq['by_confusion'] and low_uq['by_confusion'][cat]['count'] > 0:
                            cat_stats = low_uq['by_confusion'][cat]
                            low_html += f"""
                <tr><td style="padding-left: 10px; font-size: 10px;">{cat}:</td><td style="font-size: 10px;">{cat_stats['mean']:.4f} (n={cat_stats['count']})</td></tr>
                """
        
        low_html += """
            </table>
        </div>
        """
        
        # High SNR metrics div
        high_snr = snr_metrics['high_snr']
        high_html = f"""
        <div style="padding: 10px; background-color: #2F3B2F; border-radius: 5px; border-left: 4px solid #FFC300;">
            <h3 style="color: white; margin-top: 0;">High SNR Performance (|SNR| ≥ {snr_threshold})</h3>
            <table style="color: white; width: 100%; font-size: 11px;">
                <tr><td>Samples:</td><td>{int(high_snr['n_samples'])}</td></tr>
                <tr><td>Accuracy:</td><td>{high_snr['accuracy']:.3f}</td></tr>
                <tr><td>Precision:</td><td>{high_snr['precision']:.3f}</td></tr>
                <tr><td>Recall:</td><td>{high_snr['recall']:.3f}</td></tr>
                <tr><td>F1-Score:</td><td>{high_snr['f1_score']:.3f}</td></tr>
                <tr><td>Specificity:</td><td>{high_snr['specificity']:.3f}</td></tr>
        """
        
        if 'roc_auc' in high_snr:
            high_html += f"""
                <tr><td>ROC AUC:</td><td>{high_snr['roc_auc']:.3f}</td></tr>
                <tr><td>PR AUC:</td><td>{high_snr['pr_auc']:.3f}</td></tr>
            """
        
        # Add uncertainty metrics if available
        if has_uncertainty:
            uq_summary = metrics.get_extended_uq_summary(snr_threshold)
            if 'by_snr' in uq_summary and 'high_snr' in uq_summary['by_snr']:
                high_uq = uq_summary['by_snr']['high_snr']
                high_html += f"""
                <tr><td colspan="2" style="padding-top: 8px;"><b>Uncertainty:</b></td></tr>
                <tr><td>Mean UQ:</td><td>{high_uq['mean']:.4f} (±{high_uq['std']:.4f})</td></tr>
                """
                if 'by_confusion' in high_uq:
                    high_html += """<tr><td colspan="2" style="font-size: 10px; padding-top: 4px;"><em>By category:</em></td></tr>"""
                    for cat in ['TP', 'FP', 'TN', 'FN']:
                        if cat in high_uq['by_confusion'] and high_uq['by_confusion'][cat]['count'] > 0:
                            cat_stats = high_uq['by_confusion'][cat]
                            high_html += f"""
                <tr><td style="padding-left: 10px; font-size: 10px;">{cat}:</td><td style="font-size: 10px;">{cat_stats['mean']:.4f} (n={cat_stats['count']})</td></tr>
                """
        
        high_html += """
            </table>
        </div>
        """
        
        # Adjust height if uncertainty is included
        div_height = 300 if has_uncertainty else 250
        
        low_snr_div = Div(text=low_html, width=300, height=div_height)
        high_snr_div = Div(text=high_html, width=300, height=div_height)
        
        return low_snr_div, high_snr_div
    
    def create_transient_types_div(self, metrics: EvaluationMetrics, 
                                   other_transients: np.ndarray) -> Div:
        """Create a summary div with transient type breakdown (SN vs Other Transients).
        
        Shows performance metrics for the two subclasses within is_injection=1:
        - SN (supernovae): is_injection=1 && other_transients=0
        - Other Transients: is_injection=1 && other_transients=1
        
        Parameters
        ----------
        metrics : EvaluationMetrics
            Evaluation metrics object
        other_transients : np.ndarray
            Array indicating transient type (0=SN, 1=other_transient, -1=missing)
            
        Returns
        -------
        Div
            Bokeh Div element with transient type metrics
        """
        if other_transients is None:
            html_content = """
            <div style="padding: 15px; background-color: #2F2F2F; border-radius: 5px;">
                <h3 style="color: white; margin-top: 0;">Transient Type Analysis</h3>
                <p style="color: white;">Not available (other_transients column not found in data)</p>
            </div>
            """
            return Div(text=html_content, width=600, height=150)
        
        # Get transient type metrics
        tt_metrics = metrics.get_transient_type_metrics(other_transients)
        sn = tt_metrics['sn']
        other = tt_metrics['other_transients']
        bogus = tt_metrics['bogus']
        
        # Start building HTML
        html_content = """
        <div style="padding: 15px; background-color: #2F2F2F; border-radius: 5px;">
            <h3 style="color: white; margin-top: 0;">Transient Type Analysis</h3>
            <p style="color: #aaa; font-size: 11px; margin-bottom: 15px;">
                Performance breakdown for injection subclasses (is_injection=1)
            </p>
            
            <div style="display: flex; gap: 20px;">
        """
        
        # SN (Supernovae) box
        html_content += f"""
                <div style="flex: 1; padding: 10px; background-color: #2F2F2F; border-radius: 5px; border: 1px solid #444;">
                    <h4 style="color: white; margin-top: 0; margin-bottom: 10px;">Supernovae (SN)</h4>
                    <p style="color: #aaa; font-size: 10px; margin-bottom: 8px;">is_injection=1 && other_transients=0</p>
                    <table style="color: white; width: 100%; font-size: 12px;">
                        <tr><td>Total Samples:</td><td><b>{sn['n_samples']}</b></td></tr>
                        <tr><td>Correctly Detected:</td><td>{sn['n_correct']}</td></tr>
                        <tr><td>Missed (FN):</td><td>{sn['n_incorrect']}</td></tr>
                        <tr><td>Recall:</td><td><b>{sn['recall']:.1%}</b></td></tr>
        """
        
        if 'mean_probability' in sn:
            html_content += f"""
                        <tr><td colspan="2" style="padding-top: 8px;"><em>Probability Stats:</em></td></tr>
                        <tr><td>Mean Prob:</td><td>{sn['mean_probability']:.3f} (±{sn['std_probability']:.3f})</td></tr>
            """
        
        if 'mean_uncertainty' in sn:
            html_content += f"""
                        <tr><td colspan="2" style="padding-top: 4px;"><em>Uncertainty Stats:</em></td></tr>
                        <tr><td>Mean UQ:</td><td>{sn['mean_uncertainty']:.4f} (±{sn['std_uncertainty']:.4f})</td></tr>
            """
        
        html_content += """
                    </table>
                </div>
        """
        
        # Other Transients box
        html_content += f"""
                <div style="flex: 1; padding: 10px; background-color: #2F2F2F; border-radius: 5px; border: 1px solid #444;">
                    <h4 style="color: white; margin-top: 0; margin-bottom: 10px;">Other Transients</h4>
                    <p style="color: #aaa; font-size: 10px; margin-bottom: 8px;">is_injection=1 && other_transients=1</p>
                    <table style="color: white; width: 100%; font-size: 12px;">
                        <tr><td>Total Samples:</td><td><b>{other['n_samples']}</b></td></tr>
                        <tr><td>Correctly Detected:</td><td>{other['n_correct']}</td></tr>
                        <tr><td>Missed (FN):</td><td>{other['n_incorrect']}</td></tr>
                        <tr><td>Recall:</td><td><b>{other['recall']:.1%}</b></td></tr>
        """
        
        if 'mean_probability' in other:
            html_content += f"""
                        <tr><td colspan="2" style="padding-top: 8px;"><em>Probability Stats:</em></td></tr>
                        <tr><td>Mean Prob:</td><td>{other['mean_probability']:.3f} (±{other['std_probability']:.3f})</td></tr>
            """
        
        if 'mean_uncertainty' in other:
            html_content += f"""
                        <tr><td colspan="2" style="padding-top: 4px;"><em>Uncertainty Stats:</em></td></tr>
                        <tr><td>Mean UQ:</td><td>{other['mean_uncertainty']:.4f} (±{other['std_uncertainty']:.4f})</td></tr>
            """
        
        html_content += """
                    </table>
                </div>
        """
        
        # Bogus/Real negatives box
        html_content += f"""
                <div style="flex: 1; padding: 10px; background-color: #2F2F2F; border-radius: 5px; border: 1px solid #444;">
                    <h4 style="color: white; margin-top: 0; margin-bottom: 10px;">Bogus (Real)</h4>
                    <p style="color: #aaa; font-size: 10px; margin-bottom: 8px;">is_injection=0</p>
                    <table style="color: white; width: 100%; font-size: 12px;">
                        <tr><td>Total Samples:</td><td><b>{bogus['n_samples']}</b></td></tr>
                        <tr><td>Correctly Rejected:</td><td>{bogus['n_correct']}</td></tr>
                        <tr><td>False Positives:</td><td>{bogus['n_fp']}</td></tr>
                        <tr><td>Specificity:</td><td><b>{bogus['specificity']:.1%}</b></td></tr>
                        <tr><td>FP Rate:</td><td>{bogus['fp_rate']:.1%}</td></tr>
        """
        
        if 'mean_probability' in bogus:
            html_content += f"""
                        <tr><td colspan="2" style="padding-top: 8px;"><em>Probability Stats:</em></td></tr>
                        <tr><td>Mean Prob:</td><td>{bogus['mean_probability']:.3f} (±{bogus['std_probability']:.3f})</td></tr>
            """
        
        if 'mean_uncertainty' in bogus:
            html_content += f"""
                        <tr><td colspan="2" style="padding-top: 4px;"><em>Uncertainty Stats:</em></td></tr>
                        <tr><td>Mean UQ:</td><td>{bogus['mean_uncertainty']:.4f} (±{bogus['std_uncertainty']:.4f})</td></tr>
            """
        
        html_content += """
                    </table>
                </div>
            </div>
        </div>
        """
        
        return Div(text=html_content, width=900, height=380)

    def create_sn_phase_analysis_div(self, phase_analysis: Dict) -> Div:
        """Create a Div showing SN phase analysis with dist_max_bright correlation.
        
        Parameters
        ----------
        phase_analysis : Dict
            Dictionary from EvaluationMetrics.get_sn_phase_analysis()
            
        Returns
        -------
        Div
            Bokeh Div with phase analysis visualization
        """
        if not phase_analysis.get('available', False):
            html_content = """
            <div style="background-color: #2a2a2a; padding: 15px; border-radius: 5px;">
                <h3 style="color: white; margin-top: 0;">SN Light Curve Phase Analysis</h3>
                <p style="color: #aaa;">Not available - insufficient SN samples with dist_max_bright data</p>
            </div>
            """
            return Div(text=html_content, width=900, height=100)
        
        html_content = """
        <div style="background-color: #2a2a2a; padding: 15px; border-radius: 5px;">
            <h3 style="color: white; margin-top: 0; margin-bottom: 15px;">SN Light Curve Phase Analysis</h3>
            <p style="color: #aaa; font-size: 11px; margin-bottom: 15px;">
                Analysis of detection performance relative to peak brightness<br>
                (dist_max_bright > 0 = before max, < 0 = after max)
            </p>
        """
        
        # Correlation section
        html_content += """
            <div style="display: flex; gap: 15px; margin-bottom: 15px;">
        """
        
        # UQ correlation box
        if 'uq_dist_correlation' in phase_analysis:
            uq_corr = phase_analysis['uq_dist_correlation']
            rho = uq_corr['spearman_rho']
            
            html_content += f"""
                <div style="flex: 1; padding: 10px; background-color: #3a3a3a; border-radius: 5px; border-left: 3px solid #4FC3F7;">
                    <h4 style="color: #4FC3F7; margin: 0 0 8px 0; font-size: 13px;">Uncertainty vs Distance from Max</h4>
                    <table style="color: white; width: 100%; font-size: 11px;">
                        <tr><td>Spearman rho:</td><td><b>{rho:.3f}</b></td></tr>
                        <tr><td>N samples:</td><td>{uq_corr['n_samples']}</td></tr>
                    </table>
                </div>
            """
        
        # Accuracy correlation box
        if 'accuracy_dist_correlation' in phase_analysis:
            acc_corr = phase_analysis['accuracy_dist_correlation']
            rho = acc_corr['spearman_rho']
            
            html_content += f"""
                <div style="flex: 1; padding: 10px; background-color: #3a3a3a; border-radius: 5px; border-left: 3px solid #81C784;">
                    <h4 style="color: #81C784; margin: 0 0 8px 0; font-size: 13px;">Accuracy vs Distance from Max</h4>
                    <table style="color: white; width: 100%; font-size: 11px;">
                        <tr><td>Spearman rho:</td><td><b>{rho:.3f}</b></td></tr>
                        <tr><td>N samples:</td><td>{acc_corr['n_samples']}</td></tr>
                    </table>
                </div>
            """
        
        html_content += """
            </div>
        """
        
        # Phase breakdown section
        if 'phase_breakdown' in phase_analysis:
            phases = phase_analysis['phase_breakdown']
            
            html_content += """
            <h4 style="color: white; margin: 15px 0 10px 0; font-size: 13px;">Detection by Light Curve Phase</h4>
            <div style="display: flex; gap: 10px; margin-bottom: 15px;">
            """
            
            phase_names = [
                ('pre_max', 'Pre-Maximum', '(> 5 days before peak)', '#FFB74D'),
                ('near_max', 'Near Maximum', '(-5 to +5 days)', '#4FC3F7'),
                ('post_max', 'Post-Maximum', '(> 5 days after peak)', '#81C784')
            ]
            
            for phase_key, phase_label, phase_desc, color in phase_names:
                phase_data = phases.get(phase_key, {'n_samples': 0, 'accuracy': 0.0})
                n = phase_data.get('n_samples', 0)
                acc = phase_data.get('accuracy', 0.0)
                n_correct = phase_data.get('n_correct', 0)
                n_incorrect = phase_data.get('n_incorrect', 0)
                
                html_content += f"""
                <div style="flex: 1; padding: 10px; background-color: #2F2F2F; border-radius: 5px; border-top: 3px solid {color};">
                    <h5 style="color: {color}; margin: 0 0 5px 0; font-size: 12px;">{phase_label}</h5>
                    <p style="color: #888; font-size: 9px; margin: 0 0 8px 0;">{phase_desc}</p>
                    <table style="color: white; width: 100%; font-size: 11px;">
                        <tr><td>Samples:</td><td><b>{n}</b></td></tr>
                        <tr><td>Correct:</td><td>{n_correct}</td></tr>
                        <tr><td>Missed:</td><td>{n_incorrect}</td></tr>
                        <tr><td>Accuracy:</td><td><b style="color: {color};">{acc:.1%}</b></td></tr>
                """
                
                if 'mean_uncertainty' in phase_data:
                    html_content += f"""
                        <tr><td>Mean UQ:</td><td>{phase_data['mean_uncertainty']:.4f}</td></tr>
                    """
                
                if 'mean_dist' in phase_data:
                    html_content += f"""
                        <tr><td>Mean dist:</td><td>{phase_data['mean_dist']:.1f} days</td></tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            html_content += """
            </div>
            """
        
        # Binned accuracy section
        if 'binned_by_distance' in phase_analysis:
            bins = phase_analysis['binned_by_distance']
            
            html_content += """
            <h4 style="color: white; margin: 15px 0 10px 0; font-size: 13px;">Accuracy by Distance from Peak (days)</h4>
            <table style="color: white; width: 100%; font-size: 11px; border-collapse: collapse;">
                <thead>
                    <tr style="border-bottom: 1px solid #555;">
                        <th style="text-align: left; padding: 5px;">Distance Range</th>
                        <th style="text-align: center; padding: 5px;">N Samples</th>
                        <th style="text-align: center; padding: 5px;">Accuracy</th>
                        <th style="text-align: center; padding: 5px;">Mean UQ</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for bin_label in ['0-5', '5-10', '10-20', '20-50', '>50']:
                if bin_label in bins:
                    bin_data = bins[bin_label]
                    uq_str = f"{bin_data['mean_uncertainty']:.4f}" if 'mean_uncertainty' in bin_data else "-"
                    acc_pct = bin_data['accuracy']
                    
                    html_content += f"""
                    <tr style="border-bottom: 1px solid #333;">
                        <td style="padding: 5px;">{bin_label} days</td>
                        <td style="text-align: center; padding: 5px;">{bin_data['n_samples']}</td>
                        <td style="text-align: center; padding: 5px;"><b>{acc_pct:.1%}</b></td>
                        <td style="text-align: center; padding: 5px;">{uq_str}</td>
                    </tr>
                    """
            
            html_content += """
                </tbody>
            </table>
            """
        
        html_content += """
        </div>
        """
        
        return Div(text=html_content, width=900, height=450)
    
    def create_lightcurve_metrics_div(self, lc_metrics: Dict) -> Div:
        """Create a summary div with lightcurve-level classification metrics.
        
        Parameters
        ----------
        lc_metrics : Dict
            Lightcurve metrics from EvaluationMetrics.get_lightcurve_metrics()
            
        Returns
        -------
        Div
            Bokeh Div element with lightcurve-level statistics
        """
        if lc_metrics is None:
            return Div(text="<div><p>Lightcurve metrics not available</p></div>", 
                      width=600, height=100)
        
        overall = lc_metrics['overall']
        confusion = lc_metrics['confusion']
        threshold = lc_metrics.get('threshold', 0.9)  # Default to 0.9 if not specified
        threshold_pct = int(threshold * 100)
        
        # Start building HTML
        html_content = f"""
        <div style="padding: 15px; background-color: #2F2F2F; border-radius: 5px;">
            <h3 style="color: white; margin-top: 0;">Lightcurve-Level Performance</h3>
            <p style="color: #aaa; font-size: 11px; margin-bottom: 10px;">
                <b>Classification logic (threshold: {threshold_pct}%):</b><br>
                • LC classified as <span style="color: #FFC300;">REAL</span> if ≥{threshold_pct}% of sources predicted as real<br>
                • LC classified as <span style="color: #C70039;">BOGUS</span> if ≥{threshold_pct}% of sources predicted as bogus<br>
                • LC marked as <span style="color: #FFA726;">UNCLASSIFIED</span> if neither reaches {threshold_pct}%
            </p>
        """
        
        # Overall statistics
        html_content += f"""
            <div style="background-color: #3F3F3F; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <h4 style="color: #FFC300; margin-top: 0;">Overall Statistics</h4>
                <table style="color: white; width: 100%; font-size: 12px;">
                    <tr><td><b>Total Lightcurves:</b></td><td>{overall['n_lightcurves']}</td></tr>
                    <tr><td><b>Classified LCs:</b></td><td>{overall['n_classified']}</td></tr>
                    <tr><td><b>Unclassified LCs:</b></td><td><span style="color: #FFA726;">{overall['n_unclassified']}</span></td></tr>
                    <tr><td colspan="2" style="padding-top: 8px; border-top: 1px solid #555;"></td></tr>
                    <tr><td><b>Correctly Classified:</b></td><td>{overall['n_correct']} ({overall['accuracy']:.1%})</td></tr>
                    <tr><td><b>Incorrectly Classified:</b></td><td>{overall['n_incorrect']}</td></tr>
                    <tr><td colspan="2" style="padding-top: 8px;"></td></tr>
                    <tr><td>Classified as Real:</td><td>{overall['n_classified_as_real']}</td></tr>
                    <tr><td>Classified as Bogus:</td><td>{overall['n_classified_as_bogus']}</td></tr>
                </table>
                <p style="color: #aaa; font-size: 9px; margin: 5px 0 0 0;">Accuracy computed only on classified LCs</p>
            </div>
        """
        
        # Confusion matrix
        html_content += f"""
            <div style="background-color: #3F3F3F; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <h4 style="color: #FFE66D; margin-top: 0;">Confusion Matrix</h4>
                <table style="color: white; width: 100%; font-size: 12px;">
                    <tr><td><b>True Positives (TP):</b></td><td>{confusion['true_positive']}</td></tr>
                    <tr><td><b>True Negatives (TN):</b></td><td>{confusion['true_negative']}</td></tr>
                    <tr><td><b>False Positives (FP):</b></td><td>{confusion['false_positive']}</td></tr>
                    <tr><td><b>False Negatives (FN):</b></td><td>{confusion['false_negative']}</td></tr>
                    <tr><td colspan="2" style="padding-top: 8px; border-top: 1px solid #555;"></td></tr>
                    <tr><td>Unclassified (True Real):</td><td><span style="color: #FFA726;">{confusion.get('unclassified_real', 0)}</span></td></tr>
                    <tr><td>Unclassified (True Bogus):</td><td><span style="color: #FFA726;">{confusion.get('unclassified_bogus', 0)}</span></td></tr>
                </table>
            </div>
        """
        
        # Class-specific breakdown
        if 'by_true_class' in lc_metrics:
            html_content += '<div style="display: flex; gap: 15px;">'
            
            # Real transients (if ground truth available)
            if 'real_transients' in lc_metrics['by_true_class']:
                real = lc_metrics['by_true_class']['real_transients']
                html_content += f"""
                <div style="flex: 1; background-color: #3B3F2F; padding: 10px; border-radius: 5px; border-left: 4px solid #FFC300;">
                    <h4 style="color: #FFC300; margin-top: 0; font-size: 14px;">Real Transient LCs</h4>
                    <table style="color: white; width: 100%; font-size: 11px;">
                        <tr><td>Total:</td><td><b>{real['n_lightcurves']}</b></td></tr>
                        <tr><td>Detected:</td><td>{real['n_correct']}</td></tr>
                        <tr><td>Missed:</td><td>{real['n_incorrect']}</td></tr>
                        <tr><td>Unclassified:</td><td><span style="color: #FFA726;">{real.get('n_unclassified', 0)}</span></td></tr>
                        <tr><td><b>Recall:</b></td><td><b>{real['recall']:.1%}</b></td></tr>
                        <tr><td colspan="2" style="padding-top: 8px; border-top: 1px solid #555;"></td></tr>
                        <tr><td>Avg Source Acc:</td><td><b>{real.get('avg_source_accuracy', 0):.1%}</b></td></tr>
                    </table>
                    <p style="color: #aaa; font-size: 9px; margin: 5px 0 0 0;">Average % of correctly classified sources within these LCs</p>
                </div>
                """
            
            # Bogus lightcurves
            if 'bogus' in lc_metrics['by_true_class']:
                bogus = lc_metrics['by_true_class']['bogus']
                html_content += f"""
                <div style="flex: 1; background-color: #3F2F3B; padding: 10px; border-radius: 5px; border-left: 4px solid #C70039;">
                    <h4 style="color: #C70039; margin-top: 0; font-size: 14px;">Bogus LCs</h4>
                    <table style="color: white; width: 100%; font-size: 11px;">
                        <tr><td>Total:</td><td><b>{bogus['n_lightcurves']}</b></td></tr>
                        <tr><td>Rejected:</td><td>{bogus['n_correct']}</td></tr>
                        <tr><td>False Alarms:</td><td>{bogus['n_incorrect']}</td></tr>
                        <tr><td>Unclassified:</td><td><span style="color: #FFA726;">{bogus.get('n_unclassified', 0)}</span></td></tr>
                        <tr><td><b>Specificity:</b></td><td><b>{bogus['specificity']:.1%}</b></td></tr>
                        <tr><td colspan="2" style="padding-top: 8px; border-top: 1px solid #555;"></td></tr>
                        <tr><td>Avg Source Acc:</td><td><b>{bogus.get('avg_source_accuracy', 0):.1%}</b></td></tr>
                    </table>
                    <p style="color: #aaa; font-size: 9px; margin: 5px 0 0 0;">Average % of correctly classified sources within these LCs</p>
                </div>
                """
            
            html_content += '</div>'
        
        html_content += '</div>'
        
        return Div(text=html_content, width=700, height=500)
    
    def plot_lightcurve_detection_distributions(self, lc_metrics: Dict, title: str = "Transient Detection Distribution") -> figure:
        """Create histogram showing distribution of % transient detection in lightcurves.
        
        Parameters
        ----------
        lc_metrics : Dict
            Lightcurve metrics from EvaluationMetrics.get_lightcurve_metrics()
        title : str
            Plot title
            
        Returns
        -------
        figure
            Bokeh figure with histogram
        """
        if lc_metrics is None or 'distributions' not in lc_metrics:
            p = figure(title=f"{title} (Data not available)", 
                      width=self.width//2, height=self.height//2)
            p.text([self.width//4], [self.height//4], text=["Distribution data not available"],
                   text_align="center", text_baseline="middle", text_font_size="14pt")
            return p
        
        dist_data = lc_metrics['distributions']['all_lcs']
        percentages = np.array(dist_data['transient_percentages']) * 100  # Convert to percentage
        true_labels = np.array(dist_data['true_labels'])
        
        # Create histogram
        hist, edges = np.histogram(percentages, bins=20, range=(0, 100))
        
        source = ColumnDataSource(data=dict(
            left=edges[:-1],
            right=edges[1:],
            top=hist
        ))
        
        p = figure(title=title,
                  x_axis_label="% Sources Predicted as Transient",
                  y_axis_label="Number of Lightcurves",
                  width=self.width//2, height=self.height//2,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
        p.quad(left='left', right='right', top='top', bottom=0, source=source,
               fill_color='#FFC300', line_color='white', alpha=0.7)
        
        # Add vertical lines at 10% and 90% thresholds
        p.line([10, 10], [0, max(hist)*1.1], line_dash="dashed", line_color="orange", 
               line_width=2, legend_label="10% threshold")
        p.line([90, 90], [0, max(hist)*1.1], line_dash="dashed", line_color="red", 
               line_width=2, legend_label="90% threshold")
        
        p.legend.location = "top_right"
        p.y_range.start = 0
        
        return p
    
    def plot_transient_lc_detection_distribution(self, lc_metrics: Dict, 
                                                 title: str = "Detection in Real Transient LCs") -> figure:
        """Create histogram showing % transient detection in real transient lightcurves only.
        
        Parameters
        ----------
        lc_metrics : Dict
            Lightcurve metrics from EvaluationMetrics.get_lightcurve_metrics()
        title : str
            Plot title
            
        Returns
        -------
        figure
            Bokeh figure with histogram
        """
        if lc_metrics is None or 'distributions' not in lc_metrics:
            p = figure(title=f"{title} (Data not available)", 
                      width=self.width//2, height=self.height//2)
            p.text([self.width//4], [self.height//4], text=["Distribution data not available"],
                   text_align="center", text_baseline="middle", text_font_size="14pt")
            return p
        
        dist_data = lc_metrics['distributions']['all_lcs']
        percentages = np.array(dist_data['transient_percentages']) * 100
        true_labels = np.array(dist_data['true_labels'])
        
        # Filter for real transient lightcurves only
        transient_mask = true_labels == 1
        if np.sum(transient_mask) == 0:
            p = figure(title=f"{title} (No real transients)", 
                      width=self.width//2, height=self.height//2)
            p.text([self.width//4], [self.height//4], text=["No real transient data available"],
                   text_align="center", text_baseline="middle", text_font_size="14pt")
            return p
        
        transient_percentages = percentages[transient_mask]
        
        # Create histogram
        hist, edges = np.histogram(transient_percentages, bins=20, range=(0, 100))
        
        source = ColumnDataSource(data=dict(
            left=edges[:-1],
            right=edges[1:],
            top=hist
        ))
        
        p = figure(title=title,
                  x_axis_label="% Sources Predicted as Transient",
                  y_axis_label="Number of Real Transient LCs",
                  width=self.width//2, height=self.height//2,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
        p.quad(left='left', right='right', top='top', bottom=0, source=source,
               fill_color='#FFC300', line_color='white', alpha=0.7)
        
        # Add vertical line at 90% threshold
        p.line([90, 90], [0, max(hist)*1.1], line_dash="dashed", line_color="red", 
               line_width=2, legend_label="90% threshold")
        
        p.legend.location = "top_left"
        p.y_range.start = 0
        
        return p
    
    def plot_bogus_lc_rejection_distribution(self, lc_metrics: Dict, 
                                            title: str = "Rejection in Bogus LCs") -> figure:
        """Create histogram showing % bogus detection (rejection) in bogus lightcurves.
        
        Parameters
        ----------
        lc_metrics : Dict
            Lightcurve metrics from EvaluationMetrics.get_lightcurve_metrics()
        title : str
            Plot title
            
        Returns
        -------
        figure
            Bokeh figure with histogram
        """
        if lc_metrics is None or 'distributions' not in lc_metrics:
            p = figure(title=f"{title} (Data not available)", 
                      width=self.width//2, height=self.height//2)
            p.text([self.width//4], [self.height//4], text=["Distribution data not available"],
                   text_align="center", text_baseline="middle", text_font_size="14pt")
            return p
        
        dist_data = lc_metrics['distributions']['all_lcs']
        percentages = np.array(dist_data['transient_percentages']) * 100
        true_labels = np.array(dist_data['true_labels'])
        
        # Filter for bogus lightcurves only
        bogus_mask = true_labels == 0
        bogus_percentages = percentages[bogus_mask]
        
        # Convert to % bogus (100 - % transient)
        bogus_rejection_percentages = 100 - bogus_percentages
        
        # Create histogram
        hist, edges = np.histogram(bogus_rejection_percentages, bins=20, range=(0, 100))
        
        source = ColumnDataSource(data=dict(
            left=edges[:-1],
            right=edges[1:],
            top=hist
        ))
        
        p = figure(title=title,
                  x_axis_label="% Sources Predicted as Bogus",
                  y_axis_label="Number of Bogus LCs",
                  width=self.width//2, height=self.height//2,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
        p.quad(left='left', right='right', top='top', bottom=0, source=source,
               fill_color='#C70039', line_color='white', alpha=0.7)
        
        # Add vertical line at 90% threshold
        p.line([90, 90], [0, max(hist)*1.1], line_dash="dashed", line_color="red", 
               line_width=2, legend_label="90% threshold")
        
        p.legend.location = "top_left"
        p.y_range.start = 0
        
        return p


class UMAPVisualizer:
    """Class for creating UMAP-based visualizations."""
    
    def __init__(self, width: int = 800, height: int = 550):
        """Initialize with plot dimensions."""
        curdoc().theme = 'dark_minimal'
        self.width = width
        self.height = height
        
        # Define consistent symbols and colors for classification results
        self.class_symbols = {
            'True Positive': 'x',
            'True Negative': 'circle', 
            'False Positive': 'star',
            'False Negative': 'square'
        }
        
        self.class_colors = {
            'True Positive': '#FFC300',    
            'True Negative': '#C70039',  
            'False Positive': '#FF5733',   
            'False Negative': '#b3b6b7'   
        }
    
    def _build_tooltip_html(self, df: pd.DataFrame, extra_fields: Dict[str, str] = None) -> str:
        """Build HTML tooltip that adapts to available image columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with data (used to detect available columns)
        extra_fields : Dict[str, str], optional
            Extra fields to include in tooltip {field_name: display_label}
            
        Returns
        -------
        str
            HTML tooltip string
        """
        # Detect available image columns (e.g., 'image', 'image_diff', 'image_coadd')
        image_columns = [col for col in df.columns if col.startswith('image')]
        
        # Start tooltip HTML
        tooltip_html = "<div>"
        
        # Add images section
        if len(image_columns) > 0:
            tooltip_html += "<div>"
            for img_col in sorted(image_columns):
                # Extract cutout type name from column (e.g., 'image_diff' -> 'Diff')
                if img_col == 'image':
                    label = "Cutout"
                else:
                    cutout_type = img_col.replace('image_', '').capitalize()
                    label = cutout_type
                
                tooltip_html += f"""
                    <div style='display: inline-block; margin: 2px;'>
                        <div style='text-align: center; font-size: 10px; color: #666;'>{label}</div>
                        <img src='@{img_col}' height="60" width="60" style='border: 1px solid #444;'></img>
                    </div>
                """
            tooltip_html += "</div>"
        
        # Add classification info
        tooltip_html += """
            <div style='margin-top: 5px;'>
                <span style='font-size: 14px; color: #224499'>Class type:</span>
                <span style='font-size: 14px'>@class_type</span><br>
                <span style='font-size: 14px; color: #224499'>True label:</span>
                <span style='font-size: 14px'>@true_label</span><br>
                <span style='font-size: 14px; color: #224499'>Prediction:</span>
                <span style='font-size: 14px'>@prediction</span><br>
        """
        
        # Add extra fields if provided
        if extra_fields:
            for field_name, display_label in extra_fields.items():
                if field_name in df.columns:
                    # Format numbers nicely
                    if df[field_name].dtype in [np.float32, np.float64]:
                        tooltip_html += f"""
                <span style='font-size: 14px; color: #224499'>{display_label}:</span>
                <span style='font-size: 14px'>@{field_name}{{0.000}}</span><br>
                        """
                    else:
                        tooltip_html += f"""
                <span style='font-size: 14px; color: #224499'>{display_label}:</span>
                <span style='font-size: 14px'>@{field_name}</span><br>
                        """
        
        tooltip_html += """
            </div>
        </div>
        """
        
        return tooltip_html
    
    def _add_classification_scatter(self, p, df, color_field=None, color_mapper=None, 
                                   size=8, alpha=0.7, legend_field=None):
        """Helper method to add scatter points with consistent TP/FP/TN/FN symbols."""
        # Create data sources for each class type
        tp_data = ColumnDataSource(df[df['class_type'] == 'True Positive'])
        tn_data = ColumnDataSource(df[df['class_type'] == 'True Negative'])
        fp_data = ColumnDataSource(df[df['class_type'] == 'False Positive'])
        fn_data = ColumnDataSource(df[df['class_type'] == 'False Negative'])
        
        # Add scatter points with appropriate symbols and colors
        renderers = []
        
        if len(tp_data.data['umap_x']) > 0:
            if color_field and color_mapper:
                color_spec = {'field': color_field, 'transform': color_mapper}
            else:
                color_spec = self.class_colors['True Positive']
            
            renderer = p.scatter('umap_x', 'umap_y', source=tp_data, 
                               marker=self.class_symbols['True Positive'],
                               color=color_spec,
                               size=size, alpha=alpha, 
                               legend_label=legend_field or "True Positive")
            renderers.append(renderer)
            
        if len(tn_data.data['umap_x']) > 0:
            if color_field and color_mapper:
                color_spec = {'field': color_field, 'transform': color_mapper}
            else:
                color_spec = self.class_colors['True Negative']
                
            renderer = p.scatter('umap_x', 'umap_y', source=tn_data,
                               marker=self.class_symbols['True Negative'], 
                               color=color_spec,
                               size=size, alpha=alpha,
                               legend_label=legend_field or "True Negative")
            renderers.append(renderer)
            
        if len(fp_data.data['umap_x']) > 0:
            if color_field and color_mapper:
                color_spec = {'field': color_field, 'transform': color_mapper}
            else:
                color_spec = self.class_colors['False Positive']
                
            renderer = p.scatter('umap_x', 'umap_y', source=fp_data,
                               marker=self.class_symbols['False Positive'],
                               color=color_spec, 
                               size=size, alpha=alpha,
                               legend_label=legend_field or "False Positive")
            renderers.append(renderer)
            
        if len(fn_data.data['umap_x']) > 0:
            if color_field and color_mapper:
                color_spec = {'field': color_field, 'transform': color_mapper}
            else:
                color_spec = self.class_colors['False Negative']
                
            renderer = p.scatter('umap_x', 'umap_y', source=fn_data,
                               marker=self.class_symbols['False Negative'],
                               color=color_spec,
                               size=size, alpha=alpha, 
                               legend_label=legend_field or "False Negative")
            renderers.append(renderer)
            
        return renderers
    
    def plot_feature_coloring(self, df: pd.DataFrame, feature_column: str,
                            title: str = None, palette: str = "Viridis256") -> figure:
        """Plot UMAP colored by a continuous feature with TP/FP/TN/FN symbols."""
        if title is None:
            title = f"UMAP: {feature_column}"
        
        # Handle missing values
        df_clean = df.dropna(subset=[feature_column])
        
        p = figure(title=title, width=self.width, height=self.height,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
        # Create color mapper
        color_mapper = LinearColorMapper(
            palette=palette,
            low=df_clean[feature_column].min(),
            high=df_clean[feature_column].max()
        )
        
        # Use helper method to add scatter points with classification symbols
        self._add_classification_scatter(
            p, df_clean, 
            color_field=feature_column, 
            color_mapper=color_mapper
        )
        
        # Add color bar
        color_bar = ColorBar(color_mapper=color_mapper, width=8, location=(0, 0))
        p.add_layout(color_bar, 'right')
        
        # Create hover tooltip using adaptive builder
        extra_fields = {feature_column: feature_column}
        tooltip_html = self._build_tooltip_html(df_clean, extra_fields=extra_fields)
        
        hover = HoverTool(tooltips=tooltip_html)
        p.add_tools(hover)
        
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        
        return p
    
    def plot_clusters(self, df: pd.DataFrame, cluster_column: str = 'cluster',
                     title: str = "UMAP: Clusters") -> figure:
        """Plot UMAP colored by cluster labels with TP/FP/TN/FN symbols."""
        p = figure(title=title, width=self.width, height=self.height,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
        # Hide axes for cleaner UMAP visualization
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        
        # Get unique clusters and create color mapping
        clusters = sorted(df[cluster_column].unique())
        n_clusters = len(clusters)
        
        if n_clusters <= 10:
            palette = Category10[max(3, n_clusters)]
        else:
            from bokeh.palettes import turbo
            palette = turbo(n_clusters)
        
        # Create color mapper for clusters
        cluster_color_mapper = CategoricalColorMapper(
            factors=clusters,
            palette=palette
        )
        
        # Use helper method with cluster coloring
        self._add_classification_scatter(
            p, df,
            color_field=cluster_column,
            color_mapper=cluster_color_mapper
        )
        
        # Create hover tooltip using adaptive builder
        extra_fields = {cluster_column: 'Cluster'}
        tooltip_html = self._build_tooltip_html(df, extra_fields=extra_fields)
        
        hover = HoverTool(tooltips=tooltip_html)
        p.add_tools(hover)
        
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        
        return p
    
    def plot_uncertainty_vs_correctness(self, df: pd.DataFrame) -> figure:
        """Plot UMAP with uncertainty and correctness combined with TP/FP/TN/FN symbols.
        
        Works with all model types:
        - Ensemble/CoTeaching: Uses true prediction uncertainty
        - Standard models: Uses distance from decision boundary (|prob - 0.5|) as proxy
        """
        # Determine what uncertainty measure to use
        # Priority: Use true model uncertainty if available, otherwise use probability-based proxy
        if 'prediction_uncertainty' in df.columns:
            uncertainty_col = 'prediction_uncertainty'
            uncertainty_title = "Ensemble Model Disagreement (std dev of the outputs)"
            title = "UMAP: Uncertainty + Prediction Correctness"
        elif 'prediction_probability' in df.columns:
            # For standard models, use distance from decision boundary as uncertainty proxy
            df = df.copy()
            df['uncertainty_proxy'] = np.abs(df['prediction_probability'] - 0.5)
            uncertainty_col = 'uncertainty_proxy'
            uncertainty_title = "Distance from Decision Boundary"
            title = "UMAP: Confidence + Prediction Correctness "
        else:
            raise ValueError("No uncertainty data or prediction probabilities available.")
        
        p = figure(title=title, width=self.width, height=self.height,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
        # Hide axes for cleaner UMAP visualization
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        
        # Debug: Print uncertainty statistics for this plot too
        print(f"\n=== Uncertainty vs Correctness Statistics ===")
        print(f"Uncertainty type: {uncertainty_title}")
        print(f"Column: {uncertainty_col}")
        print(f"Min: {df[uncertainty_col].min():.4f}")
        print(f"Max: {df[uncertainty_col].max():.4f}")
        if uncertainty_col == 'uncertainty_proxy':
            print(f"Probability range: {df['prediction_probability'].min():.4f} to {df['prediction_probability'].max():.4f}")
        print("=" * 45)
        
        # Create uncertainty color mapper
        uncertainty_range = (df[uncertainty_col].min(), df[uncertainty_col].max())
        color_mapper = LinearColorMapper(
            palette=Viridis256,
            low=uncertainty_range[0],
            high=uncertainty_range[1]
        )
        
        # Use helper method with uncertainty coloring
        self._add_classification_scatter(
            p, df,
            color_field=uncertainty_col,
            color_mapper=color_mapper
        )
        
        # Add color bar for uncertainty
        color_bar = ColorBar(color_mapper=color_mapper, width=8, location=(0, 0),
                           title=uncertainty_title)
        p.add_layout(color_bar, 'right')
        
        # Build extra fields for tooltip
        extra_fields = {
            'correct': 'Correct',
            uncertainty_col: uncertainty_title
        }
        if 'prediction_probability' in df.columns:
            extra_fields['prediction_probability'] = 'Probability'
        
        # Create hover tooltip using adaptive builder
        tooltip_html = self._build_tooltip_html(df, extra_fields=extra_fields)
        
        hover = HoverTool(tooltips=tooltip_html)
        p.add_tools(hover)
        
        # Hide legend for cleaner visualization
        p.legend.visible = False
        
        return p
    
    def plot_uncertainty_coloring(self, df: pd.DataFrame) -> figure:
        """Plot UMAP colored by prediction uncertainty with TP/FP/TN/FN symbols.
        
        Works with all model types using uncertainty proxy for standard models.
        """
        # Determine what uncertainty measure to use
        if 'prediction_uncertainty' in df.columns:
            uncertainty_col = 'prediction_uncertainty'
            uncertainty_title = "std dev of the outputs"
            title = "UMAP: Ensemble Model Disagreement (std dev of the outputs)"
        elif 'prediction_probability' in df.columns:
            # For standard models, use distance from decision boundary as uncertainty proxy
            df = df.copy()
            df['uncertainty_proxy'] = np.abs(df['prediction_probability'] - 0.5)
            uncertainty_col = 'uncertainty_proxy'
            uncertainty_title = "Distance from Decision Boundary"
            title = "UMAP: Distance from Decision Boundary (|prob - 0.5|)"
        else:
            raise ValueError("No uncertainty data or prediction probabilities available.")
        
        # Handle missing values
        df_clean = df.dropna(subset=[uncertainty_col])
        
        p = figure(title=title, width=self.width, height=self.height,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
        # Hide axes for cleaner UMAP visualization
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        
        # Create color mapper for uncertainty
        color_mapper = LinearColorMapper(
            palette=Viridis256,
            low=df_clean[uncertainty_col].min(),
            high=df_clean[uncertainty_col].max()
        )
        
        # Use helper method with uncertainty coloring
        self._add_classification_scatter(
            p, df_clean,
            color_field=uncertainty_col,
            color_mapper=color_mapper
        )
        
        # Add color bar
        color_bar = ColorBar(color_mapper=color_mapper, width=8, location=(0, 0),
                           title=uncertainty_title)
        p.add_layout(color_bar, 'right')
        
        # Build extra fields for tooltip
        extra_fields = {uncertainty_col: uncertainty_title}
        if 'prediction_probability' in df.columns:
            extra_fields['prediction_probability'] = 'Probability'
        
        # Create hover tooltip using adaptive builder
        tooltip_html = self._build_tooltip_html(df_clean, extra_fields=extra_fields)
        
        hover = HoverTool(tooltips=tooltip_html)
        p.add_tools(hover)
        
        # Hide legend for cleaner visualization
        p.legend.visible = False
        
        return p
    
    def plot_snr_coloring(self, df: pd.DataFrame, 
                          title: str = "UMAP: Signal-to-Noise Ratio") -> figure:
        """Plot UMAP colored by SNR values with TP/FP/TN/FN symbols."""
        
        # Check if SNR data is available
        if 'snr' not in df.columns:
            # Create a placeholder plot with message
            p = figure(title=f"{title} (SNR data not available)", 
                      width=self.width, height=self.height,
                      tools=['pan', 'wheel_zoom', 'reset', 'save'])
            p.text([self.width/2], [self.height/2], text=["SNR data not available"],
                   text_align="center", text_baseline="middle", text_font_size="16pt")
            return p
        
        # Handle missing values and log-scale SNR for better visualization
        df_clean = df.dropna(subset=['snr'])
        
        # Cap the SNR values for color mapping at 15 to better distinguish low SNR values
        df_clean['snr_capped'] = np.minimum(df_clean['snr'], 15.0)
        
        p = figure(title=title, width=self.width, height=self.height,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
        # Hide axes for cleaner UMAP visualization
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        
        # Create color mapper for capped SNR (normal scale, not log)
        color_mapper = LinearColorMapper(
            palette=Viridis256,
            low=df_clean['snr_capped'].min(),
            high=15.0  # Cap at 15
        )
        
        # Use helper method with SNR coloring (using capped values for color mapping)
        self._add_classification_scatter(
            p, df_clean,
            color_field='snr_capped',
            color_mapper=color_mapper
        )
        
        # Add simple color bar with actual SNR values
        color_bar = ColorBar(color_mapper=color_mapper, width=8, location=(0, 0),
                           title="SNR (capped at 15)")
        p.add_layout(color_bar, 'right')
        
        # Create hover tooltip using adaptive builder
        extra_fields = {
            'snr': 'SNR',
            'snr_capped': 'SNR (capped)'
        }
        tooltip_html = self._build_tooltip_html(df_clean, extra_fields=extra_fields)
        
        hover = HoverTool(tooltips=tooltip_html)
        p.add_tools(hover)
        
        # Hide legend for cleaner visualization
        p.legend.visible = False
        
        return p
    
    def plot_classification_results(self, df: pd.DataFrame, 
                                   title: str = "UMAP: Classification Results") -> figure:
        """Plot UMAP with fixed classification colors and symbols for TP/FP/TN/FN."""
        p = figure(title=title, width=self.width, height=self.height,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
        # Hide axes for cleaner UMAP visualization
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
         
        # Create data sources for each class type
        tp_data = ColumnDataSource(df[df['class_type'] == 'True Positive'])
        tn_data = ColumnDataSource(df[df['class_type'] == 'True Negative'])
        fp_data = ColumnDataSource(df[df['class_type'] == 'False Positive'])
        fn_data = ColumnDataSource(df[df['class_type'] == 'False Negative'])
        
        # Add scatter points with fixed colors and symbols
        if len(tp_data.data['umap_x']) > 0:
            p.scatter('umap_x', 'umap_y', source=tp_data, 
                     marker=self.class_symbols['True Positive'],
                     color=self.class_colors['True Positive'],
                     size=8, alpha=0.7, 
                     legend_label="True Positive")
            
        if len(tn_data.data['umap_x']) > 0:
            p.scatter('umap_x', 'umap_y', source=tn_data,
                     marker=self.class_symbols['True Negative'], 
                     color=self.class_colors['True Negative'],
                     size=8, alpha=0.7,
                     legend_label="True Negative")
            
        if len(fp_data.data['umap_x']) > 0:
            p.scatter('umap_x', 'umap_y', source=fp_data,
                     marker=self.class_symbols['False Positive'],
                     color=self.class_colors['False Positive'], 
                     size=8, alpha=0.7,
                     legend_label="False Positive")
            
        if len(fn_data.data['umap_x']) > 0:
            p.scatter('umap_x', 'umap_y', source=fn_data,
                     marker=self.class_symbols['False Negative'],
                     color=self.class_colors['False Negative'],
                     size=8, alpha=0.7, 
                     legend_label="False Negative")
        
        # Create hover tooltip using helper function that adapts to available image columns
        tooltip_html = self._build_tooltip_html(df)
        
        hover = HoverTool(tooltips=tooltip_html)
        p.add_tools(hover)
        
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        
        return p
    
    def plot_uncertainty_vs_snr_scatter(self, df: pd.DataFrame, 
                                        title: str = "Uncertainty vs SNR") -> figure:
        """Plot scatter of uncertainty vs SNR with TP/FP/TN/FN symbols.
        
        This plot helps identify if model uncertainty correlates with signal quality (SNR).
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'snr', 'prediction_uncertainty' (or proxy), and classification data
        title : str
            Title for the plot
            
        Returns
        -------
        figure
            Bokeh figure with uncertainty vs SNR scatter plot
        """
        # Check if SNR data is available
        if 'snr' not in df.columns:
            p = figure(title=f"{title} (SNR data not available)", 
                      width=self.width, height=self.height,
                      tools=['pan', 'wheel_zoom', 'reset', 'save'])
            p.text([self.width/2], [self.height/2], text=["SNR data not available"],
                   text_align="center", text_baseline="middle", text_font_size="16pt")
            return p
        
        # Determine what uncertainty measure to use
        if 'prediction_uncertainty' in df.columns:
            uncertainty_col = 'prediction_uncertainty'
            uncertainty_title = "Model Uncertainty"
            plot_title = f"{title} (Ensemble/CoTeaching)"
        elif 'prediction_probability' in df.columns:
            # Create proxy uncertainty from probability
            df = df.copy()
            df['uncertainty_proxy'] = np.abs(df['prediction_probability'] - 0.5)
            uncertainty_col = 'uncertainty_proxy'
            uncertainty_title = "Distance from Decision Boundary"
            plot_title = f"{title} (Confidence Proxy)"
        else:
            p = figure(title=f"{title} (No uncertainty data)", 
                      width=self.width, height=self.height,
                      tools=['pan', 'wheel_zoom', 'reset', 'save'])
            p.text([self.width/2], [self.height/2], text=["Uncertainty data not available"],
                   text_align="center", text_baseline="middle", text_font_size="16pt")
            return p
        
        # Cap SNR at 15 for better visualization
        df = df.copy()
        df['snr_capped'] = np.minimum(df['snr'], 15.0)
        
        # Create figure with capped x-axis
        p = figure(title=plot_title,
                  x_axis_label="Signal-to-Noise Ratio (capped at 15)",
                  y_axis_label=uncertainty_title,
                  width=self.width, height=self.height,
                  x_range=(0, 15),
                  tools=['pan', 'wheel_zoom', 'reset', 'save', 'box_select'])
        
        # Plot each class type separately with different symbols and colors
        for class_type in ['True Positive', 'True Negative', 'False Positive', 'False Negative']:
            class_data = df[df['class_type'] == class_type]
            if len(class_data) > 0:
                source = ColumnDataSource(class_data)
                p.scatter('snr_capped', uncertainty_col, source=source,
                         color=self.class_colors[class_type],
                         marker=self.class_symbols[class_type],
                         alpha=0.6, size=8,
                         legend_label=class_type)
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("SNR (actual)", "@snr{0.00}"),
            ("Uncertainty", f"@{uncertainty_col}{{0.000}}"),
            ("Class", "@class_type"),
            ("Prediction", "@prediction"),
            ("True Label", "@true_label")
        ])
        p.add_tools(hover)
        
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        
        return p

def create_evaluation_dashboard(metrics: EvaluationMetrics, output_path: Path = None,
                              title: str = "Model Evaluation Dashboard", 
                              snr_threshold: float = 5.0,
                              other_transients: np.ndarray = None,
                              sn_phase_analysis: Dict = None,
                              lc_metrics: Dict = None,
                              include_lc_panels: bool = True) -> Tuple[TabPanel, str]:
    """Create comprehensive evaluation dashboard tab with all standard metrics plots.
    
    Parameters
    ----------
    metrics : EvaluationMetrics
        Evaluation metrics object containing predictions, labels, and optionally probabilities
    output_path : Path, optional
        Path to save the HTML dashboard file (only used when called standalone)
    title : str
        Title for the dashboard
    snr_threshold : float
        SNR threshold to separate low and high SNR samples (default: 5.0)
    other_transients : np.ndarray, optional
        Array indicating transient type (0=SN, 1=other_transient, -1=missing)
        If provided, adds transient type analysis panel
    sn_phase_analysis : Dict, optional
        SN phase analysis results from EvaluationMetrics.get_sn_phase_analysis()
        If provided, adds phase-based accuracy and UQ correlation analysis
    include_lc_panels : bool
        If False, excludes LC-related panels (transient types, SN phase, LC metrics)
        for use when those panels are in a separate tab (default: True)
        
    Returns
    -------
    tuple
        (TabPanel for use in combined dashboard, layout for standalone use)
    """
    # Initialize plotting class
    plots = BokehEvaluationPlots()
    
    # Create individual plots
    confusion_plot = plots.plot_confusion_matrix(metrics, title="Confusion Matrix")
    metrics_div = plots.create_metrics_summary_div(metrics)
    
    # Create SNR-based metric divs if SNR data is available
    low_snr_div, high_snr_div = plots.create_snr_metrics_divs(metrics, snr_threshold)
    
    # Create UQ metrics div if uncertainty data is available
    uq_div = plots.create_uncertainty_metrics_div(metrics)
    
    # Create distribution plot
    pred_dist_plot = plots.plot_prediction_distribution(
        metrics.predictions, 
        metrics.labels, 
        metrics.probabilities,
        title="Prediction Distribution"
    )
    
    plots_layout = [
        row(confusion_plot, metrics_div),
        Spacer(height=20),  # Add spacing between rows
        row(pred_dist_plot, uq_div),
        Spacer(height=20),  # Add spacing between rows
        row(low_snr_div, high_snr_div)
    ]
    
    # Add ROC and PR curves if probabilities are available
    if metrics.probabilities is not None:
        try:
            roc_plot = plots.plot_roc_curve(metrics, title="ROC Curve")
            pr_plot = plots.plot_precision_recall_curve(metrics, title="Precision-Recall Curve")
            plots_layout.append(Spacer(height=40))  # Add extra spacing to prevent overlap
            plots_layout.append(row(roc_plot, pr_plot))
        except Exception as e:
            print(f"Warning: Could not create ROC/PR curves: {e}")
    
    # Only include LC-related panels if requested (when not in separate tab)
    if include_lc_panels:
        # Add transient type analysis panel if other_transients data is provided
        if other_transients is not None:
            transient_types_div = plots.create_transient_types_div(metrics, other_transients)
            plots_layout.append(Spacer(height=30))  # Increased spacing
            plots_layout.append(transient_types_div)
        
        # Add SN phase analysis panel if data is provided
        if sn_phase_analysis is not None and sn_phase_analysis.get('available', False):
            phase_analysis_div = plots.create_sn_phase_analysis_div(sn_phase_analysis)
            plots_layout.append(Spacer(height=35))  # Increased spacing
            plots_layout.append(phase_analysis_div)
        
        # Add lightcurve metrics panel if data is provided
        if lc_metrics is not None:
            lc_metrics_div = plots.create_lightcurve_metrics_div(lc_metrics)
            plots_layout.append(Spacer(height=140))  # Extra spacing to prevent overlap
            plots_layout.append(lc_metrics_div)
    
    # Create final layout
    dashboard_layout = column(*plots_layout)
    
    # Create tab panel for combined dashboard
    tab_panel = TabPanel(child=dashboard_layout, title="Model Metrics")
    
    # Save to file if output path provided (standalone mode)
    if output_path:
        output_path = Path(output_path)
        output_file(str(output_path), title=title)
        save(dashboard_layout)
        print(f"Evaluation dashboard saved to: {output_path}")
        return tab_panel, str(output_path)
    else:
        return tab_panel, dashboard_layout

def create_interpretability_dashboard(interpreter, data_loader, predictions: np.ndarray = None, 
                                    labels: np.ndarray = None, output_path: Path = None,
                                    additional_features: Dict = None, config: Dict = None,
                                    probabilities: np.ndarray = None, uncertainties: np.ndarray = None,
                                    show_all_cutouts: bool = False) -> Tuple[TabPanel, str]:
    """Create UMAP-based interpretability dashboard tab.
    
    Parameters
    ----------
    interpreter : UMAPInterpreter
        Initialized UMAP interpreter
    data_loader : DataLoader
        Data loader for feature extraction and image embedding
    predictions : np.ndarray, optional
        Model predictions (if None, will be computed from data_loader via inference)
    labels : np.ndarray, optional
        True labels (if None, will be extracted from data_loader)
    output_path : Path, optional
        Path to save the HTML dashboard file (only used when called standalone)
    additional_features : Dict, optional
        Additional features to include in the analysis
    config : Dict, optional
        Configuration dictionary with interpretability settings
    probabilities : np.ndarray, optional
        Pre-computed prediction probabilities 
    uncertainties : np.ndarray, optional
        Pre-computed prediction uncertainties
    show_all_cutouts : bool, optional
        If True, show all cutout types (diff, coadd, etc.) in hover tooltip.
        If False, show only the first channel (default, backward compatible).
        
    Returns
    -------
    tuple
        (TabPanel for use in combined dashboard, layout for standalone use)
    """
    import time
    
    # Use default config if none provided
    if config is None:
        config = {}
    
    interp_config = config.get('interpretability', {})
    umap_config = interp_config.get('umap', {})
    clustering_config = interp_config.get('clustering', {})
    
    print("Step 1: Extracting features and predictions (single pass)...")
    start_time = time.time()
    
    # Extract features (this processes all batches once)
    features = interpreter.extract_features(
        data_loader,
        layer_name=interp_config.get('layer_name', 'fc1'),
        max_samples=interp_config.get('max_samples', 3000)
    )
    
    sample_indices_used = interpreter.sample_indices  # Indices into data_loader
    print(f"Feature extraction used {len(sample_indices_used)} samples from data_loader")
    
    # Create a subset data loader with only the sampled indices
    print(f"Running inference on {len(sample_indices_used)} sampled indices (avoiding duplicate processing)...")
    inference_start = time.time()
    
    # Import required modules
    from ML4transients.evaluation.inference import infer
    from torch.utils.data import Subset, DataLoader
    
    # Create subset dataset with only sampled indices
    subset_dataset = Subset(data_loader.dataset, sample_indices_used)
    subset_loader = DataLoader(
        subset_dataset,
        batch_size=data_loader.batch_size,
        shuffle=False,  # Keep order for alignment
        num_workers=0
    )
    
    print(f"  Processing {len(subset_loader)} batches ...")
    inference_results = infer(
        inference_loader=subset_loader,
        trainer=interpreter.trainer,
        return_preds=True,
        compute_metrics=False,
        device=interpreter.device,
        return_probabilities=True
    )
    
    # Extract results 
    sampled_predictions = inference_results['y_pred']
    sampled_labels = inference_results['y_true']
    sampled_probabilities = inference_results.get('y_prob', None)
    sampled_uncertainties = inference_results.get('uncertainty', None)
    
    print(f"Inference completed in {time.time() - inference_start:.2f}s")
    print(f"Perfect alignment achieved: {len(sampled_predictions)} samples")
    print(f"  - Features shape: {features.shape}")
    print(f"  - Predictions: {len(sampled_predictions)}")
    print(f"  - Labels: {len(sampled_labels)}")
    if sampled_probabilities is not None:
        print(f"  - Probabilities: {len(sampled_probabilities)}")
    if sampled_uncertainties is not None:
        print(f"  - Uncertainties: {len(sampled_uncertainties)}")
    
    if hasattr(interpreter, 'umap_reducer') and interpreter.umap_reducer is not None:
        print("Transforming features with loaded UMAP reducer...")
        embedding = interpreter.transform_with_umap(features)
        interpreter.umap_embedding = embedding
    else:
        print("Fitting UMAP embedding...")
        embedding = interpreter.fit_umap(
            n_neighbors=umap_config.get('n_neighbors', 15),
            min_dist=umap_config.get('min_dist', 0.1),
            optimize_params=umap_config.get('optimize_params', False)
        )
    print(f"Feature extraction and UMAP fitting completed in {time.time() - start_time:.2f}s")
    
    # Optional clustering
    if clustering_config.get('enabled', False):
        print("Step 3: Performing high-dimensional clustering...")
        start_time = time.time()
        clusters = interpreter.cluster_high_dimensional_features(
            min_cluster_size=clustering_config.get('min_cluster_size', 10),
            min_samples=clustering_config.get('min_samples', 5)
        )
        print(f"Clustering completed in {time.time() - start_time:.2f}s")
    
    # Create interpretability dataframe
    print("Step 4: Creating interpretability dataframe...")
    start_time = time.time()
    df = interpreter.create_interpretability_dataframe(
        sampled_predictions, sampled_labels, data_loader,
        additional_features=additional_features,
        probabilities=sampled_probabilities,  # Use sampled values
        uncertainties=sampled_uncertainties,   # Use sampled values
        separate_channel_images=show_all_cutouts  # Create separate image columns if requested
    )
    print(f"Dataframe creation completed in {time.time() - start_time:.2f}s")
    
    # Initialize visualizer
    visualizer = UMAPVisualizer()
    
    # Create plots
    print("Step 5: Creating UMAP visualizations...")
    start_time = time.time()
    
    plots_list = []
    
    basic_plot = visualizer.plot_classification_results(
        df, 
        title="UMAP: Classification Results"
    )
    plots_list.append(basic_plot)
    
    # Uncertainty plot (for all model types now)
    try:
        uncertainty_plot = visualizer.plot_uncertainty_coloring(df)
        plots_list.append(uncertainty_plot)
    except Exception as e:
        print(f"Warning: Could not create uncertainty plot: {e}")
    
    # SNR plot (if SNR data is available)
    try:
        snr_plot = visualizer.plot_snr_coloring(df)
        plots_list.append(snr_plot)
    except Exception as e:
        print(f"Warning: Could not create SNR plot: {e}")
    
    # UQ vs SNR scatter plot (if both SNR and uncertainty data available)
    try:
        if 'snr' in df.columns and ('prediction_uncertainty' in df.columns or 'prediction_probability' in df.columns):
            uq_snr_scatter = visualizer.plot_uncertainty_vs_snr_scatter(df, title="Uncertainty vs SNR")
            plots_list.append(uq_snr_scatter)
    except Exception as e:
        print(f"Warning: Could not create UQ vs SNR scatter plot: {e}")
    
    # Clustering plot if available
    if 'high_dim_cluster' in df.columns:
        try:
            cluster_plot = visualizer.plot_clusters(df, 'high_dim_cluster', title="UMAP: High-Dim Clusters")
            plots_list.append(cluster_plot)
        except Exception as e:
            print(f"Warning: Could not create cluster plot: {e}")
    
    # Additional feature plots (excluding SNR which has its own dedicated plot)
    if additional_features:
        for feature_name in additional_features.keys():
            if feature_name in df.columns and feature_name != 'snr':  # Skip SNR - it has its own plot
                try:
                    feature_plot = visualizer.plot_feature_coloring(
                        df, feature_name, title=f"UMAP: {feature_name}"
                    )
                    plots_list.append(feature_plot)
                except Exception as e:
                    print(f"Warning: Could not create plot for {feature_name}: {e}")
    
    print(f"Created {len(plots_list)} UMAP visualizations in {time.time() - start_time:.2f}s")
    
    # Create dashboard layout
    if len(plots_list) <= 2:
        dashboard_layout = column(*plots_list)
    elif len(plots_list) <= 4:
        # Arrange in 2x2 grid, but only include existing plots
        first_row = [plots_list[0]]
        if len(plots_list) > 1:
            first_row.append(plots_list[1])
        
        second_row = []
        if len(plots_list) > 2:
            second_row.append(plots_list[2])
        if len(plots_list) > 3:
            second_row.append(plots_list[3])
        
        rows = [row(*first_row)]
        if second_row:
            rows.append(row(*second_row))
        dashboard_layout = column(*rows)
    else:
        # Arrange in rows of 2
        rows = []
        for i in range(0, len(plots_list), 2):
            row_plots = [plots_list[i]]
            if i + 1 < len(plots_list):
                row_plots.append(plots_list[i + 1])
            rows.append(row(*row_plots))
        dashboard_layout = column(*rows)
    
    # Create tab panel for combined dashboard
    tab_panel = TabPanel(child=dashboard_layout, title="Model Interpretability")
    
    # Save to file if output path provided (standalone mode)
    if output_path:
        output_path = Path(output_path)
        output_file(str(output_path), title="Model Interpretability Dashboard")
        save(dashboard_layout)
        print(f"Interpretability dashboard saved to: {output_path}")
        return tab_panel, str(output_path)
    else:
        return tab_panel, dashboard_layout

def create_lightcurve_analysis_tab(metrics: EvaluationMetrics = None,
                                   other_transients: np.ndarray = None,
                                   sn_phase_analysis: Dict = None,
                                   lc_metrics: Dict = None,
                                   output_path: Path = None) -> Tuple[TabPanel, str]:
    """Create lightcurve analysis tab with all LC-related visualizations.
    
    Parameters
    ----------
    metrics : EvaluationMetrics, optional
        Evaluation metrics object (needed for transient type analysis)
    other_transients : np.ndarray, optional
        Array indicating transient type for the transient type analysis panel
    sn_phase_analysis : Dict, optional
        SN phase analysis results from EvaluationMetrics.get_sn_phase_analysis()
    lc_metrics : Dict, optional
        Lightcurve metrics from EvaluationMetrics.get_lightcurve_metrics()
    output_path : Path, optional
        Path to save the HTML dashboard file (only used when called standalone)
        
    Returns
    -------
    tuple
        (TabPanel for use in combined dashboard, layout for standalone use)
    """
    # Initialize plotting class
    plots = BokehEvaluationPlots()
    
    panels_layout = []
    
    # Add transient type analysis panel if other_transients data is provided
    if other_transients is not None and metrics is not None:
        transient_types_div = plots.create_transient_types_div(metrics, other_transients)
        panels_layout.append(transient_types_div)
        panels_layout.append(Spacer(height=30))
    
    # Prepare LC metrics and SN phase analysis for side-by-side display
    if lc_metrics is not None and sn_phase_analysis is not None and sn_phase_analysis.get('available', False):
        # Display LC metrics and SN phase analysis side by side
        lc_metrics_div = plots.create_lightcurve_metrics_div(lc_metrics)
        phase_analysis_div = plots.create_sn_phase_analysis_div(sn_phase_analysis)
        panels_layout.append(row(lc_metrics_div, phase_analysis_div))
        panels_layout.append(Spacer(height=30))
    elif lc_metrics is not None:
        # Only LC metrics
        lc_metrics_div = plots.create_lightcurve_metrics_div(lc_metrics)
        panels_layout.append(lc_metrics_div)
        panels_layout.append(Spacer(height=30))
    elif sn_phase_analysis is not None and sn_phase_analysis.get('available', False):
        # Only SN phase analysis
        phase_analysis_div = plots.create_sn_phase_analysis_div(sn_phase_analysis)
        panels_layout.append(phase_analysis_div)
        panels_layout.append(Spacer(height=30))
    
    # Add distribution plots if LC metrics data is available
    if lc_metrics is not None and 'distributions' in lc_metrics:
        dist_all = plots.plot_lightcurve_detection_distributions(
            lc_metrics, 
            title="Detection Distribution (All LCs)"
        )
        dist_transient = plots.plot_transient_lc_detection_distribution(
            lc_metrics,
            title="Detection in Real Transient LCs"
        )
        dist_bogus = plots.plot_bogus_lc_rejection_distribution(
            lc_metrics,
            title="Rejection in Bogus LCs"
        )
        
     
        panels_layout.append(Spacer(height=300))  
        panels_layout.append(row(dist_all, dist_transient, dist_bogus))
    
    # Create final layout
    if len(panels_layout) == 0:
        # No data available
        panels_layout.append(Div(text="<h3>No lightcurve analysis data available</h3>", 
                                width=800, height=100))
    
    dashboard_layout = column(*panels_layout)
    
    # Create tab panel for combined dashboard
    tab_panel = TabPanel(child=dashboard_layout, title="Lightcurve Analysis")
    
    # Save to file if output path provided (standalone mode)
    if output_path:
        output_path = Path(output_path)
        output_file(str(output_path), title="Lightcurve Analysis Dashboard")
        save(dashboard_layout)
        print(f"Lightcurve analysis dashboard saved to: {output_path}")
        return tab_panel, str(output_path)
    else:
        return tab_panel, dashboard_layout

def create_combined_dashboard(metrics: EvaluationMetrics, 
                            interpreter=None, data_loader=None, predictions=None, labels=None,
                            output_path: Path = None, additional_features: Dict = None, 
                            config: Dict = None, title: str = "Model Evaluation Dashboard",
                            probabilities: np.ndarray = None, uncertainties: np.ndarray = None,
                            snr_threshold: float = 5.0, show_all_cutouts: bool = False,
                            other_transients: np.ndarray = None,
                            sn_phase_analysis: Dict = None,
                            lc_metrics: Dict = None) -> str:
    """Create combined dashboard with both evaluation metrics and interpretability in tabs.
    
    Parameters
    ----------
    metrics : EvaluationMetrics
        Evaluation metrics object
    interpreter : UMAPInterpreter, optional
        UMAP interpreter for interpretability analysis
    data_loader : DataLoader, optional
        Data loader for interpretability analysis
    predictions : np.ndarray, optional
        Model predictions for interpretability
    labels : np.ndarray, optional  
        True labels for interpretability
    output_path : Path
        Path to save the combined HTML dashboard
    additional_features : Dict, optional
        Additional features for interpretability
    config : Dict, optional
        Configuration for interpretability
    title : str
        Title for the dashboard
    probabilities : np.ndarray, optional
        Pre-computed prediction probabilities (avoids recomputation)
    uncertainties : np.ndarray, optional
        Pre-computed prediction uncertainties (avoids recomputation)
    snr_threshold : float
        SNR threshold to separate low and high SNR samples (default: 5.0)
    show_all_cutouts : bool
        If True, show all cutout types (diff, coadd, etc.) in hover tooltips (default: False)
    other_transients : np.ndarray, optional
        Array indicating transient type (0=SN, 1=other_transient) for subclass analysis
    sn_phase_analysis : Dict, optional
        SN phase analysis results from EvaluationMetrics.get_sn_phase_analysis()
        
    Returns
    -------
    str
        Path to the saved HTML file
    """
    tabs = []
    
    # Check if we have LC-related data for a separate tab
    has_lc_data = (other_transients is not None or 
                   (sn_phase_analysis is not None and sn_phase_analysis.get('available', False)) or
                   lc_metrics is not None)
    
    # Always create metrics tab (exclude LC panels if they'll be in separate tab)
    print("Creating metrics dashboard tab...")
    metrics_tab, _ = create_evaluation_dashboard(
        metrics, title="Model Metrics", snr_threshold=snr_threshold,
        other_transients=other_transients if not has_lc_data else None,
        sn_phase_analysis=sn_phase_analysis if not has_lc_data else None,
        lc_metrics=lc_metrics if not has_lc_data else None,
        include_lc_panels=not has_lc_data  # Exclude if separate tab will be created
    )
    tabs.append(metrics_tab)
    
    # Create interpretability tab if data is available
    if interpreter is not None and data_loader is not None and predictions is not None and labels is not None:
        print("Creating interpretability dashboard tab...")
        try:
            interp_tab, _ = create_interpretability_dashboard(
                interpreter, data_loader, predictions, labels,
                additional_features=additional_features, config=config,
                probabilities=probabilities,  # Pass pre-computed values
                uncertainties=uncertainties,   # Pass pre-computed values
                show_all_cutouts=show_all_cutouts  # Pass cutout display preference
            )
            tabs.append(interp_tab)
        except Exception as e:
            print(f"Warning: Could not create interpretability tab: {e}")
            print("Proceeding with metrics tab only...")
    
    # Create lightcurve analysis tab if LC-related data is available
    if has_lc_data:
        print("Creating lightcurve analysis tab...")
        try:
            lc_tab, _ = create_lightcurve_analysis_tab(
                metrics=metrics,
                other_transients=other_transients,
                sn_phase_analysis=sn_phase_analysis,
                lc_metrics=lc_metrics
            )
            tabs.append(lc_tab)
        except Exception as e:
            print(f"Warning: Could not create lightcurve analysis tab: {e}")
            print("Proceeding without lightcurve analysis tab...")
    
    # Create tabbed layout
    tabbed_layout = Tabs(tabs=tabs)
    
    # Save to file
    if output_path:
        output_path = Path(output_path)
        output_file(str(output_path), title=title)
        save(tabbed_layout)
        print(f"Combined dashboard saved to: {output_path}")
        return str(output_path)
    else:
        return tabbed_layout