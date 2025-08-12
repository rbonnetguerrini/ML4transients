import numpy as np
import pandas as pd
from bokeh.plotting import figure, save, output_file
from bokeh.models import (
    HoverTool, ColumnDataSource, ColorBar, LinearColorMapper,
    Title, Div, CustomJS, Tabs, TabPanel, CategoricalColorMapper
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
        """Create ROC curve plot."""
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
        
        hover = HoverTool(tooltips=[("FPR", "@fpr{0.000}"), ("TPR", "@tpr{0.000}")])
        p.add_tools(hover)
        
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
        
        # Create hover tooltip
        tooltip_html = f"""
        <div>
            <div>
                <img 
                    src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #224499'>{feature_column}:</span>
                <span style='font-size: 14px'>@{feature_column}</span><br>
                <span style='font-size: 14px; color: #224499'>Class type:</span>
                <span style='font-size: 14px'>@class_type</span><br>
                <span style='font-size: 14px; color: #224499'>True label:</span>
                <span style='font-size: 14px'>@true_label</span><br>
                <span style='font-size: 14px; color: #224499'>Prediction:</span>
                <span style='font-size: 14px'>@prediction</span><br>
            </div>
        </div>
        """
        
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
        
        # Create hover tooltip
        tooltip_html = f"""
        <div>
            <div>
                <img 
                    src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #224499'>Cluster:</span>
                <span style='font-size: 14px'>@{cluster_column}</span><br>
                <span style='font-size: 14px; color: #224499'>Class type:</span>
                <span style='font-size: 14px'>@class_type</span><br>
                <span style='font-size: 14px; color: #224499'>True label:</span>
                <span style='font-size: 14px'>@true_label</span><br>
                <span style='font-size: 14px; color: #224499'>Prediction:</span>
                <span style='font-size: 14px'>@prediction</span><br>
            </div>
        </div>
        """
        
        hover = HoverTool(tooltips=tooltip_html)
        p.add_tools(hover)
        
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        
        return p
    
    def plot_uncertainty_vs_correctness(self, df: pd.DataFrame,
                                      title: str = "UMAP: Uncertainty vs Correctness") -> figure:
        """Plot UMAP with uncertainty and correctness combined with TP/FP/TN/FN symbols.
        
        Works with all model types:
        - Ensemble/CoTeaching: Uses true prediction uncertainty
        - Standard models: Uses distance from decision boundary (|prob - 0.5|) as proxy
        """
        # Determine what uncertainty measure to use
        if 'prediction_uncertainty' in df.columns:
            uncertainty_col = 'prediction_uncertainty'
            uncertainty_title = "Prediction Uncertainty"
        elif 'prediction_probability' in df.columns:
            # For standard models, use distance from decision boundary as uncertainty proxy
            df = df.copy()
            df['uncertainty_proxy'] = np.abs(df['prediction_probability'] - 0.5)
            uncertainty_col = 'uncertainty_proxy'
            uncertainty_title = "Uncertainty Proxy (|prob - 0.5|)"
        else:
            raise ValueError("No uncertainty data or prediction probabilities available.")
        
        p = figure(title=title, width=self.width, height=self.height,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
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
        
        # Combined hover tooltip
        tooltip_html = f"""
        <div>
            <div>
                <img 
                    src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #224499'>Correct:</span>
                <span style='font-size: 14px'>@correct</span><br>
                <span style='font-size: 14px; color: #224499'>Class type:</span>
                <span style='font-size: 14px'>@class_type</span><br>
                <span style='font-size: 14px; color: #224499'>{uncertainty_title}:</span>
                <span style='font-size: 14px'>@{uncertainty_col}{{0.000}}</span><br>
        """
        
        if 'prediction_probability' in df.columns:
            tooltip_html += """
                <span style='font-size: 14px; color: #224499'>Probability:</span>
                <span style='font-size: 14px'>@prediction_probability{0.000}</span><br>
            """
        
        tooltip_html += """
            </div>
        </div>
        """
        
        hover = HoverTool(tooltips=tooltip_html)
        p.add_tools(hover)
        
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        
        return p
    
    def plot_uncertainty_coloring(self, df: pd.DataFrame, 
                                title: str = "UMAP: Prediction Uncertainty") -> figure:
        """Plot UMAP colored by prediction uncertainty with TP/FP/TN/FN symbols.
        
        Works with all model types using uncertainty proxy for standard models.
        """
        # Determine what uncertainty measure to use
        if 'prediction_uncertainty' in df.columns:
            uncertainty_col = 'prediction_uncertainty'
            uncertainty_title = "Prediction Uncertainty"
        elif 'prediction_probability' in df.columns:
            # For standard models, use distance from decision boundary as uncertainty proxy
            df = df.copy()
            df['uncertainty_proxy'] = np.abs(df['prediction_probability'] - 0.5)
            uncertainty_col = 'uncertainty_proxy'
            uncertainty_title = "Uncertainty Proxy (|prob - 0.5|)"
            title = "UMAP: Uncertainty Proxy (Distance from Decision Boundary)"
        else:
            raise ValueError("No uncertainty data or prediction probabilities available.")
        
        # Handle missing values
        df_clean = df.dropna(subset=[uncertainty_col])
        
        p = figure(title=title, width=self.width, height=self.height,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
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
        
        # Create hover tooltip
        tooltip_html = f"""
        <div>
            <div>
                <img 
                    src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #224499'>{uncertainty_title}:</span>
                <span style='font-size: 14px'>@{uncertainty_col}{{0.000}}</span><br>
                <span style='font-size: 14px; color: #224499'>Class type:</span>
                <span style='font-size: 14px'>@class_type</span><br>
        """
        
        if 'prediction_probability' in df.columns:
            tooltip_html += """
                <span style='font-size: 14px; color: #224499'>Probability:</span>
                <span style='font-size: 14px'>@prediction_probability{0.000}</span><br>
            """
        
        tooltip_html += """
                <span style='font-size: 14px; color: #224499'>True label:</span>
                <span style='font-size: 14px'>@true_label</span><br>
                <span style='font-size: 14px; color: #224499'>Prediction:</span>
                <span style='font-size: 14px'>@prediction</span><br>
            </div>
        </div>
        """
        
        hover = HoverTool(tooltips=tooltip_html)
        p.add_tools(hover)
        
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        
        return p
    
    def plot_classification_results(self, df: pd.DataFrame, 
                                   title: str = "UMAP: Classification Results") -> figure:
        """Plot UMAP with fixed classification colors and symbols for TP/FP/TN/FN."""
        p = figure(title=title, width=self.width, height=self.height,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
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
        
        # Create hover tooltip
        tooltip_html = """
        <div>
            <div>
                <img 
                    src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #224499'>Class type:</span>
                <span style='font-size: 14px'>@class_type</span><br>
                <span style='font-size: 14px; color: #224499'>True label:</span>
                <span style='font-size: 14px'>@true_label</span><br>
                <span style='font-size: 14px; color: #224499'>Prediction:</span>
                <span style='font-size: 14px'>@prediction</span><br>
            </div>
        </div>
        """
        
        hover = HoverTool(tooltips=tooltip_html)
        p.add_tools(hover)
        
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        
        return p

def create_evaluation_dashboard(metrics: EvaluationMetrics, output_path: Path = None,
                              title: str = "Model Evaluation Dashboard") -> Tuple[TabPanel, str]:
    """Create comprehensive evaluation dashboard tab with all standard metrics plots.
    
    Parameters
    ----------
    metrics : EvaluationMetrics
        Evaluation metrics object containing predictions, labels, and optionally probabilities
    output_path : Path, optional
        Path to save the HTML dashboard file (only used when called standalone)
    title : str
        Title for the dashboard
        
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
    
    # Create distribution plot
    pred_dist_plot = plots.plot_prediction_distribution(
        metrics.predictions, 
        metrics.labels, 
        metrics.probabilities,
        title="Prediction Distribution"
    )
    
    plots_layout = [
        row(confusion_plot, metrics_div),
        row(pred_dist_plot)
    ]
    
    # Add ROC and PR curves if probabilities are available
    if metrics.probabilities is not None:
        try:
            roc_plot = plots.plot_roc_curve(metrics, title="ROC Curve")
            pr_plot = plots.plot_precision_recall_curve(metrics, title="Precision-Recall Curve")
            plots_layout.append(row(roc_plot, pr_plot))
        except Exception as e:
            print(f"Warning: Could not create ROC/PR curves: {e}")
    
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

def create_interpretability_dashboard(interpreter, data_loader, predictions: np.ndarray, 
                                    labels: np.ndarray, output_path: Path = None,
                                    additional_features: Dict = None, config: Dict = None,
                                    probabilities: np.ndarray = None, uncertainties: np.ndarray = None) -> Tuple[TabPanel, str]:
    """Create UMAP-based interpretability dashboard tab.
    
    Parameters
    ----------
    interpreter : UMAPInterpreter
        Initialized UMAP interpreter
    data_loader : DataLoader
        Data loader for feature extraction and image embedding
    predictions : np.ndarray
        Model predictions
    labels : np.ndarray
        True labels
    output_path : Path, optional
        Path to save the HTML dashboard file (only used when called standalone)
    additional_features : Dict, optional
        Additional features to include in the analysis
    config : Dict, optional
        Configuration dictionary with interpretability settings
    probabilities : np.ndarray, optional
        Pre-computed prediction probabilities (avoids recomputation)
    uncertainties : np.ndarray, optional
        Pre-computed prediction uncertainties (avoids recomputation)
        
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
    
    # Extract features
    print("Step 1: Extracting features...")
    start_time = time.time()
    features = interpreter.extract_features(
        data_loader,
        layer_name=interp_config.get('layer_name', 'fc1'),
        max_samples=interp_config.get('max_samples', 3000)
    )
    print(f"Feature extraction completed in {time.time() - start_time:.2f}s")
    
    # Fit UMAP
    print("Step 2: Fitting UMAP embedding...")
    start_time = time.time()
    embedding = interpreter.fit_umap(
        n_neighbors=umap_config.get('n_neighbors', 15),
        min_dist=umap_config.get('min_dist', 0.1),
        optimize_params=umap_config.get('optimize_params', False)
    )
    print(f"UMAP fitting completed in {time.time() - start_time:.2f}s")
    
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
        predictions, labels, data_loader, 
        additional_features=additional_features,
        probabilities=probabilities,  # Pass pre-computed values
        uncertainties=uncertainties   # Pass pre-computed values
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
    
    # Uncertainty vs correctness plot (for all model types now)
    try:
        uncertainty_correctness_plot = visualizer.plot_uncertainty_vs_correctness(df)
        plots_list.append(uncertainty_correctness_plot)
    except Exception as e:
        print(f"Warning: Could not create uncertainty vs correctness plot: {e}")
    
    # Clustering plot if available
    if 'high_dim_cluster' in df.columns:
        try:
            cluster_plot = visualizer.plot_clusters(df, 'high_dim_cluster', title="UMAP: High-Dim Clusters")
            plots_list.append(cluster_plot)
        except Exception as e:
            print(f"Warning: Could not create cluster plot: {e}")
    
    # Additional feature plots
    if additional_features:
        for feature_name in additional_features.keys():
            if feature_name in df.columns:
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

def create_combined_dashboard(metrics: EvaluationMetrics, 
                            interpreter=None, data_loader=None, predictions=None, labels=None,
                            output_path: Path = None, additional_features: Dict = None, 
                            config: Dict = None, title: str = "Model Evaluation Dashboard",
                            probabilities: np.ndarray = None, uncertainties: np.ndarray = None) -> str:
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
        
    Returns
    -------
    str
        Path to the saved HTML file
    """
    tabs = []
    
    # Always create metrics tab
    print("Creating metrics dashboard tab...")
    metrics_tab, _ = create_evaluation_dashboard(metrics, title="Model Metrics")
    tabs.append(metrics_tab)
    
    # Create interpretability tab if data is available
    if interpreter is not None and data_loader is not None and predictions is not None and labels is not None:
        print("Creating interpretability dashboard tab...")
        try:
            interp_tab, _ = create_interpretability_dashboard(
                interpreter, data_loader, predictions, labels,
                additional_features=additional_features, config=config,
                probabilities=probabilities,  # Pass pre-computed values
                uncertainties=uncertainties   # Pass pre-computed values
            )
            tabs.append(interp_tab)
        except Exception as e:
            print(f"Warning: Could not create interpretability tab: {e}")
            print("Proceeding with metrics tab only...")
    
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