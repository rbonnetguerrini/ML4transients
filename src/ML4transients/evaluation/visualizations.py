import numpy as np
import pandas as pd
from bokeh.plotting import figure, save, output_file
from bokeh.models import (
    HoverTool, ColumnDataSource, ColorBar, LinearColorMapper,
    Title, Div, CustomJS, Tabs, TabPanel
)
from bokeh.layouts import column, row, gridplot
from bokeh.palettes import Spectral11, Category10, Viridis256
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.io import curdoc
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
                   'Sensitivity', 'Specificity', 'Precision', 'NPV', 'Accuracy'],
            counts=[stats['true_positive'], stats['false_negative'], 
                   stats['false_positive'], stats['true_negative'],
                   None, None, None, None, None],
            percentages=[tp_pct, fn_pct, fp_pct, tn_pct, 
                        sensitivity_pct, specificity_pct, precision_pct, npv_pct, accuracy_pct],
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
    
    def plot_classification_results(self, df: pd.DataFrame, 
                              title: str = "UMAP: Classification Results") -> figure:
        """Plot UMAP colored by classification results."""
        # Create data sources for each class
        tp_data = ColumnDataSource(df[df['class_type'] == 'True Positive'])
        tn_data = ColumnDataSource(df[df['class_type'] == 'True Negative'])
        fp_data = ColumnDataSource(df[df['class_type'] == 'False Positive'])
        fn_data = ColumnDataSource(df[df['class_type'] == 'False Negative'])
        
        p = figure(title=title, width=self.width, height=self.height,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
        # Add points with different markers for each class using scatter()
        p.scatter('umap_x', 'umap_y', source=tp_data, color='#FFC300', marker='x',
                 size=8, alpha=0.7, legend_label="True Positive")
        p.scatter('umap_x', 'umap_y', source=tn_data, color='#C70039', marker='circle',
                 size=8, alpha=0.7, legend_label="True Negative")
        p.scatter('umap_x', 'umap_y', source=fp_data, color='#FF5733', marker='star',
                 size=8, alpha=0.7, legend_label="False Positive")
        p.scatter('umap_x', 'umap_y', source=fn_data, color='#b3b6b7', marker='square',
                 size=8, alpha=0.7, legend_label="False Negative")
        
        # Create hover tooltip HTML that exactly matches your working notebook format
        tooltip_html = """
        <div>
            <div>
                <img 
                    src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #224499'>Predicted class:</span>
                <span style='font-size: 14px'>@prediction</span><br>
                <span style='font-size: 14px; color: #224499'>True label:</span>
                <span style='font-size: 14px'>@true_label</span><br>
                <span style='font-size: 14px; color: #224499'>Class type:</span>
                <span style='font-size: 14px'>@class_type</span><br>
        """
        
        # Add conditional tooltips based on available columns
        if 'diaObjectId' in df.columns:
            tooltip_html += """
                <span style='font-size: 14px; color: #224499'>Obj id:</span>
                <span style='font-size: 14px'>@diaObjectId</span><br>
            """
        
        if 'nDiaSources' in df.columns:
            tooltip_html += """
                <span style='font-size: 14px; color: #224499'>Nbr src in LC:</span>
                <span style='font-size: 14px'>@nDiaSources</span><br>
            """
        
        tooltip_html += """
            </div>
        </div>
        """
        
        # Add hover tool with the exact HTML format from your working notebook
        hover = HoverTool(tooltips=tooltip_html)
        p.add_tools(hover)
        
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        
        return p
    
    def plot_feature_coloring(self, df: pd.DataFrame, feature_column: str,
                        title: str = None, palette: str = "Viridis256") -> figure:
        """Plot UMAP colored by a continuous feature."""
        if title is None:
            title = f"UMAP: {feature_column}"
        
        # Handle missing values
        df_clean = df.dropna(subset=[feature_column])
        source = ColumnDataSource(df_clean)
        
        p = figure(title=title, width=self.width, height=self.height,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
        # Create color mapper
        color_mapper = LinearColorMapper(
            palette=palette,
            low=df_clean[feature_column].min(),
            high=df_clean[feature_column].max()
        )
        
        p.scatter('umap_x', 'umap_y', source=source, size=8, alpha=0.7,
                 color={'field': feature_column, 'transform': color_mapper})
        
        # Add color bar
        color_bar = ColorBar(color_mapper=color_mapper, width=8, location=(0, 0))
        p.add_layout(color_bar, 'right')
        
        # Create hover tooltip HTML that matches your working format
        tooltip_html = f"""
        <div>
            <div>
                <img 
                    src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #224499'>{feature_column}:</span>
                <span style='font-size: 14px'>@{feature_column}</span><br>
                <span style='font-size: 14px; color: #224499'>True label:</span>
                <span style='font-size: 14px'>@true_label</span><br>
                <span style='font-size: 14px; color: #224499'>Prediction:</span>
                <span style='font-size: 14px'>@prediction</span><br>
            </div>
        </div>
        """
        
        hover = HoverTool(tooltips=tooltip_html)
        p.add_tools(hover)
        
        return p
    
    def plot_clusters(self, df: pd.DataFrame, cluster_column: str = 'cluster',
                 title: str = "UMAP: Clusters") -> figure:
        """Plot UMAP colored by cluster labels."""
        source = ColumnDataSource(df)
        
        p = figure(title=title, width=self.width, height=self.height,
                  tools=['pan', 'wheel_zoom', 'reset', 'save'])
        
        # Get unique clusters and ensure we have enough colors
        clusters = sorted(df[cluster_column].unique())
        n_clusters = len(clusters)
        
        # Choose appropriate palette based on number of clusters
        if n_clusters <= 10:
            palette = Category10[max(3, n_clusters)]  # Category10 needs at least 3 colors
        else:
            # Use a larger palette for more clusters
            from bokeh.palettes import turbo
            palette = turbo(n_clusters)
        
        p.scatter('umap_x', 'umap_y', source=source, size=8, alpha=0.7,
                 color=factor_cmap(cluster_column, palette, clusters),
                 legend_group=cluster_column)
        
        # Create hover tooltip HTML that matches your working format
        tooltip_html = f"""
        <div>
            <div>
                <img 
                    src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #224499'>Cluster:</span>
                <span style='font-size: 14px'>@{cluster_column}</span><br>
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

# Dashboard creation functions
def create_evaluation_dashboard(metrics: EvaluationMetrics,
                              output_path: Optional[Path] = None,
                              title: str = "Model Evaluation Dashboard") -> Tabs:
    """Assemble interactive evaluation dashboard.

    Parameters
    ----------
    metrics : EvaluationMetrics
        Metrics object (with or without probability scores).
    output_path : Path, optional
        If provided, writes HTML file.
    title : str
        Dashboard title.

    Returns
    -------
    Tabs
        Bokeh Tabs object.
    """
    plotter = BokehEvaluationPlots()
    
    # Tab 1: Overview with confusion matrix and summary
    confusion_plot = plotter.plot_confusion_matrix(metrics)
    summary_div = plotter.create_metrics_summary_div(metrics)
    
    if metrics.probabilities is not None:
        dist_plot = plotter.plot_prediction_distribution(
            metrics.predictions, metrics.labels, metrics.probabilities
        )
        overview_layout = column(
            row(confusion_plot, summary_div),
            dist_plot
        )
    else:
        overview_layout = row(confusion_plot, summary_div)
    
    overview_tab = TabPanel(child=overview_layout, title="Overview")
    
    tabs = [overview_tab]
    
    # Tab 2: ROC and PR curves (if probabilities available)
    if metrics.probabilities is not None:
        roc_plot = plotter.plot_roc_curve(metrics)
        pr_plot = plotter.plot_precision_recall_curve(metrics)
        curves_layout = row(roc_plot, pr_plot)
        curves_tab = TabPanel(child=curves_layout, title="ROC & PR Curves")
        tabs.append(curves_tab)
    
    dashboard = Tabs(tabs=tabs)
    
    if output_path:
        output_file(str(output_path))
        save(dashboard)
        print(f"Dashboard saved to {output_path}")
    
    return dashboard

def create_interpretability_dashboard(interpreter,
                                    data_loader,
                                    predictions: np.ndarray,
                                    labels: np.ndarray,
                                    output_path: Optional[Path] = None,
                                    additional_features: Optional[Dict[str, np.ndarray]] = None,
                                    config: Optional[Dict] = None) -> Tabs:
    """Create UMAP-based interpretability dashboard.

    Notes
    -----
    Separates sample count for:
      - UMAP computation (max_samples)
      - Visualization (max_visualization_samples)
    """
    # Get configuration parameters
    if config and 'interpretability' in config:
        interp_config = config['interpretability']
        feature_config = interp_config.get('feature_extraction', {})
        layer_name = feature_config.get('layer_name', 'fc1')
        max_umap_samples = interp_config.get('max_samples', 30000)  # Main parameter for UMAP computation
        max_viz_samples = feature_config.get('max_visualization_samples', 100)  # For final visualization
        umap_config = interp_config.get('umap', {})
        clustering_config = interp_config.get('clustering', {})
    else:
        layer_name = 'fc1'
        max_umap_samples = 30000
        max_viz_samples = 100
        umap_config = {}
        clustering_config = {}
    
    try:
        # Extract features using max_umap_samples for UMAP computation
        print("Extracting features...")
        interpreter.extract_features(data_loader, layer_name=layer_name, 
                                   max_samples=max_umap_samples)
        
        # Fit UMAP with optional parameter optimization
        print("Fitting UMAP...")
        optimize_umap = umap_config.get('optimize_params', False)
        umap_embedding = interpreter.fit_umap(
            n_neighbors=umap_config.get('n_neighbors', 15),
            min_dist=umap_config.get('min_dist', 0.1),
            n_components=umap_config.get('n_components', 2),
            random_state=umap_config.get('random_state', 42),
            optimize_params=optimize_umap
        )
        
        # Perform high-dimensional clustering (optional)
        clustering_enabled = clustering_config.get('enabled', False)
        if clustering_enabled:
            print("Performing high-dimensional clustering...")
            min_cluster_size = clustering_config.get('min_cluster_size', 10)
            min_samples = clustering_config.get('min_samples', 5)
            interpreter.cluster_high_dimensional_features(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples
            )
        else:
            print("High-dimensional clustering disabled - skipping clustering step")
        
        # Create visualization dataframe - potentially subsample further for visualization
        print(f"Creating visualization dataframe...")
        df = interpreter.create_interpretability_dataframe(
            predictions, labels, data_loader, 
            additional_features=additional_features
        )
        
        # Subsample for visualization if needed (separate from UMAP computation)
        if len(df) > max_viz_samples:
            print(f"Subsampling {max_viz_samples} from {len(df)} samples for visualization performance")
            df = df.sample(n=max_viz_samples, random_state=42).copy()
        
        print(f"Created visualization dataframe with {len(df)} samples using {layer_name} features")
        print(f"UMAP was computed on up to {max_umap_samples} samples")
        
        # Create visualizer and plots
        visualizer = UMAPVisualizer()
        tabs = []
        
        # Tab 1: Classification results
        title_suffix = f" (Layer: {layer_name})"
        class_plot = visualizer.plot_classification_results(
            df, title=f"UMAP: Classification Results{title_suffix}"
        )
        tabs.append(TabPanel(child=class_plot, title="Classification Results"))
        
        # Tab 2: Feature-based coloring (if additional features provided)
        if additional_features:
            feature_plots = []
            for feature_name in additional_features.keys():
                if feature_name in df.columns:
                    plot = visualizer.plot_feature_coloring(
                        df, feature_name, title=f"UMAP: {feature_name}{title_suffix}"
                    )
                    feature_plots.append(plot)
            
            if feature_plots:
                if len(feature_plots) == 1:
                    feature_layout = feature_plots[0]
                else:
                    feature_layout = gridplot([feature_plots[i:i+2] for i in range(0, len(feature_plots), 2)])
                tabs.append(TabPanel(child=feature_layout, title="Feature Analysis"))
        
        # Tab 3: High-dimensional clusters (if available and enabled)
        if clustering_enabled and 'high_dim_cluster' in df.columns:
            cluster_plot = visualizer.plot_clusters(
                df, cluster_column='high_dim_cluster',
                title=f"UMAP: High-Dimensional Clusters{title_suffix}"
            )
            tabs.append(TabPanel(child=cluster_plot, title="High-Dim Clusters"))
        
        dashboard = Tabs(tabs=tabs)
        
        if output_path:
            output_file(str(output_path))
            save(dashboard)
            print(f"Interpretability dashboard saved to {output_path}")
        
        return dashboard
        
    except Exception as e:
        print(f"Error creating interpretability dashboard: {e}")
        import traceback
        traceback.print_exc()
        raise