#!/usr/bin/env python
"""
Enhanced Lightcurve Visualization with Inference Integration

This module provides enhanced lightcurve visualization capabilities that integrate seamlessly 
with the ML4transients inference infrastructure. 

Usage:
    # View existing inference results
    python lightcurve_visualization.py 12345 --weights-path /path/to/model --data-path /path/to/data

"""
import os
import sys
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, Div
from bokeh.io import curdoc
from bokeh.palettes import Category10

# Import ML4transients data loaders
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ML4transients.data_access.dataset_loader import DatasetLoader
from ML4transients.evaluation.interpretability import embeddable_image

# --- CONFIG ---
# Default data path - can be overridden via command line
DEFAULT_DATA_PATH = Path("/sps/lsst/groups/transients/HSC/fouchez/raphael/data/UDEEP_norm")

# --- FUNCTIONS ---
def load_dataset_loader(data_path):
    """Initialize DatasetLoader with the given data path."""
    return DatasetLoader(data_path)

def run_inference(dataset_loader, visits, weights_path, force=False):
    """Run inference for the specified visits if not already available.
    
    Parameters
    ----------
    dataset_loader : DatasetLoader
        The dataset loader instance
    visits : list
        List of visits to run inference for
    weights_path : str
        Path to the model weights
        
    Returns
    -------
    bool
        True if inference was run successfully, False otherwise
    """
    missing_visits = []
    for visit in visits:
        try:
            inference_loader = dataset_loader.get_inference_loader(
                visit=visit,
                weights_path=weights_path
            )
            if not inference_loader.has_inference_results() or force:
                missing_visits.append(visit)
        except Exception:
            missing_visits.append(visit)
    
    if not missing_visits and not force:
        return True
    
    # Run inference for missing visits
    for visit in missing_visits:
        try:
            inference_loader = dataset_loader.get_inference_loader(
                visit=visit,
                weights_path=weights_path
            )
            inference_loader.run_inference(dataset_loader, force=force)
        except Exception:
            return False
    
    print("Inference completed successfully")
    return True

def load_diasource_index(diasource_index_file):
    """
    Load diasource index from HDF5 file, handling different dataset structures.
    
    Parameters
    ----------
    diasource_index_file : Path
        Path to the diasource_patch_index.h5 file
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with diasource index data, or None if file not found/readable
    """
    if not diasource_index_file.exists():
        print(f"Diasource patch index not found at {diasource_index_file}")
        return None
    
    # Try different possible keys that different datasets might use
    possible_keys = ['diasource_index', 'index']
    
    for key in possible_keys:
        try:
            print(f"Trying to load diasource index with key '{key}'...")
            diasource_index = pd.read_hdf(diasource_index_file, key=key)
            print(f"Successfully loaded diasource index with key '{key}'")
            print(f"Index shape: {diasource_index.shape}, Columns: {diasource_index.columns.tolist()}")
            return diasource_index
        except (KeyError, Exception):
            continue
    
    # If none of the standard keys work, try to inspect the file structure
    try:
        import h5py
        with h5py.File(diasource_index_file, 'r') as f:
            if len(f.keys()) > 0:
                first_key = list(f.keys())[0]
                diasource_index = pd.read_hdf(diasource_index_file, key=first_key)
                return diasource_index
    except Exception:
        pass
    
    return None

def get_lightcurve_and_inference_data(dataset_loader, dia_object_id, weights_path=None, model_hash=None):
    """
    Load lightcurve data and inference results for a specific diaObjectId.
    
    Args:
        dataset_loader: DatasetLoader instance
        dia_object_id: The diaObjectId to retrieve
        weights_path: Path to model weights (for inference)
        model_hash: Model hash (for existing inference results)
    
    Returns:
        Dict containing lightcurve data, cutouts, and inference results
    """
    import pandas as pd
    
    # Load the diasource patch index directly
    data_path = dataset_loader.data_paths[0]
    diasource_index_file = data_path / "lightcurves" / "diasource_patch_index.h5"
    
    # Use robust loading function
    diasource_index = load_diasource_index(diasource_index_file)
    if diasource_index is None:
        return None
    
    # Find all diaSourceIds for this diaObjectId
    sources_for_object = diasource_index[diasource_index['diaObjectId'] == dia_object_id]
    
    if len(sources_for_object) == 0:
        print(f"No diaSourceIds found for diaObjectId {dia_object_id}")
        print(f"Available diaObjectIds (sample): {diasource_index['diaObjectId'].unique()[:5]}")
        return None
    
    source_ids = sources_for_object.index.tolist()
    visit_groups = sources_for_object.groupby('visit')['diaObjectId'].count().to_dict()
    
    # Load features and create a lightcurve-like structure
    lightcurve_points = []
    cutouts = {}
    
    for visit in visit_groups.keys():
        if visit in dataset_loader.features:
            feature_loader = dataset_loader.features[visit]
            
            # Get features for sources from this visit that belong to our object
            visit_sources = sources_for_object[sources_for_object['visit'] == visit].index.tolist()
            
            for src_id in visit_sources:
                # Get features for this source
                try:
                    features = feature_loader.get_by_id(src_id)
                    if features is not None and len(features) > 0:
                        # Extract lightcurve-relevant information
                        feature_row = features.iloc[0]
                        
                        # Create a lightcurve point from features
                        lc_point = {
                            'diaSourceId': src_id,
                            'diaObjectId': dia_object_id,
                            'visit': visit,                        }
                        
                        # Add flux/magnitude information if available
                        for col in ['psFlux', 'psfFlux', 'flux']:
                            if col in feature_row:
                                lc_point['psFlux'] = feature_row[col]
                                break
                        
                        for col in ['psFluxErr', 'psfFluxErr', 'fluxErr']:
                            if col in feature_row:
                                lc_point['psFluxErr'] = feature_row[col]
                                break
                                
                        for col in ['mag', 'psfMag']:
                            if col in feature_row:
                                lc_point['mag'] = feature_row[col]
                                break
                                
                        for col in ['magErr', 'psfMagErr']:
                            if col in feature_row:
                                lc_point['magErr'] = feature_row[col]
                                break
                        
                        for col in ['midpointMjdTai', 'mjd', 'time']:
                            if col in feature_row:
                                lc_point['midpointMjdTai'] = feature_row[col]
                                break
                        
                        for col in ['band', 'filter']:
                            if col in feature_row:
                                lc_point['band'] = feature_row[col]
                                break
                        
                        lightcurve_points.append(lc_point)
                        
                        # Load cutout for this source
                        if visit in dataset_loader.cutouts:
                            cutout = dataset_loader.cutouts[visit].get_by_id(src_id)
                            if cutout is not None:
                                cutouts[src_id] = cutout
                                
                except Exception:
                    continue
    
    if not lightcurve_points:
        print(f"No lightcurve points could be constructed for diaObjectId {dia_object_id}")
        return None
    
    # Convert to DataFrame and sort by time if available
    lc_data = pd.DataFrame(lightcurve_points)
    if 'midpointMjdTai' in lc_data.columns:
        lc_data = lc_data.sort_values('midpointMjdTai').reset_index(drop=True)
    
    # Convert large integers to strings to avoid Bokeh precision warnings
    if 'diaSourceId' in lc_data.columns:
        lc_data['diaSourceId_str'] = lc_data['diaSourceId'].astype(str)
    if 'diaObjectId' in lc_data.columns:
        lc_data['diaObjectId_str'] = lc_data['diaObjectId'].astype(str)
    
    # Add base64-encoded cutouts to the lightcurve data for hover tool
    print("encoding images")
    lc_data['cutout_base64'] = ""
    for idx, row in lc_data.iterrows():
        src_id = row['diaSourceId']
        if src_id in cutouts:
            lc_data.loc[idx, 'cutout_base64'] = embeddable_image(cutouts[src_id])
    
    print(f"Constructed lightcurve with {len(lc_data)} points")
    print(f"Loaded {len(cutouts)} cutouts")
    
    # Load inference results if available
    inference_data = {}
    if weights_path or model_hash:
        print(f"Loading inference results for {len(source_ids)} sources...")
        start_time = time.time()
        
        # Group source IDs by visit for efficient loading
        visit_groups = {}
        for src_id in source_ids:
            visit = sources_for_object[sources_for_object.index == src_id]['visit'].iloc[0]
            if visit not in visit_groups:
                visit_groups[visit] = []
            visit_groups[visit].append(src_id)
        
        # Load inference loaders for each visit
        for visit, visit_source_ids in visit_groups.items():
            try:
                # Get inference loader for this visit
                inference_loader = dataset_loader.get_inference_loader(
                    visit=visit,
                    weights_path=weights_path,
                    model_hash=model_hash
                )
                
                if inference_loader and inference_loader.has_inference_results():
                    # Get results for each source ID in this visit
                    for src_id in visit_source_ids:
                        result = inference_loader.get_results_by_id(src_id)
                        if result is not None:
                            inference_data[src_id] = result
                    
            except Exception:
                continue
        
        print(f"Loaded inference results for {len(inference_data)} sources in {time.time() - start_time:.2f}s")
    
    print(f"Loaded {len(inference_data)} inference results")
    
    return {
        'lightcurve': lc_data,
        'cutouts': cutouts,
        'inference': inference_data,
        'object_id': dia_object_id
    }

def plot_lightcurve_bokeh(data_dict):
    """
    Plot the lightcurve with Bokeh and overlay inference info.
    
    Args:
        data_dict: Dictionary containing lightcurve, cutouts, and inference data
    
    Returns:
        Bokeh figure with lightcurve plot
    """
    from bokeh.io import curdoc
    
    # Apply dark theme to match UMAP style
    curdoc().theme = 'dark_minimal'
    
    lc_df = data_dict['lightcurve']
    inference_data = data_dict['inference']
    dia_object_id = data_dict['object_id']
    
    # Determine time and flux columns
    time_col = 'midpointMjdTai' if 'midpointMjdTai' in lc_df.columns else 'mjd'
    flux_col = 'psFlux' if 'psFlux' in lc_df.columns else 'flux'
    band_col = 'band' if 'band' in lc_df.columns else None
    
    p = figure(
        title=f"Lightcurve for diaObjectId {dia_object_id}",
        x_axis_label=time_col, 
        y_axis_label=flux_col, 
        width=900, 
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        background_fill_color="#2F2F2F",
        border_fill_color="#2F2F2F"
    )
    
    # Plot lightcurve points by band if available (only if no inference data to avoid overlap)
    if not inference_data:
        if band_col and band_col in lc_df.columns:
            bands = lc_df[band_col].unique()
            colors = Category10[max(len(bands), 3)]
            
            for i, band in enumerate(bands):
                band_data = lc_df[lc_df[band_col] == band].copy()
                source = ColumnDataSource(band_data)
                
                # Plot observations as scatter points only (no connecting lines)
                p.scatter(time_col, flux_col, source=source, 
                        size=10, color=colors[i % len(colors)], 
                        legend_label=f"Band {band}", alpha=0.8,
                        line_color="white", line_width=1)
        else:
            # Plot without band separation
            source = ColumnDataSource(lc_df)
            p.scatter(time_col, flux_col, source=source, 
                    size=10, color='#5b75cd', 
                    legend_label="Observations", alpha=0.8,
                    line_color="white", line_width=1)
    
    # Overlay inference information with symbols and enhanced visualization
    if inference_data:
        inference_points = []
        for src_id, inf_result in inference_data.items():
            # Find corresponding lightcurve point
            lc_point = lc_df[lc_df['diaSourceId'] == src_id]
            if len(lc_point) > 0:
                point_data = {
                    time_col: lc_point[time_col].iloc[0],
                    flux_col: lc_point[flux_col].iloc[0],
                    'diaSourceId_str': str(src_id),
                    'prediction': inf_result.get('prediction', 'Unknown'),
                    'label': inf_result.get('label', 'Unknown'),
                    'probability': inf_result.get('probability', 0.0),
                    'uncertainty': inf_result.get('uncertainty', 0.0),
                    'cutout_base64': lc_point['cutout_base64'].iloc[0] if 'cutout_base64' in lc_point.columns else ""
                }
                
                # Add band information if available
                if band_col and band_col in lc_point.columns:
                    point_data['band'] = lc_point[band_col].iloc[0]
                else:
                    point_data['band'] = 'unknown'
                
                # Add classification status
                if point_data['prediction'] != 'Unknown' and point_data['label'] != 'Unknown':
                    is_correct = point_data['prediction'] == point_data['label']
                    if is_correct:
                        if point_data['prediction'] == 1:
                            point_data['classification_status'] = 'True Positive'
                        else:
                            point_data['classification_status'] = 'True Negative'
                    else:
                        if point_data['prediction'] == 1:
                            point_data['classification_status'] = 'False Positive'
                        else:
                            point_data['classification_status'] = 'False Negative'
                else:
                    point_data['classification_status'] = 'Unknown'
                
                inference_points.append(point_data)
        
        if inference_points:
            inf_df = pd.DataFrame(inference_points)
            
            # UMAP-style symbols with band-based coloring
            # TP='x', TN='circle', FP='star', FN='square' 
            
            # Band colors for astronomical observations
            band_colors = {
                'u': '#56b4e9',  # Light blue
                'g': '#009e73',  # Green  
                'r': '#e69f00',  # Orange
                'i': '#cc79a7',  # Pink
                'z': '#d55e00',  # Red-orange
                'y': '#f0e442'   # Yellow
            }
            
            # Helper function to assign colors based on band
            def get_band_colors(data, fallback_color='#666666'):
                if 'band' in data.columns:
                    return [band_colors.get(str(band), fallback_color) for band in data['band']]
                else:
                    return [fallback_color] * len(data)
            
            # Create masks for different classification results
            tp_mask = inf_df['classification_status'] == 'True Positive'
            tn_mask = inf_df['classification_status'] == 'True Negative'
            fp_mask = inf_df['classification_status'] == 'False Positive'
            fn_mask = inf_df['classification_status'] == 'False Negative'
            
            # True Positives - X symbols (Injected data)
            if tp_mask.any():
                tp_data = inf_df[tp_mask].copy()
                tp_data['plot_color'] = get_band_colors(tp_data)
                tp_source = ColumnDataSource(tp_data)
                p.scatter(time_col, flux_col, source=tp_source, 
                         marker='x', size=12, color='plot_color', legend_label="True Positive",
                         line_width=1, alpha=0.8)
            
            # True Negatives - Circle symbols 
            if tn_mask.any():
                tn_data = inf_df[tn_mask].copy()
                tn_data['plot_color'] = get_band_colors(tn_data)
                tn_source = ColumnDataSource(tn_data)
                p.scatter(time_col, flux_col, source=tn_source, 
                         marker='circle', size=8, color='plot_color', legend_label="True Negative",
                         line_width=1, alpha=0.8)
            
            # False Positives - Star symbols 
            if fp_mask.any():
                fp_data = inf_df[fp_mask].copy()
                fp_data['plot_color'] = get_band_colors(fp_data)
                fp_source = ColumnDataSource(fp_data)
                p.scatter(time_col, flux_col, source=fp_source, 
                         marker='star', size=12, color='plot_color', legend_label="False Positive",
                         line_width=1, alpha=0.8)
            
            # False Negatives - Square symbols 
            if fn_mask.any():
                fn_data = inf_df[fn_mask].copy()
                fn_data['plot_color'] = get_band_colors(fn_data)
                fn_source = ColumnDataSource(fn_data)
                p.scatter(time_col, flux_col, source=fn_source, 
                         marker='square', size=10, color='plot_color', legend_label="False Negative",
                         line_width=1, alpha=0.8)
    
    # Add enhanced hover tool with inference information
    hover_tooltips = [
        (time_col, f"@{time_col}{{0.0000}}"),
        (flux_col, f"@{flux_col}{{0.000}}"),
        ("diaSourceId", "@diaSourceId_str"),
        ("Band", "@band"),
    ]
    
    # Add inference information if available
    if inference_data:
        hover_tooltips.extend([
            ("Prediction", "@prediction"),
            ("True Label", "@label"),
            ("Probability", "@probability{0.000}"),
            ("Uncertainty", "@uncertainty{0.000}"),
            ("Status", "@classification_status")
        ])
    
    # Add band information if available
    if band_col and band_col in lc_df.columns:
        hover_tooltips.append(("Band", f"@{band_col}"))
    
    # Enhanced hover with cutout and inference info
    if 'cutout_base64' in lc_df.columns or inference_data:
        inference_info_html = ""
        if inference_data:
            inference_info_html = """
                <span style='font-size: 14px; color: #870000'>Prediction:</span>
                <span style='font-size: 14px; color: white'>@prediction</span><br>
                <span style='font-size: 14px; color: #870000'>True Label:</span>
                <span style='font-size: 14px; color: white'>@label</span><br>
                <span style='font-size: 14px; color: #870000'>Probability:</span>
                <span style='font-size: 14px; color: white'>@probability{0.000}</span><br>
                <span style='font-size: 14px; color: #870000'>Uncertainty:</span>
                <span style='font-size: 14px; color: white'>@uncertainty{0.000}</span><br>
                <span style='font-size: 14px; color: #870000'>Status:</span>
                <span style='font-size: 14px; color: white'>@classification_status</span><br>
            """
        
        hover_tooltip_html = f"""
        <div style="background-color: #2F2F2F; padding: 10px; border-radius: 5px;">
            <div>
                <img 
                    src='@cutout_base64' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px; border: 1px solid #666;'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #870000'>{time_col}:</span>
                <span style='font-size: 14px; color: white'>@{time_col}{{0.0000}}</span><br>
                <span style='font-size: 14px; color: #870000'>{flux_col}:</span>
                <span style='font-size: 14px; color: white'>@{flux_col}{{0.000}}</span><br>
                <span style='font-size: 14px; color: #870000'>diaSourceId:</span>
                <span style='font-size: 14px; color: white'>@diaSourceId_str</span><br>
                {inference_info_html}
            </div>
        </div>
        """
        
        # Add band information if available
        if band_col and band_col in lc_df.columns:
            hover_tooltip_html = hover_tooltip_html.replace(
                '</div>\n        </div>',
                f"""<span style='font-size: 14px; color: #870000'>Band:</span>
                <span style='font-size: 14px; color: white'>@{band_col}</span><br>
            </div>
        </div>"""
            )
        
        hover = HoverTool(tooltips=hover_tooltip_html)
    else:
        # Fallback to simple tooltips if no cutout available
        hover_tooltips = [
            (time_col, f"@{time_col}{{0.0000}}"),
            (flux_col, f"@{flux_col}{{0.000}}"),
            ("diaSourceId", "@diaSourceId_str"),
        ]
        
        # Add band information if available
        if band_col and band_col in lc_df.columns:
            hover_tooltips.append(("Band", f"@{band_col}"))
        
        hover = HoverTool(tooltips=hover_tooltips)
    p.add_tools(hover)
    
    # Style the legend and axes for dark theme
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.background_fill_color = "#2F2F2F"
    p.legend.background_fill_alpha = 0.8
    p.legend.border_line_color = "#666"
    p.legend.label_text_color = "white"
    
    # Add band color information to the plot title if inference data is available
    if inference_data and band_col and band_col in lc_df.columns:
        # Get unique bands from the lightcurve data
        unique_bands = sorted(lc_df[band_col].unique())
        
        # Band colors for reference
        band_colors = {
            'u': '#56b4e9',  # Light blue
            'g': '#009e73',  # Green  
            'r': '#e69f00',  # Orange
            'i': '#cc79a7',  # Pink
            'z': '#d55e00',  # Red-orange
            'y': '#f0e442'   # Yellow
        }
        
        # Create band color description
        band_info = " | Bands: " + ", ".join([f"{band}" for band in unique_bands if band in band_colors])
        current_title = p.title.text
        p.title.text = current_title + band_info
    
    # Style the axes for dark theme
    p.title.text_color = "white"
    p.xaxis.axis_label_text_color = "white"
    p.yaxis.axis_label_text_color = "white"
    p.xaxis.major_label_text_color = "white"
    p.yaxis.major_label_text_color = "white"
    p.xaxis.axis_line_color = "#666"
    p.yaxis.axis_line_color = "#666"
    p.xaxis.major_tick_line_color = "#666"
    p.yaxis.major_tick_line_color = "#666"
    p.xaxis.minor_tick_line_color = "#666"
    p.yaxis.minor_tick_line_color = "#666"
    p.grid.grid_line_color = "#444"
    p.grid.grid_line_alpha = 0.5
    
    return p

def create_inference_summary(data_dict):
    """
    Create a summary display of inference results for the lightcurve.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing lightcurve data and inference results
        
    Returns
    -------
    Div
        Bokeh Div element with formatted inference summary
    """
    inference_data = data_dict.get('inference', {})
    dia_object_id = data_dict.get('object_id', 'Unknown')
    
    if not inference_data:
        summary_html = f"""
        <div style="background-color: #2F2F2F; padding: 15px; border-radius: 5px; border: 1px solid #666; margin-bottom: 10px;">
            <h3 style="color: #870000; margin-top: 0;">Inference Summary for diaObjectId {dia_object_id}</h3>
            <p style="color: white; margin: 0;">No inference results available for this lightcurve.</p>
        </div>
        """
        return Div(text=summary_html, width=900)
    
    # Calculate classification statistics
    total_sources = len(inference_data)
    predictions = [result.get('prediction', 'Unknown') for result in inference_data.values()]
    labels = [result.get('label', 'Unknown') for result in inference_data.values()]
    probabilities = [result.get('probability', 0.0) for result in inference_data.values()]
    uncertainties = [result.get('uncertainty', 0.0) for result in inference_data.values()]
    
    # Count classification results
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    
    real_predictions = sum(1 for p in predictions if p == 1)
    bogus_predictions = sum(1 for p in predictions if p == 0)
    
    # Calculate statistics
    accuracy = (tp + tn) / total_sources if total_sources > 0 else 0
    avg_probability = np.mean(probabilities) if probabilities else 0
    avg_uncertainty = np.mean(uncertainties) if uncertainties else 0
    
    summary_html = f"""
    <div style="background-color: #2F2F2F; padding: 15px; border-radius: 5px; border: 1px solid #666; margin-bottom: 10px;">
        <h3 style="color: #870000; margin-top: 0;">Inference Summary for diaObjectId {dia_object_id}</h3>
        
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            <div style="color: white;">
                <h4 style="color: #870000; margin-bottom: 8px;">Classification Overview</h4>
                <p><strong>Total Sources:</strong> {total_sources}</p>
                <p><strong>Predicted Real:</strong> {real_predictions}</p>
                <p><strong>Predicted Bogus:</strong> {bogus_predictions}</p>
                <p><strong>Overall Accuracy:</strong> {accuracy:.3f}</p>
            </div>
            
            <div style="color: white;">
                <h4 style="color: #870000; margin-bottom: 8px;">Classification Breakdown</h4>
                <p><strong>True Positives:</strong> {tp}</p>  
                <p><strong>True Negatives:</strong> {tn}</p>  
                <p><strong>False Positives:</strong> {fp}</p>  
                <p><strong>False Negatives:</strong> {fn}</p>  
            </div>
            
            <div style="color: white;">
                <h4 style="color: #870000; margin-bottom: 8px;">Model Confidence</h4>
                <p><strong>Avg Probability:</strong> {avg_probability:.3f}</p>
                <p><strong>Avg Uncertainty:</strong> {avg_uncertainty:.3f}</p>
                <p><strong>Prob Range:</strong> {min(probabilities):.3f} - {max(probabilities):.3f}</p>
            </div>
        </div>
        
        <div style="color: white; font-size: 12px; margin-top: 10px; border-top: 1px solid #666; padding-top: 10px;">
            <strong>Legend:</strong><br>
                <span style="color: #56b4e9;">&#x25CF;</span> <strong>u:</strong> Light blue &nbsp; 
                <span style="color: #009e73;">&#x25CF;</span> <strong>g:</strong> Green &nbsp; 
                <span style="color: #e69f00;">&#x25CF;</span> <strong>r:</strong> Orange &nbsp; 
                <span style="color: #cc79a7;">&#x25CF;</span> <strong>i:</strong> Pink &nbsp; 
                <span style="color: #d55e00;">&#x25CF;</span> <strong>z:</strong> Red &nbsp; 
                <span style="color: #f0e442;">&#x25CF;</span> <strong>y:</strong> Yellow
        </div>
    </div>
    """

    return Div(text=summary_html, width=900)

def create_cutout_display(data_dict, max_cutouts=24):
    """
    Create a display of cutout images arranged in a grid, ordered by date.
    
    Args:
        data_dict: Dictionary containing cutout data and lightcurve
        max_cutouts: Maximum number of cutouts to display
    
    Returns:
        Bokeh layout with cutout images
    """
    cutouts = data_dict['cutouts']
    lc_data = data_dict['lightcurve']
    
    if not cutouts:
        return Div(text='<div style="color: white; background-color: #2F2F2F; padding: 15px; border-radius: 5px;"><h3>No cutout images available</h3></div>')
    
    # Create a list of cutouts with their corresponding time information
    cutout_with_time = []
    time_col = 'midpointMjdTai' if 'midpointMjdTai' in lc_data.columns else 'mjd'
    
    for src_id, cutout_arr in cutouts.items():
        if cutout_arr is not None and cutout_arr.size > 0:
            # Find the corresponding lightcurve point for this source
            lc_point = lc_data[lc_data['diaSourceId'] == src_id]
            if len(lc_point) > 0 and time_col in lc_point.columns:
                time_value = lc_point[time_col].iloc[0]
                cutout_with_time.append((src_id, cutout_arr, time_value))
    
    # Sort by time (date order)
    cutout_with_time.sort(key=lambda x: x[2])
    
    # Select a subset to display
    cutout_items = cutout_with_time[:max_cutouts]
    
    cutout_plots = []
    for src_id, cutout_arr, time_value in cutout_items:
        # Use the embeddable_image function to create base64 image with proper colormap
        base64_img = embeddable_image(cutout_arr)
        
        if base64_img:
            # Find the filter/band information for this source
            lc_point = lc_data[lc_data['diaSourceId'] == src_id]
            band_info = ""
            if len(lc_point) > 0:
                if 'band' in lc_point.columns:
                    band_info = lc_point['band'].iloc[0]
                elif 'filter' in lc_point.columns:
                    band_info = lc_point['filter'].iloc[0]
            
            # Create a compact Div with just the image and filter info
            cutout_html = f"""
            <div style="background-color: #2F2F2F; padding: 5px; margin: 2px; border-radius: 3px; border: 1px solid #666; text-align: center; width: 80px;">
                <img src="{base64_img}" style="width: 60px; height: 60px; border: 1px solid #666;">
                <div style="color: #870000; font-size: 10px; margin-top: 2px; font-weight: bold;">{band_info}</div>
            </div>
            """
            cutout_div = Div(text=cutout_html, width=90, height=80)
            cutout_plots.append(cutout_div)
    
    if not cutout_plots:
        return Div(text='<div style="color: white; background-color: #2F2F2F; padding: 15px; border-radius: 5px;"><h3>No valid cutout images to display</h3></div>')
    
    # Add a compact header for the cutout section
    header_html = f"""
    <div style="background-color: #2F2F2F; padding: 8px; border-radius: 5px; border: 1px solid #666; margin-bottom: 5px;">
        <h4 style="color: #870000; margin: 0; text-align: center;">Cutouts (ordered by date)</h4>
        <p style="color: white; margin: 2px 0 0 0; text-align: center; font-size: 10px;">Showing {len(cutout_plots)} of {len(cutouts)} available</p>
    </div>
    """
    header_div = Div(text=header_html, width=900)
    
    # Arrange cutouts in a compact grid (8 columns for maximum compactness)
    rows = [header_div]
    for i in range(0, len(cutout_plots), 8):
        row_plots = cutout_plots[i:i+8]
        rows.append(row(*row_plots))
    
    return column(*rows)

def visualize(dia_object_id, data_path=None, weights_path=None, model_hash=None, 
              output_file_name=None, run_inference_flag=False):
    """
    Visualize lightcurve, cutouts, and inference results for a given diaObjectId.
    
    Parameters
    ----------
    dia_object_id : int
        The diaObjectId to visualize
    data_path : Path, optional
        Path to data directory (defaults to DEFAULT_DATA_PATH)
    weights_path : str, optional
        Path to model weights for inference
    model_hash : str, optional
        Model hash for existing inference results
    output_file_name : str, optional
        Custom output HTML filename
    run_inference_flag : bool, optional
        Whether to run inference if not available (requires weights_path)
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    
    # Load dataset
    dataset_loader = load_dataset_loader(data_path)
    
    # Check if we need to run inference
    if weights_path and run_inference_flag:
        # First, get the lightcurve data to determine which visits we need
        temp_data = get_lightcurve_and_inference_data(
            dataset_loader, dia_object_id, weights_path=None, model_hash=model_hash
        )
        
        if temp_data is None:
            print(f"No lightcurve data found for diaObjectId {dia_object_id}")
            return
        
        # Get unique visits from the lightcurve data
        lc_data = temp_data['lightcurve']
        if 'visit' in lc_data.columns:
            visits = lc_data['visit'].unique().tolist()
        else:
            # Fallback: try to extract visits from the diasource index
            visits = []
            for data_path_item in dataset_loader.data_paths:
                diasource_index_file = data_path_item / "lightcurves" / "diasource_patch_index.h5"
                if diasource_index_file.exists():
                    diasource_index = load_diasource_index(diasource_index_file)
                    if diasource_index is not None:
                        sources_for_object = diasource_index[diasource_index['diaObjectId'] == dia_object_id]
                        visits.extend(sources_for_object['visit'].unique().tolist())
            visits = list(set(visits))  # Remove duplicates
        
        # Run inference for these visits
        if not run_inference(dataset_loader, visits, weights_path):
            print("Failed to run inference, continuing with available data...")
    
    # Load lightcurve and inference data
    data_dict = get_lightcurve_and_inference_data(
        dataset_loader, dia_object_id, weights_path, model_hash
    )
    
    if data_dict is None:
        print(f"No data found for diaObjectId {dia_object_id}")
        return
    
    # Create plots
    lc_plot = plot_lightcurve_bokeh(data_dict)
    inference_summary = create_inference_summary(data_dict)
    cutout_display = create_cutout_display(data_dict)
    
    # Create overall summary information
    lc_data = data_dict['lightcurve']
    num_observations = len(lc_data)
    num_cutouts = len(data_dict['cutouts'])
    num_inference = len(data_dict['inference'])
    
    time_col = 'midpointMjdTai' if 'midpointMjdTai' in lc_data.columns else 'mjd'
    if time_col in lc_data.columns:
        time_span = lc_data[time_col].max() - lc_data[time_col].min()
        summary_text = f"""
        <div style="background-color: #2F2F2F; padding: 15px; border-radius: 5px; border: 1px solid #666; margin-bottom: 10px;">
            <h3 style="color: #870000; margin-top: 0;">Lightcurve Overview for diaObjectId {dia_object_id}</h3>
            <div style="display: flex; gap: 30px;">
                <div style="color: white;">
                    <p><strong>Observations:</strong> {num_observations}</p>
                    <p><strong>Time span:</strong> {time_span:.2f} days</p>
                </div>
                <div style="color: white;">
                    <p><strong>Cutouts available:</strong> {num_cutouts}</p>
                    <p><strong>Inference results:</strong> {num_inference}</p>
                </div>
            </div>
        </div>
        """
    else:
        summary_text = f"""
        <div style="background-color: #2F2F2F; padding: 15px; border-radius: 5px; border: 1px solid #666; margin-bottom: 10px;">
            <h3 style="color: #870000; margin-top: 0;">Lightcurve Overview for diaObjectId {dia_object_id}</h3>
            <div style="display: flex; gap: 30px;">
                <div style="color: white;">
                    <p><strong>Observations:</strong> {num_observations}</p>
                    <p><strong>Cutouts available:</strong> {num_cutouts}</p>
                </div>
                <div style="color: white;">
                    <p><strong>Inference results:</strong> {num_inference}</p>
                </div>
            </div>
        </div>
        """
    
    summary_div = Div(text=summary_text, width=900)
    
    # Create comprehensive layout
    layout_elements = [summary_div]
    
    # Add inference summary if available
    if data_dict['inference']:
        layout_elements.append(inference_summary)
    
    layout_elements.extend([lc_plot, cutout_display])
    layout = column(*layout_elements)
    
    # Set output file
    if output_file_name is None:
        output_file_name = f"lightcurve_{dia_object_id}.html"
    
    output_file(output_file_name)
    show(layout)
    
    print(f"Visualization complete! Open {output_file_name} in your browser.")
    print(f"Summary: {num_observations} observations, {num_cutouts} cutouts, {num_inference} inference results")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize lightcurve for a given diaObjectId with enhanced inference capabilities")
    parser.add_argument("dia_object_id", type=int, help="The diaObjectId to visualize")
    parser.add_argument("--data-path", type=str, help="Path to data directory")
    parser.add_argument("--weights-path", type=str, help="Path to model weights for inference")
    parser.add_argument("--model-hash", type=str, help="Model hash for existing inference results")
    parser.add_argument("--output", type=str, help="Output HTML filename")
    parser.add_argument("--run-inference", action="store_true",
                       help="Run inference if not available (requires weights-path)")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path) if args.data_path else None
    
    visualize(
        dia_object_id=args.dia_object_id,
        data_path=data_path,
        weights_path=args.weights_path,
        model_hash=args.model_hash,
        output_file_name=args.output,
        run_inference_flag=args.run_inference
    )
