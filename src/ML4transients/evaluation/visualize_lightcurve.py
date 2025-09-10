#!/usr/bin/env python
"""
Visualize a lightcurve for a given diaObjectId using Bokeh, display cutout images, and overlay inference information with symbols.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem, ImageRGBA, Div
from bokeh.io import curdoc
from bokeh.palettes import Category10

# Import ML4transients data loaders

import sys
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
    
    # For datasets that only have indices (no actual patch files), we need to work differently
    # Load the diasource patch index directly
    data_path = dataset_loader.data_paths[0]
    diasource_index_file = data_path / "lightcurves" / "diasource_patch_index.h5"
    
    if not diasource_index_file.exists():
        print(f"Diasource patch index not found at {diasource_index_file}")
        return None
    
    # Load the diasource index
    diasource_index = pd.read_hdf(diasource_index_file, key='index')
    
    # Find all diaSourceIds for this diaObjectId
    sources_for_object = diasource_index[diasource_index['diaObjectId'] == dia_object_id]
    
    if len(sources_for_object) == 0:
        print(f"No diaSourceIds found for diaObjectId {dia_object_id}")
        print(f"Available diaObjectIds (sample): {diasource_index['diaObjectId'].unique()[:5]}")
        return None
    
    source_ids = sources_for_object.index.tolist()
    print(f"Found {len(source_ids)} diaSourceIds for diaObjectId {dia_object_id}")
    
    # Since there are no actual lightcurve patch files in this dataset,
    # we'll create a mock lightcurve from the feature data
    # Group source IDs by visit for efficient loading
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
                            'visit': visit,
                            'ccdVisitId': visit,  # Approximation
                        }
                        
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
                                
                except Exception as e:
                    print(f"Warning: Could not load data for source {src_id}: {e}")
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
        for src_id in source_ids:
            inference = dataset_loader.get_inference_results_by_id(
                src_id, weights_path=weights_path, model_hash=model_hash
            )
            if inference is not None:
                inference_data[src_id] = inference
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
    
    # Plot lightcurve points by band if available
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
    
    # Overlay inference information with symbols
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
                    'probability': inf_result.get('probability', 0.0),
                    'cutout_base64': lc_point['cutout_base64'].iloc[0] if 'cutout_base64' in lc_point.columns else ""
                }
                inference_points.append(point_data)
        
        if inference_points:
            inf_df = pd.DataFrame(inference_points)
            inf_source = ColumnDataSource(inf_df)
            
            # Use different symbols based on prediction
            # Assuming binary classification: 0=bogus, 1=real
            real_mask = inf_df['prediction'] == 1
            bogus_mask = inf_df['prediction'] == 0
            
            if real_mask.any():
                real_source = ColumnDataSource(inf_df[real_mask])
                p.square(time_col, flux_col, source=real_source, 
                        size=15, color='#5C0002', legend_label="Real (Inference)",
                        line_color='white', line_width=2, alpha=0.9)
            
            if bogus_mask.any():
                bogus_source = ColumnDataSource(inf_df[bogus_mask])
                p.x(time_col, flux_col, source=bogus_source, 
                   size=15, color='#FF6B6B', legend_label="Bogus (Inference)",
                   line_color='white', line_width=3, alpha=0.9)
    
    # Add hover tool with cutout display
    hover_tooltips = [
        (time_col, f"@{time_col}{{0.0000}}"),
        (flux_col, f"@{flux_col}{{0.000}}"),
        ("diaSourceId", "@diaSourceId_str"),
    ]
    
    # Add band information if available
    if band_col and band_col in lc_df.columns:
        hover_tooltips.append(("Band", f"@{band_col}"))
    
    # Add cutout image to hover if available
    if 'cutout_base64' in lc_df.columns:
        hover_tooltip_html = f"""
        <div style="background-color: #2F2F2F; padding: 10px; border-radius: 5px;">
            <div>
                <img 
                    src='@cutout_base64' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px; border: 1px solid #666;'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #5C0002'>{time_col}:</span>
                <span style='font-size: 14px; color: white'>@{time_col}{{0.0000}}</span><br>
                <span style='font-size: 14px; color: #5C0002'>{flux_col}:</span>
                <span style='font-size: 14px; color: white'>@{flux_col}{{0.000}}</span><br>
                <span style='font-size: 14px; color: #5C0002'>diaSourceId:</span>
                <span style='font-size: 14px; color: white'>@diaSourceId_str</span><br>
            </div>
        </div>
        """
        
        # Add band information if available
        if band_col and band_col in lc_df.columns:
            hover_tooltip_html = hover_tooltip_html.replace(
                '</div>\n        </div>',
                f"""<span style='font-size: 14px; color: #5C0002'>Band:</span>
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
                <div style="color: #5C0002; font-size: 10px; margin-top: 2px; font-weight: bold;">{band_info}</div>
            </div>
            """
            cutout_div = Div(text=cutout_html, width=90, height=80)
            cutout_plots.append(cutout_div)
    
    if not cutout_plots:
        return Div(text='<div style="color: white; background-color: #2F2F2F; padding: 15px; border-radius: 5px;"><h3>No valid cutout images to display</h3></div>')
    
    # Add a compact header for the cutout section
    header_html = f"""
    <div style="background-color: #2F2F2F; padding: 8px; border-radius: 5px; border: 1px solid #666; margin-bottom: 5px;">
        <h4 style="color: #5C0002; margin: 0; text-align: center;">Cutouts (ordered by date)</h4>
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

def visualize(dia_object_id, data_path=None, weights_path=None, model_hash=None, output_file_name=None):
    """
    Visualize lightcurve, cutouts, and inference results for a given diaObjectId.
    
    Args:
        dia_object_id: The diaObjectId to visualize
        data_path: Path to data directory (defaults to DEFAULT_DATA_PATH)
        weights_path: Path to model weights for inference
        model_hash: Model hash for existing inference results
        output_file_name: Custom output HTML filename
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    
    try:
        # Initialize dataset loader
        print(f"Loading dataset from: {data_path}")
        dataset_loader = load_dataset_loader(data_path)
        
        # Get all data for this object
        print(f"Retrieving data for diaObjectId: {dia_object_id}")
        data_dict = get_lightcurve_and_inference_data(
            dataset_loader, dia_object_id, weights_path, model_hash
        )
        
        if data_dict is None:
            print(f"No data found for diaObjectId {dia_object_id}")
            return
        
        # Create plots
        print("Creating lightcurve plot...")
        lc_plot = plot_lightcurve_bokeh(data_dict)
        
        print("Creating cutout display...")
        cutout_display = create_cutout_display(data_dict)
        
        # Create summary information
        lc_data = data_dict['lightcurve']
        num_observations = len(lc_data)
        num_cutouts = len(data_dict['cutouts'])
        num_inference = len(data_dict['inference'])
        
        time_col = 'midpointMjdTai' if 'midpointMjdTai' in lc_data.columns else 'mjd'
        if time_col in lc_data.columns:
            time_span = lc_data[time_col].max() - lc_data[time_col].min()
            summary_text = f"""
            <div style="background-color: #2F2F2F; padding: 15px; border-radius: 5px; border: 1px solid #666;">
                <h3 style="color: #5C0002; margin-top: 0;">Summary for diaObjectId {dia_object_id}</h3>
                <ul style="color: white; list-style-type: none; padding-left: 0;">
                    <li style="margin: 8px 0;"><span style="color: #5C0002; font-weight: bold;">Observations:</span> {num_observations}</li>
                    <li style="margin: 8px 0;"><span style="color: #5C0002; font-weight: bold;">Time span:</span> {time_span:.2f} days</li>
                    <li style="margin: 8px 0;"><span style="color: #5C0002; font-weight: bold;">Cutouts available:</span> {num_cutouts}</li>
                    <li style="margin: 8px 0;"><span style="color: #5C0002; font-weight: bold;">Inference results:</span> {num_inference}</li>
                </ul>
            </div>
            """
        else:
            summary_text = f"""
            <div style="background-color: #2F2F2F; padding: 15px; border-radius: 5px; border: 1px solid #666;">
                <h3 style="color: #5C0002; margin-top: 0;">Summary for diaObjectId {dia_object_id}</h3>
                <ul style="color: white; list-style-type: none; padding-left: 0;">
                    <li style="margin: 8px 0;"><span style="color: #5C0002; font-weight: bold;">Observations:</span> {num_observations}</li>
                    <li style="margin: 8px 0;"><span style="color: #5C0002; font-weight: bold;">Cutouts available:</span> {num_cutouts}</li>
                    <li style="margin: 8px 0;"><span style="color: #5C0002; font-weight: bold;">Inference results:</span> {num_inference}</li>
                </ul>
            </div>
            """
        
        summary_div = Div(text=summary_text, width=300)
        
        # Create layout with cutout display below the lightcurve and summary
        top_section = column(lc_plot, summary_div)
        layout = column(top_section, cutout_display)
        
        # Set output file
        if output_file_name is None:
            output_file_name = f"saved/lc/lightcurve_{dia_object_id}.html"
        
        output_file(output_file_name)
        
        print(f"Saving visualization to: {output_file_name}")
        show(layout)
        
        print(f"Visualization complete! Open {output_file_name} in your browser.")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize lightcurve for a given diaObjectId")
    parser.add_argument("dia_object_id", type=int, help="The diaObjectId to visualize")
    parser.add_argument("--data-path", type=str, help="Path to data directory")
    parser.add_argument("--weights-path", type=str, help="Path to model weights for inference")
    parser.add_argument("--model-hash", type=str, help="Model hash for existing inference results")
    parser.add_argument("--output", type=str, help="Output HTML filename")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path) if args.data_path else None
    
    visualize(
        dia_object_id=args.dia_object_id,
        data_path=data_path,
        weights_path=args.weights_path,
        model_hash=args.model_hash,
        output_file_name=args.output
    )
