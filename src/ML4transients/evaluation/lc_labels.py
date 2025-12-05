#!/usr/bin/env python
"""
Lightcurve Labels Visualization

This module provides a band-based lightcurve visualization that displays cutouts in rows
organized by filter band. Each row shows: coadd template - mean science - mean difference.

Usage:
    python lc_labels.py 12345 --weights-path /path/to/model --data-path /path/to/data
"""
import os
import sys
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import base64
from io import BytesIO

# Import ML4transients data loaders
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ML4transients.data_access.dataset_loader import DatasetLoader
from ML4transients.evaluation.interpretability import embeddable_image
from ML4transients.evaluation.lightcurve_visualization import (
    load_diasource_index, 
    run_inference
)



def get_lightcurve_and_cutouts_by_band(dataset_loader, dia_object_id, weights_path=None, model_hash=None):
    """
    Load lightcurve data and cutouts organized by band.
    
    Args:
        dataset_loader: DatasetLoader instance
        dia_object_id: The diaObjectId to retrieve
        weights_path: Path to model weights (for inference)
        model_hash: Model hash (for existing inference results)
    
    Returns:
        Dict containing lightcurve data organized by band with cutouts
    """
    # Load the diasource patch index
    data_path = dataset_loader.data_paths[0]
    diasource_index_file = data_path / "lightcurves" / "diasource_patch_index.h5"
    
    diasource_index = load_diasource_index(diasource_index_file)
    if diasource_index is None:
        return None
    
    # Find all diaSourceIds for this diaObjectId
    sources_for_object = diasource_index[diasource_index['diaObjectId'] == dia_object_id]
    
    if len(sources_for_object) == 0:
        print(f"No diaSourceIds found for diaObjectId {dia_object_id}")
        return None
    
    source_ids = sources_for_object.index.tolist()
    visit_groups = sources_for_object.groupby('visit')['diaObjectId'].count().to_dict()
    
    # Data structures organized by band
    band_data = {}  # band -> list of observations
    
    for visit in visit_groups.keys():
        if visit in dataset_loader.features:
            feature_loader = dataset_loader.features[visit]
            
            # Get features for sources from this visit that belong to our object
            visit_sources = sources_for_object[sources_for_object['visit'] == visit].index.tolist()
            
            for src_id in visit_sources:
                try:
                    features = feature_loader.get_by_id(src_id)
                    if features is not None and len(features) > 0:
                        feature_row = features.iloc[0]
                        
                        # Determine the band
                        band = None
                        for col in ['band', 'filter']:
                            if col in feature_row:
                                band = feature_row[col]
                                break
                        
                        if band is None:
                            continue  # Skip if no band information
                        
                        # Initialize band entry if needed
                        if band not in band_data:
                            band_data[band] = {
                                'observations': [],
                                'diff_cutouts': [],
                                'science_cutouts': [],
                                'coadd_cutout': None  # Single template per band
                            }
                        
                        # Extract observation data
                        obs = {
                            'diaSourceId': src_id,
                            'visit': visit,
                        }
                        
                        # Add flux/magnitude information
                        for col in ['psFlux', 'psfFlux', 'flux']:
                            if col in feature_row:
                                obs['flux'] = feature_row[col]
                                break
                        
                        for col in ['psFluxErr', 'psfFluxErr', 'fluxErr']:
                            if col in feature_row:
                                obs['flux_err'] = feature_row[col]
                                break
                        
                        for col in ['midpointMjdTai', 'mjd', 'time']:
                            if col in feature_row:
                                obs['time'] = feature_row[col]
                                break
                        
                        band_data[band]['observations'].append(obs)
                        
                        # Load cutouts for this source
                        if visit in dataset_loader.cutouts:
                            cutout_loader = dataset_loader.cutouts[visit]
                            
                            # Load difference cutout
                            diff_cutout = cutout_loader.get_by_id(src_id, cutout_type='diff')
                            if diff_cutout is not None:
                                band_data[band]['diff_cutouts'].append(diff_cutout)
                            
                            # Load science cutout
                            science_cutout = cutout_loader.get_by_id(src_id, cutout_type='science')
                            if science_cutout is not None:
                                band_data[band]['science_cutouts'].append(science_cutout)
                            
                            # Load coadd template (only once per band)
                            if band_data[band]['coadd_cutout'] is None:
                                if 'coadd' in cutout_loader.available_types:
                                    coadd_cutout = cutout_loader.get_by_id(src_id, cutout_type='coadd')
                                    if coadd_cutout is not None:
                                        band_data[band]['coadd_cutout'] = coadd_cutout
                
                except Exception as e:
                    print(f"Error processing source {src_id}: {e}")
                    continue
    
    # Convert observations to DataFrames and sort by time
    for band in band_data:
        if band_data[band]['observations']:
            df = pd.DataFrame(band_data[band]['observations'])
            if 'time' in df.columns:
                df = df.sort_values('time').reset_index(drop=True)
            band_data[band]['observations'] = df
    
    # Load inference results if available
    inference_data = {}
    if weights_path or model_hash:
        print(f"Loading inference results for {len(source_ids)} sources...")
        
        # Group source IDs by visit for efficient loading
        visit_groups_dict = {}
        for src_id in source_ids:
            visit = sources_for_object[sources_for_object.index == src_id]['visit'].iloc[0]
            if visit not in visit_groups_dict:
                visit_groups_dict[visit] = []
            visit_groups_dict[visit].append(src_id)
        
        # Load inference loaders for each visit
        for visit, visit_source_ids in visit_groups_dict.items():
            try:
                inference_loader = dataset_loader.get_inference_loader(
                    visit=visit,
                    weights_path=weights_path,
                    model_hash=model_hash
                )
                
                if inference_loader and inference_loader.has_inference_results():
                    visit_inference = inference_loader.get_multiple_by_ids(visit_source_ids)
                    inference_data.update(visit_inference)
            except Exception as e:
                print(f"Error loading inference for visit {visit}: {e}")
                continue
    
    return {
        'band_data': band_data,
        'inference': inference_data,
        'object_id': dia_object_id
    }


def create_band_row_visualization(band, band_info, figsize_per_cutout=1.2):
    """
    Create visualization for one band: coadd - mean_science - mean_diff (no lightcurve).
    
    Args:
        band: Filter band identifier
        band_info: Dictionary with 'observations', 'diff_cutouts', 'science_cutouts', 'coadd_cutout'
        figsize_per_cutout: Size in inches for each cutout display
        
    Returns:
        matplotlib Figure
    """
    # Calculate mean cutouts
    mean_science = None
    mean_diff = None
    
    if band_info['science_cutouts']:
        mean_science = np.mean(band_info['science_cutouts'], axis=0)
    
    if band_info['diff_cutouts']:
        mean_diff = np.mean(band_info['diff_cutouts'], axis=0)
    
    coadd = band_info['coadd_cutout']
    
    # Determine how many cutouts we'll show (1-3)
    num_cutouts = sum([coadd is not None, mean_science is not None, mean_diff is not None])
    
    if num_cutouts == 0:
        return None
    
    # Create figure with just cutouts (compact)
    fig = plt.figure(figsize=(num_cutouts * figsize_per_cutout, figsize_per_cutout))
    gs = GridSpec(1, num_cutouts, figure=fig)
    
    # Dark background
    fig.patch.set_facecolor('#2F2F2F')
    
    cutout_idx = 0
    
    # Display coadd
    if coadd is not None:
        ax = fig.add_subplot(gs[0, cutout_idx])
        im = ax.imshow(coadd, cmap='gray', origin='lower')
        ax.set_title(f'{band} - Coadd', color='white', fontsize=9)
        ax.axis('off')
        cutout_idx += 1
    
    # Display mean science
    if mean_science is not None:
        ax = fig.add_subplot(gs[0, cutout_idx])
        im = ax.imshow(mean_science, cmap='gray', origin='lower')
        ax.set_title(f'{band} - Mean Science', color='white', fontsize=9)
        ax.axis('off')
        cutout_idx += 1
    
    # Display mean diff
    if mean_diff is not None:
        ax = fig.add_subplot(gs[0, cutout_idx])
        im = ax.imshow(mean_diff, cmap='gray', origin='lower')
        ax.set_title(f'{band} - Mean Diff', color='white', fontsize=9)
        ax.axis('off')
        cutout_idx += 1
    
    plt.tight_layout()
    return fig


def create_combined_lightcurve(band_data, inference_data, figsize=(6, 4)):
    """
    Create a single lightcurve plot with all bands combined.
    
    Args:
        band_data: Dictionary of band -> band_info
        inference_data: Dictionary of inference results
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#2F2F2F')
    ax.set_facecolor('#2F2F2F')
    
    # Band-specific colors
    band_colors = {
        'u': '#56b4e9',
        'g': '#009e73',
        'r': '#e69f00',
        'i': '#cc79a7',
        'z': '#d55e00',
        'y': '#f0e442'
    }
    
    # Sort bands for consistent ordering
    sorted_bands = sorted(band_data.keys())
    
    # Plot each band
    for band in sorted_bands:
        band_info = band_data[band]
        obs_df = band_info['observations']
        
        if len(obs_df) > 0 and 'time' in obs_df.columns and 'flux' in obs_df.columns:
            times = obs_df['time'].values
            fluxes = obs_df['flux'].values
            flux_errs = obs_df.get('flux_err', pd.Series([0] * len(obs_df))).values
            
            color = band_colors.get(band, '#5b75cd')
            
            # Plot observations with connecting lines (no error bars for cleaner view)
            ax.plot(times, fluxes, '-o', color=color, alpha=0.8, linewidth=1.2, 
                   markersize=5, markeredgecolor='white', markeredgewidth=0.5,
                   label=f'Band {band}')
            
            # Overlay inference information if available
            if inference_data:
                for idx, row in obs_df.iterrows():
                    src_id = row['diaSourceId']
                    if src_id in inference_data:
                        inf_result = inference_data[src_id]
                        
                        # Determine marker based on classification
                        prediction = inf_result.get('prediction', None)
                        label = inf_result.get('label', None)
                        
                        if prediction is not None and label is not None:
                            # Classification status
                            if prediction == 1 and label == 1:
                                marker = 'x'  # True Positive
                                marker_color = color
                            elif prediction == 0 and label == 0:
                                marker = 'o'  # True Negative
                                marker_color = color
                            elif prediction == 1 and label == 0:
                                marker = '*'  # False Positive
                                marker_color = '#ff0000'
                            elif prediction == 0 and label == 1:
                                marker = 's'  # False Negative
                                marker_color = '#ffaa00'
                            else:
                                continue
                            
                            ax.scatter(row['time'], row['flux'], marker=marker, 
                                    s=100, color=marker_color, edgecolor='white', 
                                    linewidths=1.5, zorder=10)
    
    ax.set_xlabel('MJD', color='white', fontsize=9)
    ax.set_ylabel('Flux', color='white', fontsize=9)
    ax.tick_params(colors='white', labelsize=8)
    ax.spines['bottom'].set_color('#666')
    ax.spines['top'].set_color('#666')
    ax.spines['left'].set_color('#666')
    ax.spines['right'].set_color('#666')
    ax.grid(True, alpha=0.3, color='#444')
    
    # Add legend
    ax.legend(loc='upper left', facecolor='#2F2F2F', edgecolor='#666', 
             labelcolor='white', fontsize=8)
    
    plt.tight_layout()
    return fig


def create_html_visualization(data_dict, output_file, all_object_ids=None):
    """
    Create complete HTML visualization with one row per band.
    
    Args:
        data_dict: Dictionary containing band_data, inference, and object_id
        output_file: Path to output HTML file
        all_object_ids: Optional list of all diaObjectIds for navigation
    """
    band_data = data_dict['band_data']
    inference_data = data_dict['inference']
    dia_object_id = data_dict['object_id']
    
    if not band_data:
        print("No band data available for visualization")
        return
    
    # Determine prev/next object IDs for navigation
    prev_id = None
    next_id = None
    if all_object_ids and len(all_object_ids) > 1:
        try:
            current_idx = all_object_ids.index(dia_object_id)
            if current_idx > 0:
                prev_id = all_object_ids[current_idx - 1]
            if current_idx < len(all_object_ids) - 1:
                next_id = all_object_ids[current_idx + 1]
        except ValueError:
            pass  # Current ID not in list
    
    # Sort bands alphabetically
    sorted_bands = sorted(band_data.keys())
    
    # Generate combined lightcurve plot
    lightcurve_img = None
    if band_data:
        lc_fig = create_combined_lightcurve(band_data, inference_data)
        if lc_fig is not None:
            buf = BytesIO()
            lc_fig.savefig(buf, format='png', dpi=120, facecolor='#2F2F2F', 
                          edgecolor='none', bbox_inches='tight')
            buf.seek(0)
            lightcurve_img = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close(lc_fig)
    
    # Generate figure for each band (cutouts only)
    band_figures = {}
    for band in sorted_bands:
        fig = create_band_row_visualization(band, band_data[band])
        if fig is not None:
            # Convert to base64
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=120, facecolor='#2F2F2F', 
                       edgecolor='none', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close(fig)
            
            band_figures[band] = img_base64
    
    # Calculate summary statistics
    total_observations = sum(len(band_data[b]['observations']) for b in band_data)
    total_inference = len(inference_data) if inference_data else 0
    
    # Build HTML
    html_parts = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        f'<title>Lightcurve Labels - diaObjectId {dia_object_id}</title>',
        '<style>',
        'body { background-color: #1a1a1a; color: white; font-family: Arial, sans-serif; margin: 20px; }',
        '.header { background-color: #2F2F2F; padding: 20px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #666; position: relative; }',
        '.header h1 { color: #870000; margin-top: 0; }',
        '.stats { display: flex; gap: 30px; }',
        '.stat-group { }',
        '.stat-group p { margin: 5px 0; }',
        '.navigation { position: absolute; top: 20px; right: 20px; display: flex; gap: 10px; }',
        '.nav-button { background-color: #870000; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; text-decoration: none; font-size: 16px; display: inline-flex; align-items: center; gap: 5px; }',
        '.nav-button:hover { background-color: #a00000; }',
        '.nav-button:disabled { background-color: #444; cursor: not-allowed; opacity: 0.5; }',
        '.nav-button:disabled:hover { background-color: #444; }',
        '.main-content { display: flex; gap: 20px; margin-bottom: 20px; }',
        '.cutouts-column { flex: 1; min-width: 400px; }',
        '.lightcurve-column { flex: 1; min-width: 500px; }',
        '.band-row { background-color: #2F2F2F; padding: 15px; margin-bottom: 15px; border-radius: 5px; border: 1px solid #666; }',
        '.band-row h3 { color: #870000; margin-top: 0; margin-bottom: 10px; }',
        '.band-row img { width: 100%; height: auto; }',
        '.lightcurve-section { background-color: #2F2F2F; padding: 15px; border-radius: 5px; border: 1px solid #666; position: sticky; top: 20px; }',
        '.lightcurve-section h2 { color: #870000; margin-top: 0; margin-bottom: 15px; }',
        '.lightcurve-section img { width: 100%; height: auto; }',
        '.legend { background-color: #2F2F2F; padding: 15px; border-radius: 5px; border: 1px solid #666; margin-top: 20px; }',
        '.legend h3 { color: #870000; margin-top: 0; }',
        '.legend p { font-size: 12px; margin: 5px 0; }',
        '</style>',
        '<script>',
        'document.addEventListener("keydown", function(event) {',
        '  if (event.key === "ArrowLeft") {',
        '    const prevBtn = document.getElementById("prev-button");',
        '    if (prevBtn && !prevBtn.disabled) prevBtn.click();',
        '  } else if (event.key === "ArrowRight") {',
        '    const nextBtn = document.getElementById("next-button");',
        '    if (nextBtn && !nextBtn.disabled) nextBtn.click();',
        '  }',
        '});',
        '</script>',
        '</head>',
        '<body>',
        '<div class="header">',
    ]
    
    # Add navigation buttons
    if prev_id or next_id:
        html_parts.append('<div class="navigation">')
        if prev_id:
            html_parts.append(f'<a href="lc_labels_{prev_id}.html" class="nav-button" id="prev-button">← Previous</a>')
        else:
            html_parts.append('<button class="nav-button" disabled id="prev-button">← Previous</button>')
        
        if next_id:
            html_parts.append(f'<a href="lc_labels_{next_id}.html" class="nav-button" id="next-button">Next →</a>')
        else:
            html_parts.append('<button class="nav-button" disabled id="next-button">Next →</button>')
        html_parts.append('</div>')
    
    html_parts.extend([
        f'<h1>Lightcurve Labels Visualization - diaObjectId {dia_object_id}</h1>',
        '<div class="stats">',
        '<div class="stat-group">',
        f'<p><strong>Total Observations:</strong> {total_observations}</p>',
        f'<p><strong>Number of Bands:</strong> {len(band_figures)}</p>',
        '</div>',
        '<div class="stat-group">',
        f'<p><strong>Inference Results:</strong> {total_inference}</p>',
        f'<p><strong>Bands:</strong> {", ".join(sorted_bands)}</p>',
        '</div>',
        '</div>',
        '</div>',
    ])
    
    # Create main content with two columns: cutouts on left, lightcurve on right
    html_parts.append('<div class="main-content">')
    
    # Left column: Band cutout rows
    html_parts.append('<div class="cutouts-column">')
    for band in sorted_bands:
        if band in band_figures:
            num_obs = len(band_data[band]['observations'])
            html_parts.extend([
                '<div class="band-row">',
                f'<h3>Band {band} ({num_obs} observations)</h3>',
                f'<img src="data:image/png;base64,{band_figures[band]}" alt="Band {band}">',
                '</div>',
            ])
    html_parts.append('</div>')
    
    # Right column: Combined lightcurve plot
    if lightcurve_img:
        html_parts.extend([
            '<div class="lightcurve-column">',
            '<div class="lightcurve-section">',
            '<h2>Combined Lightcurve</h2>',
            f'<img src="data:image/png;base64,{lightcurve_img}" alt="Combined Lightcurve">',
            '</div>',
            '</div>',
        ])
    
    html_parts.append('</div>')  # Close main-content
    
    # Add legend
    if inference_data:
        html_parts.extend([
            '<div class="legend">',
            '<h3>Inference Legend</h3>',
            '<p><strong>Markers on Lightcurve:</strong></p>',
            '<p>✕ True Positive (correct real detection)</p>',
            '<p>○ True Negative (correct bogus detection)</p>',
            '<p>★ False Positive (incorrectly classified as real)</p>',
            '<p>◼ False Negative (incorrectly classified as bogus)</p>',
            '</div>',
        ])
    
    html_parts.extend([
        '</body>',
        '</html>'
    ])
    
    # Write HTML file
    with open(output_file, 'w') as f:
        f.write('\n'.join(html_parts))
    
    print(f"Visualization saved to {output_file}")


def visualize(dia_object_id, data_path=None, weights_path=None, model_hash=None, 
              output_file_name=None, run_inference_flag=False, all_object_ids=None):
    """
    Create band-based lightcurve visualization.
    
    Parameters
    ----------
    dia_object_id : int
        The diaObjectId to visualize
    data_path : Path, optional
        Path to data directory 
    weights_path : str, optional
        Path to model weights for inference
    model_hash : str, optional
        Model hash for existing inference results
    output_file_name : str, optional
        Custom output HTML filename
    run_inference_flag : bool, optional
        Whether to run inference if not available (requires weights_path)
    all_object_ids : list, optional
        List of all diaObjectIds for navigation
    """
    
    # Load dataset
    dataset_loader = DatasetLoader(data_path)
    
    # Check if we need to run inference
    if weights_path and run_inference_flag:
        temp_data = get_lightcurve_and_cutouts_by_band(
            dataset_loader, dia_object_id, weights_path=None, model_hash=model_hash
        )
        
        if temp_data is None:
            print(f"No data found for diaObjectId {dia_object_id}")
            return
        
        # Get unique visits from all bands
        visits = set()
        for band_info in temp_data['band_data'].values():
            if 'observations' in band_info and isinstance(band_info['observations'], pd.DataFrame):
                if 'visit' in band_info['observations'].columns:
                    visits.update(band_info['observations']['visit'].unique())
        
        visits = list(visits)
        
        # Run inference for these visits
        if not run_inference(dataset_loader, visits, weights_path):
            print("Warning: Inference execution failed")
    
    # Load lightcurve and cutout data
    data_dict = get_lightcurve_and_cutouts_by_band(
        dataset_loader, dia_object_id, weights_path, model_hash
    )
    
    if data_dict is None:
        print(f"No data found for diaObjectId {dia_object_id}")
        return
    
    # Create visualization
    if output_file_name is None:
        output_file_name = f"lc_labels_{dia_object_id}.html"
    
    create_html_visualization(data_dict, output_file_name, all_object_ids)
    
    print(f"Visualization complete! Open {output_file_name} in your browser.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize lightcurve with band-based cutout rows (coadd - science - diff)"
    )
    parser.add_argument("dia_object_id", type=int, help="The diaObjectId to visualize")
    parser.add_argument("--data-path", type=str, help="Path to data directory")
    parser.add_argument("--weights-path", type=str, help="Path to model weights for inference")
    parser.add_argument("--model-hash", type=str, help="Model hash for existing inference results")
    parser.add_argument("--output", type=str, help="Output HTML filename")
    parser.add_argument("--run-inference", action="store_true",
                       help="Run inference if not available (requires weights-path)")
    parser.add_argument("--all-ids", type=str, help="Comma-separated list of all diaObjectIds for navigation")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path) if args.data_path else None
    
    # Parse all object IDs if provided
    all_ids = None
    if args.all_ids:
        all_ids = [int(x.strip()) for x in args.all_ids.split(',')]
    
    visualize(
        dia_object_id=args.dia_object_id,
        data_path=data_path,
        weights_path=args.weights_path,
        model_hash=args.model_hash,
        output_file_name=args.output,
        run_inference_flag=args.run_inference,
        all_object_ids=all_ids
    )
