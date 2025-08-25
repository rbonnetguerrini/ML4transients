"""
Lightcurve extraction and processing module for ML4transients.

This module provides functionality to extract and cache astronomical lightcurves 
from LSST data, optimized for efficient multi-source queries.
"""

from lsst.daf.butler import Butler
import numpy as np
import pandas as pd
import os
import h5py
from typing import Dict, List, Optional, Set
from pathlib import Path
import time

def extract_and_save_lightcurves(config: dict):
    """
    Extract lightcurves from LSST data and save them efficiently organized by patch.
    
    Args:
        config (dict): Configuration dictionary containing:
            - repo: Path to data repository
            - collection: Data collection name
            - path: Base output path
            - patches: Optional list of specific patches to process
    """
    repo = config["repo"]
    collection = config["collection"]
    path_lightcurves = f"{config['path']}/lightcurves"
    
    os.makedirs(path_lightcurves, exist_ok=True)
    
    butler = Butler(repo, collections=collection)
    registry = butler.registry
    
    print("Querying available lightcurve datasets...")
    # Query all available lightcurve datasets
    datasetRefs = list(registry.queryDatasets(
        datasetType='goodSeeingDiff_assocDiaSrcTable',
        collections=collection
    ))
    
    print(f"Found {len(datasetRefs)} lightcurve datasets")
    
    # Group by patch for efficient processing
    patches_refs = {}
    for ref in datasetRefs:
        patch_key = f"{ref.dataId['tract']}_{ref.dataId['patch']}"
        if patch_key not in patches_refs:
            patches_refs[patch_key] = []
        patches_refs[patch_key].append(ref)
    
    print(f"Organized into {len(patches_refs)} patches")
    
    # Process specified patches or all if none specified
    target_patches = config.get("patches", list(patches_refs.keys()))
    
    processed_count = 0
    failed_patches = []
    
    for i, patch_key in enumerate(target_patches):
        if patch_key not in patches_refs:
            print(f"Warning: Patch {patch_key} not found in data")
            continue
            
        print(f"Processing patch {patch_key} ({i+1}/{len(target_patches)})...")
        
        try:
            # Combine all lightcurve data for this patch
            all_lightcurves = []
            for ref in patches_refs[patch_key]:
                try:
                    dia_source_table = butler.get('goodSeeingDiff_assocDiaSrcTable', dataId=ref.dataId)
                    
                    # Add patch information for reference
                    dia_source_table['tract'] = ref.dataId['tract']
                    dia_source_table['patch'] = ref.dataId['patch']
                    
                    all_lightcurves.append(dia_source_table)
                except Exception as e:
                    print(f"  Warning: Failed to load data for {ref.dataId}: {e}")
                    continue
            
            if all_lightcurves:
                # Combine all data for this patch
                combined_lc = pd.concat(all_lightcurves, ignore_index=True)
                
                # Save to HDF5 with efficient indexing
                output_file = os.path.join(path_lightcurves, f"patch_{patch_key}.h5")
                save_lightcurves_hdf5(combined_lc, output_file)
                
                print(f"  Saved patch {patch_key}: {len(combined_lc)} lightcurve points, "
                      f"{combined_lc['diaObjectId'].nunique()} unique objects")
                processed_count += 1
            else:
                print(f"  No data found for patch {patch_key}")
                failed_patches.append(patch_key)
                
        except Exception as e:
            print(f"  Failed to process patch {patch_key}: {e}")
            failed_patches.append(patch_key)
            continue
    
    print(f"\nCompleted: {processed_count}/{len(target_patches)} patches processed successfully")
    if failed_patches:
        print(f"Failed patches: {failed_patches}")

def save_lightcurves_hdf5(lightcurves_df: pd.DataFrame, path: str):
    """
    Save lightcurves DataFrame to HDF5 with optimized indexing.
    
    Args:
        lightcurves_df (pd.DataFrame): DataFrame containing lightcurve data
        path (str): Output file path for HDF5 file
    """
    # Ensure proper data types for efficient storage and indexing
    id_columns = ["diaSourceId", "diaObjectId"]
    for col in id_columns:
        if col in lightcurves_df.columns:
            lightcurves_df[col] = lightcurves_df[col].astype(np.int64)
    
    # Add time sorting for better performance
    if 'midpointMjdTai' in lightcurves_df.columns:
        lightcurves_df = lightcurves_df.sort_values(['diaObjectId', 'midpointMjdTai']).reset_index(drop=True)
    
    # Use table format with data_columns for efficient querying
    lightcurves_df.to_hdf(path, key="lightcurves", mode="w", format='table', 
                         data_columns=['diaObjectId'], index=False)

def create_lightcurve_index(config: dict) -> pd.DataFrame:
    """
    Create a cross-reference index mapping diaObjectId to patch locations.
    This enables efficient lightcurve lookups without scanning all patches.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: Index with columns [diaObjectId, tract, patch, patch_key]
    """
    repo = config["repo"]
    collection = config["collection"]
    
    butler = Butler(repo, collections=collection)
    registry = butler.registry
    
    print("Querying datasets for index creation...")
    # Query all available lightcurve datasets
    datasetRefs = list(registry.queryDatasets(
        datasetType='goodSeeingDiff_assocDiaSrcTable',
        collections=collection
    ))
    
    index_data = []
    total_refs = len(datasetRefs)
    print(f"Building lightcurve index from {total_refs} datasets...")
    
    start_time = time.time()
    for i, ref in enumerate(datasetRefs):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total_refs - i - 1) / rate if rate > 0 else 0
            print(f"  Processing dataset {i+1}/{total_refs} - {ref.dataId['tract']}_{ref.dataId['patch']} "
                  f"(Rate: {rate:.1f}/s, ETA: {eta/60:.1f}min)")
        
        try:
            # Get unique diaObjectIds for this patch
            dia_source_table = butler.get('goodSeeingDiff_assocDiaSrcTable', dataId=ref.dataId)
            unique_objects = dia_source_table['diaObjectId'].unique()
            
            patch_key = f"{ref.dataId['tract']}_{ref.dataId['patch']}"
            
            # Batch append for efficiency
            batch_data = [{
                'diaObjectId': obj_id,
                'tract': ref.dataId['tract'],
                'patch': ref.dataId['patch'],
                'patch_key': patch_key
            } for obj_id in unique_objects]
            
            index_data.extend(batch_data)
            
        except Exception as e:
            print(f"  Warning: Failed to process {ref.dataId}: {e}")
            continue
    
    if not index_data:
        raise RuntimeError("No lightcurve data found - check repository and collection settings")
    
    print(f"Creating index DataFrame with {len(index_data)} entries...")
    index_df = pd.DataFrame(index_data)
    index_df['diaObjectId'] = index_df['diaObjectId'].astype(np.int64)
    
    # Remove duplicates (same object appearing in multiple visits of same patch)
    print(f"Removing duplicates...")
    original_size = len(index_df)
    index_df = index_df.drop_duplicates(subset=['diaObjectId'], keep='first')
    final_size = len(index_df)
    print(f"Index created: {final_size} unique objects ({original_size - final_size} duplicates removed)")
    
    return index_df

def create_diasource_patch_index(config: dict) -> pd.DataFrame:
    """
    Create an index mapping diaSourceId to patch locations.
    This enables efficient cutout→lightcurve lookups.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: Index with columns [diaSourceId, diaObjectId, tract, patch, patch_key, visit]
    """
    repo = config["repo"]
    collection = config["collection"]
    
    butler = Butler(repo, collections=collection)
    registry = butler.registry
    
    print("Creating diaSourceId→patch index...")
    # Query all available lightcurve datasets
    datasetRefs = list(registry.queryDatasets(
        datasetType='goodSeeingDiff_assocDiaSrcTable',
        collections=collection
    ))
    
    index_data = []
    total_refs = len(datasetRefs)
    print(f"Building diaSourceId index from {total_refs} datasets...")
    
    start_time = time.time()
    for i, ref in enumerate(datasetRefs):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total_refs - i - 1) / rate if rate > 0 else 0
            print(f"  Processing dataset {i+1}/{total_refs} - {ref.dataId['tract']}_{ref.dataId['patch']} "
                  f"(Rate: {rate:.1f}/s, ETA: {eta/60:.1f}min)")
        
        try:
            dia_source_table = butler.get('goodSeeingDiff_assocDiaSrcTable', dataId=ref.dataId)
            patch_key = f"{ref.dataId['tract']}_{ref.dataId['patch']}"
            
            # Handle case where diaSourceId might be index or column
            if 'diaSourceId' in dia_source_table.columns:
                # diaSourceId is a column
                source_ids = dia_source_table['diaSourceId']
                object_ids = dia_source_table['diaObjectId']
                visits = dia_source_table.get('visit', [None] * len(dia_source_table))
            else:
                # diaSourceId is likely the index
                if hasattr(dia_source_table, 'index'):
                    source_ids = dia_source_table.index
                    object_ids = dia_source_table['diaObjectId']
                    visits = dia_source_table.get('visit', [None] * len(dia_source_table))
                else:
                    # Fallback: try to get from the table structure
                    print(f"  Warning: Cannot find diaSourceId in {ref.dataId}. Available columns: {list(dia_source_table.columns)}")
                    continue
            
            # Create index entries for each diaSourceId
            for j, src_id in enumerate(source_ids):
                index_data.append({
                    'diaSourceId': src_id,
                    'diaObjectId': object_ids.iloc[j] if hasattr(object_ids, 'iloc') else object_ids[j],
                    'tract': ref.dataId['tract'],
                    'patch': ref.dataId['patch'],
                    'patch_key': patch_key,
                    'visit': visits.iloc[j] if hasattr(visits, 'iloc') else (visits[j] if hasattr(visits, '__getitem__') else None)
                })
            
        except Exception as e:
            print(f"  Warning: Failed to process {ref.dataId}: {e}")
            # Print more debug info
            try:
                dia_source_table = butler.get('goodSeeingDiff_assocDiaSrcTable', dataId=ref.dataId)
                print(f"    Table type: {type(dia_source_table)}")
                print(f"    Table columns: {list(dia_source_table.columns) if hasattr(dia_source_table, 'columns') else 'No columns attr'}")
                print(f"    Has index: {hasattr(dia_source_table, 'index')}")
                if hasattr(dia_source_table, 'index'):
                    print(f"    Index name: {dia_source_table.index.name}")
            except Exception as debug_e:
                print(f"    Debug error: {debug_e}")
            continue
    
    if not index_data:
        raise RuntimeError("No diaSource data found - check repository and collection settings")
    
    print(f"Creating diaSourceId index DataFrame with {len(index_data)} entries...")
    index_df = pd.DataFrame(index_data)
    
    # Ensure proper data types
    index_df['diaSourceId'] = index_df['diaSourceId'].astype(np.int64)
    index_df['diaObjectId'] = index_df['diaObjectId'].astype(np.int64)
    
    print(f"DiaSourceId index created: {len(index_df)} entries")
    return index_df

def save_lightcurve_index(index_df: pd.DataFrame, path: str):
    """Save lightcurve index to HDF5."""
    # Sort by diaObjectId for better query performance
    index_df = index_df.sort_values('diaObjectId').reset_index(drop=True)
    index_df.set_index('diaObjectId').to_hdf(path, key="index", mode="w", format='table', 
                                            data_columns=['patch_key', 'tract', 'patch'])

def save_diasource_patch_index(index_df: pd.DataFrame, path: str):
    """Save diaSourceId→patch index to HDF5."""
    # Sort by diaSourceId for better query performance
    index_df = index_df.sort_values('diaSourceId').reset_index(drop=True)
    index_df.set_index('diaSourceId').to_hdf(path, key="diasource_index", mode="w",     format='table', data_columns=['diaObjectId', 'patch_key', 'tract', 'patch', 'visit'])

def extract_and_save_lightcurves_with_index(config: dict):
    """
    Extract lightcurves and create both object and source indices for efficient lookups.
    """
    start_time = time.time()
    
    # Validate config path
    if 'path' not in config or not config['path']:
        raise ValueError("Config must contain 'path' key with valid output directory")
    
    output_path = config['path']
    print(f"Using output path: {output_path}")
    
    # First create the diaObjectId index
    print("=== Creating diaObjectId→patch index ===")
    object_index_df = create_lightcurve_index(config)
    
    # Then create the diaSourceId index
    print("\n=== Creating diaSourceId→patch index ===")
    source_index_df = create_diasource_patch_index(config)
    
    # Save both indices using the config path
    path_lightcurves = f"{output_path}/lightcurves"
    print(f"Saving to lightcurve path: {path_lightcurves}")
    os.makedirs(path_lightcurves, exist_ok=True)
    
    object_index_path = os.path.join(path_lightcurves, "lightcurve_index.h5")
    source_index_path = os.path.join(path_lightcurves, "diasource_patch_index.h5")
    
    save_lightcurve_index(object_index_df, object_index_path)
    save_diasource_patch_index(source_index_df, source_index_path)
    
    print(f"Object index saved to {object_index_path}")
    print(f"Source index saved to {source_index_path}")
    
    # Then extract lightcurves as before
    print("\n=== Extracting lightcurve data ===")
    extract_and_save_lightcurves(config)
    
    total_time = time.time() - start_time
    print(f"\n=== Completed ===")
    print(f"Created diaObjectId index with {len(object_index_df)} object→patch mappings")
    print(f"Created diaSourceId index with {len(source_index_df)} source→patch mappings")
    print(f"Total processing time: {total_time/60:.1f} minutes")
    
    # Save extraction summary 
    summary = {
        'total_objects': len(object_index_df),
        'total_sources': len(source_index_df),
        'unique_patches': object_index_df['patch_key'].nunique(),
        'processing_time_minutes': total_time / 60,
        'config': config,
        'created_at': pd.Timestamp.now().isoformat()
    }
    
    summary_path = os.path.join(path_lightcurves, "extraction_summary.yaml")
    import yaml
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f"Extraction summary saved to {summary_path}")

def get_lightcurve_for_object(dia_object_id: int, config: dict) -> Optional[pd.DataFrame]:
    """
    Quick function to get lightcurve for a single object (for testing).
    
    Args:
        dia_object_id: The diaObjectId to retrieve
        config: Configuration dictionary
        
    Returns:
        pd.DataFrame or None: Lightcurve data if found
    """
    from .data_access.data_loaders import LightCurveLoader
    
    path_lightcurves = f"{config['path']}/lightcurves"
    if not os.path.exists(path_lightcurves):
        print(f"Lightcurve directory not found: {path_lightcurves}")
        return None
    
    loader = LightCurveLoader(Path(path_lightcurves))
    return loader.get_lightcurve(dia_object_id)

