"""
Cutout extraction and processing module for ML4transients.

This module provides functionality to extract astronomical image cutouts from LSST data, apply transformations, and save the results in HDF5 format for machine learning workflows.
"""

from lsst.daf.butler import Butler
from astropy.nddata import Cutout2D
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
import os
import h5py

def extract_all_single_visit(visit_id: int, collection: str, repo: str, prefix: str) -> list:
    """
    Extract all visits references for a single visit from the LSST data repository.
    
    Args:
        visit_id (int): The visit ID to query for
        collection (str): The data collection name
        repo (str): Path to the data repository
        prefix (str): Prefix for dataset type (e.g., 'injected_' or '')
        
    Returns:
        list: List of dataId dictionaries for the visit
    """
    butler = Butler(repo, collections=collection)
    registry = butler.registry
    
    # Query for difference exposure datasets for the specific visit
    result = registry.queryDatasets(
        datasetType=f'{prefix}goodSeeingDiff_differenceExp',
        collections=collection,
        where=f"visit = {visit_id}"
    )
    return [ref.dataId for ref in result]

def batch_normalize_cutouts(batch: np.ndarray) -> np.ndarray :
    """
    Apply normalization process:
        1. Remove outliers by clipping them. Done by visits to enhance the speed. 
        2. Scale the values using arcsinh
        3. Apply Z-score normalization to maintain negative value information
    """
    # batch shape: (N, H, W)
    # Independant percentile to keep a cutout level clipping 
    p1 = np.percentile(batch, 1, axis=(1,2), keepdims=True)
    p99 = np.percentile(batch, 99, axis=(1,2), keepdims=True)
    clipped = np.clip(batch, p1, p99)

    scaled = np.arcsinh(clipped)

    median = np.median(scaled, axis=(1,2), keepdims=True)
    mad = np.median(np.abs(scaled - median), axis=(1,2), keepdims=True)
    mad[mad == 0] = 1  # Avoid division by zero

    normalized = (scaled - median) / mad
    return normalized

def apply_rotations(cutout: np.ndarray, angles: list) -> list:
    """
    Apply rotation transformations to a cutout image.
    
    Args:
        cutout (np.ndarray): Input image cutout
        angles (list): List of rotation angles in degrees
        
    Returns:
        list: List of rotated cutout images
    """
    return [rotate(cutout, angle, reshape=False) for angle in angles]

def save_features_hdf5(features_df: pd.DataFrame, path: str):
    """
    Save astronomical features DataFrame to HDF5 format.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing astronomical features
        path (str): Output file path for HDF5 file
        
    Raises:
        ValueError: If diaSourceId is not found in DataFrame
    """
    # Ensure diaSourceId is set as index for efficient lookups
    if features_df.index.name != "diaSourceId":
        if "diaSourceId" in features_df.columns:
            features_df.set_index("diaSourceId", inplace=True)
        else:
            raise ValueError("diaSourceId not found in columns or index of features_df")
    
    # Use table format for efficient partial loading and querying
    features_df.to_hdf(path, key="features", mode="w", format='table', data_columns=True)

def save_cutouts_hdf5(cutouts: np.ndarray, diaSourceIds: np.ndarray, path: str):
    """
    Save image cutouts and their corresponding IDs to HDF5 format.
    
    Args:
        cutouts (np.ndarray): Array of image cutouts
        diaSourceIds (np.ndarray): Array of corresponding source IDs
        path (str): Output file path for HDF5 file
    """
    with h5py.File(path, "w") as f:
        # Save cutouts with compression to reduce file size
        f.create_dataset("cutouts", data=cutouts, compression="gzip")
        # Save corresponding source IDs as int64 to prevent overflow
        f.create_dataset("diaSourceId", data=np.array(diaSourceIds, dtype="int64"))

def create_global_cutout_index(config: dict):
    """
    Create a global index mapping diaSourceId to visit for efficient lookups.
    This should be called after all cutouts are processed.
    
    Args:
        config (dict): Configuration dictionary
    """
    print("Creating global cutout index...")
    
    path_features = f"{config['path']}/features"
    
    if not os.path.exists(path_features):
        print(f"Features directory not found: {path_features}")
        return
    
    index_data = []
    
    # Scan all feature files to build the index
    for feature_file in os.listdir(path_features):
        if feature_file.startswith("visit_") and feature_file.endswith("_features.h5"):
            # Extract visit number from filename
            try:
                visit = int(feature_file.split("_")[1])
            except (IndexError, ValueError):
                continue
                
            feature_path = os.path.join(path_features, feature_file)
            
            try:
                # Load just the diaSourceIds from the features file
                with pd.HDFStore(feature_path, 'r') as store:
                    dia_source_ids = store.select_column('features', 'index')
                
                # Add to index
                for dia_source_id in dia_source_ids:
                    index_data.append({
                        'diaSourceId': dia_source_id,
                        'visit': visit
                    })
                    
                print(f"  Indexed visit {visit}: {len(dia_source_ids)} sources")
                
            except Exception as e:
                print(f"  Warning: Failed to process {feature_file}: {e}")
                continue
    
    if not index_data:
        print("No cutout data found for indexing")
        return
    
    # Create DataFrame and save
    index_df = pd.DataFrame(index_data)
    index_df['diaSourceId'] = index_df['diaSourceId'].astype(np.int64)
    index_df['visit'] = index_df['visit'].astype(np.int32)
    
    # Sort by diaSourceId for efficient lookups
    index_df = index_df.sort_values('diaSourceId').reset_index(drop=True)
    
    # Save to HDF5 with diaSourceId as index
    index_path = os.path.join(config['path'], "cutout_global_index.h5")
    index_df.set_index('diaSourceId').to_hdf(
        index_path, 
        key="global_index", 
        mode="w", 
        format='table', 
        data_columns=['visit']
    )
    
    print(f"Global cutout index saved: {len(index_df)} entries in {index_path}")
    return index_df

def save_cutouts(config: dict):
    """
    Main function to extract, process, and save astronomical cutouts and features.
    
    This function orchestrates the entire cutout extraction pipeline:
    1. Queries data repository for available visits
    2. Extracts cutouts from difference images
    3. Processes astronomical source catalogs
    4. Saves results in HDF5 format
    
    Args:
        config (dict): Configuration dictionary containing:
            - repo: Path to data repository
            - collection: Data collection name
            - injection: Boolean flag for injection data
            - cutout: Dictionary with size, save, and return_data settings
            - path: Base output path
            - visits: Optional list of specific visits to process
    """
    # Extract configuration parameters
    repo = config["repo"]
    collection = config["collection"]
    injection = config["injection"]
    cutout_size = config["cutout"]["size"]
    save_file = config["cutout"]["save"]
    save_features_only = config["cutout"].get("save_features_only", False)
    return_file = config["cutout"]["return_data"]

    # Set up output directories
    path_cutouts = f"{config['path']}/cutouts"
    path_features = f"{config['path']}/features"

    # Determine dataset prefix based on injection flag
    if injection:
        prefix = "injected_"  # For simulated/injected sources
    else:
        prefix = ""  # For real astronomical sources
        
    # Create output directories if they don't exist
    os.makedirs(path_cutouts, exist_ok=True)
    os.makedirs(path_features, exist_ok=True)

    # Initialize Butler and registry for data access
    butler = Butler(repo, collections=collection)
    registry = butler.registry
    
    # Query all available dataset references
    datasetRefs = registry.queryDatasets(
        datasetType=f'{prefix}goodSeeingDiff_differenceExp',
        collections=collection
    )
    
    # Get all available visits or use specified subset
    all_visits = sorted(set(ref.dataId['visit'] for ref in datasetRefs))
    visits = config.get("visits", all_visits)
    
    # Process each visit individually
    for visit in visits:
        # Initialize storage for current visit
        all_cutouts = []  # Store image cutout arrays
        all_features = []  # Store astronomical features
        diaSourceIds = []  # Store source identifiers

        # Get all dataset references for current visit
        ref_ids = extract_all_single_visit(visit, collection, repo, prefix)
        
        # Process each detector/exposure in the visit
        for ref in ref_ids:
            # Load difference image data
            diff_array = butler.get(f'{prefix}goodSeeingDiff_differenceExp', dataId=ref).getImage().array
            
            # Load source catalog (astronomical detections)
            dia_src = butler.get(f'{prefix}goodSeeingDiff_diaSrcTable', dataId=ref)
            
            # Handle injection flag for simulated vs real sources
            if injection:
                # For injection runs, mark which sources are injected
                matched_src = butler.get(f'{prefix}goodSeeingDiff_matchDiaSrc', dataId=ref)
                dia_src['is_injection'] = dia_src.diaSourceId.isin(matched_src.diaSourceId)
            else:
                # For real data, no sources are injections
                dia_src['is_injection'] = 0
            
            # Initialize rotation angle (for potential augmentation)
            dia_src['rotation_angle'] = 0

            # Extract cutouts for each detected source
            for i in range(len(dia_src)):
                # Create cutout centered on source position
                cutout = Cutout2D(diff_array, (dia_src['x'][i], dia_src['y'][i]), cutout_size)
                
                # Quality control: skip cutouts with wrong size or NaN values
                if cutout.data.shape != (cutout_size, cutout_size) or np.isnan(cutout.data).any():
                    continue
                # Store valid cutouts and corresponding metadata
                all_cutouts.append(cutout.data)
                all_features.append(dia_src.iloc[i])
                diaSourceIds.append(dia_src.iloc[i]["diaSourceId"])

        # Convert features to DataFrame for easier handling
        features_df = pd.DataFrame(all_features)

        # Data type management: ensure ID columns are not subject to floating point errors
        id_columns = ["diaSourceId", "diaObjectId"]  

        for col in id_columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].astype(np.int64)

        # Ensure diaSourceIds array is int64 for cutout saving
        diaSourceIds = np.array(diaSourceIds, dtype=np.int64)
        cutouts = np.stack(all_cutouts)  # Shape: (N, 30, 30)
        normalized_cutouts = batch_normalize_cutouts(cutouts)
        # Save data based on configuration flags
        if save_file and not save_features_only:
            # Save both cutout images and features
            save_cutouts_hdf5(np.array(normalized_cutouts), diaSourceIds, os.path.join(path_cutouts, f"visit_{visit}.h5"))
            save_features_hdf5(features_df, os.path.join(path_features, f"visit_{visit}_features.h5"))
        elif save_features_only:
            # Save only the feature catalog (no image data)
            save_features_hdf5(features_df, os.path.join(path_features, f"visit_{visit}_features.h5"))

        # Progress reporting
        print(f"Saved visit {visit}: {len(normalized_cutouts)} cutouts, {len(features_df)} features")
    
    # Create global index after all visits are processed - but only if not in batch mode
    skip_global_index = config.get("skip_global_index", False)
    if not skip_global_index:
        print("\n=== Creating global cutout index ===")
        create_global_cutout_index(config)
    else:
        print("\n=== Skipping global cutout index creation (batch mode) ===")