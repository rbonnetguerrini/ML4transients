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
import gc
import sys
import time
from datetime import timedelta

def rotate_xy_ccw(x, y, w, h, nq):
    """
    Rotate pixel coordinates counter-clockwise by nq quarter-turns.
    
    Args:
        x, y: Original pixel coordinates
        w, h: Width and height of the original (unrotated) array
        nq: Number of quarter-turns counter-clockwise
        
    Returns:
        tuple: Rotated (x, y) coordinates
    """
    nq = nq % 4
    if nq == 0:
        return x, y
    elif nq == 1:   # 90° CCW
        return (h - 1 - y), x
    elif nq == 2:   # 180°
        return (w - 1 - x), (h - 1 - y)
    else:           # 270° CCW
        return y, (w - 1 - x)

def rot90_array(a, nq):
    """
    Rotate array counter-clockwise by nq quarter-turns.
    
    Args:
        a: Input array
        nq: Number of quarter-turns counter-clockwise
        
    Returns:
        np.ndarray: Rotated array
    """
    return np.rot90(a, k=nq % 4)

def compute_xy(I0: int, R0: int, r: float, search_radius: int = 200) -> tuple:
    """
    Calculate optimal number of samples to relabel and remove for noise injection.
    
    Args:
        I0 (int): Initial number of injections
        R0 (int): Initial number of real sources
        r (float): Desired noise rate in the REAL class (fraction of former injections)
        search_radius (int): How far around the continuous solution to search for an integer one
        
    Returns:
        tuple: (x_best, y_best, F_best, noise_best) where:
            - x_best: number of injections to relabel as real
            - y_best: number of real samples to remove
            - F_best: final size of each class (both equal to F_best)
            - noise_best: actual achieved noise rate in the real class
    """
    # Continuous (ideal) solution from the formulas
    F_cont = I0 / (1 + r)
    x_cont = r * F_cont

    # Integer search around the continuous solution
    start_x = max(0, int(round(x_cont)) - search_radius)
    end_x   = min(I0, int(round(x_cont)) + search_radius)

    best = None  # (error, x, y, F, noise)

    for x in range(start_x, end_x + 1):
        F = I0 - x               # final injections
        if F <= 0:
            continue

        y = R0 + x - F           # from R0 + x - y = F  --> y = R0 + x - F
        if y < 0:
            continue

        noise = x / F            # fraction of former injections in real
        error = abs(noise - r)

        if best is None or error < best[0]:
            best = (error, x, y, F, noise)

    if best is None:
        raise ValueError("No feasible integer solution found. Try another r.")

    _, x_best, y_best, F_best, noise_best = best
    return x_best, y_best, F_best, noise_best

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

def batch_normalize_cutouts(batch: np.ndarray) -> np.ndarray:
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


def minmax_normalize_cutouts(batch: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization per image.
    
    Scales each image independently to the range [0, 1] based on its
    minimum and maximum values.
    
    Args:
        batch (np.ndarray): Input batch of shape (N, H, W)
        
    Returns:
        np.ndarray: Normalized batch with values in [0, 1]
    """
    # batch shape: (N, H, W)
    min_vals = batch.min(axis=(1, 2), keepdims=True)
    max_vals = batch.max(axis=(1, 2), keepdims=True)
    
    # Avoid division by zero for constant images
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    
    normalized = (batch - min_vals) / range_vals
    return normalized


def normalize_cutouts(batch: np.ndarray, method: str = "batch") -> np.ndarray:
    """
    Apply normalization to cutouts using the specified method.
    
    Args:
        batch (np.ndarray): Input batch of shape (N, H, W)
        method (str): Normalization method. Options:
            - "batch": Percentile clipping + arcsinh + z-score (default)
            - "minmax": Min-max normalization per image to [0, 1]
            
    Returns:
        np.ndarray: Normalized batch
        
    Raises:
        ValueError: If unknown normalization method is specified
    """
    if method == "batch":
        return batch_normalize_cutouts(batch)
    elif method == "minmax":
        return minmax_normalize_cutouts(batch)
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'batch' or 'minmax'.")

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

def save_cutouts_hdf5(cutouts: np.ndarray, diaSourceIds: np.ndarray, path: str, 
                      coadd_cutouts: np.ndarray = None, science_cutouts: np.ndarray = None):
    """
    Save image cutouts and their corresponding IDs to HDF5 format.
    
    Args:
        cutouts (np.ndarray): Array of difference image cutouts
        diaSourceIds (np.ndarray): Array of corresponding source IDs
        path (str): Output file path for HDF5 file
        coadd_cutouts (np.ndarray, optional): Array of coadd (template) image cutouts
        science_cutouts (np.ndarray, optional): Array of science (calexp) image cutouts
    """
    with h5py.File(path, "w") as f:
        # Save difference cutouts with compression to reduce file size
        f.create_dataset("cutouts", data=cutouts, compression="gzip")
        # Save coadd cutouts if provided
        if coadd_cutouts is not None:
            f.create_dataset("coadd_cutouts", data=coadd_cutouts, compression="gzip")
        # Save science cutouts if provided
        if science_cutouts is not None:
            f.create_dataset("science_cutouts", data=science_cutouts, compression="gzip")
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
    
    # Load pre-computed noise perturbation IDs if available
    noise_perturbation = None
    perturbation_file = os.path.join(config['path'], 'noise_perturbation.yaml')
    
    if os.path.exists(perturbation_file):
        import yaml
        with open(perturbation_file, 'r') as f:
            noise_perturbation = yaml.safe_load(f)
        
        print(f"\n{'='*70}")
        print(f"Loaded noise perturbation plan from: {perturbation_file}")
        print(f"  Target noise rate: {noise_perturbation['noise_rate']}")
        print(f"  Injections to relabel: {noise_perturbation['x_relabel']}")
        print(f"  Real samples to remove: {noise_perturbation['y_remove']}")
        print(f"  Final class size: {noise_perturbation['F_final']}")
        print(f"  Actual noise rate: {noise_perturbation['noise_actual']:.6f}")
        print(f"{'='*70}\n")
        sys.stdout.flush()
        
        # Convert ID lists to sets for fast lookup
        relabel_ids_set = set(noise_perturbation['relabel_ids'])
        remove_ids_set = set(noise_perturbation['remove_ids'])
        
        noise_perturbation['relabel_ids_set'] = relabel_ids_set
        noise_perturbation['remove_ids_set'] = remove_ids_set
    
    # Track processing statistics
    processed_visits = []
    skipped_visits = []
    total_visits = len(visits)
    total_sources_processed = 0
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"PROCESSING SUMMARY")
    print(f"Total visits to process: {total_visits}")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    sys.stdout.flush()
    
    # Process each visit individually
    for visit_idx, visit in enumerate(visits, 1):
        # Check if visit has already been processed (checkpoint)
        visit_cutout_file = os.path.join(path_cutouts, f"visit_{visit}.h5")
        visit_feature_file = os.path.join(path_features, f"visit_{visit}_features.h5")
        
        if save_file and not save_features_only:
            # Check if both cutout and feature files exist
            if os.path.exists(visit_cutout_file) and os.path.exists(visit_feature_file):
                print(f"Skipping visit {visit}: already processed")
                sys.stdout.flush()
                skipped_visits.append(visit)
                continue
        elif save_features_only:
            # Check if feature file exists
            if os.path.exists(visit_feature_file):
                print(f"Skipping visit {visit}: already processed")
                sys.stdout.flush()
                skipped_visits.append(visit)
                continue
        
        # Initialize storage for current visit
        all_cutouts = []  # Store difference image cutout arrays
        all_coadd_cutouts = []  # Store coadd (template) image cutout arrays
        all_science_cutouts = []  # Store science (calexp) image cutout arrays
        all_features = []  # Store astronomical features
        diaSourceIds = []  # Store source identifiers

        # Get all dataset references for current visit
        ref_ids = extract_all_single_visit(visit, collection, repo, prefix)
        visit_start_time = time.time()
        
        # Calculate progress statistics
        elapsed_time = time.time() - start_time
        visits_processed = visit_idx - 1  # Visits completed so far
        visits_remaining = total_visits - visit_idx
        
        if visits_processed > 0:
            avg_time_per_visit = elapsed_time / visits_processed
            estimated_remaining_time = avg_time_per_visit * visits_remaining
            eta_str = str(timedelta(seconds=int(estimated_remaining_time)))
        else:
            eta_str = "Calculating..."
        
        print(f"\n{'='*70}")
        print(f"VISIT {visit_idx}/{total_visits}: {visit} ({len(ref_ids)} detectors)")
        print(f"Progress: {visit_idx/total_visits*100:.1f}% | Elapsed: {str(timedelta(seconds=int(elapsed_time)))} | ETA: {eta_str}")
        if total_sources_processed > 0:
            print(f"Sources processed so far: {total_sources_processed:,}")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        # Get camera for detector orientation information
        camera = butler.get("camera", instrument="HSC")
        
        # Process each detector/exposure in the visit
        for detector_idx, ref in enumerate(ref_ids):
            # Get detector orientation (number of quarter-turns)
            det = camera[ref["detector"]]
            nq = det.getOrientation().getNQuarter()
            
            # Load arrays (unrotated from raw detector frame)
            diff_array_u = butler.get(f'{prefix}goodSeeingDiff_differenceExp', dataId=ref).getImage().array
            coadd_array_u = butler.get(f'{prefix}goodSeeingDiff_templateExp', dataId=ref).getImage().array
            science_array_u = butler.get(f'{prefix}calexp', dataId=ref).getImage().array
            
            # Rotate arrays to focal-plane-aligned orientation
            diff_array = rot90_array(diff_array_u, nq)
            coadd_array = rot90_array(coadd_array_u, nq)
            science_array = rot90_array(science_array_u, nq)
            
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
            
            # Precompute shapes for coordinate rotation
            h_d, w_d = diff_array_u.shape
            h_s, w_s = science_array_u.shape
            h_c, w_c = coadd_array_u.shape

            # Extract cutouts for each detected source
            for i in range(len(dia_src)):
                # Get source coordinates in unrotated frame
                x_u = float(dia_src['x'][i])
                y_u = float(dia_src['y'][i])
                
                # Rotate source coordinates for diff/science images
                x_d, y_d = rotate_xy_ccw(x_u, y_u, w_d, h_d, nq)
                x_s, y_s = rotate_xy_ccw(x_u, y_u, w_s, h_s, nq)
                
                # Coadd uses magic +20,+20 offset in UNROTATED frame, then rotate
                # Magic offset: due to convolution for the coadd. Coadd is 20 pixels larger on each side
                # in this version of the pipeline: (4176, 2048) normal CCD VS (4216, 2088) for Coadded image.
                x_c_u = x_u + 20.0
                y_c_u = y_u + 20.0
                x_c, y_c = rotate_xy_ccw(x_c_u, y_c_u, w_c, h_c, nq)
                
                # Create cutout centered on source position from difference image
                cutout = Cutout2D(diff_array, (x_d, y_d), cutout_size)
                
                # Quality control: skip cutouts with wrong size or NaN values
                if cutout.data.shape != (cutout_size, cutout_size) or np.isnan(cutout.data).any():
                    continue
                
                # Create cutouts from coadd (template) image
                coadd_cutout = Cutout2D(coadd_array, (x_c, y_c), cutout_size)
                
                # Create cutouts from science (calexp) image
                science_cutout = Cutout2D(science_array, (x_s, y_s), cutout_size)
                
                # Quality control: skip if any cutout has wrong size or NaN values
                if (coadd_cutout.data.shape != (cutout_size, cutout_size) or np.isnan(coadd_cutout.data).any() or
                    science_cutout.data.shape != (cutout_size, cutout_size) or np.isnan(science_cutout.data).any()):
                    continue
                
                # Store valid cutouts and corresponding metadata
                all_cutouts.append(cutout.data)
                all_coadd_cutouts.append(coadd_cutout.data)
                all_science_cutouts.append(science_cutout.data)
                all_features.append(dia_src.iloc[i])
                diaSourceIds.append(dia_src.iloc[i]["diaSourceId"])
            
            # Clean up large arrays after each detector to free memory
            del diff_array_u, coadd_array_u, science_array_u
            del diff_array, coadd_array, science_array, dia_src
            
            # Progress update every 10 detectors to show detector-level progress
            if (detector_idx + 1) % 10 == 0 or (detector_idx + 1) == len(ref_ids):
                det_progress = (detector_idx + 1) / len(ref_ids) * 100
                print(f"  Detectors: {detector_idx + 1}/{len(ref_ids)} ({det_progress:.0f}%) | Cutouts: {len(all_cutouts):,}", end='\r')
                sys.stdout.flush()
        
        # Clear the progress line and print final count
        print(f"\n  \u2713 Extracted {len(all_cutouts):,} cutouts from {len(ref_ids)} detectors")
        sys.stdout.flush()
        
        # Skip visit if no valid cutouts were extracted
        if len(all_cutouts) == 0:
            print(f"Skipping visit {visit}: no valid cutouts extracted")
            sys.stdout.flush()
            skipped_visits.append(visit)
            continue

        # Convert features to DataFrame for easier handling
        features_df = pd.DataFrame(all_features)
        
        # Data type management: ensure ID columns are not subject to floating point errors
        id_columns = ["diaSourceId", "diaObjectId"]  

        for col in id_columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].astype(np.int64)

        # Apply noise perturbation if pre-computed perturbation exists
        if noise_perturbation is not None:
            # Save original labels to spy_injected (ground truth)
            features_df['spy_injected'] = features_df['is_injection'].astype(np.int8)
            
            # Convert is_injection to int8 for modification
            features_df['is_injection'] = features_df['is_injection'].astype(np.int8)
            
            relabel_ids_set = noise_perturbation['relabel_ids_set']
            remove_ids_set = noise_perturbation['remove_ids_set']
            
            # Check which sources in this visit should be relabeled
            # Relabel injections as real in is_injection (the working label)
            relabel_mask = features_df['diaSourceId'].isin(relabel_ids_set)
            n_relabeled = relabel_mask.sum()
            if n_relabeled > 0:
                features_df.loc[relabel_mask, 'is_injection'] = np.int8(0)
                print(f"  Relabeled {n_relabeled} injections as real (is_injection modified)")
                sys.stdout.flush()
            
            # Check which sources in this visit should be removed
            remove_mask = features_df['diaSourceId'].isin(remove_ids_set)
            remove_indices = features_df[remove_mask].index.tolist()
            n_removed = len(remove_indices)
            
            if n_removed > 0:
                # Remove from features_df
                features_df = features_df.drop(remove_indices)
                
                # Also remove corresponding cutouts and IDs
                keep_mask = np.ones(len(all_cutouts), dtype=bool)
                keep_mask[remove_indices] = False
                
                # Filter cutouts
                all_cutouts = [cutout for i, cutout in enumerate(all_cutouts) if keep_mask[i]]
                all_coadd_cutouts = [cutout for i, cutout in enumerate(all_coadd_cutouts) if keep_mask[i]]
                all_science_cutouts = [cutout for i, cutout in enumerate(all_science_cutouts) if keep_mask[i]]
                diaSourceIds = [sid for i, sid in enumerate(diaSourceIds) if keep_mask[i]]
                
                print(f"  Removed {n_removed} real samples")
                sys.stdout.flush()
            
            # Reset index after removal
            features_df = features_df.reset_index(drop=True)

        # Ensure diaSourceIds array is int64 for cutout saving
        diaSourceIds = np.array(diaSourceIds, dtype=np.int64)
        
        # Convert to numpy arrays and normalize in chunks to save memory
        print(f"Normalizing cutouts...")
        sys.stdout.flush()
        
        # Stack arrays - this is memory intensive
        cutouts = np.stack(all_cutouts)  # Shape: (N, 30, 30)
        del all_cutouts  # Free memory immediately
        
        coadd_cutouts = np.stack(all_coadd_cutouts)  # Shape: (N, 30, 30)
        del all_coadd_cutouts  # Free memory immediately
        
        science_cutouts = np.stack(all_science_cutouts)  # Shape: (N, 30, 30)
        del all_science_cutouts  # Free memory immediately
        
        # Get normalization method from config (default: "batch")
        normalization_method = config.get("cutout", {}).get("normalization", "batch")
        
        # Normalize all three types
        print(f"Normalizing {len(cutouts)} cutouts using '{normalization_method}' method")
        sys.stdout.flush()
        normalized_cutouts = normalize_cutouts(cutouts, method=normalization_method)
        del cutouts  # Free memory after normalization
        
        normalized_coadd_cutouts = normalize_cutouts(coadd_cutouts, method=normalization_method)
        del coadd_cutouts  # Free memory after normalization
        
        normalized_science_cutouts = normalize_cutouts(science_cutouts, method=normalization_method)
        del science_cutouts  # Free memory after normalization
        
        # Save data based on configuration flags
        if save_file and not save_features_only:
            # Save all cutout images (difference, coadd, science) and features
            print(f"Saving to disk...")
            sys.stdout.flush()
            save_cutouts_hdf5(
                np.array(normalized_cutouts), 
                diaSourceIds, 
                os.path.join(path_cutouts, f"visit_{visit}.h5"),
                coadd_cutouts=np.array(normalized_coadd_cutouts),
                science_cutouts=np.array(normalized_science_cutouts)
            )
            save_features_hdf5(features_df, os.path.join(path_features, f"visit_{visit}_features.h5"))
        elif save_features_only:
            # Save only the feature catalog (no image data)
            save_features_hdf5(features_df, os.path.join(path_features, f"visit_{visit}_features.h5"))

        # Progress reporting
        print(f"\nSaved visit {visit}:")
        print(f"  - {len(normalized_cutouts)} difference cutouts")
        print(f"  - {len(normalized_coadd_cutouts)} coadd cutouts")
        print(f"  - {len(normalized_science_cutouts)} science cutouts")
        print(f"  - {len(features_df)} features")
        sys.stdout.flush()
        
        # Clean up memory after saving
        # Update statistics BEFORE deleting variables
        visit_time = time.time() - visit_start_time
        visit_sources = len(diaSourceIds)
        total_sources_processed += visit_sources
        
        del normalized_cutouts, normalized_coadd_cutouts, normalized_science_cutouts
        del features_df, diaSourceIds
        gc.collect()
        
        print(f"Visit {visit} completed in {timedelta(seconds=int(visit_time))} ({visit_sources:,} sources)")
        sys.stdout.flush()
        
        # Mark visit as successfully processed
        processed_visits.append(visit)
    
    # Print final summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total visits processed: {len(processed_visits)}/{total_visits}")
    print(f"Visits skipped (already done): {len(skipped_visits)}")
    print(f"Total sources processed: {total_sources_processed:,}")
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    if len(processed_visits) > 0:
        print(f"Average time per visit: {timedelta(seconds=int(total_time/len(processed_visits)))}")
        print(f"Average sources per visit: {total_sources_processed/len(processed_visits):.1f}")
    print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    # Create global index after all visits are processed - but only if not in batch mode
    skip_global_index = config.get("skip_global_index", False)
    if not skip_global_index:
        create_global_cutout_index(config)
    else:
        print("Skipping global index creation (batch mode)")
        sys.stdout.flush()