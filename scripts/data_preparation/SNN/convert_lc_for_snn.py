import os
import glob
import pandas as pd
import argparse
import numpy as np

def convert_single(input_path, output_path, stats=None):
    """Convert a single HDF5 lightcurve file to a cleaned CSV.
    
    NOTE: SNR, time window, and quality filtering is now performed earlier in the pipeline
    by filter_snr_and_quality.py. This function only handles format conversion and normalization.

    Args:
        input_path (str): Path to the input HDF5 file (already filtered).
        output_path (str): Path to save the output CSV file.
        stats (dict): Dictionary to update with conversion counts.

    Returns:
        None
    """
    print(f"Processing: {input_path}")
    
    # Try reading the HDF5 file
    try:
        df = pd.read_hdf(input_path, key='lightcurves')
    except Exception as e:
        print(f" Failed to read {input_path}: {e}")
        return  # Skip this file

    # Create the DataFrame with explicit data types to prevent precision loss
    des_phot = pd.DataFrame({
        "SNID": df["diaObjectId"].astype(str),  # Use string to prevent precision loss
        "MJD": df["midpointMjdTai"].astype(np.float64),
        "FLUXCAL": df["psfFlux"].astype(np.float64),
        "FLUXCALERR": df["psfFluxErr"].astype(np.float64),
        "FLT": df["band"].str.upper()
    })
        
    # Sort and reset index
    des_phot = des_phot.sort_values(by=["SNID", "MJD"]).reset_index(drop=True)

    # Remove rows with NaN flux values (minimal - most filtering already done)
    initial_size = len(des_phot)
    initial_objects = des_phot['SNID'].nunique()
    des_phot = des_phot.dropna(subset=['FLUXCAL', 'FLUXCALERR']).reset_index(drop=True)
    if initial_size > len(des_phot):
        print(f"  Removed {initial_size - len(des_phot)} rows with NaN flux or flux error values")
    
    n_objects = des_phot['SNID'].nunique()
    
    if stats is not None:
        stats['objects_converted'] += n_objects
        stats['sources_converted'] += len(des_phot)

    # Sort again for neatness
    des_phot = des_phot.sort_values(by=["SNID", "MJD"]).reset_index(drop=True)

    # Map filters and drop non-standard ones
    filter_mapping = {'G': 'g', 'R': 'r', 'I': 'i', 'Z': 'z', 'Y': 'z'}
    df_processed = des_phot.copy()
    df_processed['FLT'] = df_processed['FLT'].map(filter_mapping)
    df_processed = df_processed.dropna(subset=['FLT'])

    # Drop any remaining NaNs in FLUXCAL/FLUXCALERR before normalization
    df_processed = df_processed.dropna(subset=['FLUXCAL', 'FLUXCALERR']).reset_index(drop=True)

    # Apply training normalization parameters
    norm_params = {
        "FLUXCAL_g": {"mean": 7.938, "std": 0.0227}, "FLUXCAL_i": {"mean": 7.938, "std": 0.0227},
        "FLUXCAL_r": {"mean": 7.938, "std": 0.0227}, "FLUXCAL_z": {"mean": 7.938, "std": 0.0227},
        "FLUXCALERR_g": {"mean": -1.767, "std": 6.202}, "FLUXCALERR_i": {"mean": -1.767, "std": 6.202},
        "FLUXCALERR_r": {"mean": -1.767, "std": 6.202}, "FLUXCALERR_z": {"mean": -1.767, "std": 6.202},
        "delta_time": {"mean": 0.787, "std": 2.482}
    }

    # Normalize fluxes and flux errors
    for flt in ['g', 'r', 'i', 'z']:
        mask = df_processed['FLT'] == flt
        if mask.any():
            flux_vals = df_processed.loc[mask, 'FLUXCAL'].values
            flux_err_vals = df_processed.loc[mask, 'FLUXCALERR'].values

            # Replace NaNs with a small value to avoid log10 warnings (should be none after dropna)
            flux_vals = np.nan_to_num(flux_vals, nan=1e-10)
            flux_err_vals = np.nan_to_num(flux_err_vals, nan=1e-10)

            # Log transform with proper handling of negative/zero values
            with np.errstate(invalid='ignore'):
                log_flux = np.where(flux_vals > 0, np.log10(flux_vals), -2000.0)
                log_flux_err = np.log10(np.maximum(flux_err_vals, 1e-10))

            # Normalize
            flux_key, err_key = f"FLUXCAL_{flt}", f"FLUXCALERR_{flt}"
            log_flux = (log_flux - norm_params[flux_key]["mean"]) / norm_params[flux_key]["std"]
            log_flux_err = (log_flux_err - norm_params[err_key]["mean"]) / norm_params[err_key]["std"]

            df_processed.loc[mask, 'FLUXCAL'] = log_flux
            df_processed.loc[mask, 'FLUXCALERR'] = log_flux_err

    # Calculate and normalize delta_time (faster: use groupby().transform)
    min_mjd = df_processed.groupby('SNID')['MJD'].transform('min')
    df_processed['delta_time'] = df_processed['MJD'] - min_mjd
    df_processed['delta_time'] = (df_processed['delta_time'] - norm_params["delta_time"]["mean"]) / norm_params["delta_time"]["std"]

    # Add source count for debugging (before host galaxy columns)
    snid_counts = df_processed['SNID'].value_counts()
    df_processed['n_sources_at_filtering'] = df_processed['SNID'].map(snid_counts)

    # Add required host galaxy columns (dummy values - not used by model)
    df_processed["HOSTGAL_PHOTOZ"] = 0.0
    df_processed["HOSTGAL_PHOTOZ_ERR"] = 0.0
    df_processed["HOSTGAL_SPECZ"] = 0.0
    df_processed["HOSTGAL_SPECZ_ERR"] = 0.0
    df_processed["MWEBV"] = 0.01
    
    print(f"Converted: {df_processed.shape[0]} observations, {df_processed['SNID'].nunique()} objects")

    # Save to CSV only if there are any lightcurves left
    if not df_processed.empty:
        # Save with explicit formatting to prevent precision loss on large integers
        df_processed.to_csv(output_path, index=False, float_format='%.6f')
        print(f"  Saved {len(df_processed)} observations for {df_processed['SNID'].nunique()} objects")
    else:
        print(f"  No valid lightcurves left after conversion, skipping save.\n")


def convert_all_patches(input_dir, output_dir):
    """Convert all .h5 files in a directory to CSVs.
    
    NOTE: Input files should already be filtered by filter_snr_and_quality.py

    Args:
        input_dir (str): Directory containing input .h5 files (already filtered).
        output_dir (str): Directory to save output CSV files.

    Returns:
        dict: Conversion statistics
    
        output_dir (str): Directory to save output CSV files.
        min_obs (int): Minimum number of observations in the time window to keep a lightcurve.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    h5_files = glob.glob(os.path.join(input_dir, "*.h5"))
    # Filter out index files that don't contain lightcurve data
    h5_files = [f for f in h5_files if not any(name in os.path.basename(f) for name in ['index', 'diasource_patch_index', 'lightcurve_index', 'metadata'])]
    
    print(f"Found {len(h5_files)} lightcurve .h5 files in {input_dir}")
    
    if not h5_files:
        print("No valid HDF5 lightcurve files found.")
        return {}

    stats = {
        'objects_converted': 0,
        'sources_converted': 0
    }
    
    for h5_file in h5_files:
        base_name = os.path.splitext(os.path.basename(h5_file))[0]
        csv_file = os.path.join(output_dir, f"{base_name}.csv")
        convert_single(h5_file, csv_file, stats=stats)
    
    # Print summary
    print(f"\n=== Conversion Summary ===")
    print(f"  Files processed: {len(h5_files)}")
    print(f"  Objects converted: {stats['objects_converted']}")
    print(f"  Sources converted: {stats['sources_converted']}")
    
    # Save summary to JSON
    import json
    from datetime import datetime
    summary_file = os.path.join(output_dir, "conversion_summary.json")
    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': input_dir,
        'output_dir': output_dir,
        'files_processed': len(h5_files),
        'objects_converted': stats['objects_converted'],
        'sources_converted': stats['sources_converted']
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nConversion summary saved to: {summary_file}")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 lightcurves to CSVs.")
    parser.add_argument("input_dir", help="Path to folder containing .h5 files (already filtered)")
    parser.add_argument("output_dir", help="Path to folder where CSVs will be saved")
    args = parser.parse_args()
    convert_all_patches(args.input_dir, args.output_dir)


