import os
import glob
import pandas as pd
import argparse
import numpy as np

def convert_single(input_path, output_path):
    """Convert a single HDF5 lightcurve file to a cleaned CSV.

    Args:
        input_path (str): Path to the input HDF5 file.
        output_path (str): Path to save the output CSV file.

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

    des_phot = pd.DataFrame({
        "SNID": df["diaObjectId"],
        "MJD": df["midpointMjdTai"],
        "FLUXCAL": df["psfFlux"],
        "FLUXCALERR": df["psfFluxErr"],
        "FLT": df["band"].str.upper()
    })

    # Sort and reset index
    des_phot = des_phot.sort_values(by=["SNID", "MJD"]).reset_index(drop=True)

    # Remove rows with NaN flux values
    initial_size = len(des_phot)
    des_phot = des_phot.dropna(subset=['FLUXCAL', 'FLUXCALERR']).reset_index(drop=True)
    print(f"  Removed {initial_size - len(des_phot)} rows with NaN flux or flux error values")

    # Sort again for neatness
    des_phot = des_phot.sort_values(by=["SNID", "MJD"]).reset_index(drop=True)

    # --- Added: Filter mapping and normalization ---
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

            # Log transform
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

    # Add required host galaxy columns (dummy values - not used by model)
    df_processed["HOSTGAL_PHOTOZ"] = 0.0
    df_processed["HOSTGAL_PHOTOZ_ERR"] = 0.0
    df_processed["HOSTGAL_SPECZ"] = 0.0
    df_processed["HOSTGAL_SPECZ_ERR"] = 0.0
    df_processed["MWEBV"] = 0.01
    print(f"Processed: {df_processed.shape[0]} observations, {df_processed['SNID'].nunique()} objects")


    # Save to CSV
    df_processed.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}\n")

def convert_all_patches(input_dir, output_dir):
    """Convert all .h5 files in a directory to CSVs.

    Args:
        input_dir (str): Directory containing input .h5 files.
        output_dir (str): Directory to save output CSV files.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(input_dir, "*.h5")))
    if not h5_files:
        print(f"No .h5 files found in {input_dir}")
        return

    print(f"Found {len(h5_files)} .h5 files in {input_dir}\n")

    for h5_file in h5_files:
        base_name = os.path.splitext(os.path.basename(h5_file))[0]
        csv_file = os.path.join(output_dir, f"{base_name}.csv")
        convert_single(h5_file, csv_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 lightcurves to CSVs.")
    parser.add_argument("input_dir", help="Path to folder containing .h5 files")
    parser.add_argument("output_dir", help="Path to folder where CSVs will be saved")

    args = parser.parse_args()

    convert_all_patches(args.input_dir, args.output_dir)


