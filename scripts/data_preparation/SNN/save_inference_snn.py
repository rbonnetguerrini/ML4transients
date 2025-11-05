import os
import h5py
import pandas as pd
import glob
import numpy as np

def load_filtered_ids(csv_dir):
    """Load the set of diaObjectIds that passed SNN filtering from all patches.
    
    Args:
        csv_dir (str): Directory containing filtered_ids.txt files.
        
    Returns:
        set: Set of diaObjectIds that should have SNN inference results.
    """
    filtered_ids = set()
    filtered_files = glob.glob(os.path.join(csv_dir, "*_filtered_ids.txt"))
    
    for file_path in filtered_files:
        try:
            with open(file_path, 'r') as f:
                ids = [int(line.strip()) for line in f if line.strip()]
                filtered_ids.update(ids)
        except Exception as e:
            print(f"Warning: Could not load filtered IDs from {file_path}: {e}")
    
    return filtered_ids

def save_predictions_to_h5(pred_dir, h5_dir, csv_dir=None):
    """Save SNN predictions from CSVs into HDF5 patch files.
    Creates complete SNN inference tables with NaN for lightcurves not processed by SNN.
    Now validates that only properly filtered lightcurves get SNN results.

    Args:
        pred_dir (str): Directory containing per-patch *_ensemble_predictions.csv files.
        h5_dir (str): Directory containing HDF5 patch files.
        csv_dir (str, optional): Directory with filtered_ids.txt files for validation.

    Returns:
        None
    """
    h5_files = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
    if not h5_files:
        print(f"No HDF5 files found in {h5_dir}")
        return

    # Load the set of diaObjectIds that should have passed SNN filtering
    snn_filtered_ids = set()
    if csv_dir:
        snn_filtered_ids = load_filtered_ids(csv_dir)
        print(f"Loaded {len(snn_filtered_ids)} diaObjectIds that passed SNN filtering")
    else:
        print("Warning: No csv_dir provided, skipping filtering validation")

    validation_errors = []

    for h5_file in h5_files:
        patch_name = os.path.splitext(os.path.basename(h5_file))[0]
        pred_csv = os.path.join(pred_dir, f"{patch_name}_ensemble_predictions.csv")
        
        # Load all lightcurves in this patch
        try:
            all_lcs = pd.read_hdf(h5_file, key="lightcurves")
            all_object_ids = all_lcs['diaObjectId'].unique()
            print(f"Patch {patch_name}: {len(all_object_ids)} total objects")
        except Exception as e:
            print(f"Failed to load lightcurves from {h5_file}: {e}")
            continue
        
        # Create complete SNN inference table
        complete_snn_df = pd.DataFrame({
            'diaObjectId': all_object_ids.astype(str),  # Use string type that preserves precision
            'prob_class0_mean': np.nan,
            'prob_class1_mean': np.nan, 
            'prob_class0_std': np.nan,
            'prob_class1_std': np.nan,
            'pred_class': -1,  # Use -1 to indicate "not processed"
            'n_sources_at_inference': np.nan  # Add debug column
        })
        
        if os.path.exists(pred_csv):
            # Load SNN predictions for objects that were processed
            patch_preds_df = pd.read_csv(pred_csv, dtype={'SNID': str, 'diaObjectId': str})
            
            # Rename SNID to diaObjectId if needed
            if "SNID" in patch_preds_df.columns:
                patch_preds_df = patch_preds_df.rename(columns={"SNID": "diaObjectId"})
            
            # Ensure diaObjectId is string to prevent precision loss
            patch_preds_df['diaObjectId'] = patch_preds_df['diaObjectId'].astype(str)
            
            # VALIDATION: Check if SNN predictions match filtering expectations
            if snn_filtered_ids:
                # Convert to int only for validation comparison, but keep strings for data handling
                patch_snn_ids = set(patch_preds_df['diaObjectId'].astype(int))
                patch_expected_ids = snn_filtered_ids.intersection(set(all_object_ids.astype(int)))
                
                # Check for unexpected SNN results (objects that shouldn't have passed filtering)
                unexpected_snn = patch_snn_ids - patch_expected_ids
                if unexpected_snn:
                    error_msg = f"Patch {patch_name}: {len(unexpected_snn)} objects have SNN results but didn't pass filtering"
                    validation_errors.append(error_msg)
                    print(f"  WARNING: {error_msg}")
                    # Sample a few for inspection
                    sample_unexpected = list(unexpected_snn)[:5]
                    print(f"    Sample unexpected objects: {sample_unexpected}")
                
                # Check for missing SNN results (objects that should have passed filtering)
                missing_snn = patch_expected_ids - patch_snn_ids
                if missing_snn:
                    print(f"  Note: {len(missing_snn)} objects passed filtering but have no SNN results (inference may have failed)")
            
            # Update the complete table with actual predictions
            processed_objects = len(patch_preds_df)
            
            # Merge the predictions
            complete_snn_df = complete_snn_df.set_index('diaObjectId')
            patch_preds_df = patch_preds_df.set_index('diaObjectId')
            
            # Update only the rows that have predictions
            for col in ['prob_class0_mean', 'prob_class1_mean', 'prob_class0_std', 'prob_class1_std', 'pred_class', 'n_sources_at_inference']:
                if col in patch_preds_df.columns:
                    complete_snn_df.loc[patch_preds_df.index, col] = patch_preds_df[col]
            
            complete_snn_df = complete_snn_df.reset_index()
            
            print(f"  SNN processed: {processed_objects}/{len(all_object_ids)} objects")
            print(f"  Objects without SNN results: {len(all_object_ids) - processed_objects}")
        else:
            print(f"  No SNN predictions found for patch {patch_name}")
            print(f"  All {len(all_object_ids)} objects will have NaN SNN results")

        # Save complete table to HDF5
        # Use a pandas-compatible approach that preserves string precision
        complete_snn_df_copy = complete_snn_df.copy()
        
        # Ensure diaObjectId stays as string
        complete_snn_df_copy['diaObjectId'] = complete_snn_df_copy['diaObjectId'].astype(str)
        
        # Save using pandas to_hdf which handles strings better
        # Use table format for better compatibility
        try:
            complete_snn_df_copy.to_hdf(h5_file, key="snn_inference", mode="a", format="table", 
                                       data_columns=True, append=False)
        except Exception as pandas_error:
            print(f"  Warning: pandas to_hdf failed ({pandas_error}), trying h5py approach...")
            
            # Fallback to h5py with variable-length strings
            with h5py.File(h5_file, "a") as h5f:
                if "snn_inference" in h5f:
                    del h5f["snn_inference"]
                
                # Create variable-length string dtype for better compatibility
                str_dt = h5py.string_dtype(encoding='utf-8')
                dt = np.dtype([
                    ('diaObjectId', str_dt),
                    ('prob_class0_mean', 'f8'),
                    ('prob_class1_mean', 'f8'),
                    ('prob_class0_std', 'f8'),
                    ('prob_class1_std', 'f8'),
                    ('pred_class', 'i4'),
                    ('n_sources_at_inference', 'f8')
                ])
                
                # Convert the array
                h5_arr = np.empty(len(complete_snn_df_copy), dtype=dt)
                h5_arr['diaObjectId'] = complete_snn_df_copy['diaObjectId'].values
                h5_arr['prob_class0_mean'] = complete_snn_df_copy['prob_class0_mean'].values
                h5_arr['prob_class1_mean'] = complete_snn_df_copy['prob_class1_mean'].values  
                h5_arr['prob_class0_std'] = complete_snn_df_copy['prob_class0_std'].values
                h5_arr['prob_class1_std'] = complete_snn_df_copy['prob_class1_std'].values
                h5_arr['pred_class'] = complete_snn_df_copy['pred_class'].values
                h5_arr['n_sources_at_inference'] = complete_snn_df_copy['n_sources_at_inference'].values
                
                h5f.create_dataset("snn_inference", data=h5_arr)

        print(f"Saved complete SNN table to {h5_file}: {len(complete_snn_df)} total objects")

    # Report validation errors
    if validation_errors:
        print(f"\n{'='*60}")
        print(f"VALIDATION ERRORS FOUND ({len(validation_errors)} patches affected):")
        print(f"{'='*60}")
        for error in validation_errors:
            print(f"  • {error}")
        print(f"\nThis indicates a mismatch between filtering and inference!")
        print(f"Consider re-running the SNN pipeline to ensure consistency.")
    else:
        print(f"\n Validation passed: SNN results match filtering expectations")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Save SNN inference results into HDF5 files, patch by patch."
    )
    parser.add_argument(
        "pred_dir", help="Directory containing per-patch *_ensemble_predictions.csv"
    )
    parser.add_argument(
        "h5_dir", help="Directory containing HDF5 patch files"
    )
    parser.add_argument(
        "--csv_dir", help="Directory containing CSV files with filtered_ids.txt for validation"
    )
    args = parser.parse_args()

    save_predictions_to_h5(args.pred_dir, args.h5_dir, args.csv_dir)
