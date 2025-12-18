import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import glob
import pandas as pd
import numpy as np
import pandas as pd
from supernnova.validation.validate_onthefly import classify_lcs, get_settings

def merge_same_night_observations(df):
    """Merge and average observations from the same night for each lightcurve.
    
    This is crucial when using DES-trained models on HSC data, as DES data is 
    nightly-averaged while HSC has multiple visits per night.
    
    Args:
        df (pd.DataFrame): DataFrame with columns [SNID, MJD, FLT, FLUXCAL, FLUXCALERR, ...]
        
    Returns:
        pd.DataFrame: DataFrame with nightly-averaged observations
    """
    print(f"  Pre-nightly averaging: {len(df)} observations")
    
    # Define the night as the integer part of MJD
    df['night'] = np.floor(df['MJD']).astype(int)
    
    # Group by SNID, night, and filter
    grouped = df.groupby(['SNID', 'night', 'FLT'])
    
    # For flux: weighted average by inverse variance
    # When used in agg(), the function receives a Series, not a DataFrame
    def weighted_avg_flux(flux_series):
        # Get the corresponding flux errors from the original df
        fluxes = flux_series.values
        # Access flux errors using the same indices
        flux_errs = df.loc[flux_series.index, 'FLUXCALERR'].values
        
        # Avoid division by zero
        weights = np.where(flux_errs > 0, 1.0 / (flux_errs ** 2), 0.0)
        
        if weights.sum() == 0:
            # If all weights are zero, use simple average
            return fluxes.mean()
        else:
            return np.average(fluxes, weights=weights)
    
    def weighted_avg_flux_err(flux_err_series):
        flux_errs = flux_err_series.values
        
        # Propagate uncertainties: 1/sigma^2 = sum(1/sigma_i^2)
        weights = np.where(flux_errs > 0, 1.0 / (flux_errs ** 2), 0.0)
        
        if weights.sum() == 0:
            # If all weights are zero, use simple average
            return flux_errs.mean()
        else:
            return 1.0 / np.sqrt(weights.sum())
    
    # Aggregate: average MJD within the night, weighted average for flux
    agg_dict = {
        'MJD': 'mean',  # Average MJD within the night
        'FLUXCAL': weighted_avg_flux,
        'FLUXCALERR': weighted_avg_flux_err,
    }
    
    # Include other columns if they exist (e.g., redshift info)
    for col in df.columns:
        if col not in ['SNID', 'MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR', 'night']:
            # For other columns, take the first value (should be constant per SNID)
            if col not in agg_dict:
                agg_dict[col] = 'first'
    
    df_nightly = grouped.agg(agg_dict).reset_index()
    
    # Drop the temporary 'night' column
    df_nightly = df_nightly.drop(columns=['night'])
    
    print(f"  Post-nightly averaging: {len(df_nightly)} observations ({len(df) - len(df_nightly)} merged)")
    
    return df_nightly

def reformat_to_df(pred_probs, ids=None):
    """Convert SuperNNova prediction output to a DataFrame.

    Args:
        pred_probs (np.ndarray): Array of predicted probabilities, shape (N, 1, 2) or (N, 2).
        ids (array-like, optional): IDs to assign to each row.

    Returns:
        pd.DataFrame: DataFrame with columns ['prob_class0', 'prob_class1', 'SNID'].
    """
    arr = np.asarray(pred_probs)
    # If shape is (N, 1, 2), squeeze to (N, 2)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    elif arr.ndim == 1:
        arr = np.stack(arr)
    df_preds = pd.DataFrame(arr, columns=['prob_class0', 'prob_class1'])
    if ids is not None:
        df_preds['SNID'] = ids
    else:
        df_preds['SNID'] = np.arange(len(df_preds))
    return df_preds

def run_inference(csv_dir, output_dir):
    """Run SuperNNova inference on all CSV files in a directory.

    Args:
        csv_dir (str): Directory containing input CSV files.
        output_dir (str): Directory to save inference results.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    # Filter out debug files - be more specific about what constitutes a lightcurve CSV
    csv_files = [f for f in csv_files if not any(pattern in os.path.basename(f) for pattern in 
                ['debug_counts', 'filtered_ids', 'inference_counts', '_ensemble_predictions', '_individual_predictions'])]
    
    print(f"Found {len(csv_files)} lightcurve CSV files to process:")
    for f in csv_files[:5]:  # Show first 5 files
        print(f"  {os.path.basename(f)}")
    if len(csv_files) > 5:
        print(f"  ... and {len(csv_files) - 5} more")
    
    if not csv_files:
        print(f"No valid lightcurve CSVs found in {csv_dir}")
        return

    base_path = "/sps/lsst/users/rbonnetguerrini/ML4_transientV5/DES_Bonnet_Guerrini"
    model_files = [
        f"{base_path}/vanilla_S_0_CLF_2_R_none_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean/vanilla_S_0_CLF_2_R_none_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean.pt",
        f"{base_path}/vanilla_S_1_CLF_2_R_none_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean/vanilla_S_1_CLF_2_R_none_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean.pt",
        f"{base_path}/vanilla_S_2_CLF_2_R_none_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean/vanilla_S_2_CLF_2_R_none_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean.pt"
    ]

    all_ensemble = []
    all_individual = []

    for csv_file in csv_files:
        patch_name = os.path.splitext(os.path.basename(csv_file))[0]
        ensemble_path = os.path.join(output_dir, f"{patch_name}_ensemble_predictions.csv")
        individual_path = os.path.join(output_dir, f"{patch_name}_individual_predictions.csv")
        # Skip if ensemble predictions already exist for this patch
        if os.path.exists(ensemble_path) and os.path.exists(individual_path):
            print(f"Skipping {patch_name}: inference already done.")
            # Optionally, load and append to all_ensemble/all_individual for global files
            all_ensemble.append(pd.read_csv(ensemble_path))
            all_individual.append(pd.read_csv(individual_path))
            continue

        print(f"Running inference for patch: {patch_name}")
        df = pd.read_csv(csv_file, dtype={'SNID': str})
        
        # Ensure SNID is kept as string to prevent precision loss
        df['SNID'] = df['SNID'].astype(str)
        print(f"  SNID dtype: {df['SNID'].dtype} (strings preserve precision)")
        
        # Track original observation count before nightly averaging
        original_obs_count = len(df)
        original_lc_count = df['SNID'].nunique()
        
        print(f"  Loaded {original_obs_count} observations for {original_lc_count} lightcurves")
        
        # CRITICAL: Merge observations from the same night
        # DES models expect nightly-averaged data, but HSC has multiple visits per night
        print(f"  Applying nightly averaging (DES model compatibility)...")
        df = merge_same_night_observations(df)
        
        # Verify we still have all lightcurves after averaging
        assert df['SNID'].nunique() == original_lc_count, "Lost lightcurves during nightly averaging!"

        all_predictions = []
        for i, model_file in enumerate(model_files):
            print(f"  Model {i+1}/3 (seed {i})...")
            settings = get_settings(model_file)
            settings.redshift_label = "none"
            ids_preds, pred_probs = classify_lcs(df, model_file, "cpu")
            model_preds = reformat_to_df(pred_probs, ids=ids_preds)
            model_preds['model_seed'] = i
            all_predictions.append(model_preds)

        # Ensemble statistics for this patch
        ensemble_df = all_predictions[0][['SNID']].copy()
        prob_matrix = np.array([pred[['prob_class0', 'prob_class1']].values for pred in all_predictions])
        ensemble_df['prob_class0_mean'] = prob_matrix[:, :, 0].mean(axis=0)
        ensemble_df['prob_class1_mean'] = prob_matrix[:, :, 1].mean(axis=0)
        ensemble_df['prob_class0_std'] = prob_matrix[:, :, 0].std(axis=0)
        ensemble_df['prob_class1_std'] = prob_matrix[:, :, 1].std(axis=0)
        ensemble_df['pred_class'] = np.argmax(prob_matrix.mean(axis=0), axis=1)
        
        # Add source count debug info to ensemble results (from the CSV)
        source_counts = df.groupby('SNID')['n_sources_at_filtering'].first().reset_index()
        source_counts.columns = ['SNID', 'n_sources_at_inference']
        ensemble_df = ensemble_df.merge(source_counts, on='SNID', how='left')

        # Save patch results with string formatting to prevent precision loss
        # Use object to preserve string dtypes when saving
        ensemble_df['SNID'] = ensemble_df['SNID'].astype(str)
        ensemble_df.to_csv(ensemble_path, index=False, float_format='%.6f')
        
        # Ensure SNID is string in individual predictions too
        individual_df = pd.concat(all_predictions)
        individual_df['SNID'] = individual_df['SNID'].astype(str)
        individual_df.to_csv(individual_path, index=False)

        all_ensemble.append(ensemble_df)
        all_individual.append(individual_df)

    # Optionally, concatenate all results for global files
    if all_ensemble:
        pd.concat(all_ensemble).to_csv(os.path.join(output_dir, "ensemble_predictions.csv"), index=False)
    if all_individual:
        pd.concat(all_individual).to_csv(os.path.join(output_dir, "individual_predictions.csv"), index=False)

    print(f"\nEnsemble results: {sum(len(df) for df in all_ensemble)} objects")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SuperNNova inference on CSV lightcurves.")
    parser.add_argument("csv_dir", help="Directory containing CSV files")
    parser.add_argument("output_dir", help="Directory to save inference results")
    args = parser.parse_args()
    run_inference(args.csv_dir, args.output_dir)
