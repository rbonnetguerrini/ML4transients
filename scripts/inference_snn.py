import os
import glob
import pandas as pd
import numpy as np
from supernnova.validation.validate_onthefly import classify_lcs, get_settings

def reformat_to_df(pred_probs, ids=None):
    """
    Convert SuperNNova prediction output to a DataFrame.
    Handles pred_probs with shape (N, 1, 2) or (N, 2).
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
    os.makedirs(output_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        print(f"No CSVs found in {csv_dir}")
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
        df = pd.read_csv(csv_file)

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

        # Save patch results
        ensemble_df.to_csv(ensemble_path, index=False)
        pd.concat(all_predictions).to_csv(individual_path, index=False)

        all_ensemble.append(ensemble_df)
        all_individual.append(pd.concat(all_predictions))

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
