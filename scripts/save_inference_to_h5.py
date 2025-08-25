import os
import h5py
import pandas as pd
import glob

def save_predictions_to_h5(pred_dir, h5_dir):
    h5_files = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
    if not h5_files:
        print(f"No HDF5 files found in {h5_dir}")
        return

    for h5_file in h5_files:
        patch_name = os.path.splitext(os.path.basename(h5_file))[0]
        pred_csv = os.path.join(pred_dir, f"{patch_name}_ensemble_predictions.csv")
        
        if not os.path.exists(pred_csv):
            print(f"No predictions found for patch {patch_name}")
            continue

        patch_preds_df = pd.read_csv(pred_csv)

        # --- Only keep diaObjectId, not SNID ---
        if "SNID" in patch_preds_df.columns:
            patch_preds_df = patch_preds_df.rename(columns={"SNID": "diaObjectId"})
        # Remove SNID if present redundantly (shouldn't be after rename, but just in case)
        if "SNID" in patch_preds_df.columns and "diaObjectId" in patch_preds_df.columns:
            patch_preds_df = patch_preds_df.drop(columns=["SNID"])

        arr = patch_preds_df.to_records(index=False)

        with h5py.File(h5_file, "a") as h5f:
            if "snn_inference" in h5f:
                del h5f["snn_inference"]
            h5f.create_dataset("snn_inference", data=arr)
        
        with h5py.File(h5_file, "r") as h5f:
            print("Keys after writing:", list(h5f.keys()))

        print(f"Saved {len(patch_preds_df)} predictions to {h5_file}")


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
    args = parser.parse_args()

    save_predictions_to_h5(args.pred_dir, args.h5_dir)
