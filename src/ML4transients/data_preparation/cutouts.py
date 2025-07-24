from lsst.daf.butler import Butler
from astropy.nddata import Cutout2D
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
import os
import h5py

def extract_all_single_visit(visit_id: int, collection: str, repo: str, prefix : str) -> list:
    butler = Butler(repo, collections=collection)
    registry = butler.registry
    result = registry.queryDatasets(
        datasetType=f'{prefix}goodSeeingDiff_differenceExp',
        collections=collection,
        where=f"visit = {visit_id}"
    )
    return [ref.dataId for ref in result]

def apply_rotations(cutout: np.ndarray, angles: list) -> list:
    return [rotate(cutout, angle, reshape=False) for angle in angles]

def save_features_hdf5(features_df, path):
    if features_df.index.name != "diaSourceId":
        if "diaSourceId" in features_df.columns:
            features_df.set_index("diaSourceId", inplace=True)
        else:
            raise ValueError("diaSourceId not found in columns or index of features_df")
    features_df.to_hdf(path, key="features", mode="w")

def save_cutouts_hdf5(cutouts, diaSourceIds, path):
    with h5py.File(path, "w") as f:
        f.create_dataset("cutouts", data=cutouts, compression="gzip")
        f.create_dataset("diaSourceId", data=np.array(diaSourceIds, dtype="int64"))

def save_cutouts(config: dict):
    repo = config["repo"]
    collection = config["collection"]
    injection = config["injection"]
    cutout_size = config["cutout"]["size"]
    save_file = config["cutout"]["save"]
    save_features_only = config["cutout"].get("save_features_only", False)
    return_file = config["cutout"]["return_data"]

    path_cutouts = f"{config['path']}/cutouts"
    path_features = f"{config['path']}/features"

    if injection:
        prefix = "injected_"
    else:
        prefix = ""
        
    os.makedirs(path_cutouts, exist_ok=True)
    os.makedirs(path_features, exist_ok=True)

    butler = Butler(repo, collections=collection)
    registry = butler.registry
    datasetRefs = registry.queryDatasets(
        datasetType=f'{prefix}goodSeeingDiff_differenceExp',
        collections=collection
    )
    
    all_visits = sorted(set(ref.dataId['visit'] for ref in datasetRefs))
    visits = config.get("visits", all_visits)
    for visit in visits:
        all_cutouts = []
        all_features = []
        diaSourceIds = []

        ref_ids = extract_all_single_visit(visit, collection, repo, prefix)
        for ref in ref_ids:
            diff_array = butler.get(f'{prefix}goodSeeingDiff_differenceExp', dataId=ref).getImage().array
            dia_src = butler.get(f'{prefix}goodSeeingDiff_diaSrcTable', dataId=ref)
            if injection:
                matched_src = butler.get(f'{prefix}goodSeeingDiff_matchDiaSrc', dataId=ref)
                dia_src['is_injection'] = dia_src.diaSourceId.isin(matched_src.diaSourceId)
            else:
                dia_src['is_injection'] = 0
            
            dia_src['rotation_angle'] = 0

            for i in range(len(dia_src)):
                cutout = Cutout2D(diff_array, (dia_src['x'][i], dia_src['y'][i]), cutout_size)
                if cutout.data.shape != (cutout_size, cutout_size) or np.isnan(cutout.data).any():
                    continue
                all_cutouts.append(cutout.data)
                all_features.append(dia_src.iloc[i])
                diaSourceIds.append(dia_src.iloc[i]["diaSourceId"])


        features_df = pd.DataFrame(all_features)

        # Making sure IDs columns  are not rounded
        id_columns = ["diaSourceId", "diaObjectId"]  

        for col in id_columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].astype(np.int64)

        # Ensure diaSourceIds array is int64 for cutout saving
        diaSourceIds = np.array(diaSourceIds, dtype=np.int64)

        if save_file and not save_features_only:
            save_cutouts_hdf5(np.array(all_cutouts), diaSourceIds, os.path.join(path_cutouts, f"visit_{visit}.h5"))
            save_features_hdf5(features_df, os.path.join(path_features, f"visit_{visit}_features.h5"))
        elif save_features_only:
            save_features_hdf5(features_df, os.path.join(path_features, f"visit_{visit}_features.h5"))

        print(f"Saved visit {visit}: {len(all_cutouts)} cutouts, {len(features_df)} features")