from lsst.daf.butler import Butler
from astropy.nddata import Cutout2D
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
import os
import h5py


def extract_all_single_visit(visit_id: int, collection: str, repo: str) -> list:
    """
    Query all datasets for a single visit.

    Parameters:
        visit_id (int): Visit ID.
        collection (str): Butler collection name.
        repo (str): Butler repo path.

    Returns:
        list: List of dataIds for the visit.
    """
    butler = Butler(repo, collections=collection)
    registry = butler.registry
    result = registry.queryDatasets(
        datasetType='injected_goodSeeingDiff_differenceExp',
        collections=collection,
        where=f"visit = {visit_id}"
    )
    return [ref.dataId for ref in result]

def apply_rotations(cutout: np.ndarray, angles: list) -> list:
    """
    Generate rotated versions of a cutout.

    Parameters:
        cutout (ndarray): The original cutout array.
        angles (list): List of rotation angles in degrees.

    Returns:
        list of ndarray: Rotated cutouts.
    """
    rotated_cutouts = []
    for angle in angles:
        rotated = rotate(cutout, angle, reshape=False)
        rotated_cutouts.append(rotated)
    return rotated_cutouts

def save_cutouts(config: dict):
    """
    Extract cutouts and features from Butler repo, apply rotations, and save to disk.

    Parameters:
        config (dict): Configuration dictionary with keys:
            - repo, collection, cutout (size, rotate, rotation_angles, coadd_science, save, return_data), paths (cutouts, csv)
    """
    repo = config["repo"]
    collection = config["collection"]
    cutout_size = config["cutout"]["size"]
    rotate_data = config["cutout"]["rotate"]
    rotation_angles = config["cutout"].get("rotation_angles", [90, 180, 270])
    coadd_science = config["cutout"]["coadd_science"]
    save_file = config["cutout"]["save"]
    return_file = config["cutout"]["return_data"]

    path_cutouts = config["paths"]["cutouts"]
    path_csv = config["paths"]["csv"]
    os.makedirs(path_cutouts, exist_ok=True)
    os.makedirs(path_csv, exist_ok=True)

    butler = Butler(repo, collections=collection)
    registry = butler.registry

    datasetRefs = registry.queryDatasets(
        datasetType='injected_goodSeeingDiff_differenceExp',
        collections=collection
    )
    
    # Get all visits from datasets
    all_visits = sorted(set(ref.dataId['visit'] for ref in datasetRefs))
    # Use visits from config if specified, else use all
    visits = config.get("visits", all_visits)

    nbr_cutout = 0

    all_cutouts_collection = []
    all_features_collection = []

    for visit in visits:
        all_cutouts = []
        all_features = []
        rotated_cutouts = []
        rotated_rows = []

        ref_ids = extract_all_single_visit(visit, collection, repo)

        for ref in ref_ids:
            diff_array = butler.get('injected_goodSeeingDiff_differenceExp', dataId=ref).getImage().array
            dia_src = butler.get('injected_goodSeeingDiff_diaSrcTable', dataId=ref)
            matched_src = butler.get('injected_goodSeeingDiff_matchDiaSrc', dataId=ref)

            dia_src['is_injection'] = dia_src.diaSourceId.isin(matched_src.diaSourceId)
            dia_src['rotation_angle'] = 0  # Set original cutout angle to 0

            for i in range(len(dia_src)):
                cutout = Cutout2D(diff_array, (dia_src['x'][i], dia_src['y'][i]), cutout_size)
                if cutout.data.shape != (cutout_size, cutout_size) or np.isnan(cutout.data).any():
                    continue

                all_cutouts.append(cutout.data)
                all_features.append(dia_src.iloc[i])
                nbr_cutout += 1

                if rotate_data:
                    for angle, rotated in zip(rotation_angles, apply_rotations(cutout.data, rotation_angles)):
                        new_row = dia_src.iloc[[i]].copy()
                        new_row["rotation_angle"] = angle
                        rotated_rows.append(new_row)
                        rotated_cutouts.append(rotated)

        features_df = pd.DataFrame(all_features)
        if rotate_data:
            rotated_df = pd.concat(rotated_rows, ignore_index=True)
            features_df = pd.concat([features_df, rotated_df], ignore_index=True)
            all_cutouts = np.concatenate((all_cutouts, rotated_cutouts))

        if save_file:
            np.save(os.path.join(path_cutouts, f"visit_{visit}.npy"), all_cutouts)
            features_df.to_csv(os.path.join(path_csv, f"visit_{visit}.csv"), index=False)

        if return_file:
            all_cutouts_collection.append(all_cutouts)
            all_features_collection.append(features_df)

        print(f"Saved visit {visit}: {len(all_cutouts)} cutouts")

    print(f"Processed {len(visits)} visits with total {nbr_cutout} cutouts.")

    if return_file:
        return all_cutouts_collection, all_features_collection