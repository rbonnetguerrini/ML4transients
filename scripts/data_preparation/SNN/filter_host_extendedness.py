#!/usr/bin/env python3
"""
Filter lightcurves based on host galaxy extendedness.

This script filters diaObjects by checking if their host galaxy (if present)
is extended (not a point source) using the extendedness criterion from coadd measurements.

For each diaObject, we:
1. Get its coordinates (ra, dec) from the lightcurve file
2. Find the nearest coadd source within a matching radius
3. Check the extendedness value: 0 = point source, 1 = extended
4. Keep the diaObject if:
   - No host found (hostless transient) OR
   - Host is extended (extendedness == 1)
5. Reject if host is a point source (extendedness == 0)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from lsst.daf.butler import Butler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_coadd_sources_for_patch(
    butler: Butler,
    tract: int,
    patch: int,
    band: str = 'i'
) -> pd.DataFrame:
    """
    Retrieve coadd source measurements for a given tract/patch.
    
    Args:
        butler: LSST Butler instance
        tract: Tract number
        Patch: Patch number
        band: Photometric band (default: 'i')
        
    Returns:
        DataFrame with columns: ra, dec, extendedness, psf_flux, psf_flux_err
    """
    try:
        data_id = {
            'tract': tract,
            'patch': patch,
            'band': band
        }
        
        logger.info(f"Loading deepCoadd_meas for tract={tract}, patch={patch}, band={band}")
        meas_catalog = butler.get('deepCoadd_meas', dataId=data_id)
        
        # Extract schema columns
        schema = meas_catalog.schema
        all_cols = [s.field.getName() for s in schema]
        
        # Extract coordinates, extendedness, and PSF flux
        ra_values = [rec.get('coord_ra').asDegrees() for rec in meas_catalog]
        dec_values = [rec.get('coord_dec').asDegrees() for rec in meas_catalog]
        ext_values = [rec.get('base_ClassificationExtendedness_value') for rec in meas_catalog]
        psf_flux_values = [rec.get('base_PsfFlux_instFlux') for rec in meas_catalog]
        psf_flux_err_values = [rec.get('base_PsfFlux_instFluxErr') for rec in meas_catalog]
        
        result_df = pd.DataFrame({
            'ra': ra_values,
            'dec': dec_values,
            'extendedness': ext_values,
            'coadd_psf_flux': psf_flux_values,
            'coadd_psf_flux_err': psf_flux_err_values
        })
        
        logger.info(f"Retrieved {len(result_df)} sources from coadd")
        logger.info(f"Extended sources: {(result_df['extendedness'] == 1.0).sum()}, "
                   f"Point sources: {(result_df['extendedness'] == 0.0).sum()}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error retrieving coadd sources: {e}")
        logger.error(f"Data ID attempted: {data_id}")
        return pd.DataFrame(columns=['ra', 'dec', 'extendedness', 'coadd_psf_flux', 'coadd_psf_flux_err'])


def match_to_host(
    diaobject_coords: SkyCoord,
    coadd_coords: SkyCoord,
    coadd_extendedness: np.ndarray,
    coadd_psf_flux: np.ndarray,
    coadd_psf_flux_err: np.ndarray,
    match_radius: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match diaObjects to their nearest coadd source (potential host).
    
    Args:
        diaobject_coords: SkyCoord array of diaObject positions
        coadd_coords: SkyCoord array of coadd source positions
        coadd_extendedness: Array of extendedness values for coadd sources
        coadd_psf_flux: Array of PSF flux values for coadd sources
        coadd_psf_flux_err: Array of PSF flux errors for coadd sources
        match_radius: Maximum matching radius in arcseconds
        
    Returns:
        Tuple of (host_extendedness, host_psf_flux, host_psf_flux_err).
        NaN indicates no match found (hostless transient).
    """
    # Find nearest coadd source for each diaObject
    idx, sep, _ = diaobject_coords.match_to_catalog_sky(coadd_coords)
    
    # Initialize result arrays with NaN (no host)
    host_extendedness = np.full(len(diaobject_coords), np.nan)
    host_coadd_psf_flux = np.full(len(diaobject_coords), np.nan)
    host_coadd_psf_flux_err = np.full(len(diaobject_coords), np.nan)
    
    # Only keep matches within the specified radius
    matched_mask = sep.arcsec < match_radius
    host_extendedness[matched_mask] = coadd_extendedness[idx[matched_mask]]
    host_coadd_psf_flux[matched_mask] = coadd_psf_flux[idx[matched_mask]]
    host_coadd_psf_flux_err[matched_mask] = coadd_psf_flux_err[idx[matched_mask]]
    
    n_matched = matched_mask.sum()
    n_hostless = (~matched_mask).sum()
    
    logger.info(f"Matched {n_matched} diaObjects to coadd sources within {match_radius}\"")
    logger.info(f"Found {n_hostless} hostless transients (no match within {match_radius}\")")
    
    return host_extendedness, host_coadd_psf_flux, host_coadd_psf_flux_err


def filter_lightcurves(
    input_path: str,
    output_path: str,
    butler_repo: str,
    collection: str,
    match_radius: float = 1.0,
    band: str = 'i',
    flux_ratio_threshold: float = 1.4
) -> Tuple[int, int, int]:
    """
    Filter lightcurves based on host galaxy extendedness and diff/coadd flux ratio.
    
    Args:
        input_path: Path to input HDF5 lightcurve file
        output_path: Path to output filtered HDF5 file
        butler_repo: Path to Butler repository
        collection: Butler collection name
        match_radius: Maximum matching radius in arcseconds
        band: Photometric band to use for coadd measurements
        flux_ratio_threshold: Minimum ratio of max(diff_flux)/coadd_flux (default: 1.2 = 20% brighter)
        
    Returns:
        Tuple of (total_objects, kept_objects, rejected_objects)
    """
    # Load lightcurves
    logger.info(f"Loading lightcurves from {input_path}")
    
    # Check if file is empty or has no lightcurves key (all filtered out in previous step)
    try:
        import h5py
        with h5py.File(input_path, 'r') as f:
            if 'lightcurves' not in f:
                logger.warning(f"File {input_path} has no lightcurves - all objects were filtered out in previous step")
                # Create empty output file
                with open(output_path, 'w') as out_f:
                    pass
                # Save metadata indicating empty file
                metadata_path = output_path.replace('.h5', '_filter_metadata.json')
                import json
                metadata = {
                    'input_file': input_path,
                    'output_file': output_path,
                    'filter_timestamp': pd.Timestamp.now().isoformat(),
                    'total_objects': 0,
                    'kept_objects': 0,
                    'rejected_objects': 0,
                    'hostless_objects': 0,
                    'extended_host_objects': 0,
                    'point_source_host_objects': 0,
                    'note': 'Input file was empty - all objects filtered in previous step'
                }
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                return 0, 0, 0
    except Exception as e:
        logger.error(f"Error checking file {input_path}: {e}")
        raise
    
    lc_df = pd.read_hdf(input_path, key='lightcurves')
    
    # Get unique diaObjects with their coordinates AND max diff flux
    # For each diaObject, find the observation with max absolute psfFlux and get its band
    diaobject_data = []
    for diaobj_id, group in lc_df.groupby('diaObjectId'):
        # Find index of max absolute flux
        max_idx = group['psfFlux'].abs().idxmax()
        max_row = group.loc[max_idx]
        
        diaobject_data.append({
            'diaObjectId': diaobj_id,
            'coord_ra': max_row['ra'],
            'coord_dec': max_row['dec'],
            'tract': max_row['tract'],
            'patch': max_row['patch'],
            'max_diff_flux': abs(max_row['psfFlux']),
            'band': max_row['band'] if 'band' in max_row else band  # Use observation band or fallback
        })
    
    diaobjects = pd.DataFrame(diaobject_data)
    
    total_objects = len(diaobjects)
    logger.info(f"Found {total_objects} unique diaObjects")
    logger.info(f"Max diff flux range: {diaobjects['max_diff_flux'].min():.1f} to {diaobjects['max_diff_flux'].max():.1f}")
    
    # Initialize Butler
    logger.info(f"Connecting to Butler repository: {butler_repo}")
    butler = Butler(butler_repo, collections=collection)
    
    # Process each patch grouped by tract, patch, AND band
    all_host_extendedness = []
    all_host_psf_flux = []
    all_host_psf_flux_err = []
    
    for (tract, patch, obj_band), group in diaobjects.groupby(['tract', 'patch', 'band']):
        logger.info(f"Processing tract={tract}, patch={patch}, band={obj_band} ({len(group)} objects)")
        
        # Get coadd sources for this patch in the same band as the max flux observation
        coadd_df = get_coadd_sources_for_patch(butler, tract, patch, obj_band)
        
        if len(coadd_df) == 0:
            logger.warning(f"No coadd sources found for tract={tract}, patch={patch}")
            # Assume all are hostless
            patch_host_ext = np.full(len(group), np.nan)
            patch_host_flux = np.full(len(group), np.nan)
            patch_host_flux_err = np.full(len(group), np.nan)
        else:
            # Convert coordinates to SkyCoord
            diaobject_coords = SkyCoord(
                ra=group['coord_ra'].values * u.deg,
                dec=group['coord_dec'].values * u.deg
            )
            
            coadd_coords = SkyCoord(
                ra=coadd_df['ra'].values * u.deg,
                dec=coadd_df['dec'].values * u.deg
            )
            
            # Match and get host properties
            patch_host_ext, patch_host_flux, patch_host_flux_err = match_to_host(
                diaobject_coords,
                coadd_coords,
                coadd_df['extendedness'].values,
                coadd_df['coadd_psf_flux'].values,
                coadd_df['coadd_psf_flux_err'].values,
                match_radius
            )
        
        all_host_extendedness.extend(patch_host_ext)
        all_host_psf_flux.extend(patch_host_flux)
        all_host_psf_flux_err.extend(patch_host_flux_err)
    
    # Add host properties to diaobjects DataFrame
    # Add host properties to diaobjects DataFrame
    diaobjects['host_extendedness'] = all_host_extendedness
    diaobjects['host_coadd_flux'] = all_host_psf_flux
    diaobjects['host_coadd_flux_err'] = all_host_psf_flux_err
    
    # Calculate flux ratio (transient / static)
    diaobjects['flux_ratio'] = diaobjects['max_diff_flux'] / diaobjects['host_coadd_flux'].abs()
    # For hostless (NaN coadd flux), set ratio to infinity
    diaobjects.loc[diaobjects['host_coadd_flux'].isna(), 'flux_ratio'] = np.inf
    
    # Apply filters:
    # 1. Extendedness filter: Keep if hostless (NaN) OR extended host (== 1.0)
    extendedness_keep = (diaobjects['host_extendedness'].isna()) | (diaobjects['host_extendedness'] == 1.0)
    
    # 2. Flux ratio filter: Keep if hostless OR flux_ratio > threshold
    #    For hostless (NaN coadd flux), flux_ratio will be inf, so check explicitly
    flux_ratio_keep = (diaobjects['host_coadd_flux'].isna()) | (diaobjects['flux_ratio'] > flux_ratio_threshold)
    
    # Combine both filters (both must pass)
    keep_mask = extendedness_keep & flux_ratio_keep
    
    kept_diaobjects = diaobjects[keep_mask]['diaObjectId'].values
    rejected_diaobjects = diaobjects[~keep_mask]['diaObjectId'].values
    
    n_kept = len(kept_diaobjects)
    n_rejected = len(rejected_diaobjects)
    
    # Extendedness statistics
    n_hostless = diaobjects['host_extendedness'].isna().sum()
    n_extended = (diaobjects['host_extendedness'] == 1.0).sum()
    n_point = (diaobjects['host_extendedness'] == 0.0).sum()
    
    # Rejection reasons
    n_rejected_point_host = (~extendedness_keep).sum()
    n_rejected_low_flux_ratio = (~flux_ratio_keep & extendedness_keep).sum()
    
    logger.info(f"\nFiltering results:")
    logger.info(f"  Total diaObjects: {total_objects}")
    logger.info(f"  Hostless transients: {n_hostless}")
    logger.info(f"  Extended hosts: {n_extended}")
    logger.info(f"  Point source hosts: {n_point}")
    logger.info(f"  Flux ratio threshold: {flux_ratio_threshold:.2f} (transient must be {(flux_ratio_threshold-1)*100:.0f}% brighter)")
    logger.info(f"\nRejection breakdown:")
    logger.info(f"  Point source hosts: {n_rejected_point_host}")
    logger.info(f"  Low flux ratio (extended host): {n_rejected_low_flux_ratio}")
    logger.info(f"  Total rejected: {n_rejected} ({100*n_rejected/total_objects:.1f}%)")
    logger.info(f"  Kept: {n_kept} ({100*n_kept/total_objects:.1f}%)")
    
    # Filter lightcurves DataFrame
    filtered_lc_df = lc_df[lc_df['diaObjectId'].isin(kept_diaobjects)]
    
    logger.info(f"\nLightcurve rows: {len(lc_df)} --> {len(filtered_lc_df)}")
    
    # Save filtered lightcurves
    logger.info(f"Saving filtered lightcurves to {output_path}")
    filtered_lc_df.to_hdf(output_path, key='lightcurves', mode='w', format='table')
    
    # Save filtering metadata
    metadata_path = output_path.replace('.h5', '_filter_metadata.json')
    import json
    metadata = {
        'input_file': input_path,
        'output_file': output_path,
        'filter_timestamp': pd.Timestamp.now().isoformat(),
        'total_objects': int(total_objects),
        'kept_objects': int(n_kept),
        'rejected_objects': int(n_rejected),
        'hostless_objects': int(n_hostless),
        'extended_host_objects': int(n_extended),
        'point_source_host_objects': int(n_point),
        'rejected_point_host': int(n_rejected_point_host),
        'rejected_low_flux_ratio': int(n_rejected_low_flux_ratio),
        'flux_ratio_threshold': float(flux_ratio_threshold),
        'match_radius_arcsec': match_radius,
        'band': band,
        'butler_repo': butler_repo,
        'butler_collection': collection
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved filter metadata to {metadata_path}")
    
    return total_objects, n_kept, n_rejected


def main():
    parser = argparse.ArgumentParser(
        description='Filter lightcurves based on host galaxy extendedness'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input HDF5 lightcurve file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output filtered HDF5 file'
    )
    parser.add_argument(
        '--repo',
        required=True,
        help='Path to Butler repository'
    )
    parser.add_argument(
        '--collection',
        required=True,
        help='Butler collection name'
    )
    parser.add_argument(
        '--match-radius',
        type=float,
        default=1.0,
        help='Maximum matching radius in arcseconds (default: 1.0)'
    )
    parser.add_argument(
        '--band',
        default='i',
        help='Photometric band for coadd measurements (default: i)'
    )
    parser.add_argument(
        '--flux-ratio-threshold',
        type=float,
        default=1.2,
        help='Minimum flux ratio (transient/static) to keep (default: 1.2 = 20%% brighter)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
        
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run filtering
    try:
        total, kept, rejected = filter_lightcurves(
            args.input,
            args.output,
            args.repo,
            args.collection,
            args.match_radius,
            args.band,
            args.flux_ratio_threshold
        )
        
        logger.info("\nFiltering complete!")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"\nFiltering failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
