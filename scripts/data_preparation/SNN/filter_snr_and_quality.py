#!/usr/bin/env python3
"""
Filter lightcurves based on SNR, time window, and quality criteria.

This script performs fast filtering on HDF5 lightcurve files before expensive
host galaxy extendedness checks:
1. Remove sources with SNR < 5.0
2. Apply time window restrictions [-30, +100] days around maximum brightness
3. Require minimum number of high SNR observations (SNR > 3)
4. Discard lightcurves with average negative flux

This reduces the dataset early in the pipeline before expensive Butler queries.
"""

import argparse
import json
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def filter_lightcurves(
    input_path: str,
    output_path: str,
    min_obs: int = 10,
    snr_threshold: float = 5.0,
    high_snr_threshold: float = 3.0,
    time_window: tuple = (-30, 100)
) -> Tuple[int, int, dict]:
    """
    Filter lightcurves based on SNR and quality criteria.
    
    Args:
        input_path: Path to input HDF5 lightcurve file
        output_path: Path to output filtered HDF5 file
        min_obs: Minimum number of high SNR observations required
        snr_threshold: Minimum SNR to keep individual sources
        high_snr_threshold: SNR threshold for counting high-quality observations
        time_window: Time window (min_days, max_days) around maximum brightness
        
    Returns:
        Tuple of (total_objects, kept_objects, detailed_stats)
    """
    logger.info(f"Loading lightcurves from {input_path}")
    lc_df = pd.read_hdf(input_path, key='lightcurves')
    
    initial_objects = lc_df['diaObjectId'].nunique()
    initial_sources = len(lc_df)
    logger.info(f"Initial: {initial_objects} objects, {initial_sources} sources")
    
    # Track filtering statistics
    stats = {
        'initial_objects': initial_objects,
        'initial_sources': initial_sources,
    }
    
    # Calculate SNR
    lc_df['SNR'] = np.abs(lc_df['psfFlux']) / lc_df['psfFluxErr']
    
    # Filter 1: Remove sources with SNR < threshold
    logger.info(f"\n=== Filter 1: SNR >= {snr_threshold} ===")
    before_snr = lc_df['diaObjectId'].nunique()
    lc_df = lc_df[lc_df['SNR'] >= snr_threshold].reset_index(drop=True)
    after_snr = lc_df['diaObjectId'].nunique()
    stats['snr_filter_discarded'] = before_snr - after_snr
    logger.info(f"Removed {before_snr - after_snr} objects with SNR < {snr_threshold}")
    logger.info(f"Remaining: {after_snr} objects, {len(lc_df)} sources")
    
    # Filter 2: Time window around maximum brightness
    logger.info(f"\n=== Filter 2: Time window {time_window} days around max ===")
    
    # Find MJD of maximum flux for each object
    max_flux_idx = lc_df.groupby('diaObjectId')['psfFlux'].idxmax()
    max_mjd_df = lc_df.loc[max_flux_idx, ['diaObjectId', 'midpointMjdTai']].rename(
        columns={'midpointMjdTai': 'mjd_max'}
    )
    
    # Merge and calculate time from max
    lc_df = lc_df.merge(max_mjd_df, on='diaObjectId', how='left')
    lc_df['dt_max'] = lc_df['midpointMjdTai'] - lc_df['mjd_max']
    
    # Check which objects have ALL sources within the window
    min_days, max_days = time_window
    objects_in_window = lc_df.groupby('diaObjectId')['dt_max'].apply(
        lambda x: ((x >= min_days) & (x <= max_days)).all()
    )
    valid_objects = objects_in_window[objects_in_window].index
    
    before_window = lc_df['diaObjectId'].nunique()
    lc_df = lc_df[lc_df['diaObjectId'].isin(valid_objects)].reset_index(drop=True)
    after_window = lc_df['diaObjectId'].nunique()
    stats['window_filter_discarded'] = before_window - after_window
    logger.info(f"Removed {before_window - after_window} objects with sources outside window")
    logger.info(f"Remaining: {after_window} objects, {len(lc_df)} sources")
    
    # Drop temporary columns
    lc_df = lc_df.drop(columns=['mjd_max', 'dt_max'])
    
    # Filter 3: Minimum number of high SNR observations
    logger.info(f"\n=== Filter 3: >= {min_obs} observations with SNR > {high_snr_threshold} ===")
    
    high_snr_counts = lc_df[lc_df['SNR'] > high_snr_threshold].groupby('diaObjectId').size()
    valid_objects = high_snr_counts[high_snr_counts >= min_obs].index
    
    before_minobs = lc_df['diaObjectId'].nunique()
    lc_df = lc_df[lc_df['diaObjectId'].isin(valid_objects)].reset_index(drop=True)
    after_minobs = lc_df['diaObjectId'].nunique()
    stats['minobs_filter_discarded'] = before_minobs - after_minobs
    logger.info(f"Removed {before_minobs - after_minobs} objects with < {min_obs} high SNR observations")
    logger.info(f"Remaining: {after_minobs} objects, {len(lc_df)} sources")
    
    # Filter 4: Average flux must be positive
    logger.info(f"\n=== Filter 4: Average flux > 0 ===")
    
    avg_flux = lc_df.groupby('diaObjectId')['psfFlux'].mean()
    valid_objects = avg_flux[avg_flux > 0].index
    
    before_flux = lc_df['diaObjectId'].nunique()
    lc_df = lc_df[lc_df['diaObjectId'].isin(valid_objects)].reset_index(drop=True)
    after_flux = lc_df['diaObjectId'].nunique()
    stats['negative_flux_discarded'] = before_flux - after_flux
    logger.info(f"Removed {before_flux - after_flux} objects with average negative flux")
    logger.info(f"Remaining: {after_flux} objects, {len(lc_df)} sources")
    
    # Drop SNR column (will be recalculated in CSV conversion if needed)
    lc_df = lc_df.drop(columns=['SNR'])
    
    # Final statistics
    final_objects = lc_df['diaObjectId'].nunique()
    final_sources = len(lc_df)
    stats['final_objects'] = final_objects
    stats['final_sources'] = final_sources
    
    total_discarded = (stats['snr_filter_discarded'] + 
                      stats['window_filter_discarded'] + 
                      stats['minobs_filter_discarded'] +
                      stats['negative_flux_discarded'])
    
    logger.info(f"\n=== Summary ===")
    logger.info(f"Initial objects: {initial_objects}")
    logger.info(f"  Discarded (SNR < {snr_threshold}): {stats['snr_filter_discarded']}")
    logger.info(f"  Discarded (time window): {stats['window_filter_discarded']}")
    logger.info(f"  Discarded (min observations): {stats['minobs_filter_discarded']}")
    logger.info(f"  Discarded (negative avg flux): {stats['negative_flux_discarded']}")
    logger.info(f"  Total discarded: {total_discarded}")
    logger.info(f"Final kept: {final_objects} objects ({100*final_objects/initial_objects:.1f}%)")
    logger.info(f"Sources: {initial_sources} → {final_sources}")
    
    # Save filtered lightcurves (even if empty, for tracking)
    logger.info(f"\nSaving filtered lightcurves to {output_path}")
    if final_objects > 0:
        lc_df.to_hdf(output_path, key='lightcurves', mode='w', format='table')
        logger.info(f"Saved {final_objects} objects to HDF5")
    else:
        # Create empty file with marker to indicate it was processed but empty
        with open(output_path, 'w') as f:
            pass  # Create empty file
        logger.info("All objects filtered out - created empty marker file")
    
    # Save metadata
    metadata_path = output_path.replace('.h5', '_snr_filter_metadata.json')
    metadata = {
        'input_file': input_path,
        'output_file': output_path,
        'filter_timestamp': pd.Timestamp.now().isoformat(),
        'initial_objects': int(initial_objects),
        'initial_sources': int(initial_sources),
        'snr_threshold': snr_threshold,
        'high_snr_threshold': high_snr_threshold,
        'min_observations': min_obs,
        'time_window_days': time_window,
        'discarded_snr_filter': int(stats['snr_filter_discarded']),
        'discarded_window_filter': int(stats['window_filter_discarded']),
        'discarded_minobs_filter': int(stats['minobs_filter_discarded']),
        'discarded_negative_flux': int(stats['negative_flux_discarded']),
        'total_discarded': int(total_discarded),
        'final_objects': int(final_objects),
        'final_sources': int(final_sources),
        'keep_rate_percent': float(100*final_objects/initial_objects) if initial_objects > 0 else 0.0
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved filter metadata to {metadata_path}")
    
    return initial_objects, final_objects, stats


def main():
    parser = argparse.ArgumentParser(
        description='Filter lightcurves based on SNR and quality criteria'
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
        '--min-obs',
        type=int,
        default=10,
        help='Minimum number of high SNR observations (default: 10)'
    )
    parser.add_argument(
        '--snr-threshold',
        type=float,
        default=5.0,
        help='Minimum SNR to keep sources (default: 5.0)'
    )
    parser.add_argument(
        '--high-snr-threshold',
        type=float,
        default=3.0,
        help='SNR threshold for counting high-quality observations (default: 3.0)'
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
        total, kept, stats = filter_lightcurves(
            args.input,
            args.output,
            args.min_obs,
            args.snr_threshold,
            args.high_snr_threshold
        )
        
        logger.info("\n Filtering complete!")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"\nFiltering failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
