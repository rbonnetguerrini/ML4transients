#!/usr/bin/env python3
"""
Filter lightcurves based on SNR, time window, and quality criteria.

This script performs fast filtering on HDF5 lightcurve files before expensive
host galaxy extendedness checks:
1. Remove sources with SNR < 5.0
2. Average flux measurements by (diaObjectId, band, night)
3. Apply time window restrictions [-30, +100] days around maximum brightness
4. Require minimum number of different nights with observations
5. Discard lightcurves with average negative flux

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
    min_nights: int = 5,
    snr_threshold: float = 5.0,
    time_window: tuple = (-30, 100)
) -> Tuple[int, int, dict]:
    """
    Filter lightcurves based on SNR and quality criteria.
    
    Args:
        input_path: Path to input HDF5 lightcurve file
        output_path: Path to output filtered HDF5 file
        min_nights: Minimum number of different nights with observations
        snr_threshold: Minimum SNR to keep individual sources
        time_window: Time window (min_days, max_days) around maximum brightness
        
    Returns:
        Tuple of (total_objects, kept_objects, detailed_stats)
    """
    logger.info(f"Loading lightcurves from {input_path}")
    lc_df = pd.read_hdf(input_path, key='lightcurves')
    
    initial_objects = lc_df['diaObjectId'].nunique()
    initial_sources = len(lc_df)
    logger.info(f"Initial: {initial_objects} objects, {initial_sources} sources")
    
    stats = {
        'initial_objects': initial_objects,
        'initial_sources': initial_sources,
    }
    
    lc_df['SNR'] = np.abs(lc_df['psfFlux']) / lc_df['psfFluxErr']
    
    logger.info(f"\n=== Filter 1: SNR >= {snr_threshold} ===")
    before_snr = lc_df['diaObjectId'].nunique()
    lc_df = lc_df[lc_df['SNR'] >= snr_threshold].reset_index(drop=True)
    after_snr = lc_df['diaObjectId'].nunique()
    stats['snr_filter_discarded'] = before_snr - after_snr
    logger.info(f"Removed {before_snr - after_snr} objects with SNR < {snr_threshold}")
    logger.info(f"Remaining: {after_snr} objects, {len(lc_df)} sources")
    
    logger.info(f"\n=== Averaging: Group by (diaObjectId, band, night) ===")
    lc_df['night'] = np.floor(lc_df['midpointMjdTai']).astype(int)
    
    before_avg = len(lc_df)
    
    # Build aggregation functions - include all columns that downstream filters need
    agg_funcs = {
        'psfFlux': 'mean',
        'psfFluxErr': lambda x: np.sqrt(np.sum(x**2)) / len(x),
        'midpointMjdTai': 'mean',
        'ra': 'mean',
        'dec': 'mean',
        'tract': 'first',
        'patch': 'first'
    }
    
    # Add coord_ra and coord_dec if they exist (needed by extendedness filter)
    if 'coord_ra' in lc_df.columns:
        agg_funcs['coord_ra'] = 'mean'
    if 'coord_dec' in lc_df.columns:
        agg_funcs['coord_dec'] = 'mean'
    
    # Add host galaxy columns if they exist
    for col in ['hostgal_ellipticity', 'hostgal_sqradius', 'hostgal_snsep', 'hostgal_zphot']:
        if col in lc_df.columns:
            agg_funcs[col] = 'first'  # These are per-object, not per-observation
    
    groupby_cols = ['diaObjectId', 'band', 'night']
    lc_df = lc_df.groupby(groupby_cols, as_index=False).agg(agg_funcs)
    
    after_avg = len(lc_df)
    logger.info(f"Averaged: {before_avg} sources --> {after_avg} nightly measurements")
    logger.info(f"Compression ratio: {before_avg/after_avg:.2f}x")
    
    logger.info(f"\n=== Filter 2: Time window {time_window} days around max ===")
    
    max_flux_idx = lc_df.groupby('diaObjectId')['psfFlux'].idxmax()
    max_mjd_df = lc_df.loc[max_flux_idx, ['diaObjectId', 'midpointMjdTai']].rename(
        columns={'midpointMjdTai': 'mjd_max'}
    )
    
    lc_df = lc_df.merge(max_mjd_df, on='diaObjectId', how='left')
    lc_df['dt_max'] = lc_df['midpointMjdTai'] - lc_df['mjd_max']
    
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
    
    lc_df = lc_df.drop(columns=['mjd_max', 'dt_max', 'night'])
    
    logger.info(f"\n=== Filter 3: >= {min_nights} different nights ===")
    
    lc_df['night'] = np.floor(lc_df['midpointMjdTai']).astype(int)
    night_counts = lc_df.groupby('diaObjectId')['night'].nunique()
    valid_objects = night_counts[night_counts >= min_nights].index
    
    before_nights = lc_df['diaObjectId'].nunique()
    lc_df = lc_df[lc_df['diaObjectId'].isin(valid_objects)].reset_index(drop=True)
    after_nights = lc_df['diaObjectId'].nunique()
    stats['min_nights_discarded'] = before_nights - after_nights
    logger.info(f"Removed {before_nights - after_nights} objects with < {min_nights} different nights")
    logger.info(f"Remaining: {after_nights} objects, {len(lc_df)} sources")
    
    lc_df = lc_df.drop(columns=['night'])
    
    logger.info(f"\n=== Filter 4: Average flux > 0 ===")
    
    avg_flux = lc_df.groupby('diaObjectId')['psfFlux'].mean()
    valid_objects = avg_flux[avg_flux > 0].index
    
    before_flux = lc_df['diaObjectId'].nunique()
    lc_df = lc_df[lc_df['diaObjectId'].isin(valid_objects)].reset_index(drop=True)
    after_flux = lc_df['diaObjectId'].nunique()
    stats['negative_flux_discarded'] = before_flux - after_flux
    logger.info(f"Removed {before_flux - after_flux} objects with average negative flux")
    logger.info(f"Remaining: {after_flux} objects, {len(lc_df)} sources")
    
    final_objects = lc_df['diaObjectId'].nunique()
    final_sources = len(lc_df)
    stats['final_objects'] = final_objects
    stats['final_sources'] = final_sources
    
    total_discarded = (stats['snr_filter_discarded'] + 
                      stats['window_filter_discarded'] + 
                      stats['min_nights_discarded'] +
                      stats['negative_flux_discarded'])
    
    logger.info(f"\n=== Summary ===")
    logger.info(f"Initial objects: {initial_objects}")
    logger.info(f"  Discarded (SNR < {snr_threshold}): {stats['snr_filter_discarded']}")
    logger.info(f"  Discarded (time window): {stats['window_filter_discarded']}")
    logger.info(f"  Discarded (min nights): {stats['min_nights_discarded']}")
    logger.info(f"  Discarded (negative avg flux): {stats['negative_flux_discarded']}")
    logger.info(f"  Total discarded: {total_discarded}")
    logger.info(f"Final kept: {final_objects} objects ({100*final_objects/initial_objects:.1f}%)")
    logger.info(f"Sources: {initial_sources} --> {final_sources} (nightly averaged)")
    
    logger.info(f"\nSaving filtered lightcurves to {output_path}")
    if final_objects > 0:
        lc_df.to_hdf(output_path, key='lightcurves', mode='w', format='table')
        logger.info(f"Saved {final_objects} objects to HDF5")
    else:
        with open(output_path, 'w') as f:
            pass
        logger.info("All objects filtered out - created empty marker file")
    
    metadata_path = output_path.replace('.h5', '_snr_filter_metadata.json')
    metadata = {
        'input_file': input_path,
        'output_file': output_path,
        'filter_timestamp': pd.Timestamp.now().isoformat(),
        'initial_objects': int(initial_objects),
        'initial_sources': int(initial_sources),
        'snr_threshold': snr_threshold,
        'min_nights': min_nights,
        'time_window_days': time_window,
        'discarded_snr_filter': int(stats['snr_filter_discarded']),
        'discarded_window_filter': int(stats['window_filter_discarded']),
        'discarded_min_nights': int(stats['min_nights_discarded']),
        'discarded_negative_flux': int(stats['negative_flux_discarded']),
        'total_discarded': int(total_discarded),
        'final_objects': int(final_objects),
        'final_sources': int(final_sources),
        'keep_rate_percent': float(100*final_objects/initial_objects) if initial_objects > 0 else 0.0,
        'nightly_averaged': True
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved filter metadata to {metadata_path}")
    
    return initial_objects, final_objects, stats


def main():
    parser = argparse.ArgumentParser(
        description='Filter lightcurves based on SNR and quality criteria'
    )
    parser.add_argument('--input', required=True, help='Input HDF5 lightcurve file')
    parser.add_argument('--output', required=True, help='Output filtered HDF5 file')
    parser.add_argument('--min-nights', type=int, default=5,
                        help='Minimum number of different nights with observations (default: 5)')
    parser.add_argument('--snr-threshold', type=float, default=5.0,
                        help='Minimum SNR to keep sources (default: 5.0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
        
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        total, kept, stats = filter_lightcurves(
            args.input,
            args.output,
            args.min_nights,
            args.snr_threshold
        )
        
        logger.info("\nFiltering complete!")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"\nFiltering failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
