#!/usr/bin/env python3
"""
Calculate what percentage of galaxies to keep to reach target injections across ALL bands.

Usage:
    python calculate_settings.py --target 44785 --repo /path/to/repo --collections DC2/runs/latest --bands g r i z y --hostless-fraction 0.10
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lsst.daf.butler import Butler
import numpy as np


def count_galaxies_all_bands(butler, collections, bands, patch=None, visit_range=None):
    """Count total number of galaxies across all bands in the repo."""
    
    all_band_counts = {}
    
    for band in bands:
        print(f"\n{'='*70}")
        print(f"Processing band: {band}")
        print(f"{'='*70}")
        
        # Build query
        where_clause = f"band = '{band}'"
        if patch:
            where_clause += f" AND patch = '{patch}'"
        if visit_range:
            min_v, max_v = visit_range
            where_clause += f" AND visit >= {min_v} AND visit <= {max_v}"
        
        print(f"Query: {where_clause}")
        
        # Get dataset references
        refs = list(butler.registry.queryDatasets(
            'sourceTable',
            collections=collections,
            where=where_clause,
            findFirst=True
        ))
        
        print(f"Found {len(refs)} CCDs for band {band}")
        
        if len(refs) == 0:
            print(f"WARNING: No datasets found for band {band}, skipping")
            continue
        
        # Sample a few CCDs to estimate total galaxies
        sample_size = min(48, len(refs))
        sample_indices = np.random.choice(len(refs), size=sample_size, replace=False)
        
        galaxy_counts = []
        print(f"Sampling {sample_size} CCDs...")
        
        for idx in sample_indices:
            ref = refs[idx]
            try:
                sourceTable = butler.get('sourceTable', dataId=ref.dataId)
                catalog = sourceTable  # Already a pandas DataFrame
                n_galaxies = len(catalog[catalog['extendedness'] == 1])
                galaxy_counts.append(n_galaxies)
            except Exception as e:
                print(f"  Warning: Failed to load CCD: {e}")
                continue
        
        if not galaxy_counts:
            print(f"ERROR: Could not count galaxies in any CCD for band {band}")
            continue
        
        avg_galaxies_per_ccd = np.mean(galaxy_counts)
        total_galaxies_band = int(avg_galaxies_per_ccd * len(refs))
        
        all_band_counts[band] = {
            'total_galaxies': total_galaxies_band,
            'n_ccds': len(refs),
            'avg_per_ccd': avg_galaxies_per_ccd
        }
        
        print(f"  Total galaxies in band {band}: {total_galaxies_band:,}")
        print(f"  Average per CCD: {avg_galaxies_per_ccd:.0f}")
        print(f"  Across {len(refs)} CCDs")
    
    return all_band_counts


def calculate_galaxy_fraction_all_bands(target_total, all_band_counts, hostless_fraction=0.05):
    """
    Calculate what fraction of galaxies to keep across all bands.
    
    Total injections = Sum over all bands of: N_galaxies_kept[band] x (1 + hostless_fraction)
    
    If we use the same galaxy_fraction for all bands:
        N_galaxies_kept[band] = galaxy_fraction x total_available_galaxies[band]
    
    So:
        target_total = galaxy_fraction x Sum(total_available[band] x (1 + hostless_fraction))
        galaxy_fraction = target_total / (Sum(total_available[band]) x (1 + hostless_fraction))
    """
    
    # Total available galaxies across all bands
    total_available = sum(counts['total_galaxies'] for counts in all_band_counts.values())
    
    # Calculate fraction needed
    # Total = galaxy_fraction x total_available x (1 + hostless_fraction)
    galaxy_fraction = target_total / (total_available * (1 + hostless_fraction))
    
    if galaxy_fraction > 1.0:
        max_possible = int(total_available * (1 + hostless_fraction))
        print(f"\n  WARNING: You need more galaxies than available!")
        print(f"   Total available galaxies: {total_available:,}")
        print(f"   Maximum possible injections: {max_possible:,}")
        print(f"   Your target: {target_total:,}")
        galaxy_fraction = 1.0
    
    return galaxy_fraction, total_available


def print_results(target, all_band_counts, galaxy_fraction, total_available, hostless_fraction):
    """Print the results."""
    
    print("\n" + "="*70)
    print("MULTI-BAND GALAXY SAMPLING CALCULATOR")
    print("="*70)
    
    print(f"\nTarget: {target:,} total injections (ALL BANDS COMBINED)")
    print(f"Hostless fraction: {hostless_fraction:.4f} ({hostless_fraction*100:.1f}%)")
    
    print("\n" + "-"*70)
    print("SETTINGS TO USE (same for all bands):")
    print("-"*70)
    print(f"  galaxy_fraction = {galaxy_fraction:.6f}  ({galaxy_fraction*100:.3f}%)")
    print(f"  hostless_fraction = {hostless_fraction:.4f}")
    print("-"*70)
    
    # Calculate per-band breakdown
    print(f"\nPer-band breakdown:")
    print("-"*70)
    
    grand_total_hosted = 0
    grand_total_hostless = 0
    
    for band, counts in sorted(all_band_counts.items()):
        galaxies_kept = int(counts['total_galaxies'] * galaxy_fraction)
        hostless = int(galaxies_kept * hostless_fraction)
        total_band = galaxies_kept + hostless
        
        grand_total_hosted += galaxies_kept
        grand_total_hostless += hostless
        
        print(f"  Band {band}:")
        print(f"    Available: {counts['total_galaxies']:,} galaxies ({counts['n_ccds']} CCDs)")
        print(f"    Keep: {galaxies_kept:,} galaxies ({galaxy_fraction*100:.3f}%)")
        print(f"    Hostless: {hostless:,}")
        print(f"    Total injections: {total_band:,}")
        print()
    
    grand_total = grand_total_hosted + grand_total_hostless
    
    print("-"*70)
    print(f"TOTAL ACROSS ALL BANDS:")
    print(f"  Available galaxies: {total_available:,}")
    print(f"  Galaxies to keep: {grand_total_hosted:,} ({galaxy_fraction*100:.3f}%)")
    print(f"  Hostless injections: {grand_total_hostless:,}")
    print(f"  GRAND TOTAL INJECTIONS: {grand_total:,}")
    
    diff = grand_total - target
    print(f"\n  Difference from target: {diff:+,} ({diff/target*100:+.2f}%)")
    
    print("="*70 + "\n")
    
    # Show config snippet
    avg_galaxies_per_ccd = np.mean([c['avg_per_ccd'] for c in all_band_counts.values()])
    min_inj = int(avg_galaxies_per_ccd * galaxy_fraction * 0.5)  # 50% of average kept
    
    print("Copy this to your config file:")
    print("-"*70)
    print("injection:")
    print(f"  galaxy_fraction: {galaxy_fraction:.6f}  # Keep {galaxy_fraction*100:.3f}% of galaxies")
    print(f"  hostless_fraction: {hostless_fraction:.4f}  # {hostless_fraction*100:.1f}% hostless")
    print(f"  min_injections: {min_inj}")
    print(f"  max_injections: null")
    print(f"  random_seed: 42")
    print("-"*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate galaxy_fraction for target injections across ALL bands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--target', '-t',
        type=int,
        required=True,
        help='Target total number of injections (ALL BANDS COMBINED)'
    )
    
    parser.add_argument(
        '--repo', '-r',
        type=str,
        required=True,
        help='Butler repository path'
    )
    
    parser.add_argument(
        '--collections',
        type=str,
        nargs='+',
        required=True,
        help='Butler collections to query'
    )
    
    parser.add_argument(
        '--bands', '-b',
        type=str,
        nargs='+',
        default=['g', 'r', 'i', 'z', 'y'],
        help='Photometric bands to process (default: g r i z y)'
    )
    
    parser.add_argument(
        '--hostless-fraction',
        type=float,
        default=0.05,
        help='Hostless fraction to use (default: 0.05 = 5%%)'
    )
    
    parser.add_argument(
        '--patch',
        type=str,
        help='Specific patch to process'
    )
    
    parser.add_argument(
        '--visit-min',
        type=int,
        help='Minimum visit number'
    )
    
    parser.add_argument(
        '--visit-max',
        type=int,
        help='Maximum visit number'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize Butler
        print(f"Initializing Butler: {args.repo}")
        print(f"Collections: {args.collections}")
        print(f"Bands: {args.bands}")
        butler = Butler(args.repo, collections=args.collections)
        
        # Count galaxies in all bands
        visit_range = None
        if args.visit_min and args.visit_max:
            visit_range = (args.visit_min, args.visit_max)
        
        all_band_counts = count_galaxies_all_bands(
            butler, 
            args.collections, 
            args.bands,
            patch=args.patch,
            visit_range=visit_range
        )
        
        if not all_band_counts:
            print("ERROR: No galaxies found in any band!")
            return 1
        
        # Calculate galaxy fraction needed across all bands
        galaxy_fraction, total_available = calculate_galaxy_fraction_all_bands(
            args.target,
            all_band_counts,
            args.hostless_fraction
        )
        
        # Print results
        print_results(
            args.target,
            all_band_counts,
            galaxy_fraction,
            total_available,
            args.hostless_fraction
        )
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


