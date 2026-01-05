#!/usr/bin/env python3
"""
Pre-compute noise perturbation IDs before batch processing.

This script analyzes the full dataset to:
1. Count total injections and real sources
2. Calculate optimal x and y values for noise injection
3. Randomly select specific diaSourceIds to relabel/remove
4. Save the perturbation plan to a file for batch jobs to use
"""

import yaml
import sys
import numpy as np
from pathlib import Path
from lsst.daf.butler import Butler

# Add project src to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ML4transients.data_preparation.cutouts import compute_xy

def collect_all_source_ids(repo, collection, visits, injection=True):
    """
    Collect all diaSourceIds across all visits, categorized by injection status.
    
    Returns:
        injection_ids: list of diaSourceIds that are injections
        real_ids: list of diaSourceIds that are real sources
    """
    prefix = "injected_" if injection else ""
    butler = Butler(repo, collections=collection)
    
    injection_ids = []
    real_ids = []
    
    print(f"Scanning {len(visits)} visits to collect source IDs...")
    sys.stdout.flush()
    
    for i, visit in enumerate(visits):
        # Get all refs for this visit
        result = butler.registry.queryDatasets(
            datasetType=f'{prefix}goodSeeingDiff_differenceExp',
            collections=collection,
            where=f"visit = {visit}"
        )
        ref_ids = [ref.dataId for ref in result]
        
        # Process each detector
        for ref in ref_ids:
            dia_src = butler.get(f'{prefix}goodSeeingDiff_diaSrcTable', dataId=ref)
            matched_src = butler.get(f'{prefix}goodSeeingDiff_matchDiaSrc', dataId=ref)
            
            is_injection = dia_src.diaSourceId.isin(matched_src.diaSourceId)
            
            # Collect IDs by category
            injection_ids.extend(dia_src[is_injection]['diaSourceId'].tolist())
            real_ids.extend(dia_src[~is_injection]['diaSourceId'].tolist())
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(visits)} visits...")
            sys.stdout.flush()
    
    return injection_ids, real_ids

def main():
    if len(sys.argv) < 2:
        print("Usage: compute_noise_perturbation.py <config.yaml>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check if noise perturbation is requested
    noise_rate = config.get("noise_rate", None)
    
    if noise_rate is None:
        print("No noise_rate specified in config - skipping noise perturbation computation")
        return
    
    if not config.get("injection", False):
        print("WARNING: noise_rate specified but injection=False - noise perturbation only works with injection data")
        return
    
    print(f"\n{'='*70}")
    print(f"Computing Noise Perturbation IDs")
    print(f"Config: {config_path}")
    print(f"Target noise rate: {noise_rate}")
    print(f"{'='*70}\n")
    sys.stdout.flush()
    
    # Get all visits
    sys.path.insert(0, str(script_dir))
    from create_batch_jobs import get_all_visits_from_collection
    
    if "visits" in config:
        visits = config["visits"]
    else:
        visits = get_all_visits_from_collection(
            config["repo"], 
            config["collection"], 
            config.get("injection", False)
        )
    
    print(f"Total visits to process: {len(visits)}")
    sys.stdout.flush()
    
    # Collect all source IDs
    injection_ids, real_ids = collect_all_source_ids(
        config["repo"],
        config["collection"],
        visits,
        config.get("injection", False)
    )
    
    I0 = len(injection_ids)
    R0 = len(real_ids)
    
    print(f"\nInitial counts:")
    print(f"  Injections (I0): {I0}")
    print(f"  Real sources (R0): {R0}")
    sys.stdout.flush()
    
    # Calculate optimal x and y values
    x_best, y_best, F_best, noise_best = compute_xy(I0, R0, noise_rate)
    
    print(f"\nNoise perturbation plan:")
    print(f"  Injections to relabel as real: {x_best}")
    print(f"  Real samples to remove: {y_best}")
    print(f"  Final class size (each): {F_best}")
    print(f"  Actual noise rate: {noise_best:.6f}")
    sys.stdout.flush()
    
    # Randomly select IDs to relabel and remove
    np.random.seed(42)  # For reproducibility
    relabel_ids = np.random.choice(injection_ids, size=x_best, replace=False).tolist()
    
    np.random.seed(43)  # Different seed for removal
    remove_ids = np.random.choice(real_ids, size=y_best, replace=False).tolist()
    
    # Prepare perturbation data
    perturbation_data = {
        'noise_rate': noise_rate,
        'I0': I0,
        'R0': R0,
        'x_relabel': x_best,
        'y_remove': y_best,
        'F_final': F_best,
        'noise_actual': float(noise_best),
        'relabel_ids': [int(id) for id in relabel_ids],  # Convert to int for YAML
        'remove_ids': [int(id) for id in remove_ids]
    }
    
    # Save to file in the same directory as the output data
    output_path = Path(config['path']) / 'noise_perturbation.yaml'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(perturbation_data, f)
    
    print(f"\nPerturbation IDs saved to: {output_path}")
    print(f"{'='*70}\n")
    sys.stdout.flush()
    


if __name__ == "__main__":
    main()
