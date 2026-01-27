#!/usr/bin/env python3
"""
Create global cutout index after batch processing is complete.
Run this once after all batch jobs have finished.

Usage:
    python scripts/data_preparation/create_global_index_post_batch.py configs/data_preparation/configs_cutout_injected.yaml
"""

import sys
import yaml
from pathlib import Path

# Add project src to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from ML4transients.data_preparation.cutouts import create_global_cutout_index
from ML4transients.data_preparation.lightcurves import (
    create_lightcurve_index, 
    create_diasource_patch_index,
    save_lightcurve_index,
    save_diasource_patch_index
)
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/data_preparation/create_global_index_post_batch.py config.yaml")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Creating Global Indices (Post-Batch)")
    print("=" * 60)
    print(f"Data directory: {config['path']}")
    print()
    
    # Remove skip flags if present
    config.pop("skip_global_index", None)
    config.pop("skip_lightcurve_index", None)
    
    output_path = config['path']
    
    try:
        # Create the global cutout index
        print("\n=== Creating global cutout index ===")
        result = create_global_cutout_index(config)
        if result is not None:
            print("  Cutout index created")
        else:
            print(" Cutout index creation returned None")
        
        # Create lightcurve indices
        print("\n=== Creating diaObjectId-->patch index ===")
        object_index_df = create_lightcurve_index(config)
        print(f"  Object index created: {len(object_index_df)} entries")
        
        print("\n=== Creating diaSourceId-->patch index ===")
        source_index_df = create_diasource_patch_index(config)
        print(f"  Source index created: {len(source_index_df)} entries")
        
        # Save lightcurve indices
        path_lightcurves = f"{output_path}/lightcurves"
        os.makedirs(path_lightcurves, exist_ok=True)
        
        object_index_path = os.path.join(path_lightcurves, "lightcurve_index.h5")
        source_index_path = os.path.join(path_lightcurves, "diasource_patch_index.h5")
        
        save_lightcurve_index(object_index_df, object_index_path)
        save_diasource_patch_index(source_index_df, source_index_path)
        
        print(f"\n Object index saved: {object_index_path}")
        print(f" Source index saved: {source_index_path}")
        
        print("\n" + "=" * 60)
        print("SUCCESS: All global indices created!")
        print("All batch jobs are now properly indexed.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: Failed to create global indices: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
