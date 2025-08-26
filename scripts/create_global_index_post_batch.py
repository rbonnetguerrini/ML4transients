#!/usr/bin/env python3
"""
Create global cutout index after batch processing is complete.
Run this once after all batch jobs have finished.

Usage:
    python scripts/create_global_index_post_batch.py configs/configs_cutout.yaml
"""

import sys
import yaml
from pathlib import Path

# Add project src to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from ML4transients.data_preparation.cutouts import create_global_cutout_index

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/create_global_index_post_batch.py config.yaml")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Creating Global Cutout Index (Post-Batch)")
    print("=" * 60)
    print(f"Data directory: {config['path']}")
    print()
    
    # Remove skip flag if present
    config.pop("skip_global_index", None)
    
    # Create the global index
    try:
        result = create_global_cutout_index(config)
        if result is not None:
            print("\n" + "=" * 60)
            print("SUCCESS: Global index created!")
            print("All batch jobs are now properly indexed.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("WARNING: Global index creation returned None")
            print("Check if all feature files exist.")
            print("=" * 60)
    except Exception as e:
        print(f"\nERROR: Failed to create global index: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
