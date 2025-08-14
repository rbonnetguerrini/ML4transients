#!/usr/bin/env python3
import yaml
import sys
from pathlib import Path

# Add project src to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from ML4transients.data_preparation.lightcurves import extract_and_save_lightcurves_with_index

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/configs_cutout.yaml"
    config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Creating lightcurve data from config: {config_path}")
    print(f"Output path from config: {config.get('path', 'NOT SPECIFIED')}")
    
    # Ensure path is correctly set
    if 'path' not in config or not config['path']:
        raise ValueError("No output path specified in config")
    
    extract_and_save_lightcurves_with_index(config)
    print("Lightcurve extraction completed!")

if __name__ == "__main__":
    main()
