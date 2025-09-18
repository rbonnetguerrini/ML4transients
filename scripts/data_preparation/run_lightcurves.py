#!/usr/bin/env python3
import sys
import yaml
import argparse
from pathlib import Path
from ML4transients.data_preparation.lightcurves import extract_and_save_lightcurves_with_index
from ML4transients.data_preparation.crossmatch import perform_crossmatching
from datetime import datetime
from ML4transients.utils import realtime_update, append_config

def parse_args():
    parser = argparse.ArgumentParser(description='Extract lightcurves and optionally perform cross-matching')
    parser.add_argument('config', help='Configuration file path')
    parser.add_argument('--crossmatch', action='store_true', help='Enable cross-matching')
    parser.add_argument('--catalogs', nargs='+', help='Paths to catalog files for cross-matching')
    parser.add_argument('--catalog-names', nargs='+', help='Names for catalogs (default: use filenames)')
    parser.add_argument('--tolerances', nargs='+', type=float, help='Matching tolerances in arcsec (default: 1.0)')
    parser.add_argument('--ra-columns', nargs='+', help='RA column names for catalogs (default: ra)')
    parser.add_argument('--dec-columns', nargs='+', help='Dec column names for catalogs (default: dec)')
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = Path(args.config)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Creating lightcurve data from config: {config_path}")
    print(f"Output path from config: {config.get('path', 'NOT SPECIFIED')}")
    
    # Ensure path is correctly set
    if 'path' not in config or not config['path']:
        raise ValueError("No output path specified in config")

    realtime_update(config_path, "running")

    try:
        print("=== Extracting lightcurves and creating indices ===")
        extract_and_save_lightcurves_with_index(config)
        
        # Perform cross-matching if enabled
        if args.crossmatch and args.catalogs:
            print("\n=== Performing cross-matching ===")
            perform_crossmatching(
                lsst_config=config,
                catalog_paths=args.catalogs,
                catalog_names=args.catalog_names,
                tolerances_arcsec=args.tolerances,
                ra_columns=args.ra_columns,
                dec_columns=args.dec_columns
            )
        elif args.crossmatch:
            print("Cross-matching enabled but no catalogs specified. Use --catalogs option.")
        else:
            print("Cross-matching not requested")
            
        config["run_info"]["status"] = "completed"
    except Exception as e:
        config["run_info"]["status"] = "failed"
        raise
    finally:
        config["run_info"]["finished"] = datetime.utcnow().isoformat()
        append_config(config)
        realtime_update(config_path, config["run_info"]["status"])

if __name__ == "__main__":
    main()
