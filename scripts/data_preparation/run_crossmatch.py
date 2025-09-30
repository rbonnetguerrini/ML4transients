#!/usr/bin/env python3
"""
Standalone script for running cross-matching on existing lightcurve dataset.

Usage:
    python run_crossmatch.py --dataset PATH --catalog_file CATALOG --output_file OUTPUT
    
Examples:
    # Basic usage with Gaia catalog (saves to crossmatch/crossmatch_results.h5)
    python run_crossmatch.py \
        --dataset /path/to/lightcurve/data \
        --catalog_file saved/source_cat_gaia.pkl
    
    # Custom output location
    python run_crossmatch.py \
        --dataset /path/to/lightcurve/data \
        --catalog_file saved/source_cat_gaia.pkl \
        --output_file lightcurves_with_gaia_crossmatch.pkl
    
    # Custom catalog with specific column names
    python run_crossmatch.py \
        --dataset /path/to/lightcurve/data \
        --catalog_file /path/to/ps1_catalog.csv \
        --catalog_name ps1 \
        --ra_column RAMean \
        --dec_column DecMean \
        --tolerance 0.5 \
        --output_file lightcurves_with_ps1_crossmatch.h5
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ML4transients.data_access.dataset_loader import DatasetLoader

def parse_args():
    parser = argparse.ArgumentParser(
        description='Cross-match lightcurve dataset with external catalogs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic Gaia cross-matching (saves to crossmatch/crossmatch_results.h5)
  %(prog)s --dataset /path/to/lightcurve/data --catalog_file saved/source_cat_gaia.pkl
  
  # Custom output location
  %(prog)s --dataset /path/to/lightcurve/data --catalog_file saved/source_cat_gaia.pkl --output_file gaia_crossmatch.pkl
  
  # Custom catalog with specific settings  
  %(prog)s --dataset /path/to/lightcurve/data --catalog_file ps1.csv \\
           --catalog_name ps1 --ra_column RAMean --dec_column DecMean \\
           --tolerance 0.5 --output_file ps1_crossmatch.h5
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', required=True, 
                       help='Path to lightcurve dataset directory')
    parser.add_argument('--catalog_file', required=True, 
                       help='Path to external catalog file (.pkl, .csv, .h5)')
    parser.add_argument('--output_file', 
                       help='Output file path for cross-match results (default: crossmatch/crossmatch_results.h5 in dataset dir)')
    
    # Optional arguments
    parser.add_argument('--catalog_name', help='Name for catalog (default: use filename)')
    parser.add_argument('--ra_column', default='ra', help='RA column name in catalog (default: ra)')
    parser.add_argument('--dec_column', default='dec', help='Dec column name in catalog (default: dec)')
    parser.add_argument('--tolerance', type=float, default=1.0, 
                       help='Matching tolerance in arcsec (default: 1.0)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate paths
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_path}")
        return 1
    
    catalog_path = Path(args.catalog_file)
    if not catalog_path.exists():
        print(f"Error: Catalog file not found: {catalog_path}")
        return 1
    
    # Set catalog name
    catalog_name = args.catalog_name or catalog_path.stem
    
    # Set default output path if not provided
    if args.output_file is None:
        # Use catalog name and tolerance in filename to avoid overwriting
        output_dir = dataset_path / "crossmatch"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"crossmatch_{catalog_name}.h5"
        output_path = output_dir / output_filename
        args.output_file = str(output_path)
    else:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=== Cross-Matching Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Catalog: {args.catalog_file}")
    print(f"Catalog Name: {catalog_name}")
    print(f"RA Column: {args.ra_column}")
    print(f"Dec Column: {args.dec_column}")
    print(f"Tolerance: {args.tolerance}\"")
    print(f"Output: {args.output_file}")

    try:
        print("\n=== Loading Dataset ===")
        dataset = DatasetLoader(args.dataset)
        
        print("\n=== Starting Cross-Matching ===")
        start_time = datetime.utcnow()
        
        results = dataset.perform_crossmatch(
            catalog_file=str(catalog_path),
            catalog_name=catalog_name,
            ra_col=args.ra_column,
            dec_col=args.dec_column,
            tolerance_arcsec=args.tolerance,
            output_file=args.output_file
        )
        
        # Only save diaObjectId and crossmatch column to output file
        match_col = f'in_{catalog_name}'
        minimal_results = results[['diaObjectId', match_col]] if match_col in results.columns else results[['diaObjectId']]
        if output_path.suffix == '.h5':
            minimal_results.to_hdf(output_path, key='crossmatch', mode='w', format='table', index=False)
        elif output_path.suffix == '.csv':
            minimal_results.to_csv(output_path, index=False)
        elif output_path.suffix == '.pkl':
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(minimal_results, f)
        else:
            print(f"Unsupported output format: {output_path.suffix}")
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n=== Cross-Matching Completed ===")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Total objects: {len(results)}")
        
        # Print summary statistics
        if match_col in results.columns:
            match_count = results[match_col].sum()
            match_rate = match_count / len(results) * 100
            print(f"Matches found: {match_count}/{len(results)} ({match_rate:.1f}%)")
        
        return 0
        
    except Exception as e:
        print(f"Error during cross-matching: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())