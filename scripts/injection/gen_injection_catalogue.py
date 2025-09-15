#!/usr/bin/env python3
"""
Script to generate injection catalogs for transient detection
"""

import argparse
import sys
import os
import time
from pathlib import Path
import yaml
from typing import List, Dict, Any, Optional

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ML4transients.injection.injection import (
    process_all_ccds, 
    InjectionConfig, 
    CatalogProcessor
)

# LSST imports
import lsst.daf.butler as dafButler
from lsst.daf.butler import Butler

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def setup_butler(config: Dict[str, Any]) -> Butler:
    """Initialize Butler with configuration"""
    print(f"Initializing Butler with repo: {config['butler']['repo']}")
    
    butler = Butler(config['butler']['repo'], collections = config['butler']['collections'])
    return butler

def get_dataset_refs(butler: Butler, config: Dict[str, Any]) -> List[Any]:
    """Get dataset references based on configuration"""
    data_config = config['data_selection']
    
    # Build data ID constraints
    constraints = {
        'band': data_config['band']
    }
    
    # Add patch constraint if specified
    if data_config.get('patch') is not None:
        constraints['patch'] = data_config['patch']
    
    print(f"Searching for datasets with constraints: {constraints}")
    # Get dataset references
    refs = list(butler.registry.queryDatasets(
        'sourceTable',
        collections=config['butler']['collections'],
        where=f"band = '{data_config['band']}'",
        findFirst=True
    ))
    
    # Filter by visit range if specified
    if data_config.get('visit_range') is not None:
        min_visit, max_visit = data_config['visit_range']
        refs = [ref for ref in refs 
                if min_visit <= ref.dataId['visit'] <= max_visit]
    
    # Filter by detector range if specified  
    if data_config.get('detector_range') is not None:
        min_det, max_det = data_config['detector_range']
        refs = [ref for ref in refs 
                if min_det <= ref.dataId['detector'] <= max_det]
    
    print(f"Found {len(refs)} dataset references")
    return refs

def create_injection_config(config: Dict[str, Any]) -> InjectionConfig:
    """Create injection configuration from dict"""
    inj_config = config['injection']
    return InjectionConfig(
        galaxy_fraction=inj_config['galaxy_fraction'],
        hostless_fraction=inj_config['hostless_fraction'],
        min_injections=inj_config['min_injections'],
        max_injections=inj_config.get('max_injections'),
        random_seed=inj_config['random_seed']
    )

def generate_output_filename(config: Dict[str, Any]) -> str:
    """Generate output filename based on configuration"""
    output_config = config['output']
    data_config = config['data_selection']
    
    # Create filename with key parameters
    filename_parts = [
        output_config['filename_prefix'],
        f"band{data_config['band']}",
        f"gfrac{config['injection']['galaxy_fraction']:.3f}",
        f"seed{config['injection']['random_seed']}"
    ]
    
    # Add visit range if specified
    if data_config.get('visit_range'):
        min_v, max_v = data_config['visit_range']
        filename_parts.append(f"visits{min_v}-{max_v}")
    
    # Add patch if specified
    if data_config.get('patch'):
        filename_parts.append(f"patch{data_config['patch']}")
    
    return "_".join(filename_parts)

def main():
    parser = argparse.ArgumentParser(
        description="Generate injection catalogs for transient detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c', 
        type=str,
        required=True,
        help='Configuration file (YAML format)'
    )
    
    parser.add_argument(
        '--band', '-b',
        type=str,
        choices=['u', 'g', 'r', 'i', 'z', 'y'],
        help='Override band from config file'
    )
    
    parser.add_argument(
        '--output-suffix',
        type=str,
        help='Add suffix to output filename'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually running'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)    
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {args.config}")
        return 1
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in configuration file: {e}")
        return 1

    # Override config with command line arguments
    if args.band:
        config['data_selection']['band'] = args.band
        print(f"Overriding band to: {args.band}")
    
    # Add suffix to filename if specified
    if args.output_suffix:
        config['output']['filename_prefix'] += f"_{args.output_suffix}"

    # Print configuration summary
    print("\n" + "="*60)
    print("INJECTION CATALOG GENERATION")
    print("="*60)
    print(f"Band: {config['data_selection']['band']}")
    print(f"Galaxy fraction: {config['injection']['galaxy_fraction']:.3f}")
    print(f"Hostless fraction: {config['injection']['hostless_fraction']:.3f}")
    print(f"Min injections per CCD: {config['injection']['min_injections']}")
    print(f"Max injections per CCD: {config['injection']['max_injections']}")
    print(f"Random seed: {config['injection']['random_seed']}")
    print(f"Output directory: {config['output']['output_dir']}")
    print("="*60)
    
    try:
        # Setup Butler
        print("\nSetting up Butler...")
        butler = setup_butler(config)
        
        # Get dataset references
        print("Getting dataset references...")
        dataset_refs = get_dataset_refs(butler, config)
        
        if len(dataset_refs) == 0:
            print("ERROR: No datasets found matching criteria!")
            return 1
        
        print(f"Found {len(dataset_refs)} CCDs to process")
        
        # Show sample of what will be processed
        print(f"\nSample dataset references:")
        for i, ref in enumerate(dataset_refs[:5]):
            print(f"  {ref.dataId}")
        if len(dataset_refs) > 5:
            print(f"  ... and {len(dataset_refs) - 5} more")
        
        if args.dry_run:
            print("\nDry run complete. No processing performed.")
            return 0
        
        # Create output directory
        output_dir = Path(config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = generate_output_filename(config)
        
        print(f"\nOutput filename: {filename}")
        
        # Create injection configuration
        injection_config = create_injection_config(config)
        
        # Process all CCDs
        print(f"\nStarting catalog generation...")
        start_time = time.time()
        
        result_catalog = process_all_ccds(
            dataset_refs,
            butler,
            config['data_selection']['band'],
            config=injection_config,
            csv=config['output']['save_csv'],
            save_filename=filename if config['output']['save_csv'] else None
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Summary
        print(f"\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print(f"Total injections generated: {len(result_catalog)}")
        print(f"Processing time: {processing_time:.1f} seconds")
        print(f"Average time per CCD: {processing_time/len(dataset_refs):.2f} seconds")
        
        if config['output']['save_csv']:
            output_file = output_dir / f"{filename}.csv"
            print(f"Catalog saved to: {output_file}")
            if output_file.exists():
                print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())