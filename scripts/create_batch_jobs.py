#!/usr/bin/env python3
import yaml
import sys
from pathlib import Path
import math

# Add project src to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from lsst.daf.butler import Butler

def get_all_visits_from_collection(repo, collection, injection=False):
    """Get all available visits from a collection."""
    prefix = "injected_" if injection else ""
    butler = Butler(repo, collections=collection)
    registry = butler.registry
    
    datasetRefs = registry.queryDatasets(
        datasetType=f'{prefix}goodSeeingDiff_differenceExp',
        collections=collection
    )
    
    all_visits = sorted(set(ref.dataId['visit'] for ref in datasetRefs))
    return all_visits

def create_batch_configs(base_config_path, batch_size=50):
    """Split a large collection into batch configs."""
    
    base_config_path = Path(base_config_path)
    
    # Load base config
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)
    
    # Get all visits if not specified or if too many
    if "visits" not in base_config or len(base_config["visits"]) > batch_size:
        all_visits = get_all_visits_from_collection(
            base_config["repo"], 
            base_config["collection"], 
            base_config.get("injection", False)
        )
    else:
        all_visits = base_config["visits"]
    
    print(f"Found {len(all_visits)} total visits")
    print(f"Creating batches of {batch_size} visits each")
    
    # Calculate number of batches
    num_batches = math.ceil(len(all_visits) / batch_size)
    
    # Create configs directory
    configs_dir = Path("configs/batches")
    configs_dir.mkdir(exist_ok=True)
    
    batch_configs = []
    
    for batch_num in range(num_batches):
        # Get visits for this batch
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(all_visits))
        batch_visits = all_visits[start_idx:end_idx]
        
        # Create batch config
        batch_config = base_config.copy()
        batch_config["visits"] = batch_visits
        # Skip global index creation in batch mode
        batch_config["skip_global_index"] = True
        
        # Save batch config
        batch_config_path = configs_dir / f"batch_{batch_num:03d}_{base_config_path.stem}.yaml"
        with open(batch_config_path, 'w') as f:
            yaml.dump(batch_config, f)
        
        batch_configs.append(batch_config_path)
        print(f"Created batch {batch_num:03d}: visits {batch_visits[0]}-{batch_visits[-1]} ({len(batch_visits)} visits)")
    
    return batch_configs, num_batches

if __name__ == "__main__":
    base_config = sys.argv[1] if len(sys.argv) > 1 else "configs/configs_cutout.yaml"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    batch_configs, num_batches = create_batch_configs(base_config, batch_size)
    
    print(f"\nCreated {len(batch_configs)} batch config files in configs/batches/")
    print(f"To submit to SLURM:")
    print(f"  ./scripts/submit_collection.sh {base_config} {batch_size}")