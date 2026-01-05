import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from torch.utils.data import DataLoader
import yaml
import argparse
import sys
from pathlib import Path

# Add the ML4transients package to path
sys.path.append('/sps/lsst/users/rbonnetguerrini/ML4transients/src')

from ML4transients.data_access.dataset_loader import DatasetLoader

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Run inference on dataset')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--weights-path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--config', type=str, help='Optional path to config file (overrides other args)')
    parser.add_argument('--force', action='store_true', help='Force re-run inference even if results exist')
    parser.add_argument('--save-summary', action='store_true', help='Save inference summary to file')
    parser.add_argument('--visits', type=str, help='Comma-separated list of visit numbers to process')
    parser.add_argument('--mc-dropout', action='store_true', 
                       help='Enable Monte Carlo Dropout for uncertainty estimation')
    parser.add_argument('--mc-samples', type=int, default=50,
                       help='Number of MC Dropout forward passes (default: 50)')
    
    args = parser.parse_args()
    
    # Use config file if provided, otherwise use command line arguments
    if args.config:
        config = load_config(args.config)
        dataset_path = config['dataset']['path']
        weights_path = config['inference']['weights_path']
        save_summary = config.get('inference', {}).get('save_summary', False)
    else:
        dataset_path = args.dataset_path
        weights_path = args.weights_path
        save_summary = args.save_summary
    
    print(f"Loading dataset from: {dataset_path}")
    print(f"Using weights from: {weights_path}")
    
    # Load model config to check for multi-channel setup
    model_config_path = Path(weights_path) / "config.yaml"
    if model_config_path.exists():
        model_config = load_config(str(model_config_path))
        cutout_types = model_config.get('data', {}).get('cutout_types', ['diff'])
        print(f"Model trained with cutout types: {cutout_types}")
        if len(cutout_types) > 1:
            print(f"→ Multi-channel model detected ({len(cutout_types)} channels)")
    
    # Load dataset
    dataset_loader = DatasetLoader(dataset_path)
    
    # Filter visits if specified
    if args.visits:
        visit_list = [int(v.strip()) for v in args.visits.split(',')]
        print(f"Processing specific visits: {visit_list}")
        
        # Pre-load the trainer once for all visits
        print("Loading model (will be cached for all visits)...")
        trainer = dataset_loader._get_or_load_trainer(weights_path)
        
        # Process each visit
        for visit in visit_list:
            if visit not in dataset_loader.visits:
                print(f"Warning: Visit {visit} not found in dataset, skipping")
                continue
                
            print(f"\nProcessing visit {visit}...")
            inference_loader = dataset_loader.get_inference_loader(visit, weights_path=weights_path)
            
            if not args.force and inference_loader.has_inference_results():
                print(f"  Inference results already exist for visit {visit}, skipping")
                continue
            
            inference_loader.run_inference(dataset_loader, trainer=trainer, force=args.force,
                                         mc_dropout=args.mc_dropout, mc_samples=args.mc_samples)
            print(f"  Completed inference for visit {visit}")
        
        print("\nAll specified visits processed!")
        
    else:
        # Run inference on all visits
        print("Running inference on all visits...")
        if args.force:
            inference_results = dataset_loader.run_inference_all_visits(
                weights_path, force=True,
                mc_dropout=args.mc_dropout, mc_samples=args.mc_samples
            )
        else:
            inference_results = dataset_loader.check_or_run_inference(
                weights_path,
                mc_dropout=args.mc_dropout, mc_samples=args.mc_samples
            )
        
        if inference_results:
            print("Inference completed successfully!")
        else:
            print("Inference failed or no results available.")

if __name__ == "__main__":
    main()