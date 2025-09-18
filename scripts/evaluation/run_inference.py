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
    
    # Load dataset
    dataset_loader = DatasetLoader(dataset_path)
    
    # Run inference
    print("Running inference...")
    if args.force:
        inference_results = dataset_loader.run_inference_all_visits(weights_path, force=True)
    else:
        inference_results = dataset_loader.check_or_run_inference(weights_path)
    
    if inference_results:
        print("Inference completed successfully!")
        
        # Handle results - could be dict with metrics or other structure
        if isinstance(inference_results, dict):
            if 'accuracy' in inference_results:
                print(f"Accuracy: {inference_results['accuracy']}")
            if 'y_pred' in inference_results:
                print(f"Predictions shape: {inference_results['y_pred'].shape}")
            
            # Save results summary if requested
            if save_summary:
                summary_path = Path(weights_path) / 'inference_summary.txt'
                with open(summary_path, 'w') as f:
                    if 'accuracy' in inference_results:
                        f.write(f"Accuracy: {inference_results['accuracy']}\n")
                    if 'y_pred' in inference_results:
                        f.write(f"Predictions shape: {inference_results['y_pred'].shape}\n")
                    if 'confusion_matrix' in inference_results:
                        f.write(f"Confusion matrix:\n{inference_results['confusion_matrix']}\n")
                print(f"Summary saved to: {summary_path}")
        else:
            # Handle other return types if needed
            print(f"Inference results type: {type(inference_results)}")
            if hasattr(inference_results, 'metrics'):
                print(f"Accuracy: {inference_results.metrics['accuracy']}")
                print(f"Predictions shape: {inference_results.predictions.shape}")
                
                if save_summary:
                    summary_path = Path(weights_path) / 'inference_summary.txt'
                    with open(summary_path, 'w') as f:
                        f.write(f"Accuracy: {inference_results.metrics['accuracy']}\n")
                        f.write(f"Predictions shape: {inference_results.predictions.shape}\n")
                    print(f"Summary saved to: {summary_path}")
    else:
        print("Inference failed or no results available.")

if __name__ == "__main__":
    main()