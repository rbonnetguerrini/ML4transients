#!/usr/bin/env python3
"""
Script to run comprehensive model evaluation including metrics and interpretability analysis.
"""

import argparse
import yaml
from pathlib import Path
import sys
import torch
import numpy as np
import gc
from torch.utils.data import DataLoader

sys.path.append('/sps/lsst/users/rbonnetguerrini/ML4transients/src')

from ML4transients.data_access import DatasetLoader
from ML4transients.evaluation.metrics import EvaluationMetrics, load_inference_metrics
from ML4transients.evaluation.visualizations import create_evaluation_dashboard, create_interpretability_dashboard
from ML4transients.evaluation.interpretability import UMAPInterpreter
from ML4transients.training.pytorch_dataset import PytorchDataset
from ML4transients.utils import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to evaluation configuration file")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to data directory")
    parser.add_argument("--weights-path", type=str, default=None,
                       help="Path to trained model weights (for new inference)")
    parser.add_argument("--model-hash", type=str, default=None,
                       help="Model hash for existing inference results")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for evaluation results")
    parser.add_argument("--visits", type=int, nargs="+", default=None,
                       help="Specific visits to evaluate (default: all)")
    parser.add_argument("--run-inference", action="store_true",
                       help="Run inference if not available (requires weights-path)")
    parser.add_argument("--interpretability", action="store_true",
                       help="Run UMAP-based interpretability analysis")
    parser.add_argument("--port", type=int, default=5006,
                       help="Port for Bokeh server (default: 5006)")
    
    return parser.parse_args()

def load_evaluation_config(config_path: Path) -> dict:
    """Load evaluation configuration.
    
    Parameters
    ----------
    config_path : Path
        Path to YAML configuration file
        
    Returns
    -------
    dict
        Loaded configuration dictionary
    """
    # Load configuration
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def collect_inference_results(dataset_loader: DatasetLoader, weights_path: str = None,
                            visits: list = None, model_hash: str = None) -> dict:
    """Collect inference results for specified visits.
    
    Parameters
    ----------
    dataset_loader : DatasetLoader
        Dataset loader instance
    weights_path : str, optional
        Path to model weights for new inference
    visits : list, optional
        List of visit numbers to collect. If None, uses all visits
    model_hash : str, optional
        Model hash for loading existing inference results
        
    Returns
    -------
    dict or None
        Dictionary mapping visit numbers to InferenceLoader instances,
        or None if collection failed or user cancelled
        
    Raises
    ------
    ValueError
        If neither weights_path nor model_hash is provided
    """
    # Validate arguments
    if not weights_path and not model_hash:
        raise ValueError("Must provide either weights_path or model_hash")
    
    if model_hash:
        print("Syncing inference registry to discover existing results...")
        dataset_loader.sync_inference_registry()

    if visits is None:
        visits = dataset_loader.visits
    
    results = {}
    missing_visits = []
    
    print(f"Collecting inference results for {len(visits)} visits...")
    
    for i, visit in enumerate(visits):
        print(f"Checking visit {visit} ({i+1}/{len(visits)})...", end=' ')
        try:
            inference_loader = dataset_loader.get_inference_loader(
                visit=visit,
                weights_path=weights_path,
                model_hash=model_hash
            )
            
            if inference_loader and inference_loader.has_inference_results():
                results[visit] = inference_loader
            else:
                missing_visits.append(visit)
                print("✗")
                
        except Exception as e:
            print(f"Error loading inference for visit {visit}: {e}")
            missing_visits.append(visit)
    
    if missing_visits:
        print(f"\nFound inference results for {len(results)} out of {len(visits)} visits")
        print(f"Missing inference results for visits: {missing_visits}")
        
        if len(results) == 0:
            print("No inference results found at all.")
            if not weights_path:
                print("Cannot run inference without weights_path")
            return None
        else:
            # Give user choice to continue with partial results
            print(f"\nOptions:")
            print(f"  - Continue evaluation with {len(results)} visits that have results")
            if weights_path:
                print(f"  - Run inference for {len(missing_visits)} missing visits")
                print(f"  - Cancel evaluation")
                
                while True:
                    response = input("\nChoose: [c]ontinue with available data, [r]un missing inference, [q]uit: ").lower().strip()
                    if response in ['c', 'continue']:
                        print(f"Continuing evaluation with {len(results)} available visits")
                        break
                    elif response in ['r', 'run']:
                        print("Will run inference for missing visits")
                        return None  # Signal to run inference
                    elif response in ['q', 'quit', 'cancel']:
                        print("Evaluation cancelled")
                        return None
                    else:
                        print("Please enter 'c', 'r', or 'q'")
            else:
                while True:
                    response = input(f"\nContinue evaluation with {len(results)} available visits? [y/n]: ").lower().strip()
                    if response in ['y', 'yes']:
                        print(f"Continuing evaluation with {len(results)} available visits")
                        break
                    elif response in ['n', 'no']:
                        print("Evaluation cancelled")
                        return None
                    else:
                        print("Please enter 'y' or 'n'")
    else:
        print(f"\nSuccessfully loaded inference results for all {len(results)} visits")
    
    return results

def aggregate_results(inference_results: dict) -> tuple:
    """Aggregate predictions and labels across all visits.
    
    Parameters
    ----------
    inference_results : dict
        Dictionary mapping visit numbers to InferenceLoader instances
        
    Returns
    -------
    predictions : np.ndarray
        Concatenated predictions from all visits
    labels : np.ndarray
        Concatenated true labels from all visits
    source_ids : np.ndarray
        Concatenated diaSourceId arrays from all visits
    """
    all_predictions = []
    all_labels = []
    all_ids = []
    
    for i, (visit, loader) in enumerate(inference_results.items()):
        print(f"Loading visit {visit} results ({i+1}/{len(inference_results)})...")
        all_predictions.append(loader.predictions)
        all_labels.append(loader.labels)
        all_ids.append(loader.ids)
    
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    source_ids = np.concatenate(all_ids)
    
    # Clear individual loader data to save memory
    del all_predictions, all_labels, all_ids
    import gc
    gc.collect()
    
    return (predictions, labels, source_ids)

def create_evaluation_data_loader(dataset_loader: DatasetLoader, visits: list,
                                max_samples: int = 3000) -> DataLoader:
    """Create a DataLoader for interpretability analysis.
    
    Parameters
    ----------
    dataset_loader : DatasetLoader
        Dataset loader instance
    visits : list
        List of visit numbers to include
    max_samples : int, default=3000
        Maximum number of samples (not currently enforced)
        
    Returns
    -------
    DataLoader
        Configured DataLoader with optimized settings for evaluation
    """
    print(f"Creating evaluation dataset from {len(visits)} visits...")
    
    # Create inference dataset for the specified visits
    eval_dataset = PytorchDataset.create_inference_dataset(
        dataset_loader, 
        visits=visits
    )
    
    # Use optimized DataLoader settings for evaluation
    return DataLoader(
        eval_dataset,
        batch_size=64,  # Larger batch size for evaluation efficiency
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing overhead for evaluation
        pin_memory=False,  # Save memory
        drop_last=False
    )

def main():
    args = parse_args()
    
    # Validate arguments
    if not args.weights_path and not args.model_hash:
        print("Error: Must provide either --weights-path or --model-hash")
        return
    
    if args.run_inference and not args.weights_path:
        print("Error: --run-inference requires --weights-path")
        return
    
    # Load configuration
    config = load_evaluation_config(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset...")
    dataset_loader = DatasetLoader(args.data_path)
    
    # Show available inference results if using model_hash
    if args.model_hash:
        print("\nAvailable inference results:")
        dataset_loader.list_available_inference()
    
    # Determine visits to evaluate
    visits_to_eval = args.visits if args.visits else dataset_loader.visits
    print(f"\nEvaluating visits: {visits_to_eval}")
    
    # Check for inference results
    print("Checking inference results...")
    inference_results = collect_inference_results(
        dataset_loader, 
        weights_path=args.weights_path, 
        visits=visits_to_eval, 
        model_hash=args.model_hash
    )
    
    if inference_results is None:
        if args.run_inference and args.weights_path:
            print("Running inference for missing visits...")
            # Run inference for all originally requested visits
            inference_results = dataset_loader.run_inference_all_visits(
                args.weights_path, force=False
            )
            
            # After running inference, collect results again
            if inference_results:
                print("Re-collecting inference results after running inference...")
                inference_results = collect_inference_results(
                    dataset_loader, 
                    weights_path=args.weights_path, 
                    visits=visits_to_eval, 
                    model_hash=args.model_hash
                )
        else:
            print("Evaluation stopped.")
            if args.weights_path and not args.run_inference:
                print("Use --run-inference to generate missing results.")
            elif not args.weights_path:
                print("Use --weights-path and --run-inference to generate them, or use --model-hash to load existing results.")
            return
    
    # Update visits_to_eval to only include visits with successful inference
    if inference_results:
        actual_visits = list(inference_results.keys())
        if set(actual_visits) != set(visits_to_eval):
            print(f"\nNote: Evaluation will proceed with {len(actual_visits)} visits instead of {len(visits_to_eval)} originally requested")
            visits_to_eval = actual_visits
    
    # Aggregate results across visits
    print("Aggregating results...")
    predictions, labels, source_ids = aggregate_results(inference_results)
    
    # Create metrics evaluation
    print("Computing evaluation metrics...")
    metrics = EvaluationMetrics(predictions, labels)
    
    # Determine model name for dashboard title
    if args.weights_path:
        model_name = Path(args.weights_path).name
    else:
        model_name = f"Model Hash {args.model_hash}"
    
    # Create metrics dashboard
    print("Creating metrics dashboard...")
    metrics_dashboard = create_evaluation_dashboard(
        metrics, 
        output_path=output_dir / "metrics_dashboard.html",
        title=f"Model Evaluation - {model_name}"
    )
    
    # Print summary
    summary = metrics.summary()
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Total samples: {len(predictions):,}")
    print(f"Visits evaluated: {len(visits_to_eval)} {visits_to_eval}")
    print("-"*50)
    for metric, value in summary.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    print("="*50)
    
    # Run interpretability analysis if requested
    if args.interpretability:
        if not args.weights_path:
            print("Warning: Interpretability analysis requires --weights-path, skipping...")
        else:
            print("\nRunning interpretability analysis...")
            
            try:
                # Create data loader for UMAP analysis
                eval_data_loader = create_evaluation_data_loader(
                    dataset_loader, visits_to_eval, 
                    max_samples=config.get('interpretability', {}).get('max_samples', 3000)
                )
                
                # Initialize UMAP interpreter
                interpreter = UMAPInterpreter(args.weights_path)
                
                # Get additional features if specified in config
                additional_features = {}
                if 'features' in config.get('interpretability', {}):
                    # Add SNR or other features here
                    pass
                
                # Create interpretability dashboard with configuration
                interp_dashboard = create_interpretability_dashboard(
                    interpreter,
                    eval_data_loader,
                    predictions[:len(eval_data_loader.dataset)],  # Match lengths
                    labels[:len(eval_data_loader.dataset)],
                    output_path=output_dir / "interpretability_dashboard.html",
                    additional_features=additional_features,
                    config=config
                )
                
                print(f"Interpretability dashboard saved to {output_dir / 'interpretability_dashboard.html'}")
                
            except Exception as e:
                print(f"Error in interpretability analysis: {e}")
                print("Continuing with basic evaluation...")
            
            finally:
                # Cleanup memory after interpretability analysis
                if 'eval_data_loader' in locals():
                    del eval_data_loader
                if 'interpreter' in locals():
                    del interpreter
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save detailed results
    results_file = output_dir / "evaluation_results.yaml"
    eval_data = {
        'visits_evaluated': visits_to_eval,
        'total_samples': len(predictions),
        'metrics': summary,
        'confusion_matrix': metrics.confusion_mat.tolist()
    }
    
    if args.weights_path:
        eval_data['model_path'] = args.weights_path
    if args.model_hash:
        eval_data['model_hash'] = args.model_hash
    
    with open(results_file, 'w') as f:
        yaml.dump(eval_data, f)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Open metrics dashboard: file://{output_dir / 'metrics_dashboard.html'}")
    if args.interpretability and args.weights_path:
        print(f"Open interpretability dashboard: file://{output_dir / 'interpretability_dashboard.html'}")

if __name__ == "__main__":
    main()