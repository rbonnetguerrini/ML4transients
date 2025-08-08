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
import time
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
    parser.add_argument("--optimize-umap", action="store_true",
                       help="Optimize UMAP parameters during interpretability analysis")
    parser.add_argument("--enable-clustering", action="store_true",
                       help="Enable HDBSCAN clustering on high-dimensional features")
    parser.add_argument("--port", type=int, default=5006,
                       help="Port for Bokeh server (default: 5006)")
    
    return parser.parse_args()

def load_evaluation_config(config_path: Path) -> dict:
    """Load YAML evaluation configuration.

    Parameters
    ----------
    config_path : Path
        Path to YAML config.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def collect_inference_results(dataset_loader: DatasetLoader, weights_path: str = None,
                            visits: list = None, model_hash: str = None) -> dict:
    """Collect (or discover) inference loaders for requested visits.

    Notes
    -----
    If some visits are missing, user is prompted (unless non-interactive usage).
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
    """Concatenate predictions / labels / ids across visits.

    Returns
    -------
    tuple
        (predictions, labels, source_ids)
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
    """Return DataLoader tailored for feature extraction / interpretability."""
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
    """Orchestrate full evaluation:
       1. Load config + dataset
       2. Collect or run inference
       3. Aggregate results + compute metrics (cached)
       4. Build dashboards
       5. Optional interpretability (UMAP)
       6. Persist summary
    """
    args = parse_args()
    
    # Start overall timing
    start_time = time.time()
    # Validate arguments
    if not args.weights_path and not args.model_hash:
        print("Error: Must provide either --weights-path or --model-hash")
        return
    
    if args.run_inference and not args.weights_path:
        print("Error: --run-inference requires --weights-path")
        return
    
    # Load configuration
    print("Loading configuration...")
    step_start = time.time()
    config = load_evaluation_config(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Configuration loaded in {time.time() - step_start:.2f}s")
    
    print("Loading dataset...")
    step_start = time.time()
    dataset_loader = DatasetLoader(args.data_path)
    print(f"Dataset loaded in {time.time() - step_start:.2f}s")
    
    # Show available inference results if using model_hash
    if args.model_hash:
        print("\nAvailable inference results:")
        dataset_loader.list_available_inference()
    
    # Determine visits to evaluate
    visits_to_eval = args.visits if args.visits else dataset_loader.visits
    print(f"\nEvaluating visits: {visits_to_eval}")
    
    # Check for inference results
    print("Checking inference results...")
    step_start = time.time()
    inference_results = collect_inference_results(
        dataset_loader, 
        weights_path=args.weights_path, 
        visits=visits_to_eval, 
        model_hash=args.model_hash
    )
    print(f"Inference results collection completed in {time.time() - step_start:.2f}s")
    
    if inference_results is None:
        if args.run_inference and args.weights_path:
            print("Running inference for missing visits...")
            step_start = time.time()
            # Run inference for all originally requested visits
            inference_results = dataset_loader.run_inference_all_visits(
                args.weights_path, force=False
            )
            print(f"Inference completed in {time.time() - step_start:.2f}s")
            
            # After running inference, collect results again
            if inference_results:
                print("Re-collecting inference results after running inference...")
                step_start = time.time()
                inference_results = collect_inference_results(
                    dataset_loader, 
                    weights_path=args.weights_path, 
                    visits=visits_to_eval, 
                    model_hash=args.model_hash
                )
                print(f"Re-collection completed in {time.time() - step_start:.2f}s")
        else:
            print("Evaluation stopped.")
            if args.weights_path and not args.run_inference:
                print("Use --run-inference to generate missing results.")
            elif not args.weights_path:
                print("Use --weights-path and --run-inference to generate them, or use --model-hash to load existing results.")
            return
    
    # Update visits_to_eval to only include visits with successful inference
    if inference_results is None:
        print("No inference results found, evaluation cannot proceed.")
        return
    actual_visits = list(inference_results.keys())
    if set(actual_visits) != set(visits_to_eval):
        print(f"\nNote: Evaluation will proceed with {len(actual_visits)} visits instead of {len(visits_to_eval)} originally requested")
        visits_to_eval = actual_visits
    
    # Aggregate results across visits
    print("Aggregating results...")
    step_start = time.time()
    predictions, labels, source_ids = aggregate_results(inference_results)
    print(f"Results aggregation completed in {time.time() - step_start:.2f}s")
    
    # Create metrics evaluation
    print("Computing evaluation metrics...")
    step_start = time.time()
    metrics = EvaluationMetrics(predictions, labels)
    print(f"Metrics computation completed in {time.time() - step_start:.2f}s")
    
    # Determine model name for dashboard title
    if args.weights_path:
        model_name = Path(args.weights_path).name
    else:
        model_name = f"Model Hash {args.model_hash}"
    
    # Create metrics dashboard
    print("Creating metrics dashboard...")
    step_start = time.time()
    metrics_dashboard = create_evaluation_dashboard(
        metrics, 
        output_path=output_dir / "metrics_dashboard.html",
        title=f"Model Evaluation - {model_name}"
    )
    print(f"Metrics dashboard created in {time.time() - step_start:.2f}s")
    
    # Print summary
    summary = metrics.summary()  # Single call (cached) used for printing + serialization
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
    
    # After dashboards, optionally release large arrays to free memory before interpretability
    # (Especially useful on constrained systems)
    del predictions
    del labels
    del source_ids
    import gc; gc.collect()
    
    # Run interpretability analysis if requested
    if args.interpretability:
        if not args.weights_path:
            print("Warning: Interpretability analysis requires --weights-path, skipping...")
        else:
            print("\nRunning interpretability analysis...")
            interp_start = time.time()
            
            try:
                # Create data loader for UMAP analysis
                print("Creating evaluation data loader...")
                step_start = time.time()
                eval_data_loader = create_evaluation_data_loader(
                    dataset_loader, visits_to_eval, 
                    max_samples=config.get('interpretability', {}).get('max_samples', 3000)
                )
                print(f"Data loader created in {time.time() - step_start:.2f}s")
                
                # Initialize UMAP interpreter
                print("Initializing UMAP interpreter...")
                step_start = time.time()
                interpreter = UMAPInterpreter(args.weights_path)
                print(f"Interpreter initialized in {time.time() - step_start:.2f}s")
                
                # Get additional features if specified in config
                additional_features = {}
                if 'features' in config.get('interpretability', {}):
                    # Add SNR or other features here
                    pass
                
                # Update config with UMAP optimization and clustering from command line
                interp_config = config.copy()
                if 'interpretability' not in interp_config:
                    interp_config['interpretability'] = {}
                if 'umap' not in interp_config['interpretability']:
                    interp_config['interpretability']['umap'] = {}
                if 'clustering' not in interp_config['interpretability']:
                    interp_config['interpretability']['clustering'] = {}
                
                # Override UMAP optimization from command line
                if args.optimize_umap:
                    interp_config['interpretability']['umap']['optimize_params'] = True
                
                # Override clustering enablement from command line
                if args.enable_clustering:
                    interp_config['interpretability']['clustering']['enabled'] = True

                # Create interpretability dashboard with configuration
                print("Creating interpretability dashboard...")
                step_start = time.time()
                interp_dashboard = create_interpretability_dashboard(
                    interpreter,
                    eval_data_loader,
                    predictions[:len(eval_data_loader.dataset)],  # Match lengths
                    labels[:len(eval_data_loader.dataset)],
                    output_path=output_dir / "interpretability_dashboard.html",
                    additional_features=additional_features,
                    config=interp_config
                )
                print(f"Interpretability dashboard created in {time.time() - step_start:.2f}s")
                print(f"Total interpretability analysis time: {time.time() - interp_start:.2f}s")
                
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
    print("Saving evaluation results...")
    step_start = time.time()
    results_file = output_dir / "evaluation_results.yaml"
    eval_data = {
        'visits_evaluated': visits_to_eval,
        'total_samples': metrics.get_confusion_matrix_stats()['total_samples'],
        'metrics': summary,
        'confusion_matrix': metrics.confusion_mat.tolist()
    }
    
    if args.weights_path:
        eval_data['model_path'] = args.weights_path
    if args.model_hash:
        eval_data['model_hash'] = args.model_hash
    
    with open(results_file, 'w') as f:
        yaml.dump(eval_data, f)
    print(f"Results saved in {time.time() - step_start:.2f}s")
    
    # Final timing summary
    total_time = time.time() - start_time
    print(f"\nEvaluation complete!")
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Results saved to: {output_dir}")
    print(f"Open metrics dashboard: file://{output_dir / 'metrics_dashboard.html'}")
    if args.interpretability and args.weights_path:
        print(f"Open interpretability dashboard: file://{output_dir / 'interpretability_dashboard.html'}")

if __name__ == "__main__":
    main()