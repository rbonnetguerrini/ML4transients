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
from ML4transients.evaluation.visualizations import create_evaluation_dashboard, create_interpretability_dashboard, create_combined_dashboard
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
        (predictions, labels, source_ids, probabilities, uncertainties)
    """
    all_predictions = []
    all_labels = []
    all_ids = []
    all_probabilities = []
    all_uncertainties = []
    
    has_probabilities = False
    has_uncertainties = False
    
    for i, (visit, loader) in enumerate(inference_results.items()):
        print(f"Loading visit {visit} results ({i+1}/{len(inference_results)})...")
        all_predictions.append(loader.predictions)
        all_labels.append(loader.labels)
        all_ids.append(loader.ids)
        
        # Check if probabilistic data is available
        if hasattr(loader, 'probabilities') and loader.probabilities is not None:
            all_probabilities.append(loader.probabilities)
            has_probabilities = True
        if hasattr(loader, 'uncertainties') and loader.uncertainties is not None:
            all_uncertainties.append(loader.uncertainties)
            has_uncertainties = True
    
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    source_ids = np.concatenate(all_ids)
    
    probabilities = None
    uncertainties = None
    if has_probabilities:
        probabilities = np.concatenate(all_probabilities)
        print(f"Loaded prediction probabilities (mean: {probabilities.mean():.4f})")
    if has_uncertainties:
        uncertainties = np.concatenate(all_uncertainties)
        print(f"Loaded prediction uncertainties (mean: {uncertainties.mean():.4f})")
    
    # Clear individual loader data to save memory
    del all_predictions, all_labels, all_ids, all_probabilities, all_uncertainties
    import gc
    gc.collect()
    
    return (predictions, labels, source_ids, probabilities, uncertainties)

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
        batch_size=128,  # Increase batch size for better GPU utilization
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing overhead for evaluation
        pin_memory=False,  # Save memory
        drop_last=False,
        prefetch_factor=2 if hasattr(DataLoader, 'prefetch_factor') else None  # Prefetch next batch
    )

def main():
    """Orchestrate full evaluation with ensemble model support."""
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
    predictions, labels, source_ids, probabilities, uncertainties = aggregate_results(inference_results)
    print(f"Results aggregation completed in {time.time() - step_start:.2f}s")
    
    # Create metrics evaluation with probabilities if available
    print("Computing evaluation metrics...")
    step_start = time.time()
    metrics = EvaluationMetrics(predictions, labels, probabilities)
    print(f"Metrics computation completed in {time.time() - step_start:.2f}s")
    
    # Print enhanced summary
    summary = metrics.summary()
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    
    # Determine model name for dashboard title
    if args.weights_path:
        model_name = Path(args.weights_path).name
        # Add model type information
        try:
            config = load_config(Path(args.weights_path) / "config.yaml")
            trainer_type = config["training"]["trainer_type"]
            if trainer_type == "ensemble":
                num_models = config["training"]["num_models"]
                model_name += f" (Ensemble-{num_models})"
            elif trainer_type == "coteaching":
                model_name += " (Co-Teaching)"
            else:
                model_name += " (Standard)"
        except:
            pass
    else:
        model_name = f"Model Hash {args.model_hash}"
    
    # Store data for interpretability
    interp_predictions = None
    interp_labels = None
    interpreter = None
    eval_data_loader = None
    
    if args.interpretability and args.weights_path:
        interp_predictions = predictions.copy()
        interp_labels = labels.copy()
        
        # Initialize interpretability components
        try:
            print("Creating evaluation data loader...")
            step_start = time.time()
            eval_data_loader = create_evaluation_data_loader(
                dataset_loader, visits_to_eval, 
                max_samples=config.get('interpretability', {}).get('max_samples', 3000)
            )
            print(f"Data loader created in {time.time() - step_start:.2f}s")
            
            print("Initializing UMAP interpreter...")
            step_start = time.time()
            interpreter = UMAPInterpreter(args.weights_path)
            print(f"Interpreter initialized in {time.time() - step_start:.2f}s")
            
            # Update config with command line overrides
            interp_config = config.copy()
            if 'interpretability' not in interp_config:
                interp_config['interpretability'] = {}
            if 'umap' not in interp_config['interpretability']:
                interp_config['interpretability']['umap'] = {}
            if 'clustering' not in interp_config['interpretability']:
                interp_config['interpretability']['clustering'] = {}
            
            if args.optimize_umap:
                interp_config['interpretability']['umap']['optimize_params'] = True
            if args.enable_clustering:
                interp_config['interpretability']['clustering']['enabled'] = True
                
        except Exception as e:
            print(f"Error initializing interpretability components: {e}")
            print("Will create metrics-only dashboard...")
            interpreter = None
            eval_data_loader = None
    
    # Create combined dashboard
    print("Creating combined dashboard...")
    step_start = time.time()
    
    # Get additional features if specified in config
    additional_features = {}
    if args.interpretability and 'features' in config.get('interpretability', {}):
        # Add SNR or other features here if available
        pass
    
    dashboard_path = create_combined_dashboard(
        metrics=metrics,
        interpreter=interpreter,
        data_loader=eval_data_loader,
        predictions=interp_predictions,
        labels=interp_labels,
        output_path=output_dir / "evaluation_dashboard.html",
        additional_features=additional_features if args.interpretability else None,
        config=interp_config if args.interpretability and interpreter else None,
        title=f"Model Evaluation - {model_name}",
        probabilities=probabilities,  # Pass pre-computed probabilities
        uncertainties=uncertainties   # Pass pre-computed uncertainties
    )
    print(f"Combined dashboard created in {time.time() - step_start:.2f}s")
    
    # After dashboards, release large arrays
    del predictions, labels, source_ids
    if probabilities is not None:
        del probabilities
    if uncertainties is not None:
        del uncertainties
    if interp_predictions is not None:
        del interp_predictions
    if interp_labels is not None:
        del interp_labels
    
    # Cleanup memory after dashboard creation
    if eval_data_loader is not None:
        del eval_data_loader
    if interpreter is not None:
        del interpreter
    import gc; gc.collect()
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
    print(f"Open dashboard: file://{output_dir / 'evaluation_dashboard.html'}")
    if args.interpretability and args.weights_path:
        print("Dashboard includes both metrics and interpretability tabs")
    else:
        print("Dashboard includes metrics tab only")

if __name__ == "__main__":
    main()