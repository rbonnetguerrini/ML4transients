#!/usr/bin/env python3
"""
Script to run comprehensive model evaluation including metrics and interpretability analysis.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import argparse
import yaml
from pathlib import Path
from typing import List, Optional
import sys
import torch
import numpy as np
import gc
import time
from torch.utils.data import DataLoader

sys.path.append('/sps/lsst/users/rbonnetguerrini/ML4transients/src')

from ML4transients.data_access import DatasetLoader
from ML4transients.evaluation.metrics import EvaluationMetrics, load_inference_metrics
from ML4transients.evaluation.visualizations import create_combined_dashboard
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
    parser.add_argument("--snr-threshold", type=float, default=5.0,
                       help="SNR threshold to separate low and high SNR samples (default: 5.0)")
    parser.add_argument("--object-ids-file", type=str, default=None,
                       help="Path to file containing diaObjectIds (one per line)")
    parser.add_argument("--umap-load-path", type=str, default=None,
                       help="Path to load an existing UMAP fit for interpretability")
    parser.add_argument("--umap-save-path", type=str, default=None,
                       help="Path to save the fitted UMAP after interpretability analysis")
    parser.add_argument("--show-all-cutouts", action="store_true",
                       help="Show all cutout types (diff, coadd, etc.) in UMAP hover tooltips (default: show only first channel)")
    
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

def load_object_ids_from_file(file_path: str) -> List[str]:
    """Load diaObjectIds from a text file (one per line).
    
    Parameters
    ----------
    file_path : str
        Path to the file containing diaObjectIds
        
    Returns
    -------
    List[str]
        List of diaObjectIds as strings to preserve precision
    """
    object_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                if line.isdigit():  # Simple validation for numeric string
                    object_ids.append(line)
                else:
                    print(f"Warning: Skipping invalid object ID: {line}")
    return object_ids

def filter_inference_results_by_object_ids(dataset_loader: DatasetLoader, 
                                         inference_results: dict, 
                                         object_ids: List[str]) -> dict:
    """Filter inference results to only include sources from specified diaObjectIds.
    
    Parameters
    ----------
    dataset_loader : DatasetLoader
        Dataset loader for accessing lightcurve data
    inference_results : dict
        Dictionary mapping visit -> InferenceLoader
    object_ids : List[str]
        List of diaObjectIds to filter by (as strings to preserve precision)
        
    Returns
    -------
    dict
        Filtered inference results containing only sources from specified objects
    """
    if not object_ids:
        return inference_results
    
    print(f"Filtering inference results to {len(object_ids)} specified diaObjectIds...")
    
    print("Step 1: Collecting all diaSourceIds for specified diaObjectIds...")
    target_source_ids = set()
    
    # Create lightcurve loader 
    lightcurve_path = None
    for data_path in dataset_loader.data_paths:
        potential_lc_path = data_path / "lightcurves"
        if potential_lc_path.exists():
            lightcurve_path = potential_lc_path
            break
    
    if lightcurve_path is None:
        print("Error: No lightcurves directory found in dataset paths")
        return {}
    
    print(f"Loading lightcurve data directly from: {lightcurve_path}")
    from ML4transients.data_access.data_loaders import LightCurveLoader
    lc_loader = LightCurveLoader(lightcurve_path)
    
    print(f"Lightcurve loader loaded with {len(lc_loader.index)} objects")
    available_objects = set(lc_loader.index.index.astype(str))
    
    # Use the new function to map diaObjectId -> diaSourceIds
    print("Using get_source_ids_for_objects to map diaObjectId -> diaSourceIds...")
    
    # Convert string object IDs to integers for the function
    object_ids_int = []
    for obj_id_str in object_ids:
        obj_id_int = int(obj_id_str)
        object_ids_int.append(obj_id_int)

    if not object_ids_int:
        print("ERROR: No valid integer object IDs found")
        return {}
    
    # Map objects to their source IDs
    object_to_sources = lc_loader.get_source_ids_for_objects(object_ids_int)
    
    # Collect all diaSourceIds
    for obj_id, source_ids in object_to_sources.items():
        target_source_ids.update(source_ids)
        
    found_objects = len(object_to_sources)
    total_sources = len(target_source_ids)
    
    print(f"  Mapped {found_objects}/{len(object_ids)} objects to {total_sources} diaSourceIds")
    
    # Filter each inference loader to only include target source IDs
    print("Step 2: Filtering inference results...")
    filtered_results = {}
    
    for visit, loader in inference_results.items():
        # Get source ID from this visit's inference results
        visit_source_ids = loader.ids
        
        # Find intersection with target source IDs
        filtered_source_ids = [sid for sid in visit_source_ids if sid in target_source_ids]
        
        if filtered_source_ids:
            print(f"  Visit {visit}: {len(filtered_source_ids)}/{len(visit_source_ids)} sources match")
            
            # Create a filtered version of the inference loader
            filtered_loader = FilteredInferenceLoader(loader, filtered_source_ids, target_source_ids)
            filtered_results[visit] = filtered_loader
        else:
            print(f"  Visit {visit}: No matching sources found")
    
    print(f"Filtering complete: {len(filtered_results)}/{len(inference_results)} visits have matching sources")
    return filtered_results

class FilteredInferenceLoader:
    """Wrapper around InferenceLoader that filters results to specific source IDs."""
    
    def __init__(self, original_loader, filtered_source_ids: List[int], target_source_ids: set):
        self.original_loader = original_loader
        self.filtered_source_ids = filtered_source_ids
        self.target_source_ids = target_source_ids
        
        # Create index mapping for filtering arrays
        all_ids = original_loader.ids
        self.filter_indices = [i for i, sid in enumerate(all_ids) if sid in target_source_ids]
        
        # Cache filtered data
        self._filtered_predictions = None
        self._filtered_labels = None
        self._filtered_probabilities = None
        self._filtered_uncertainties = None
        self._filtered_ids = None
    
    @property
    def predictions(self):
        if self._filtered_predictions is None:
            original = self.original_loader.predictions
            self._filtered_predictions = original[self.filter_indices] if original is not None else None
        return self._filtered_predictions
    
    @property
    def labels(self):
        if self._filtered_labels is None:
            original = self.original_loader.labels
            self._filtered_labels = original[self.filter_indices] if original is not None else None
        return self._filtered_labels
    
    @property
    def probabilities(self):
        if self._filtered_probabilities is None:
            original = self.original_loader.probabilities
            self._filtered_probabilities = original[self.filter_indices] if original is not None else None
        return self._filtered_probabilities
    
    @property
    def uncertainties(self):
        if self._filtered_uncertainties is None:
            original = self.original_loader.uncertainties
            self._filtered_uncertainties = original[self.filter_indices] if original is not None else None
        return self._filtered_uncertainties
    
    @property
    def ids(self):
        if self._filtered_ids is None:
            original = self.original_loader.ids
            self._filtered_ids = original[self.filter_indices] if original is not None else None
        return self._filtered_ids

def collect_inference_results(dataset_loader: DatasetLoader, weights_path: str = None,
                            visits: list = None, model_hash: str = None, 
                            auto_run_inference: bool = False) -> dict:
    """Collect (or discover) inference loaders for requested visits.

    Parameters
    ----------
    auto_run_inference : bool
        If True, automatically return None to trigger inference instead of prompting user
    
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
            # Automatic mode for batch/SLURM jobs
            if auto_run_inference:
                if weights_path:
                    print(f"Auto mode: Will run inference for {len(missing_visits)} missing visits")
                    return None  # Signal to run inference
                else:
                    print(f"Auto mode: Continuing with {len(results)} available visits (no weights to run inference)")
                    return results
            
            # Check if we're in a non-interactive environment (no TTY)
            import sys
            if not sys.stdin.isatty():
                print(f"Non-interactive mode detected: Continuing with {len(results)} available visits")
                return results
            
            # Interactive mode - give user choice to continue with partial results
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

def extract_snr_values(dataset_loader: DatasetLoader, source_ids: np.ndarray) -> np.ndarray:
    """Extract SNR values for given source IDs.
    
    Parameters
    ----------
    dataset_loader : DatasetLoader
        Dataset loader instance
    source_ids : np.ndarray
        Array of diaSourceId values
        
    Returns
    -------
    np.ndarray
        SNR values for each source (calculated as abs(flux) / flux_error)
    """
    print("Extracting SNR values for evaluation...")
    start_time = time.time()
    
    # Get feature data for all source IDs
    try:
        # Group source IDs by visit for efficient loading
        visit_to_source_ids = {}
        for source_id in source_ids:
            visit = dataset_loader.find_visit(int(source_id))
            if visit:
                if visit not in visit_to_source_ids:
                    visit_to_source_ids[visit] = []
                visit_to_source_ids[visit].append(int(source_id))
        
        print(f"Found {len(visit_to_source_ids)} visits for {len(source_ids)} source IDs")
        
        snr_values = []
        snr_dict = {}  # To map source_id to SNR value
        
        # Load features by visit
        for visit, visit_source_ids in visit_to_source_ids.items():
            if visit in dataset_loader.features:
                feature_loader = dataset_loader.features[visit]
                feature_data = feature_loader.get_multiple_by_ids(visit_source_ids)
                
                for source_id in visit_source_ids:
                    if source_id in feature_data:
                        df = feature_data[source_id]
                        # Calculate SNR from psfFlux and psfFluxErr
                        if 'psfFlux' in df.columns and 'psfFluxErr' in df.columns:
                            flux = df['psfFlux'].iloc[0]  # Take first measurement
                            flux_err = df['psfFluxErr'].iloc[0]
                            if flux_err > 0:
                                snr = abs(flux) / flux_err
                            else:
                                snr = 0.0
                        else:
                            snr = 0.0
                    else:
                        snr = 0.0
                    snr_dict[source_id] = snr
            else:
                # No feature data for this visit
                for source_id in visit_source_ids:
                    snr_dict[source_id] = 0.0
        
        # Create SNR array in the same order as source_ids
        snr_values = [snr_dict.get(int(source_id), 0.0) for source_id in source_ids]
        snr_array = np.array(snr_values)
        
        print(f"SNR extraction completed in {time.time() - start_time:.2f}s")
        print(f"SNR statistics: mean={snr_array.mean():.2f}, std={snr_array.std():.2f}, min={snr_array.min():.2f}, max={snr_array.max():.2f}")
        return snr_array
        
    except Exception as e:
        print(f"Warning: Could not extract SNR values: {e}")
        print("SNR-based metrics will not be available")
        return None

def create_evaluation_data_loader(dataset_loader: DatasetLoader, visits: list,
                                max_samples: int = 3000, 
                                object_ids: Optional[List[str]] = None,
                                weights_path: Optional[str] = None) -> DataLoader:
    """Return DataLoader tailored for feature extraction / interpretability.
    
    Parameters
    ----------
    dataset_loader : DatasetLoader
        Dataset loader instance
    visits : list
        List of visits to include
    max_samples : int
        Maximum number of samples for UMAP
    object_ids : Optional[List[str]]
        If provided, only include sources from these diaObjectIds
    weights_path : Optional[str]
        Path to model weights (to extract cutout_types from config)
    """
    print(f"Creating evaluation dataset from {len(visits)} visits...")
    
    # Extract cutout_types from model config if weights_path provided
    cutout_types = ['diff']  # Default to single channel
    if weights_path:
        try:
            config = load_config(Path(weights_path) / "config.yaml")
            cutout_types = config.get('data', {}).get('cutout_types', ['diff'])
            print(f"Using {len(cutout_types)} channel(s) from model config: {cutout_types}")
        except Exception as e:
            print(f"Warning: Could not load cutout_types from config: {e}")
            print("Defaulting to single channel ['diff']")
    
    # Create inference dataset for the specified visits
    eval_dataset = PytorchDataset.create_inference_dataset(
        dataset_loader, 
        visits=visits,
        cutout_types=cutout_types
    )
    
    # Filter dataset by object IDs if specified
    if object_ids:
        print(f"Filtering dataset to {len(object_ids)} specified diaObjectIds...")
        eval_dataset = filter_pytorch_dataset_by_object_ids(
            eval_dataset, dataset_loader, object_ids
        )
        print(f"Filtered dataset contains {len(eval_dataset)} samples")
    
    # Use optimized DataLoader settings for evaluation
    return DataLoader(
        eval_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=2 if hasattr(DataLoader, 'prefetch_factor') else None
    )

def filter_pytorch_dataset_by_object_ids(dataset, dataset_loader: DatasetLoader, 
                                        object_ids: List[str]):
    """Filter a PyTorch dataset to only include sources from specified diaObjectIds.
    
    Parameters
    ----------
    dataset : PytorchDataset
        Original dataset
    dataset_loader : DatasetLoader
        Dataset loader for accessing lightcurve data
    object_ids : List[str]
        List of diaObjectIds to filter by
        
    Returns
    -------
    FilteredPytorchDataset
        Filtered dataset containing only specified objects
    """
    # Get lightcurve loader
    lightcurve_path = None
    for data_path in dataset_loader.data_paths:
        potential_lc_path = data_path / "lightcurves"
        if potential_lc_path.exists():
            lightcurve_path = potential_lc_path
            break
    
    if lightcurve_path is None:
        print("Error: No lightcurves directory found")
        return dataset
    
    from ML4transients.data_access.data_loaders import LightCurveLoader
    lc_loader = LightCurveLoader(lightcurve_path)
    
    # Convert object IDs to integers and get source IDs
    object_ids_int = [int(obj_id) for obj_id in object_ids]
    object_to_sources = lc_loader.get_source_ids_for_objects(object_ids_int)
    
    # Collect all target source IDs
    target_source_ids = set()
    for source_ids in object_to_sources.values():
        target_source_ids.update(source_ids)
    
    print(f"Found {len(target_source_ids)} diaSourceIds for {len(object_ids)} diaObjectIds")
    
    # Access the underlying dataset if this is already a Subset
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    
    # Find indices in dataset that correspond to target source IDs
    filtered_indices = []
    
    # Check if the dataset has _sample_index (lazy loading) or dia_source_ids attribute
    if hasattr(base_dataset, '_sample_index'):
        # Lazy loading dataset - check source IDs in sample index
        for idx in range(len(dataset)):
            # Get actual index if this is a Subset
            actual_idx = dataset.indices[idx] if hasattr(dataset, 'indices') else idx
            visit, source_id = base_dataset._sample_index[actual_idx]
            if source_id in target_source_ids:
                filtered_indices.append(idx)
    elif hasattr(base_dataset, 'dia_source_ids'):
        # Pre-loaded dataset - use dia_source_ids array
        for idx in range(len(dataset)):
            actual_idx = dataset.indices[idx] if hasattr(dataset, 'indices') else idx
            source_id = base_dataset.dia_source_ids[actual_idx]
            if source_id in target_source_ids:
                filtered_indices.append(idx)
    else:
        print("Warning: Cannot determine dataset structure, trying direct access...")
        # Fallback: try accessing samples directly
        for idx in range(len(dataset)):
            try:
                # This will trigger __getitem__ which should work
                _, _, meta_idx = dataset[idx]
                # meta_idx should be the index, we can use it to get source_id
                if hasattr(base_dataset, '_sample_index'):
                    actual_idx = dataset.indices[idx] if hasattr(dataset, 'indices') else idx
                    _, source_id = base_dataset._sample_index[actual_idx]
                elif hasattr(base_dataset, 'dia_source_ids'):
                    actual_idx = dataset.indices[idx] if hasattr(dataset, 'indices') else idx
                    source_id = base_dataset.dia_source_ids[actual_idx]
                else:
                    continue
                    
                if source_id in target_source_ids:
                    filtered_indices.append(idx)
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                continue
    
    print(f"Filtered dataset from {len(dataset)} to {len(filtered_indices)} samples")
    
    if len(filtered_indices) == 0:
        print("ERROR: No matching samples found! Check that object IDs exist in the dataset.")
        return dataset  # Return original to avoid complete failure
    
    # Create a custom filtered dataset that preserves source_ids access
    return FilteredPytorchDataset(dataset, filtered_indices, target_source_ids, base_dataset)


class FilteredPytorchDataset:
    """Wrapper that filters a PyTorch dataset while preserving metadata access."""
    
    def __init__(self, dataset, filtered_indices, target_source_ids, base_dataset):
        self.dataset = dataset
        self.filtered_indices = filtered_indices
        self.target_source_ids = target_source_ids
        self.base_dataset = base_dataset
        
        # Create mapping of filtered index to original index
        self.index_mapping = {i: filtered_indices[i] for i in range(len(filtered_indices))}
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        return self.dataset[original_idx]
    
    @property
    def source_ids(self):
        """Expose source_ids for efficient filtering."""
        # Build source_ids array on demand
        source_ids = []
        for idx in self.filtered_indices:
            # Get actual index in base dataset
            actual_idx = self.dataset.indices[idx] if hasattr(self.dataset, 'indices') else idx
            
            if hasattr(self.base_dataset, '_sample_index'):
                _, source_id = self.base_dataset._sample_index[actual_idx]
            elif hasattr(self.base_dataset, 'dia_source_ids'):
                source_id = self.base_dataset.dia_source_ids[actual_idx]
            else:
                source_id = -1
                
            source_ids.append(source_id)
        
        return np.array(source_ids)

def main():
    """Orchestrate full evaluation process."""
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
    print("Loading configuration...", flush=True)
    step_start = time.time()
    config = load_evaluation_config(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print detailed configuration
    print(f"\nConfiguration loaded in {time.time() - step_start:.2f}s", flush=True)
    print("\n" + "="*70, flush=True)
    print("EVALUATION CONFIGURATION", flush=True)
    print("="*70, flush=True)
    print(f"Config file: {args.config}", flush=True)
    print(f"Data path: {args.data_path}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    
    if args.weights_path:
        print(f"Model weights: {args.weights_path}", flush=True)
    if args.model_hash:
        print(f"Model hash: {args.model_hash}", flush=True)
    
    if args.visits:
        print(f"Specific visits: {args.visits}", flush=True)
    else:
        print("Visits: All available", flush=True)
    
    print(f"Run inference: {args.run_inference}", flush=True)
    print(f"Interpretability: {args.interpretability}", flush=True)
    
    if args.interpretability:
        print(f"  - Optimize UMAP: {args.optimize_umap}", flush=True)
        print(f"  - Enable clustering: {args.enable_clustering}", flush=True)
        print(f"  - Show all cutouts: {args.show_all_cutouts}", flush=True)
        if args.umap_load_path:
            print(f"  - UMAP load path: {args.umap_load_path}", flush=True)
        if args.umap_save_path:
            print(f"  - UMAP save path: {args.umap_save_path}", flush=True)
    
    print(f"SNR threshold: {args.snr_threshold}", flush=True)
    
    if args.object_ids_file:
        print(f"Object IDs file: {args.object_ids_file}", flush=True)
    
    # Print config file contents if available
    if config:
        print("\nConfig file settings:", flush=True)
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"  {key}:", flush=True)
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}", flush=True)
            else:
                print(f"  {key}: {value}", flush=True)
    
    print("="*70 + "\n", flush=True)
    
    print("Loading dataset...")
    step_start = time.time()
    dataset_loader = DatasetLoader(args.data_path)
    print(f"Dataset loaded in {time.time() - step_start:.2f}s")
    
    # Process object IDs filtering if specified
    object_ids = None

    if args.object_ids_file:  # only if user provided the path
        object_ids = load_object_ids_from_file(args.object_ids_file)
        print(f"Loaded {len(object_ids)} object IDs from file: {args.object_ids_file}")

    
    if object_ids:
        print(f"Evaluation will be filtered to {len(object_ids)} specified diaObjectIds")
        if args.run_inference:
            print("\n" + "="*70)
            print("WARNING: --run-inference with --object-ids-file is inefficient!")
            print("Inference will run on ALL samples first, then filter to your objects.")
            print("For large datasets, it's better to:")
            print("  1. Run inference once WITHOUT --object-ids-file (inference cached)")
            print("  2. Then run evaluation WITH --object-ids-file (no --run-inference)")
            print("="*70 + "\n")
    else:
        print("Evaluation will include all available data")
    
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
        model_hash=args.model_hash,
        auto_run_inference=args.run_inference  # Pass the flag for non-interactive mode
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
                    model_hash=args.model_hash,
                    auto_run_inference=False  # No need to prompt again after running inference
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
    
    # Apply object ID filtering if specified
    if object_ids:
        print(f"Applying diaObjectId filtering to {len(object_ids)} objects...")
        step_start = time.time()
        inference_results = filter_inference_results_by_object_ids(
            dataset_loader, inference_results, object_ids
        )
        print(f"Object ID filtering completed in {time.time() - step_start:.2f}s")
        
        if not inference_results:
            print("Error: No inference results remain after object ID filtering")
            return
        
        # Update visits list after filtering
        actual_visits = list(inference_results.keys())
        print(f"After filtering: {len(actual_visits)} visits contain matching sources")
    
    # Aggregate results across visits
    print("Aggregating results...")
    step_start = time.time()
    predictions, labels, source_ids, probabilities, uncertainties = aggregate_results(inference_results)
    print(f"Results aggregation completed in {time.time() - step_start:.2f}s")
    
    # Extract SNR values for SNR-based metrics
    print("Extracting SNR values...")
    step_start = time.time()
    snr_values = extract_snr_values(dataset_loader, source_ids)
    print(f"SNR extraction completed in {time.time() - step_start:.2f}s")
    
    # Create metrics evaluation with probabilities, SNR, and uncertainties if available
    print("Computing evaluation metrics...")
    step_start = time.time()
    metrics = EvaluationMetrics(predictions, labels, probabilities, snr_values, uncertainties)
    print(f"Metrics computation completed in {time.time() - step_start:.2f}s")
    
    # Print enhanced summary
    summary = metrics.summary()
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    
    # Print SNR-based summary if available
    if snr_values is not None:
        try:
            snr_summary = metrics.get_snr_based_metrics(args.snr_threshold)
            print("\n" + "="*50)
            print("SNR-BASED PERFORMANCE BREAKDOWN")
            print("="*50)
            
            low_snr = snr_summary['low_snr']
            high_snr = snr_summary['high_snr']
            
            print(f"Low SNR (|SNR| < {args.snr_threshold}): {int(low_snr['n_samples'])} samples")
            if low_snr['n_samples'] > 0:
                print(f"  Accuracy: {low_snr['accuracy']:.4f}")
                print(f"  Precision: {low_snr['precision']:.4f}")
                print(f"  Recall: {low_snr['recall']:.4f}")
                print(f"  F1-Score: {low_snr['f1_score']:.4f}")
            
            print(f"High SNR (|SNR| ≥ {args.snr_threshold}): {int(high_snr['n_samples'])} samples")
            if high_snr['n_samples'] > 0:
                print(f"  Accuracy: {high_snr['accuracy']:.4f}")
                print(f"  Precision: {high_snr['precision']:.4f}")
                print(f"  Recall: {high_snr['recall']:.4f}")
                print(f"  F1-Score: {high_snr['f1_score']:.4f}")
                
        except Exception as e:
            print(f"Warning: Could not compute SNR-based metrics: {e}")
    
    print("="*50)
    
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
            print("Creating evaluation data loader (with object ID filtering if specified)...")
            step_start = time.time()
            eval_data_loader = create_evaluation_data_loader(
                dataset_loader, visits_to_eval, 
                max_samples=config.get('interpretability', {}).get('max_samples', 3000),
                object_ids=object_ids,  # Pass object_ids to filter the data loader
                weights_path=args.weights_path  # Pass weights_path to extract cutout_types
            )
            print(f"Data loader created in {time.time() - step_start:.2f}s")
            
            # Verify the data loader has samples
            if len(eval_data_loader.dataset) == 0:
                print("ERROR: Filtered data loader is empty! Skipping interpretability analysis.")
                interpreter = None
                eval_data_loader = None
            else:
                print(f"Data loader contains {len(eval_data_loader.dataset)} samples")
                
                print("Initializing UMAP interpreter...")
                step_start = time.time()
                interpreter = UMAPInterpreter(args.weights_path, load_path=args.umap_load_path, save_path=args.umap_save_path)
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
                # Add UMAP load/save paths to config
                if args.umap_load_path:
                    interp_config['interpretability']['umap']['load_path'] = args.umap_load_path
                if args.umap_save_path:
                    interp_config['interpretability']['umap']['save_path'] = args.umap_save_path
                
        except Exception as e:
            print(f"Error initializing interpretability components: {e}")
            import traceback
            traceback.print_exc()
            print("Will create metrics-only dashboard...")
            interpreter = None
            eval_data_loader = None
    
    # Create combined dashboard
    print("Creating combined dashboard...")
    step_start = time.time()
    
    # Get additional features if specified in config
    additional_features = {}
    if args.interpretability:
        # Add SNR values if available
        if snr_values is not None:
            additional_features['snr'] = snr_values
        
        # Add other features if specified in config
        if 'features' in config.get('interpretability', {}):
            # Add additional features here if available
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
        uncertainties=uncertainties,   # Pass pre-computed uncertainties
        snr_threshold=args.snr_threshold,  # Pass SNR threshold
        show_all_cutouts=args.show_all_cutouts  # Pass cutout display preference
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
    
    # Get comprehensive confusion matrix statistics
    cm_stats = metrics.get_confusion_matrix_stats()
    
    # Build comprehensive evaluation data dictionary
    eval_data = {
        'visits_evaluated': visits_to_eval,
        'total_samples': cm_stats['total_samples'],
        
        # Basic metrics (accuracy, precision, recall, F1, specificity)
        'metrics': summary,
        
        # Confusion matrix 
        'confusion_matrix': {
            'matrix': metrics.confusion_mat.tolist(),
            'true_positive': int(cm_stats['true_positive']),
            'true_negative': int(cm_stats['true_negative']),
            'false_positive': int(cm_stats['false_positive']),
            'false_negative': int(cm_stats['false_negative']),
            'true_positive_rate': float(cm_stats['true_positive_rate']),
            'true_negative_rate': float(cm_stats['true_negative_rate']),
            'false_positive_rate': float(cm_stats['false_positive_rate']),
            'false_negative_rate': float(cm_stats['false_negative_rate']),
            'positive_predictive_value': float(cm_stats['positive_predictive_value']),
            'negative_predictive_value': float(cm_stats['negative_predictive_value'])
        }
    }
    
    # Add SNR-based metrics if available (use metrics.snr_values since local var was deleted)
    if metrics.snr_values is not None:
        try:
            snr_metrics = metrics.get_snr_based_metrics(snr_threshold=args.snr_threshold)
            eval_data['snr_based_metrics'] = {
                'threshold': args.snr_threshold,
                'low_snr': {
                    'n_samples': int(snr_metrics['low_snr']['n_samples']),
                    'accuracy': float(snr_metrics['low_snr']['accuracy']),
                    'precision': float(snr_metrics['low_snr']['precision']),
                    'recall': float(snr_metrics['low_snr']['recall']),
                    'f1_score': float(snr_metrics['low_snr']['f1_score']),
                    'specificity': float(snr_metrics['low_snr']['specificity'])
                },
                'high_snr': {
                    'n_samples': int(snr_metrics['high_snr']['n_samples']),
                    'accuracy': float(snr_metrics['high_snr']['accuracy']),
                    'precision': float(snr_metrics['high_snr']['precision']),
                    'recall': float(snr_metrics['high_snr']['recall']),
                    'f1_score': float(snr_metrics['high_snr']['f1_score']),
                    'specificity': float(snr_metrics['high_snr']['specificity'])
                }
            }
            
            # Add ROC AUC and PR AUC for SNR-based metrics if probabilities available
            if 'roc_auc' in snr_metrics['low_snr']:
                eval_data['snr_based_metrics']['low_snr']['roc_auc'] = float(snr_metrics['low_snr']['roc_auc'])
                eval_data['snr_based_metrics']['low_snr']['pr_auc'] = float(snr_metrics['low_snr']['pr_auc'])
            if 'roc_auc' in snr_metrics['high_snr']:
                eval_data['snr_based_metrics']['high_snr']['roc_auc'] = float(snr_metrics['high_snr']['roc_auc'])
                eval_data['snr_based_metrics']['high_snr']['pr_auc'] = float(snr_metrics['high_snr']['pr_auc'])
        except Exception as e:
            print(f"Warning: Could not compute SNR-based metrics: {e}")
    
    # Add uncertainty quantification metrics if available (use metrics.uncertainties since local var was deleted)
    if metrics.uncertainties is not None:
        try:
            uq_summary = metrics.get_extended_uq_summary(snr_threshold=args.snr_threshold)
            eval_data['uncertainty_quantification'] = {
                'overall': {
                    'mean': float(uq_summary['overall']['mean']),
                    'std': float(uq_summary['overall']['std']),
                    'median': float(uq_summary['overall']['median'])
                },
                'by_correctness': {},
                'by_confusion_category': {}
            }
            
            # Add correctness-based UQ
            if 'correct' in uq_summary['by_correctness']:
                eval_data['uncertainty_quantification']['by_correctness']['correct'] = {
                    'mean': float(uq_summary['by_correctness']['correct']['mean']),
                    'std': float(uq_summary['by_correctness']['correct']['std']),
                    'median': float(uq_summary['by_correctness']['correct']['median'])
                }
            if 'incorrect' in uq_summary['by_correctness']:
                eval_data['uncertainty_quantification']['by_correctness']['incorrect'] = {
                    'mean': float(uq_summary['by_correctness']['incorrect']['mean']),
                    'std': float(uq_summary['by_correctness']['incorrect']['std']),
                    'median': float(uq_summary['by_correctness']['incorrect']['median'])
                }
            
            # Add confusion category UQ
            for category in ['TP', 'TN', 'FP', 'FN']:
                if category in uq_summary['by_confusion_category']:
                    eval_data['uncertainty_quantification']['by_confusion_category'][category] = {
                        'mean': float(uq_summary['by_confusion_category'][category]['mean']),
                        'std': float(uq_summary['by_confusion_category'][category]['std']),
                        'median': float(uq_summary['by_confusion_category'][category]['median'])
                    }
            
            # Add SNR-UQ correlation if available
            if 'snr_uq_correlation' in uq_summary:
                correlation_data = uq_summary['snr_uq_correlation']
                eval_data['uncertainty_quantification']['snr_correlation'] = {
                    'correlation': float(correlation_data['correlation']),
                    'p_value': float(correlation_data['p_value']),
                    'n_samples': int(correlation_data['n_samples'])
                }
                
                # Add per-category correlations
                if 'snr_uq_correlation_by_category' in uq_summary:
                    eval_data['uncertainty_quantification']['snr_correlation_by_category'] = {}
                    for category, corr_data in uq_summary['snr_uq_correlation_by_category'].items():
                        if corr_data['n_samples'] >= 3:  # Only include if we have enough samples
                            eval_data['uncertainty_quantification']['snr_correlation_by_category'][category] = {
                                'correlation': float(corr_data['correlation']),
                                'p_value': float(corr_data['p_value']),
                                'n_samples': int(corr_data['n_samples'])
                            }
            
            # Add SNR-based UQ breakdown if available
            if 'by_snr' in uq_summary:
                eval_data['uncertainty_quantification']['by_snr'] = {
                    'threshold': args.snr_threshold,
                    'low_snr': {},
                    'high_snr': {}
                }
                
                for snr_cat in ['low_snr', 'high_snr']:
                    if snr_cat in uq_summary['by_snr']:
                        for conf_cat in ['TP', 'TN', 'FP', 'FN']:
                            if conf_cat in uq_summary['by_snr'][snr_cat]:
                                cat_data = uq_summary['by_snr'][snr_cat][conf_cat]
                                if cat_data['n_samples'] > 0:
                                    eval_data['uncertainty_quantification']['by_snr'][snr_cat][conf_cat] = {
                                        'mean': float(cat_data['mean']),
                                        'std': float(cat_data['std']),
                                        'median': float(cat_data['median']),
                                        'n_samples': int(cat_data['n_samples'])
                                    }
        except Exception as e:
            print(f"Warning: Could not compute UQ metrics: {e}")
    
    # Add model information
    if args.weights_path:
        eval_data['model_path'] = args.weights_path
    if args.model_hash:
        eval_data['model_hash'] = args.model_hash
    
    # Add object ID filtering info if used
    if object_ids:
        eval_data['filtering'] = {
            'filtered_by_object_ids': True,
            'n_objects': len(object_ids)
        }
    
    with open(results_file, 'w') as f:
        yaml.dump(eval_data, f, default_flow_style=False, sort_keys=False)
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