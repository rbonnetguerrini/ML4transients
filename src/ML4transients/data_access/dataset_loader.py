"""
Main dataset loader for transient detection data.

This module provides the primary interface for accessing multi-visit astronomical datasets, including cutouts, features, and inference results. It supports lazy loading, efficient memory management, and automatic discovery of data files.
"""

import yaml
import json
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
from .data_loaders import CutoutLoader, FeatureLoader, LightCurveLoader, InferenceLoader

class DatasetLoader:
    """Main interface for accessing astronomical transient detection datasets.
    
    Provides unified access to cutout images, tabular features, and model inference results across multiple visits. Supports lazy loading and efficient memory
    management for large-scale datasets.
    
    Parameters
    ----------
    data_paths : str, Path, or list
        Single path or list of paths to data directories
        
    Attributes
    ----------
    data_paths : list of Path
        List of data directory paths
    visits : list of int
        Available visit numbers in the dataset
    """

    def __init__(self, data_paths: Union[str, Path, List[Union[str, Path]]]):
        # Normalize paths to list of Path objects
        if isinstance(data_paths, (str, Path)):
            data_paths = [Path(data_paths)]
        else:
            data_paths = [Path(p) for p in data_paths]
        
        self.data_paths = data_paths
        
        # Initialize internal storage
        self._cutout_loaders = {}
        self._feature_loaders = {}
        self._lightcurve_loaders = {}
        self._inference_loaders = {}
        self._visits = None
        self._config_summary = None
        self._global_id_index = None 
        self._cached_trainers = {}  # Cache trained models
        self._inference_registry = {}  # Registry of inference files

        # Discover available data files
        self._discover_data()
        self._load_inference_registries()

    def _build_global_index(self):
        """Build global index mapping diaSourceId to visit number.
        
        Creates an efficient lookup table for finding which visit contains a specific diaSourceId,enabling fast cross-visit searches.
        """
        if self._global_id_index is not None:
            return
        
        print("Building global index...")
        self._global_id_index = {}
        
        for visit, loader in self._feature_loaders.items():
            for dia_id in loader.ids:
                self._global_id_index[dia_id] = visit
                        
    @property
    def global_index(self) -> Dict[int, int]:
        """Get global ID index (lazy built).
        
        Returns
        -------
        dict
            Mapping from diaSourceId to visit number
        """
        if self._global_id_index is None:
            self._build_global_index()
        return self._global_id_index
    
    def get_cutout_by_id(self, dia_source_id: int) -> Optional[np.ndarray]:
        """Get cutout by diaSourceId (searches all visits).
        
        Parameters
        ----------
        dia_source_id : int
            The diaSourceId to retrieve
            
        Returns
        -------
        np.ndarray or None
            Cutout array if found, None otherwise
        """
        visit = self.global_index.get(dia_source_id)
        if visit and visit in self._cutout_loaders:
            return self._cutout_loaders[visit].get_by_id(dia_source_id)
        return None
    
    def get_features_by_id(self, dia_source_id: int) -> Optional[pd.DataFrame]:
        """Get features by diaSourceId (searches all visits).
        
        Parameters
        ----------
        dia_source_id : int
            The diaSourceId to retrieve
            
        Returns
        -------
        pd.DataFrame or None
            Features DataFrame if found, None otherwise
        """
        visit = self.global_index.get(dia_source_id)
        if visit and visit in self._feature_loaders:
            return self._feature_loaders[visit].get_by_id(dia_source_id)
        return None
    
    def find_visit(self, dia_source_id: int) -> Optional[int]:
        """Find which visit contains a specific diaSourceId.
        
        Parameters
        ----------
        dia_source_id : int
            The diaSourceId to search for
            
        Returns
        -------
        int or None
            Visit number containing the ID, None if not found
        """
        return self.global_index.get(dia_source_id)

    def get_inference_loader(self, visit: int, weights_path: str = None, 
                           model_hash: str = None, data_path: Path = None) -> Optional[InferenceLoader]:
        """Get or create InferenceLoader for a specific visit.
        
        Parameters
        ----------
        visit : int
            Visit number
        weights_path : str, optional
            Path to model weights directory (for new inference)
        model_hash : str, optional
            Model hash for existing inference results
        data_path : Path, optional
            Specific data path (defaults to first data path)
            
        Returns
        -------
        InferenceLoader or None
            Configured inference loader, None if not found/creatable
            
        Raises
        ------
        ValueError
            If neither weights_path nor model_hash provided
        """
        if not data_path:
            data_path = self.data_paths[0]
        
        if not weights_path and not model_hash:
            raise ValueError("Must provide either weights_path or model_hash")
        
        # Case 1: Loading existing results by model hash
        if model_hash:
            data_path_str = str(data_path)
            
            # Check registry for existing results
            if (data_path_str in self._inference_registry and 
                str(visit) in self._inference_registry[data_path_str] and
                model_hash in self._inference_registry[data_path_str][str(visit)]):
                
                cache_key = f"{data_path}_{visit}_{model_hash}"
                
                if cache_key not in self._inference_loaders:
                    # Create loader for existing results
                    inference_loader = InferenceLoader(data_path, visit, weights_path=None)
                    
                    # Set inference file from registry
                    inference_info = self._inference_registry[data_path_str][str(visit)][model_hash]
                    inference_file = data_path / "inference" / inference_info['filename']
                    
                    if inference_file.exists():
                        inference_loader._inference_file = inference_file
                        self._inference_loaders[cache_key] = inference_loader
                        print(f"Loaded inference loader for visit {visit}, model {model_hash}")
                    else:
                        print(f"Warning: Registry points to missing file: {inference_file}")
                        return None
                
                return self._inference_loaders[cache_key]
            else:
                print(f"No inference results found in registry for visit {visit}, model hash {model_hash}")
                return None
        
        # Case 2: Creating new inference or loading by weights_path
        else:
            cache_key = f"{data_path}_{visit}_{weights_path}"
            
            if cache_key not in self._inference_loaders:
                self._inference_loaders[cache_key] = InferenceLoader(data_path, visit, weights_path)
            
            return self._inference_loaders[cache_key]

    def run_inference_all_visits(self, weights_path: str, data_path: Path = None, force: bool = False) -> Dict[int, InferenceLoader]:
        """Run inference on all visits sequentially for memory efficiency.
        
        Parameters
        ----------
        weights_path : str 
            Path to model weights directory
        data_path : Path, optional
            Specific data path (defaults to first data path)
        force : bool
            Force re-run even if results exist
            
        Returns
        -------
        dict
            Mapping from visit number to InferenceLoader with results
        """
        if not data_path:
            data_path = self.data_paths[0]
        
        # Pre-load model once for efficiency
        print("Loading model (will be cached for all visits)...")
        trainer = self._get_or_load_trainer(weights_path)
        
        results = {}
        failed_visits = []
        
        print(f"Running inference on {len(self.visits)} visits...")
        
        for i, visit in enumerate(self.visits):
            print(f"\n--- Processing visit {visit} ({i+1}/{len(self.visits)}) ---")
            
            try:
                inference_loader = self.get_inference_loader(
                    visit, weights_path=weights_path, data_path=data_path
                )
                
                if inference_loader.has_inference_results() and not force:
                    print(f"Inference results already exist for visit {visit}")
                    results[visit] = inference_loader
                    continue
                
                # Run inference using cached trainer
                inference_loader.run_inference(self, trainer=trainer, force=force)
                results[visit] = inference_loader
                print(f"✓ Completed inference for visit {visit}")
                
            except Exception as e:
                print(f"✗ Error running inference for visit {visit}: {e}")
                failed_visits.append(visit)
                continue
            
            finally:
                # Force memory cleanup after each visit
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if failed_visits:
            print(f"\n⚠️  Failed visits: {failed_visits}")
        
        print(f"\nCompleted inference for {len(results)}/{len(self.visits)} visits")
        return results

    def check_or_run_inference(self, weights_path: str, data_path: Path = None):
        """
        Check for existing inference results across all visits and optionally run inference.
        
        Args:
            weights_path: Path to trained model weights directory
            data_path: Specific data path to use (defaults to first data path)
            
        Returns:
            Dict[int, InferenceLoader]: Inference loaders with results
        """
        if not data_path:
            data_path = self.data_paths[0]
        
        # Check which visits have inference results
        missing_visits = []
        existing_loaders = {}
        
        for visit in self.visits:
            inference_loader = self.get_inference_loader(visit, weights_path=weights_path, data_path=data_path)
            if inference_loader.has_inference_results():
                existing_loaders[visit] = inference_loader
            else:
                missing_visits.append(visit)
        
        if not missing_visits:
            print(f"Found existing inference results for all {len(self.visits)} visits")
            return existing_loaders
        
        print(f"Missing inference results for visits: {missing_visits}")
        response = input(f"Would you like to run inference for {len(missing_visits)} visits? (Y/n): ").lower().strip()
        
        if response in ['', 'y', 'yes']:
            try:
                all_results = self.run_inference_all_visits(weights_path, data_path)
                return all_results
            except Exception as e:
                print(f"Error running inference: {e}")
                return existing_loaders
        else:
            print("Inference not run.")
            return existing_loaders

    def _discover_data(self):
        """Discover available data files in directories.
        
        Scans data paths for cutout and feature files, automatically detecting visit numbers from filenames and creating appropriate loaders.
        """
        for data_path in self.data_paths:
            if not data_path.exists():
                continue
                
            # Discover cutout files
            cutout_dir = data_path / "cutouts"
            if cutout_dir.exists():
                for file in cutout_dir.glob("visit_*.h5"):
                    visit = self._extract_visit_from_filename(file.name)
                    if visit:
                        self._cutout_loaders[visit] = CutoutLoader(file)
            
            # Discover feature files
            feature_dir = data_path / "features"
            if feature_dir.exists():
                for file in feature_dir.glob("visit_*_features.h5"):
                    visit = self._extract_visit_from_filename(file.name)
                    if visit:
                        self._feature_loaders[visit] = FeatureLoader(file)

            # Load configuration summary if available
            config_file = data_path / "config_summary.yaml"
            if config_file.exists() and self._config_summary is None:
                with open(config_file) as f:
                    self._config_summary = yaml.safe_load(f)

    def _extract_visit_from_filename(self, filename: str) -> Optional[int]:
        """Extract visit number from filename pattern.
        
        Parameters
        ----------
        filename : str
            Filename to parse (e.g., 'visit_12345.h5')
            
        Returns
        -------
        int or None
            Visit number if successfully extracted, None otherwise
        """
        try:
            if filename.startswith("visit_"):
                visit_part = filename.split("_")[1].split(".")[0]
                if visit_part != "features":  # Skip "visit_12345_features.h5"
                    return int(visit_part)
        except (IndexError, ValueError):
            pass
        return None
    
    @property
    def visits(self) -> List[int]:
        """Get sorted list of available visits.
        
        Returns
        -------
        list of int
            Available visit numbers
        """
        if self._visits is None:
            all_visits = set(self._cutout_loaders.keys()) | set(self._feature_loaders.keys())
            self._visits = sorted(all_visits)
        return self._visits
    
    @property
    def cutouts(self) -> Dict[int, CutoutLoader]:
        """Access cutout loaders by visit number.
        
        Returns
        -------
        dict
            Mapping from visit number to CutoutLoader
        """
        return self._cutout_loaders
    
    @property
    def features(self) -> Dict[int, FeatureLoader]:
        """Access feature loaders by visit number.
        
        Returns
        -------
        dict
            Mapping from visit number to FeatureLoader
        """
        return self._feature_loaders
    
    def get_cutout(self, visit: int, dia_source_id: int):
        """Get specific cutout by visit and diaSourceId."""
        if visit in self._cutout_loaders:
            return self._cutout_loaders[visit].get_by_id(dia_source_id)
        return None
    
    def get_features(self, visit: int, dia_source_id: int):
        """Get specific features by visit and diaSourceId."""
        if visit in self._feature_loaders:
            return self._feature_loaders[visit].get_by_id(dia_source_id)
        return None
    
    def get_all_cutouts(self, visit: int):
        """Get all cutouts for a visit."""
        if visit in self._cutout_loaders:
            return self._cutout_loaders[visit].data
        return None
    
    def get_all_features(self, visit: int):
        """Get all features for a visit."""
        if visit in self._feature_loaders:
            return self._feature_loaders[visit].data
        return None

    def _get_or_load_trainer(self, weights_path: str):
        """Get cached trainer or load it if not cached.
        
        Parameters
        ----------
        weights_path : str
            Path to model weights directory
            
        Returns
        -------
        trainer
            Loaded and cached trainer instance
        """
        if weights_path not in self._cached_trainers:
            print(f"Loading model from {weights_path}...")
            
            from ML4transients.training import get_trainer
            from ML4transients.utils import load_config
            import torch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config = load_config(f"{weights_path}/config.yaml")
            trainer_type = config["training"]["trainer_type"]
            trainer = get_trainer(trainer_type, config["training"])
            
            # Load models based on trainer type
            if trainer_type == "ensemble":
                num_models = config["training"]["num_models"]
                print(f"Ensemble model, loading {num_models} models.")
                for i in range(num_models):
                    model_path = f"{weights_path}/ensemble_model_{i}_best.pth"
                    print(f"Loading model {i} from {model_path}")
                    state_dict = torch.load(model_path, map_location=device)
                    trainer.models[i].load_state_dict(state_dict)
                    trainer.models[i].to(device)
                    trainer.models[i].eval()
            elif trainer_type == "coteaching":
                state_dict1 = torch.load(f"{weights_path}/model1_best.pth", map_location=device)
                state_dict2 = torch.load(f"{weights_path}/model2_best.pth", map_location=device)
                trainer.model1.load_state_dict(state_dict1)
                trainer.model2.load_state_dict(state_dict2)
                trainer.model1.to(device)
                trainer.model2.to(device)
                trainer.model1.eval()
                trainer.model2.eval()
            else:
                state_dict = torch.load(f"{weights_path}/model_best.pth", map_location=device)
                trainer.model.load_state_dict(state_dict)
                trainer.model.to(device)
                trainer.model.eval()
            
            self._cached_trainers[weights_path] = trainer
            print(f"Model loaded and cached")
        
        return self._cached_trainers[weights_path]

    def _load_inference_registries(self):
        """Load inference registries from all data paths.
        
        Loads JSON registry files that track available inference results
        for efficient discovery and management.
        """
        for data_path in self.data_paths:
            registry_file = data_path / "inference" / "inference_registry.json"
            if registry_file.exists():
                try:
                    with open(registry_file, 'r') as f:
                        registry = json.load(f)
                    self._inference_registry[str(data_path)] = registry
                    print(f"Loaded inference registry from {registry_file}")
                except Exception as e:
                    print(f"Error loading inference registry from {registry_file}: {e}")
            else:
                # Create empty registry if none exists
                self._inference_registry[str(data_path)] = {}

    def _save_inference_registry(self, data_path: Path):
        """Save inference registry for a specific data path.
        
        Parameters
        ----------
        data_path : Path
            Data path to save registry for
        """
        registry_file = data_path / "inference" / "inference_registry.json"
        registry_file.parent.mkdir(exist_ok=True)
        
        registry_data = self._inference_registry.get(str(data_path), {})
        
        try:
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            print(f"Saved inference registry to {registry_file}")
        except Exception as e:
            print(f"Error saving inference registry to {registry_file}: {e}")

    def _register_inference_file(self, data_path: Path, visit: int, model_hash: str, 
                                 weights_path: str = None, metadata: Dict = None):
        """Register a new inference file in the registry.
        
        Parameters
        ----------
        data_path : Path
            Data path where inference file is located
        visit : int
            Visit number
        model_hash : str
            Model hash identifier
        weights_path : str, optional
            Path to model weights
        metadata : dict, optional
            Additional metadata to store
        """
        data_path_str = str(data_path)
        
        if data_path_str not in self._inference_registry:
            self._inference_registry[data_path_str] = {}
        
        if visit not in self._inference_registry[data_path_str]:
            self._inference_registry[data_path_str][visit] = {}
        
        inference_info = {
            'model_hash': model_hash,
            'filename': f"visit_{visit}_inference_{model_hash}.h5",
            'created_at': pd.Timestamp.now().isoformat(),
        }
        
        if weights_path:
            inference_info['weights_path'] = str(weights_path)
        
        if metadata:
            inference_info['metadata'] = metadata
        
        self._inference_registry[data_path_str][visit][model_hash] = inference_info
        self._save_inference_registry(data_path)

    def list_available_inference(self, data_path: Path = None):
        """List all available inference files from the registry.
        
        Parameters
        ----------
        data_path : Path, optional
            Specific data path to check (defaults to first data path)
            
        Returns
        -------
        dict
            Registry data for the specified path
        """
        if not data_path:
            data_path = self.data_paths[0]
        
        data_path_str = str(data_path)
        
        if data_path_str not in self._inference_registry or not self._inference_registry[data_path_str]:
            print(f"No inference files found in registry for {data_path}")
            return {}
        
        registry = self._inference_registry[data_path_str]
        
        print(f"Available inference files in {data_path}:")
        for visit, models in registry.items():
            print(f"  Visit {visit}:")
            for model_hash, info in models.items():
                weights_info = f" (from {info.get('weights_path', 'unknown')})" if 'weights_path' in info else ""
                created_info = f" created {info.get('created_at', 'unknown')}" if 'created_at' in info else ""
                print(f"    Model hash: {model_hash}{weights_info}{created_info}")
                print(f"      File: {info['filename']}")
        
        return registry

    def sync_inference_registry(self, data_path: Path = None):
        """Sync the inference registry with actual files on disk.
        
        This helps recover from registry corruption or manual file operations
        by scanning the filesystem and updating the registry accordingly.
        
        Parameters
        ----------
        data_path : Path, optional
            Data path to sync (defaults to first data path)
        """
        if not data_path:
            data_path = self.data_paths[0]
        
        print(f"Syncing inference registry for {data_path}...")
        
        data_path_str = str(data_path)
        inference_dir = data_path / "inference"
        
        if not inference_dir.exists():
            print(f"No inference directory found at {inference_dir}")
            return
        
        # Discover files on disk
        discovered_files = {}
        for file in inference_dir.glob("visit_*_inference_*.h5"):
            parts = file.stem.split('_')
            if len(parts) >= 4 and parts[0] == 'visit' and parts[2] == 'inference':
                try:
                    visit = int(parts[1])
                    model_hash = parts[3]
                    
                    if visit not in discovered_files:
                        discovered_files[visit] = {}
                    
                    discovered_files[visit][model_hash] = {
                        'filename': file.name,
                        'file_size': file.stat().st_size,
                        'modified_at': pd.Timestamp.fromtimestamp(file.stat().st_mtime).isoformat()
                    }
                except ValueError:
                    continue
        
        # Update registry
        current_registry = self._inference_registry.get(data_path_str, {})
        
        # Add missing files to registry
        added_count = 0
        for visit, models in discovered_files.items():
            for model_hash, file_info in models.items():
                if (str(visit) not in current_registry or 
                    model_hash not in current_registry.get(str(visit), {})):
                    
                    if str(visit) not in current_registry:
                        current_registry[str(visit)] = {}
                    
                    current_registry[str(visit)][model_hash] = {
                        'model_hash': model_hash,
                        'filename': file_info['filename'],
                        'discovered_at': pd.Timestamp.now().isoformat(),
                        'file_size': file_info['file_size']
                    }
                    added_count += 1
        
        # Remove registry entries for missing files
        removed_count = 0
        visits_to_remove = []
        for visit, models in current_registry.items():
            models_to_remove = []
            for model_hash in models.keys():
                if (int(visit) not in discovered_files or 
                    model_hash not in discovered_files[int(visit)]):
                    models_to_remove.append(model_hash)
                    removed_count += 1
            
            for model_hash in models_to_remove:
                del models[model_hash]
            
            if not models:  # Remove empty visit entries
                visits_to_remove.append(visit)
        
        for visit in visits_to_remove:
            del current_registry[visit]
        
        self._inference_registry[data_path_str] = current_registry
        self._save_inference_registry(data_path)
        
        print(f"Registry sync complete: added {added_count}, removed {removed_count} entries")

    def __repr__(self):
        """String representation with dataset statistics"""
        total_cutouts = 0
        total_features = 0
        visits_with_cutouts = 0
        visits_with_features = 0
        
        for visit in self.visits:
            if visit in self._cutout_loaders:
                total_cutouts += len(self._cutout_loaders[visit].ids)
                visits_with_cutouts += 1
                
            if visit in self._feature_loaders:
                total_features += len(self._feature_loaders[visit].ids)
                visits_with_features += 1
        
        return (f"DatasetLoader({len(self.visits)} visits, {len(self.data_paths)} paths)\n"
                f"  Cutouts: {total_cutouts} across {visits_with_cutouts} visits\n"
                f"  Features: {total_features} across {visits_with_features} visits")
