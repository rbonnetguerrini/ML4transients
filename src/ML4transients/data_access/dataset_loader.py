import yaml
import json
import torch
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Optional, Union
from .data_loaders import CutoutLoader, FeatureLoader, LightCurveLoader, InferenceLoader

class DatasetLoader:
    """Lazy loading dataset class data."""

    def __init__(self, data_paths: Union[str, Path, List[Union[str, Path]]]):
        """
        Initialize with one or more data directories.
        
        Args:
            data_paths: Single path or list of paths to data directories
        """

        if isinstance(data_paths, (str, Path)):
            data_paths = [Path(data_paths)]
        else:
            data_paths = [Path(p) for p in data_paths]
        

        self.data_paths = data_paths
        self._cutout_loaders = {}
        self._feature_loaders = {}
        self._lightcurve_loaders = {}
        self._inference_loaders = {}
        self._visits = None
        self._config_summary = None
        self._global_id_index = None 
        self._cached_trainers = {}  # Cache trainers by weights_path
        self._inference_registry = {}  # Registry of available inference files

        self._discover_data()
        self._load_inference_registries()

    def _load_global_index(self):
        """Load the persistent global index from disk.
        
        Loads the pre-built index mapping diaSourceId to visit number
        that was created during cutout processing.
        """
        if self._global_id_index is not None:
            return
        
        # Try to find global index in any of the data paths
        for data_path in self.data_paths:
            index_file = data_path / "cutout_global_index.h5"
            if index_file.exists():
                try:
                    print(f"Loading global cutout index from {index_file}...")
                    index_df = pd.read_hdf(index_file, key="global_index")
                    
                    # Convert to dictionary for fast lookups
                    self._global_id_index = dict(zip(index_df.index, index_df['visit']))
                    print(f"Loaded global index with {len(self._global_id_index)} entries")
                    return
                    
                except Exception as e:
                    print(f"Error loading global index from {index_file}: {e}")
                    continue
        
        # Only reach here if NO index file was found or ALL failed to load
        print("Warning: No persistent global index found. Building dynamically...")
        print("Consider running cutout extraction with index creation for better performance.")
        self._build_global_index_fallback()

    def _build_global_index_fallback(self):
        """Fallback method to build global index dynamically (legacy behavior)."""
        print("Building global index from feature files...")
        self._global_id_index = {}
        
        # Index from feature loaders
        for visit, loader in self._feature_loaders.items():
            for dia_id in loader.ids:
                self._global_id_index[dia_id] = visit
                        
    @property
    def global_index(self):
        """Lazy load and return global ID index."""
        if self._global_id_index is None:
            self._load_global_index()
        return self._global_id_index
    
    def get_cutout_by_id(self, dia_source_id: int):
        """Get cutout by diaSourceId (searches all visits).
        
        Parameters
        ----------
        dia_source_id : int
            The diaSourceId to retrieve
            
        Returns
        -------
        np.ndarray or None
            Cutout array, or None if not found
        """
        visit = self.global_index.get(dia_source_id)
        if visit and visit in self._cutout_loaders:
            return self._cutout_loaders[visit].get_by_id(dia_source_id)
        return None
    
    def get_features_by_id(self, dia_source_id: int):
        """Get features by diaSourceId (searches all visits).
        
        Parameters
        ----------
        dia_source_id : int
            The diaSourceId to retrieve
            
        Returns
        -------
        pd.DataFrame or None
            Features DataFrame, or None if not found
        """
        visit = self.global_index.get(dia_source_id)
        if visit and visit in self._feature_loaders:
            return self._feature_loaders[visit].get_by_id(dia_source_id)
        return None
    
    def find_visit(self, dia_source_id: int) -> Optional[int]:
        """Find which visit contains this diaSourceId.
        
        Parameters
        ----------
        dia_source_id : int
            The diaSourceId to search for
            
        Returns
        -------
        int or None
            Visit number containing the ID, or None if not found
        """
        return self.global_index.get(dia_source_id)

    def get_inference_loader(self, visit: int, weights_path: str = None, model_hash: str = None, data_path: Path = None):
        """
        Get or create an InferenceLoader for the given visit.
        Can work with either weights_path (for new inference) or model_hash (for existing results).
        
        Args:
            visit: Visit number
            weights_path: Path to trained model weights directory (for new inference)
            model_hash: Model hash from existing inference filename (for loading existing results)
            data_path: Specific data path to use (defaults to first data path)
            
        Returns:
            InferenceLoader: Configured inference loader for the visit
        """
        if not data_path:
            data_path = self.data_paths[0]
        
        # Ensure we have either weights_path or model_hash
        if not weights_path and not model_hash:
            raise ValueError("Must provide either weights_path or model_hash")
        
        # Case 1: Loading existing results by model hash
        if model_hash:
            data_path_str = str(data_path)
            
            # Check registry first
            if (data_path_str in self._inference_registry and 
                str(visit) in self._inference_registry[data_path_str] and
                model_hash in self._inference_registry[data_path_str][str(visit)]):
                
                # Create cache key for this specific inference file
                cache_key = f"{data_path}_{visit}_{model_hash}"
                
                if cache_key not in self._inference_loaders:
                    # Create InferenceLoader without weights_path since we're loading existing results
                    inference_loader = InferenceLoader(data_path, visit, weights_path=None)
                    
                    # Manually set the inference file based on the registry info
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
            # Use weights path and visit as key to cache inference loaders
            cache_key = f"{data_path}_{visit}_{weights_path}"
            
            if cache_key not in self._inference_loaders:
                self._inference_loaders[cache_key] = InferenceLoader(data_path, visit, weights_path)
            
            return self._inference_loaders[cache_key]

    def run_inference_all_visits(self, weights_path: str, data_path: Path = None, force=False):
        """
        Run inference on all visits sequentially for memory efficiency.
        Uses cached model to avoid reloading.
        """
        if not data_path:
            data_path = self.data_paths[0]
        
        # Pre-load the model once
        print("Loading model (will be cached for all visits)...")
        trainer = self._get_or_load_trainer(weights_path)
        
        results = {}
        failed_visits = []
        
        print(f"Running inference on {len(self.visits)} visits...")
        
        for i, visit in enumerate(self.visits):
            print(f"\n--- Processing visit {visit} ({i+1}/{len(self.visits)}) ---")
            
            try:
                inference_loader = self.get_inference_loader(visit, weights_path=weights_path, data_path=data_path)
                
                if inference_loader.has_inference_results() and not force:
                    print(f"Inference results already exist for visit {visit}")
                    results[visit] = inference_loader
                    continue
                
                # Pass the cached trainer to avoid reloading
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
        """Discover available data files in the directories."""
        for data_path in self.data_paths:
            if not data_path.exists():
                continue
                
            # Find cutout files
            cutout_dir = data_path / "cutouts"
            if cutout_dir.exists():
                for file in cutout_dir.glob("visit_*.h5"):
                    visit = self._extract_visit_from_filename(file.name)
                    if visit:
                        self._cutout_loaders[visit] = CutoutLoader(file)
            
            # Find feature files
            feature_dir = data_path / "features"
            if feature_dir.exists():
                for file in feature_dir.glob("visit_*_features.h5"):
                    visit = self._extract_visit_from_filename(file.name)
                    if visit:
                        self._feature_loaders[visit] = FeatureLoader(file)

            # Find lightcurve directory
            lightcurve_dir = data_path / "lightcurves"
            if lightcurve_dir.exists() and (lightcurve_dir / "lightcurve_index.h5").exists():
                self._lightcurve_loaders[str(data_path)] = LightCurveLoader(lightcurve_dir)
            
            # Load config summary if available
            config_file = data_path / "config_summary.yaml"
            if config_file.exists() and self._config_summary is None:
                with open(config_file) as f:
                    self._config_summary = yaml.safe_load(f)
        
        # Note: Global index is now loaded on-demand, not built here
    def get_complete_lightcurve_data(
        self, 
        dia_source_id: int, 
        data_path: Path = None, 
        columns: list = None, 
        load_cutouts: bool = False
    ) -> Optional[Dict]:
        """Get all data for sources in a lightcurve given any diaSourceId from that lightcurve.
        
        This method efficiently retrieves all related data for a complete lightcurve
        by using a diaSourceId as the entry point. It only loads the requested columns
        from the patch file for optimal performance.
        
        Args:
            dia_source_id: Any diaSourceId from the desired lightcurve
            data_path: Path to data directory. If None, uses first available lightcurve loader
            columns: List of columns to load from the patch file. If None, uses default
                    lightcurve columns: ["diaSourceId", "midpointMjdTai", "psFlux", 
                    "psFluxErr", "mag", "magErr", "band", "ccdVisitId", "visit"]
            load_cutouts: If True, also loads cutouts for all sources in the lightcurve
        
        Returns:
            Optional[Dict]: Dictionary containing:
                - 'lightcurve': pd.DataFrame with lightcurve data sorted by time
                - 'source_ids': List[int] of all diaSourceIds in the lightcurve
                - 'cutouts': Dict[int, np.ndarray] mapping diaSourceId to cutout (if load_cutouts=True)
                - 'object_id': int diaObjectId for this lightcurve
                - 'num_sources': int number of sources in the lightcurve
            
            Returns None if the diaSourceId is not found or data cannot be loaded.
        
        Example:
            >>> data = dataset_loader.get_complete_lightcurve_data(
            ...     dia_source_id=12345,
            ...     columns=["diaSourceId", "midpointMjdTai", "psFlux", "band"],
            ...     load_cutouts=True
            ... )
            >>> print(f"Lightcurve has {data['num_sources']} sources")
            >>> print(f"Object ID: {data['object_id']}")
        """
        start_time = time.time()
        
        # Determine data path to use
        if not data_path:
            if self._lightcurve_loaders:
                data_path = Path(list(self._lightcurve_loaders.keys())[0])
            else:
                return None
        
        path_str = str(data_path)
        if path_str not in self._lightcurve_loaders:
            return None
        
        lc_loader = self._lightcurve_loaders[path_str]

        # Step 1: Get all source IDs in this lightcurve using the diasource index
        step1_start = time.time()
        source_ids = lc_loader.get_all_source_ids_in_lightcurve(dia_source_id)
        if not source_ids:
            return None
        step1_time = time.time() - step1_start
        print(f"Step 1 - Get source IDs: {step1_time:.3f}s ({len(source_ids)} sources found)")
        
        # Step 2: Get object ID for this lightcurve
        step2_start = time.time()
        object_id = lc_loader.get_object_id_for_source(dia_source_id)
        step2_time = time.time() - step2_start
        print(f"Step 2 - Get object ID: {step2_time:.3f}s (object_id: {object_id})")
        
        # Step 3: Load only the requested columns from the patch file
        step3_start = time.time()
        if columns is None:
            # Default columns for lightcurve analysis
            columns = [
                "diaSourceId", "midpointMjdTai", "psFlux", "psFluxErr", 
                "mag", "magErr", "band", "ccdVisitId", "visit"
            ]
        
        # Find the patch containing this lightcurve
        patch_key = lc_loader.find_patch_by_source_id(dia_source_id)
        if not patch_key:
            # Fallback: get patch from object ID
            patch_key = lc_loader.find_patch(object_id)
        
        if not patch_key:
            print(f"No patch found for source {dia_source_id} or object {object_id}")
            return None
            
        patch_file = lc_loader.lightcurve_path / f"patch_{patch_key}.h5"
        if not patch_file.exists():
            print(f"Patch file not found: {patch_file}")
            return None

        try:
            # Load only the rows for this diaObjectId and requested columns
            lc_df = pd.read_hdf(
                patch_file, 
                key="lightcurves", 
                where=f"diaObjectId=={object_id}", 
                columns=columns
            )
            
            # Sort by time if available for proper lightcurve ordering
            if 'midpointMjdTai' in lc_df.columns and len(lc_df) > 0:
                lc_df = lc_df.sort_values('midpointMjdTai').reset_index(drop=True)
                
        except Exception as e:
            print(f"Error loading lightcurve from {patch_file}: {e}")
            return None
        
        step3_time = time.time() - step3_start
        lc_points = len(lc_df) if lc_df is not None else 0
        print(f"Step 3 - Get lightcurve data: {step3_time:.3f}s ({lc_points} lightcurve points)")

        # Step 4: Optionally load cutouts for all sources in the lightcurve
        cutouts = {}
        if load_cutouts and len(source_ids) > 0:
            step4_start = time.time()
            
            # Group source IDs by visit for efficient batch loading
            visit_groups = {}
            for src_id in source_ids:
                visit = self.find_visit(src_id)
                if visit:
                    if visit not in visit_groups:
                        visit_groups[visit] = []
                    visit_groups[visit].append(src_id)
            
            print(f"Step 4a - Group by visit: ({len(visit_groups)} visits)")
            
            # Load cutouts visit by visit using batch operations
            for visit, src_ids_in_visit in visit_groups.items():
                if visit in self._cutout_loaders:
                    batch_cutouts = self._cutout_loaders[visit].get_multiple_by_ids(src_ids_in_visit)
                    cutouts.update(batch_cutouts)
            
            step4_time = time.time() - step4_start
            print(f"Step 4 - Get cutouts: {step4_time:.3f}s ({len(cutouts)} cutouts)")

        total_time = time.time() - start_time
        print(f"Total lightcurve data retrieval time: {total_time:.3f}s")
        
        return {
            'lightcurve': lc_df,
            'source_ids': source_ids,
            'cutouts': cutouts if load_cutouts else None,
            'object_id': object_id,
            'num_sources': len(source_ids)
        }

    

    def get_inference_for_lightcurve(self, dia_source_id: int, weights_path: str = None, 
                                   model_hash: str = None, data_path: Path = None) -> Dict[int, Dict]:
        """Get inference results for all sources in a lightcurve.
        
        Args:
            dia_source_id: Any diaSourceId from the lightcurve
            weights_path: Path to model weights (for new inference)
            model_hash: Model hash (for existing results)
            data_path: Specific data path to search
            
        Returns:
            Dict[int, Dict]: Dictionary mapping diaSourceId to inference results for the entire lightcurve
        """
        source_ids = self.get_lightcurve_source_ids(dia_source_id, data_path)
        
        inference_results = {}
        for src_id in source_ids:
            result = self.get_inference_results_by_id(src_id, weights_path, model_hash)
            if result is not None:
                inference_results[src_id] = result
        
        return inference_results

    def get_inference_results_by_id(self, dia_source_id: int, weights_path: str = None, model_hash: str = None):
        """Get inference results by diaSourceId across all visits."""
        visit = self.find_visit(dia_source_id)
        if visit:
            inference_loader = self.get_inference_loader(visit, weights_path=weights_path, model_hash=model_hash)
            if inference_loader and inference_loader.has_inference_results():
                return inference_loader.get_results_by_id(dia_source_id)
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
            
            # Import here to avoid circular imports
            from ML4transients.training import get_trainer
            from ML4transients.utils import load_config
            import torch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config = load_config(f"{weights_path}/config.yaml")
            trainer_type = config["training"]["trainer_type"]
            trainer = get_trainer(trainer_type, config["training"])
            
            # Load models based on trainer type (same logic as inference.py)
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
                # Load both models for co-teaching
                state_dict1 = torch.load(f"{weights_path}/model1_best.pth", map_location=device)
                state_dict2 = torch.load(f"{weights_path}/model2_best.pth", map_location=device)
                trainer.model1.load_state_dict(state_dict1)
                trainer.model2.load_state_dict(state_dict2)
                trainer.model1.to(device)
                trainer.model2.to(device)
                trainer.model1.eval()
                trainer.model2.eval()
            else:
                # Standard single model
                state_dict = torch.load(f"{weights_path}/model_best.pth", map_location=device)
                trainer.model.load_state_dict(state_dict)
                trainer.model.to(device)
                trainer.model.eval()
            
            self._cached_trainers[weights_path] = trainer
            print(f"Model loaded and cached for {weights_path}")
        
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

    @property
    def config_summary(self) -> Optional[Dict]:
        """Get config summary if available."""
        return self._config_summary
        
    def _extract_visit_from_filename(self, filename: str) -> Optional[int]:
        """Extract visit number from filename like 'visit_12345.h5'.
        
        Parameters
        ----------
        filename : str
            Filename to parse
            
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
        """Get list of available visits."""
        if self._visits is None:
            all_visits = set(self._cutout_loaders.keys()) | set(self._feature_loaders.keys())
            self._visits = sorted(all_visits)
        return self._visits
    
    @property
    def cutouts(self) -> Dict[int, CutoutLoader]:
        """Access cutout loaders by visit."""
        return self._cutout_loaders
    
    @property
    def features(self) -> Dict[int, FeatureLoader]:
        """Access feature loaders by visit."""
        return self._feature_loaders
    
    @property
    def lightcurves(self):
        """Access lightcurve loaders by path."""
        # If only one path, return the loader directly for convenience
        if len(self._lightcurve_loaders) == 1:
            return list(self._lightcurve_loaders.values())[0]
        return self._lightcurve_loaders
    
    @property
    def inference(self) -> Dict[str, InferenceLoader]:
        """Access inference loaders by cache key."""
        return self._inference_loaders

    def get_lightcurve_by_object_id(self, dia_object_id: int, data_path: Path = None) -> Optional[pd.DataFrame]:
        """Get lightcurve by diaObjectId from any available data path.
        
        Parameters
        ----------
        dia_object_id : int
            The diaObjectId to retrieve lightcurve for
        data_path : Path, optional
            Specific data path to search (defaults to all paths)
            
        Returns
        -------
        pd.DataFrame or None
            Lightcurve DataFrame, or None if not found
        """
        paths_to_search = [data_path] if data_path else self.data_paths
        
        for path in paths_to_search:
            path_str = str(path)
            if path_str in self._lightcurve_loaders:
                lc = self._lightcurve_loaders[path_str].get_lightcurve(dia_object_id)
                if lc is not None:
                    return lc
        
        return None
    
    def get_multiple_lightcurves_by_object_id(self, dia_object_ids: List[int], data_path: Path = None) -> Dict[int, pd.DataFrame]:
        """Efficiently get lightcurves for multiple diaObjectIds.
        
        Parameters
        ----------
        dia_object_ids : List[int]
            List of diaObjectIds to retrieve
        data_path : Path, optional
            Specific data path to search (defaults to first available)
            
        Returns
        -------
        Dict[int, pd.DataFrame]
            Dictionary mapping diaObjectId to lightcurve DataFrame
        """
        if not data_path:
            # Use first available lightcurve loader
            if self._lightcurve_loaders:
                data_path = Path(list(self._lightcurve_loaders.keys())[0])
            else:
                return {}
        
        path_str = str(data_path)
        if path_str in self._lightcurve_loaders:
            return self._lightcurve_loaders[path_str].get_multiple_lightcurves(dia_object_ids)
        
        return {}
    
    def get_lightcurve_stats(self, dia_object_id: int, data_path: Path = None) -> Optional[Dict]:
        """Get lightcurve statistics without loading full data."""
        paths_to_search = [data_path] if data_path else self.data_paths
        
        for path in paths_to_search:
            path_str = str(path)
            if path_str in self._lightcurve_loaders:
                stats = self._lightcurve_loaders[path_str].get_lightcurve_stats(dia_object_id)
                if stats is not None:
                    return stats
        
        return None

    def __repr__(self):
        total_cutouts = 0
        total_features = 0
        visits_with_cutouts = 0
        visits_with_features = 0
        total_lightcurve_objects = 0
        
        for visit in self.visits:
            if visit in self._cutout_loaders:
                total_cutouts += len(self._cutout_loaders[visit].ids)
                visits_with_cutouts += 1
                
            if visit in self._feature_loaders:
                total_features += len(self._feature_loaders[visit].ids)
                visits_with_features += 1
        
        # Count lightcurve objects
        for loader in self._lightcurve_loaders.values():
            try:
                total_lightcurve_objects += len(loader.index)
                break  # Just count once since all paths should have same objects
            except:
                pass
        
        lc_info = f"  Lightcurves: {total_lightcurve_objects} objects" if total_lightcurve_objects > 0 else "  Lightcurves: Not available"
        
        return (f"DatasetLoader({len(self.visits)} visits, {len(self.data_paths)} paths)\n"
                f"  Cutouts: {total_cutouts} across {visits_with_cutouts} visits\n"
                f"  Features: {total_features} across {visits_with_features} visits\n"
                f"{lc_info}")

    def __str__(self):
        return self.__repr__()

