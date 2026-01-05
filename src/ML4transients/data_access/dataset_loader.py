import yaml
import json
import pandas as pd
import numpy as np
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
        self._discovery_done = False  # Track if discovery has been performed
        self._crossmatch_data = None  # Cache for cross-match results

        # Only load critical components immediately
        self._load_config_summary()
        # Defer expensive operations until needed

    def _load_config_summary(self):
        """Load config summary only if available."""
        for data_path in self.data_paths:
            config_file = data_path / "config_summary.yaml"
            if config_file.exists() and self._config_summary is None:
                with open(config_file) as f:
                    self._config_summary = yaml.safe_load(f)
                break

    def _ensure_discovery(self):
        """Ensure data discovery has been performed."""
        if not self._discovery_done:
            print("Discovering data files...")
            self._discover_data()
            self._load_inference_registries()
            self._discovery_done = True

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
        """Fallback method to build global index dynamically."""
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
    
    def get_cutout_by_id(self, dia_source_id: int, cutout_type: str = 'diff'):
        """Get cutout by diaSourceId (searches all visits).
        
        Parameters
        ----------
        dia_source_id : int
            The diaSourceId to retrieve
        cutout_type : str, default 'diff'
            Type of cutout to retrieve: 'diff', 'coadd', or 'science'
            
        Returns
        -------
        np.ndarray or None
            Cutout array, or None if not found
        """
        visit = self.global_index.get(dia_source_id)
        if visit and visit in self._cutout_loaders:
            return self._cutout_loaders[visit].get_by_id(dia_source_id, cutout_type=cutout_type)
        return None
    
    def get_cutout(self, visit: int, dia_source_id: int, cutout_type: str = 'diff'):
        """Get specific cutout by visit and diaSourceId.
        
        Parameters
        ----------
        visit : int
            Visit number
        dia_source_id : int
            The diaSourceId to retrieve
        cutout_type : str, default 'diff'
            Type of cutout to retrieve: 'diff', 'coadd', or 'science'
            
        Returns
        -------
        np.ndarray or None
            Cutout array, or None if not found
        """
        if visit in self._cutout_loaders:
            return self._cutout_loaders[visit].get_by_id(dia_source_id, cutout_type=cutout_type)
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

    def get_inference_loader(self, visit: int, weights_path: str = None, model_hash: str = None, data_path: Path = None, mc_dropout: bool = False, mc_samples: int = 50):
        """
        Get or create an InferenceLoader for the given visit.
        Can work with either weights_path (for new inference) or model_hash (for existing results).
        
        Args:
            visit: Visit number
            weights_path: Path to trained model weights directory (for new inference)
            model_hash: Model hash from existing inference filename (for loading existing results)
            data_path: Specific data path to use (defaults to first data path)
            mc_dropout: Enable Monte Carlo Dropout (affects hash generation for new inference)
            mc_samples: Number of MC Dropout samples (affects hash generation for new inference)
            
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
                    # MC dropout params don't matter when loading existing results
                    inference_loader = InferenceLoader(data_path, visit, weights_path=None, mc_dropout=False, mc_samples=50)
                    
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
            # Use weights path, visit, and MC dropout config as key to cache inference loaders
            mc_suffix = f"_mcd{mc_samples}" if mc_dropout else ""
            cache_key = f"{data_path}_{visit}_{weights_path}{mc_suffix}"
            
            if cache_key not in self._inference_loaders:
                self._inference_loaders[cache_key] = InferenceLoader(data_path, visit, weights_path, mc_dropout=mc_dropout, mc_samples=mc_samples)
            
            return self._inference_loaders[cache_key]

    def run_inference_all_visits(self, weights_path: str, data_path: Path = None, force=False, mc_dropout=False, mc_samples=50):
        """
        Run inference on all visits sequentially for memory efficiency.
        Uses cached model to avoid reloading.
        
        Args:
            weights_path: Path to model weights
            data_path: Optional data path override
            force: Force re-run even if results exist
            mc_dropout: Enable Monte Carlo Dropout for uncertainty estimation
            mc_samples: Number of MC Dropout forward passes
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
                inference_loader = self.get_inference_loader(visit, weights_path=weights_path, data_path=data_path,
                                                            mc_dropout=mc_dropout, mc_samples=mc_samples)
                
                if inference_loader.has_inference_results() and not force:
                    print(f"Inference results already exist for visit {visit}")
                    results[visit] = inference_loader
                    continue
                
                # Pass the cached trainer to avoid reloading
                inference_loader.run_inference(self, trainer=trainer, force=force, 
                                             mc_dropout=mc_dropout, mc_samples=mc_samples)
                results[visit] = inference_loader
                print(f" Completed inference for visit {visit}")
                
            except Exception as e:
                print(f" Error running inference for visit {visit}: {e}")
                failed_visits.append(visit)
                continue
            
            finally:
                # Force memory cleanup after each visit
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass  # torch not available in this environment
        
        if failed_visits:
            print(f"\n  Failed visits: {failed_visits}")
        
        print(f"\nCompleted inference for {len(results)}/{len(self.visits)} visits")
        return results

    def check_or_run_inference(self, weights_path: str, data_path: Path = None, mc_dropout=False, mc_samples=50):
        """
        Check for existing inference results across all visits and optionally run inference.
        
        Args:
            weights_path: Path to trained model weights directory
            data_path: Specific data path to use (defaults to first data path)
            mc_dropout: Enable Monte Carlo Dropout for uncertainty estimation
            mc_samples: Number of MC Dropout forward passes
            
        Returns:
            Dict[int, InferenceLoader]: Inference loaders with results
        """
        if not data_path:
            data_path = self.data_paths[0]
        
        # Check which visits have inference results
        missing_visits = []
        existing_loaders = {}
        
        for visit in self.visits:
            inference_loader = self.get_inference_loader(visit, weights_path=weights_path, data_path=data_path,
                                                        mc_dropout=mc_dropout, mc_samples=mc_samples)
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
                all_results = self.run_inference_all_visits(weights_path, data_path,
                                                           mc_dropout=mc_dropout, mc_samples=mc_samples)
                return all_results
            except Exception as e:
                print(f"Error running inference: {e}")
                return existing_loaders
        else:
            print("Inference not run.")
            return existing_loaders

    def _discover_data(self):
        """Discover available data files in the directories."""
        start_time = time.time()
        
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
        
        discovery_time = time.time() - start_time
        num_cutouts = len(self._cutout_loaders)
        num_features = len(self._feature_loaders)
        num_lc_paths = len(self._lightcurve_loaders)
        
        print(f"Data discovery completed in {discovery_time:.2f}s: "
              f"{num_cutouts} cutout visits, {num_features} feature visits, "
              f"{num_lc_paths} lightcurve paths")
        
        # Note: Global index is now loaded on-demand, not built here
    def get_complete_lightcurve_data(
        self, 
        dia_source_id: int, 
        data_path: Path = None, 
        columns: list = None, 
        load_cutouts: bool = False,
        cutout_type: str = 'diff'
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
            cutout_type: Type of cutout to load: 'diff' (default), 'coadd', or 'science'
        
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
                    batch_cutouts = self._cutout_loaders[visit].get_multiple_by_ids(
                        src_ids_in_visit, cutout_type=cutout_type
                    )
                    cutouts.update(batch_cutouts)
            
            step4_time = time.time() - step4_start
            print(f"Step 4 - Load {cutout_type} cutouts: {step4_time:.3f}s ({len(cutouts)} cutouts)")

        total_time = time.time() - start_time
        print(f"Total lightcurve data retrieval time: {total_time:.3f}s")
        
        return {
            'lightcurve': lc_df,
            'source_ids': source_ids,
            'cutouts': cutouts if load_cutouts else None,
            'object_id': object_id,
            'num_sources': len(source_ids),
            'cutout_type': cutout_type if load_cutouts else None
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
            
            # Expand base_filters and base_dropout if present (for new architecture)
            model_params = config["training"].get("model_params", {})
            if "base_filters" in model_params:
                F = model_params["base_filters"]
                model_params["filters_1"] = F
                model_params["filters_2"] = 2 * F
                model_params["filters_3"] = 4 * F
                print(f"[Inference] Expanded base_filters={F} to filters: ({F}, {2*F}, {4*F})")
                del model_params["base_filters"]
            
            if "base_dropout" in model_params:
                DR = model_params["base_dropout"]
                model_params["dropout_1"] = 0.5 * DR
                model_params["dropout_2"] = 0.5 * DR
                model_params["dropout_3"] = DR
                print(f"[Inference] Expanded base_dropout={DR:.3f} to dropouts: ({0.5*DR:.3f}, {0.5*DR:.3f}, {DR:.3f})")
                del model_params["base_dropout"]
            
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
            elif trainer_type == "repulsive_ensemble":
                num_models = config["training"]["num_models"]
                print(f"Ensemble model, loading {num_models} models.")
                for i in range(num_models):
                    model_path = f"{weights_path}/repulsive_ensemble_model_{i}_best.pth"
                    print(f"Loading model {i} from {model_path}")
                    state_dict = torch.load(model_path, map_location=device)
                    trainer.models[i].load_state_dict(state_dict)
                    trainer.models[i].to(device)
                    trainer.models[i].eval()
            elif (trainer_type == "coteaching" or 
                  trainer_type == "coteaching_asym" or 
                  trainer_type == "stochastic_coteaching"):
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
        self._ensure_discovery()
        if self._visits is None:
            all_visits = set(self._cutout_loaders.keys()) | set(self._feature_loaders.keys())
            self._visits = sorted(all_visits)
        return self._visits
    
    @property
    def cutouts(self) -> Dict[int, CutoutLoader]:
        """Access cutout loaders by visit."""
        self._ensure_discovery()
        return self._cutout_loaders
    
    @property
    def features(self) -> Dict[int, FeatureLoader]:
        """Access feature loaders by visit."""
        self._ensure_discovery()
        return self._feature_loaders
    
    @property
    def lightcurves(self):
        """Access lightcurve loaders by path.
        
        Returns:
            LightCurveLoader: If only one data path exists
            Dict[str, LightCurveLoader]: If multiple data paths exist (keyed by path string)
            
        Note:
            When multiple loaders exist, you can:
            - Access specific loader: dataset.lightcurves[path]
            - Get first loader: list(dataset.lightcurves.values())[0]
            - Use get_primary_lightcurve_loader() for the main loader
        """
        self._ensure_discovery()
        # If only one path, return the loader directly for convenience
        if len(self._lightcurve_loaders) == 1:
            return list(self._lightcurve_loaders.values())[0]
        return self._lightcurve_loaders
    
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

    def get_dataset_statistics(self, detailed: bool = False, plot: bool = False) -> Dict:
        """Get comprehensive statistics about the dataset.
        
        Provides statistics including:
        - Total visits, cutouts, features, lightcurves
        - Percentage of injection vs real sources per visit
        - Cutouts per visit distribution
        - Label distribution across the dataset
        - Inference results summary (if available)
        
        Parameters
        ----------
        detailed : bool, default False
            If True, return per-visit statistics
        plot : bool, default False
            If True, display visualization plots
            
        Returns
        -------
        Dict
            Dictionary containing dataset statistics
        """
        self._ensure_discovery()
        
        print("Computing dataset statistics...")
        
        stats = {
            'summary': {},
            'per_visit': {},
            'labels': {},
            'noise': {},
            'inference': {}
        }
        
        # Summary statistics
        total_cutouts = 0
        total_features = 0
        total_injections = 0
        total_real = 0
        visits_with_cutouts = 0
        visits_with_features = 0
        
        # Noise statistics (when spy_injected is available)
        has_noise_info = False
        total_ground_truth_injections = 0
        total_ground_truth_real = 0
        total_mislabeled = 0  # Injections labeled as real
        
        per_visit_stats = {}
        
        # Compute per-visit statistics
        for visit in self.visits:
            visit_stat = {
                'cutouts': 0,
                'features': 0,
                'injections': 0,
                'real': 0,
                'injection_pct': 0.0,
                'real_pct': 0.0,
                'ground_truth_injections': 0,
                'ground_truth_real': 0,
                'mislabeled': 0,
                'noise_rate': 0.0
            }
            
            # Cutout statistics
            if visit in self._cutout_loaders:
                visit_stat['cutouts'] = len(self._cutout_loaders[visit].ids)
                total_cutouts += visit_stat['cutouts']
                visits_with_cutouts += 1
            
            # Feature statistics with labels
            if visit in self._feature_loaders:
                loader = self._feature_loaders[visit]
                visit_stat['features'] = len(loader.ids)
                total_features += visit_stat['features']
                visits_with_features += 1
                
                # Get labels to count injections vs real
                if loader.labels is not None:
                    labels = loader.labels
                    # Use .item() for numpy arrays or proper int conversion for Series
                    injections_count = (labels == 1).sum()
                    real_count = (labels == 0).sum()
                    visit_stat['injections'] = int(injections_count.item()) if hasattr(injections_count, 'item') else int(injections_count)
                    visit_stat['real'] = int(real_count.item()) if hasattr(real_count, 'item') else int(real_count)
                    total_injections += visit_stat['injections']
                    total_real += visit_stat['real']
                    
                    if visit_stat['features'] > 0:
                        visit_stat['injection_pct'] = (visit_stat['injections'] / visit_stat['features']) * 100
                        visit_stat['real_pct'] = (visit_stat['real'] / visit_stat['features']) * 100
                
                # Check for spy_injected (ground truth) column to compute noise statistics
                try:
                    if loader._check_table_format():
                        with pd.HDFStore(loader.file_path, 'r') as store:
                            # Check if spy_injected column exists
                            storer = store.get_storer('features')
                            if hasattr(storer, 'attrs') and hasattr(storer.attrs, 'data_columns'):
                                data_cols = storer.attrs.data_columns
                            else:
                                # Try to get column names from a small sample
                                sample = store.select('features', start=0, stop=1)
                                data_cols = sample.columns.tolist()
                            
                            if 'spy_injected' in data_cols:
                                has_noise_info = True
                                spy_labels = store.select('features', columns=['spy_injected', 'is_injection'])
                                
                                # Ground truth counts
                                gt_inj_count = (spy_labels['spy_injected'] == 1).sum()
                                gt_real_count = (spy_labels['spy_injected'] == 0).sum()
                                visit_stat['ground_truth_injections'] = int(gt_inj_count.item()) if hasattr(gt_inj_count, 'item') else int(gt_inj_count)
                                visit_stat['ground_truth_real'] = int(gt_real_count.item()) if hasattr(gt_real_count, 'item') else int(gt_real_count)
                                total_ground_truth_injections += visit_stat['ground_truth_injections']
                                total_ground_truth_real += visit_stat['ground_truth_real']
                                
                                # Mislabeled: spy_injected=1 but is_injection=0 (injection labeled as real)
                                mislabeled_mask = (spy_labels['spy_injected'] == 1) & (spy_labels['is_injection'] == 0)
                                mislabeled_count = mislabeled_mask.sum()
                                visit_stat['mislabeled'] = int(mislabeled_count.item()) if hasattr(mislabeled_count, 'item') else int(mislabeled_count)
                                total_mislabeled += visit_stat['mislabeled']
                                
                                # Noise rate in the "real" class for this visit
                                if visit_stat['real'] > 0:
                                    visit_stat['noise_rate'] = (visit_stat['mislabeled'] / visit_stat['real']) * 100
                except Exception as e:
                    # If we can't read spy_injected, just continue without noise stats
                    pass
            
            per_visit_stats[visit] = visit_stat
        
        # Overall summary
        stats['summary'] = {
            'num_visits': len(self.visits),
            'num_data_paths': len(self.data_paths),
            'total_cutouts': total_cutouts,
            'total_features': total_features,
            'visits_with_cutouts': visits_with_cutouts,
            'visits_with_features': visits_with_features,
            'avg_cutouts_per_visit': total_cutouts / visits_with_cutouts if visits_with_cutouts > 0 else 0,
            'avg_features_per_visit': total_features / visits_with_features if visits_with_features > 0 else 0,
        }
        
        # Label distribution (working labels - potentially noisy)
        stats['labels'] = {
            'total_injections': total_injections,
            'total_real': total_real,
            'total_labeled': total_injections + total_real,
            'injection_pct': (total_injections / (total_injections + total_real) * 100) if (total_injections + total_real) > 0 else 0,
            'real_pct': (total_real / (total_injections + total_real) * 100) if (total_injections + total_real) > 0 else 0,
        }
        
        # Noise statistics (if spy_injected column is available)
        if has_noise_info:
            noise_rate_in_real = (total_mislabeled / total_real * 100) if total_real > 0 else 0
            noise_rate_overall = (total_mislabeled / (total_injections + total_real) * 100) if (total_injections + total_real) > 0 else 0
            
            stats['noise'] = {
                'has_noise_perturbation': True,
                'ground_truth_injections': total_ground_truth_injections,
                'ground_truth_real': total_ground_truth_real,
                'total_mislabeled': total_mislabeled,
                'noise_rate_in_real_class': noise_rate_in_real,
                'noise_rate_overall': noise_rate_overall,
                'label_agreement_rate': 100.0 - noise_rate_overall,
            }
        else:
            stats['noise'] = {
                'has_noise_perturbation': False,
            }
        
        # Lightcurve statistics
        total_lightcurve_objects = 0
        for loader in self._lightcurve_loaders.values():
            if hasattr(loader, 'index') and loader.index is not None:
                total_lightcurve_objects += len(loader.index)
                break  # Just count once since all paths should have same objects
        
        stats['summary']['total_lightcurve_objects'] = total_lightcurve_objects
        
        # Inference statistics (if available)
        if self._inference_registry:
            total_inference_results = 0
            models_used = set()
            
            for data_path_str, visits_data in self._inference_registry.items():
                for visit, models in visits_data.items():
                    total_inference_results += len(models)
                    models_used.update(models.keys())
            
            stats['inference'] = {
                'total_inference_files': total_inference_results,
                'num_models': len(models_used),
                'visits_with_inference': len([v for v in self._inference_registry.values() for _ in v]),
            }
        
        if detailed:
            stats['per_visit'] = per_visit_stats
        
        # Print summary
        self._print_statistics_summary(stats)
        
        # Optionally plot
        if plot:
            self._plot_statistics(stats)
        
        return stats
    
    def _print_statistics_summary(self, stats: Dict):
        """Print formatted statistics summary."""
        print("\n" + "="*70)
        print("DATASET STATISTICS SUMMARY")
        print("="*70)
        
        summary = stats['summary']
        print(f"\nData Overview:")
        print(f"  Total visits: {summary['num_visits']}")
        print(f"  Data paths: {summary['num_data_paths']}")
        print(f"  Visits with cutouts: {summary['visits_with_cutouts']}")
        print(f"  Visits with features: {summary['visits_with_features']}")
        
        print(f"\nCutout Statistics:")
        print(f"  Total cutouts: {summary['total_cutouts']:,}")
        print(f"  Average per visit: {summary['avg_cutouts_per_visit']:.1f}")
        
        print(f"\nFeature Statistics:")
        print(f"  Total features: {summary['total_features']:,}")
        print(f"  Average per visit: {summary['avg_features_per_visit']:.1f}")
        
        labels = stats['labels']
        if labels['total_labeled'] > 0:
            print(f"\nLabel Distribution (Working Labels):")
            print(f"  Total labeled sources: {labels['total_labeled']:,}")
            print(f"  Injections: {labels['total_injections']:,} ({labels['injection_pct']:.1f}%)")
            print(f"  Real sources: {labels['total_real']:,} ({labels['real_pct']:.1f}%)")
        
        # Noise statistics (if available)
        noise = stats.get('noise', {})
        if noise.get('has_noise_perturbation', False):
            print(f"\nNoise Perturbation Statistics:")
            print(f"  Ground Truth Distribution:")
            print(f"    True injections: {noise['ground_truth_injections']:,}")
            print(f"    True real sources: {noise['ground_truth_real']:,}")
            print(f"  Label Noise:")
            print(f"    Mislabeled samples: {noise['total_mislabeled']:,} (injections labeled as real)")
            print(f"    Noise rate in 'real' class: {noise['noise_rate_in_real_class']:.2f}%")
            print(f"    Overall noise rate: {noise['noise_rate_overall']:.2f}%")
            print(f"    Label agreement rate: {noise['label_agreement_rate']:.2f}%")
        
        if summary.get('total_lightcurve_objects', 0) > 0:
            print(f"\nLightcurve Statistics:")
            print(f"  Total objects: {summary['total_lightcurve_objects']:,}")
        
        if stats.get('inference') and stats['inference']:
            inf = stats['inference']
            print(f"\nInference Results:")
            print(f"  Inference files: {inf['total_inference_files']}")
            print(f"  Models used: {inf['num_models']}")
            print(f"  Visits with inference: {inf['visits_with_inference']}")
        
        print("="*70 + "\n")
    
    def _plot_statistics(self, stats: Dict):
        """Create visualization plots for statistics."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if 'per_visit' not in stats or not stats['per_visit']:
            print("Detailed per-visit statistics required for plotting. Run with detailed=True")
            return
        
        per_visit = stats['per_visit']
        visits = sorted(per_visit.keys())
        
        # Prepare data for plotting
        cutouts_per_visit = [per_visit[v]['cutouts'] for v in visits]
        features_per_visit = [per_visit[v]['features'] for v in visits]
        injection_pct = [per_visit[v]['injection_pct'] for v in visits if per_visit[v]['features'] > 0]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Cutouts per visit
        ax1 = axes[0, 0]
        ax1.bar(range(len(visits)), cutouts_per_visit, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Visit Index')
        ax1.set_ylabel('Number of Cutouts')
        ax1.set_title('Cutouts per Visit')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Features per visit
        ax2 = axes[0, 1]
        ax2.bar(range(len(visits)), features_per_visit, alpha=0.7, color='coral')
        ax2.set_xlabel('Visit Index')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Features per Visit')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Injection percentage distribution
        ax3 = axes[1, 0]
        if injection_pct:
            ax3.hist(injection_pct, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(np.mean(injection_pct), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(injection_pct):.1f}%')
            ax3.set_xlabel('Injection Percentage (%)')
            ax3.set_ylabel('Number of Visits')
            ax3.set_title('Distribution of Injection Percentage per Visit')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Overall label distribution pie chart
        ax4 = axes[1, 1]
        labels_data = stats['labels']
        if labels_data['total_labeled'] > 0:
            sizes = [labels_data['total_injections'], labels_data['total_real']]
            labels = [f"Injections\n{labels_data['total_injections']:,}\n({labels_data['injection_pct']:.1f}%)",
                     f"Real\n{labels_data['total_real']:,}\n({labels_data['real_pct']:.1f}%)"]
            colors = ['#ff9999', '#66b3ff']
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Overall Label Distribution')
        
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        if not self._discovery_done:
            return (f"DatasetLoader({len(self.data_paths)} paths)\n"
                   f"  Data discovery: Not yet performed (will be done on first access)\n"
                   f"  Use .visits, .cutouts, or .features properties to trigger discovery")
        
        total_cutouts = 0
        total_features = 0
        visits_with_cutouts = 0
        visits_with_features = 0
        total_lightcurve_objects = 0
        
        # Only compute stats if discovery has been done
        for visit in self._visits or []:
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
        
        num_visits = len(self._visits) if self._visits else 0
        lc_info = f"  Lightcurves: {total_lightcurve_objects} objects" if total_lightcurve_objects > 0 else "  Lightcurves: Not available"
        
        return (f"DatasetLoader({num_visits} visits, {len(self.data_paths)} paths)\n"
                f"  Cutouts: {total_cutouts} across {visits_with_cutouts} visits\n"
                f"  Features: {total_features} across {visits_with_features} visits\n"
                f"{lc_info}")

    def __str__(self):
        return self.__repr__()

    def load_crossmatch_data(self, data_path: Path = None, catalog_name: str = None) -> Optional[pd.DataFrame]:
        """
        Load cross-match results for diaObjectIds.
        
        Parameters
        ----------
        data_path : Path, optional
            Specific data path to use. If None, uses first available.
            
        Returns
        -------
        pd.DataFrame or None
            Cross-match results with diaObjectId and catalog flags
        """
        if self._crossmatch_data is not None:
            return self._crossmatch_data
            
        if not data_path:
            data_path = self.data_paths[0]
            
        crossmatch_file = data_path / "crossmatch" / f"crossmatch_{catalog_name}.h5"
        
        if not crossmatch_file.exists():
            print(f"No cross-match results found at {crossmatch_file}")
            return None
            
        try:
            self._crossmatch_data = pd.read_hdf(crossmatch_file, key='crossmatch')
            print(f"Loaded cross-match data for {len(self._crossmatch_data)} objects")
            return self._crossmatch_data
        except Exception as e:
            print(f"Error loading cross-match data: {e}")
            return None
    
    def get_crossmatch_info(self, dia_object_id: int, data_path: Path = None, catalog_name: str = None) -> Optional[Dict]:
        """
        Get cross-match information for a specific diaObjectId.
        
        Parameters
        ----------
        dia_object_id : int
            The diaObjectId to query
        data_path : Path, optional
            Specific data path to use
            
        Returns
        -------
        Dict or None
            Cross-match information including catalog flags
        """
        crossmatch_data = self.load_crossmatch_data(data_path, catalog_name)
        
        if crossmatch_data is None:
            return None
            
        match_row = crossmatch_data[crossmatch_data['diaObjectId'] == dia_object_id]
        
        if len(match_row) == 0:
            return None
            
        return match_row.iloc[0].to_dict()
    
    def filter_by_crossmatch(self, catalog_name: str, matched: bool = True, 
                           data_path: Path = None) -> List[int]:
        """
        Get list of diaObjectIds based on cross-match results.
        
        Parameters
        ----------
        catalog_name : str
            Name of catalog to filter by (e.g., 'gaia')
        matched : bool
            If True, return objects matched to catalog. If False, return unmatched.
        data_path : Path, optional
            Specific data path to use
            
        Returns
        -------
        List[int]
            List of diaObjectIds matching the filter criteria
        """
        crossmatch_data = self.load_crossmatch_data(data_path, catalog_name)
        
        if crossmatch_data is None:
            print("No cross-match data available")
            return []
            
        flag_column = f'in_{catalog_name}'
        
        if flag_column not in crossmatch_data.columns:
            print(f"No cross-match results for catalog '{catalog_name}'")
            available_catalogs = [col.replace('in_', '') for col in crossmatch_data.columns 
                                if col.startswith('in_')]
            print(f"Available catalogs: {available_catalogs}")
            return []
            
        if matched:
            filtered_data = crossmatch_data[crossmatch_data[flag_column] == True]
        else:
            filtered_data = crossmatch_data[crossmatch_data[flag_column] == False]
            
        return filtered_data['diaObjectId'].tolist()
    
    def get_crossmatch_summary(self, data_path: Path = None) -> Optional[Dict]:
        """
        Get summary statistics of cross-match results.
        
        Parameters
        ----------
        data_path : Path, optional
            Specific data path to use
            
        Returns
        -------
        Dict or None
            Summary statistics including match counts and rates
        """
        if not data_path:
            data_path = self.data_paths[0]
            
        summary_file = data_path / "crossmatch" / "crossmatch_summary.pkl"
        
        if not summary_file.exists():
            # Generate summary from data if summary file doesn't exist
            crossmatch_data = self.load_crossmatch_data(data_path)
            if crossmatch_data is None:
                return None
                
            summary = {'total_objects': len(crossmatch_data)}
            
            # Calculate match rates for each catalog
            for col in crossmatch_data.columns:
                if col.startswith('in_'):
                    catalog_name = col.replace('in_', '')
                    match_count = crossmatch_data[col].sum()
                    match_rate = match_count / len(crossmatch_data) * 100
                    summary[f'{catalog_name}_matches'] = match_count
                    summary[f'{catalog_name}_match_rate_percent'] = match_rate
                    
            return summary
            
        try:
            import pickle
            with open(summary_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cross-match summary: {e}")
            return None

    def extract_coordinates(self) -> pd.DataFrame:
        """
        Extract unique coordinates (ra, dec) for all diaObjectIds from lightcurve data.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns [diaObjectId, coord_ra, coord_dec]
            One row per unique diaObjectId with representative coordinates.
        """
        self._ensure_discovery()
        
        if not hasattr(self, 'lightcurves') or not self.lightcurves:
            raise ValueError("No lightcurve data available for coordinate extraction")
        
        print("Extracting coordinates from lightcurve data...")
        
        all_coords = []
        
        # Get list of available patches
        available_patches = self.lightcurves.list_available_patches()
        
        if not available_patches:
            raise ValueError("No lightcurve patch files found")
        
        print(f"Found {len(available_patches)} patches to process")
        
        # Process each patch
        for i, patch_id in enumerate(available_patches):
            if i % 10 == 0:
                print(f"  Processing patch {i+1}/{len(available_patches)}: {patch_id}")
            
            try:
                patch_data = self.lightcurves._load_patch_data(patch_id)
                
                if patch_data is None or patch_data.empty:
                    continue
                
                # Check for required columns
                if 'diaObjectId' not in patch_data.columns:
                    print(f"    Warning: No diaObjectId column in patch {patch_id}")
                    continue
                
                # Try coord_ra/coord_dec first, then fallback to ra/dec
                ra_col = 'coord_ra' if 'coord_ra' in patch_data.columns else 'ra'
                dec_col = 'coord_dec' if 'coord_dec' in patch_data.columns else 'dec'
                
                if ra_col not in patch_data.columns or dec_col not in patch_data.columns:
                    print(f"    Warning: No coordinate columns in patch {patch_id}")
                    continue
                
                # Get unique coordinates per diaObjectId (taking first occurrence)
                coords_subset = patch_data[['diaObjectId', ra_col, dec_col]].drop_duplicates(
                    subset=['diaObjectId'], keep='first'
                )
                
                # Rename columns to standard names
                coords_subset = coords_subset.rename(columns={
                    ra_col: 'coord_ra',
                    dec_col: 'coord_dec'
                })
                
                all_coords.append(coords_subset)
                
            except Exception as e:
                print(f"    Error processing patch {patch_id}: {e}")
                continue
        
        if not all_coords:
            raise RuntimeError("No coordinate data extracted from any patch")
        
        # Combine all coordinates
        coords_df = pd.concat(all_coords, ignore_index=True)
        
        # Remove duplicates across patches (keep first occurrence)
        coords_df = coords_df.drop_duplicates(subset=['diaObjectId'], keep='first')
        
        # Ensure diaObjectId is int64
        coords_df['diaObjectId'] = coords_df['diaObjectId'].astype(np.int64)
        
        print(f"Extracted coordinates for {len(coords_df)} unique diaObjectIds")
        return coords_df

    def perform_crossmatch(self, catalog_file: str, catalog_name: str = None,
                          ra_col: str = 'ra', dec_col: str = 'dec',
                          tolerance_arcsec: float = 1.0,
                          output_file: str = None) -> pd.DataFrame:
        """
        Perform cross-matching with an external catalog at the DatasetLoader level.
        
        Parameters
        ----------
        catalog_file : str
            Path to catalog file (.pkl, .csv, .h5)
        catalog_name : str, optional
            Name for the catalog (default: use filename)
        ra_col : str, default 'ra'
            RA column name in external catalog
        dec_col : str, default 'dec'
            Dec column name in external catalog
        tolerance_arcsec : float, default 1.0
            Matching tolerance in arcseconds
        output_file : str, optional
            Output file path for results
            
        Returns
        -------
        pd.DataFrame
            Cross-match results with columns [diaObjectId, coord_ra, coord_dec, in_{catalog_name}]
        """
        # Import cross-matching functionality
        from ..data_preparation.crossmatch import CrossMatcher
        
        if catalog_name is None:
            catalog_name = Path(catalog_file).stem
        
        print(f"=== Cross-matching with {catalog_name} ===")
        
        # Extract coordinates from lightcurve data
        coords_df = self.extract_coordinates()
        
        # Initialize CrossMatcher
        crossmatcher = CrossMatcher({})
        
        # Load external catalog
        print(f"Loading external catalog: {catalog_file}")
        crossmatcher.load_catalog(catalog_name, catalog_file, ra_col, dec_col)
        
        # Perform cross-matching
        results = crossmatcher.crossmatch_objects(
            coords_df, catalog_name, tolerance_arcsec
        )
        
        # Save results if output file specified
        if output_file:
            output_path = Path(output_file)
            if not output_path.is_absolute():
                # Save relative to first data path
                output_path = self.data_paths[0] / output_file
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crossmatcher.save_crossmatch_results(results, output_path)
            
            print(f"Results saved to: {output_path}")
        
        # Cache results for later use
        self._crossmatch_data = results
        
        return results
