import yaml
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

        self._discover_data()

    def _build_global_index(self):
        """Build a global index mapping diaSourceId to visit."""
        if self._global_id_index is not None:
            return
        
        print("Building global index...")
        self._global_id_index = {}
        
        # Index feature
        for visit, loader in self._feature_loaders.items():
            for dia_id in loader.ids:
                self._global_id_index[dia_id] = visit
                        
    @property
    def global_index(self):
        """Lazy build and return global ID index."""
        if self._global_id_index is None:
            self._build_global_index()
        return self._global_id_index
    
    def get_cutout_by_id(self, dia_source_id: int):
        """Get cutout by diaSourceId (searches all visits)."""
        visit = self.global_index.get(dia_source_id)
        if visit and visit in self._cutout_loaders:
            return self._cutout_loaders[visit].get_by_id(dia_source_id)
        return None
    
    def get_features_by_id(self, dia_source_id: int):
        """Get features by diaSourceId (searches all visits)."""
        visit = self.global_index.get(dia_source_id)
        if visit and visit in self._feature_loaders:
            return self._feature_loaders[visit].get_by_id(dia_source_id)
        return None
    
    def find_visit(self, dia_source_id: int) -> Optional[int]:
        """Find which visit contains this diaSourceId."""
        return self.global_index.get(dia_source_id)

    def get_inference_loader(self, weights_path: str, visit: int, data_path: Path = None):
        """
        Get or create an InferenceLoader for the given weights and visit.
        
        Args:
            weights_path: Path to trained model weights directory
            visit: Visit number
            data_path: Specific data path to use (defaults to first data path)
            
        Returns:
            InferenceLoader: Configured inference loader for the visit
        """
        if not data_path:
            data_path = self.data_paths[0]
        
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
        trainer = self._get_or_load_trainer(weights_path)
        
        results = {}
        
        for i, visit in enumerate(self.visits):
            print(f"\n--- Processing visit {visit} ({i+1}/{len(self.visits)}) ---")
            inference_loader = self.get_inference_loader(weights_path, visit, data_path)
            
            if inference_loader.has_inference_results() and not force:
                print(f"Inference results already exist for visit {visit}")
                results[visit] = inference_loader
                continue
            
            try:
                # Pass the cached trainer to avoid reloading
                inference_loader.run_inference(self, trainer=trainer, force=force)
                results[visit] = inference_loader
                print(f"Completed inference for visit {visit}")
                
                # Force memory cleanup after each visit
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"Error running inference for visit {visit}: {e}")
                continue
        
        print(f"\nCompleted inference for {len(results)} visits")
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
            inference_loader = self.get_inference_loader(weights_path, visit, data_path)
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

            
            # Load config summary if available
            config_file = data_path / "config_summary.yaml"
            if config_file.exists() and self._config_summary is None:
                with open(config_file) as f:
                    self._config_summary = yaml.safe_load(f)
    @property
    def config_summary(self) -> Optional[Dict]:
        """Get config summary if available."""
        return self._config_summary
        
    def _extract_visit_from_filename(self, filename: str) -> Optional[int]:
        """Extract visit number from filename like 'visit_12345.h5'."""
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
    def lightcurves(self) -> Dict[int, LightCurveLoader]:
        """Access lightcurve loaders by visit (placeholder)."""
        return self._lightcurve_loaders
    
    @property
    def inference(self) -> Dict[str, InferenceLoader]:
        """Access inference loaders by cache key."""
        return self._inference_loaders
    
    @property
    def config_summary(self) -> Optional[Dict]:
        """Get config summary if available."""
        return self._config_summary
    
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

    def get_inference_results_by_id(self, dia_source_id: int, weights_path: str):
        """Get inference results by diaSourceId across all visits."""
        visit = self.find_visit(dia_source_id)
        if visit:
            inference_loader = self.get_inference_loader(weights_path, visit)
            if inference_loader.has_inference_results():
                return inference_loader.get_results_by_id(dia_source_id)
        return None

    def _get_or_load_trainer(self, weights_path: str):
        """Get cached trainer or load it if not cached."""
        if weights_path not in self._cached_trainers:
            print(f"Loading model from {weights_path}...")
            
            # Import here to avoid circular imports
            from ML4transients.training import get_trainer
            from ML4transients.utils import load_config
            import torch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config = load_config(f"{weights_path}/config.yaml")
            trainer = get_trainer(config["training"]["trainer_type"], config["training"])
            state_dict = torch.load(f"{weights_path}/model_best.pth", map_location=device)
            trainer.model.load_state_dict(state_dict)
            trainer.model.to(device)
            trainer.model.eval()
            
            self._cached_trainers[weights_path] = trainer
            print(f"Model loaded and cached for {weights_path}")
        
        return self._cached_trainers[weights_path]

    def __repr__(self):
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

    def __str__(self):
        return self.__repr__()
