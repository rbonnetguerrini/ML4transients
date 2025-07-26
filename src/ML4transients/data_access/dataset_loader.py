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
    def inference(self) -> Dict[int, InferenceLoader]:
        """Access inference loaders by visit (placeholder)."""
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
