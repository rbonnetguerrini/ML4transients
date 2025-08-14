import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import yaml
import torch  

class CutoutLoader:
    """Lazy loader for cutout data.
    
    Provides efficient access to astronomical image cutouts stored in HDF5 format.
    Data is loaded on-demand to minimize memory usage.
    """
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
        self._ids = None
    
    @property
    def data(self):
        """Lazy load cutout arrays."""
        if self._data is None:
            with h5py.File(self.file_path, 'r') as f:
                self._data = f['cutouts'][:]
        return self._data
    
    @property
    def ids(self):
        """Lazy load diaSourceIds."""
        if self._ids is None:
            with h5py.File(self.file_path, 'r') as f:
                self._ids = f['diaSourceId'][:]
        return self._ids
    
    def get_by_id(self, dia_source_id: int):
        """Get specific cutout by diaSourceId.
        
        Parameters
        ----------
        dia_source_id : int
            The diaSourceId to retrieve
            
        Returns
        -------
        np.ndarray or None
            Cutout array, or None if ID not found
        """
        idx = np.where(self.ids == dia_source_id)[0]
        if len(idx) == 0:
            return None
        return self.data[idx[0]]

class FeatureLoader:
    """Lazy loader for feature data stored in HDF5/Pandas format.
    
    Provides efficient access to tabular features with on-demand loading
    of specific columns or rows to minimize memory usage.
    """
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
        self._ids = None
        self._labels = None
    
    @property
    def ids(self):
        """Get diaSourceIds without loading full data."""
        if self._ids is None:
            with pd.HDFStore(self.file_path, 'r') as store:
                self._ids = store.select_column('features', 'index')
        return self._ids
    
    @property
    def labels(self):
        """Get only labels without loading all features."""
        if self._labels is None:
            try:
                with pd.HDFStore(self.file_path, 'r') as store:
                    # Load only the is_injection column
                    self._labels = store.select('features', columns=['is_injection'])
            except (ValueError, TypeError):
                # Fall back for fixed format
                print(f"Warning: Loading full features for labels from {self.file_path}")
                self._labels = self.data[['is_injection']]
        return self._labels
    
    def get_by_id(self, dia_source_id: int):
        """Get specific features by diaSourceId.
        
        Parameters
        ----------
        dia_source_id : int
            The diaSourceId to retrieve
            
        Returns
        -------
        pd.DataFrame
            Features for the specified ID
        """
        with pd.HDFStore(self.file_path, 'r') as store:
            # Efficient query without loading full data
            return store.select('features', where=f'index == {dia_source_id}')
    
    def get_object_id(self):
        """
        Get only diaObjectId without loading all features.
            
        Returns
        -------
        pd.DataFrame
            diaObjectId 
        """

        with pd.HDFStore(self.file_path, 'r') as store:
            # Load only the diaObjectId column
            dia_obj = store.select('features', columns=['diaObjectId'])
        return dia_obj

            
class InferenceLoader:
    """Handles inference results loading and running inference on datasets."""
    
    def __init__(self, data_path: Path, visit: int, weights_path: str = None):
        """
        Initialize InferenceLoader.
        
        Args:
            data_path: Path to data directory
            visit: Visit number
            weights_path: Path to model weights (optional, for running inference)
        """
        self.data_path = Path(data_path)
        self.visit = visit
        self.weights_path = weights_path
        
        # Generate model hash from weights path if provided
        self.model_hash = None
        if weights_path:
            import hashlib
            self.model_hash = hashlib.md5(str(weights_path).encode()).hexdigest()[:8]
        
        # Set up inference file path
        self.inference_dir = self.data_path / "inference"
        self.inference_dir.mkdir(exist_ok=True)
        
        if self.model_hash:
            self._inference_file = self.inference_dir / f"visit_{visit}_inference_{self.model_hash}.h5"
        else:
            self._inference_file = None
            
        # Lazy loading attributes
        self._predictions = None
        self._labels = None
        self._probabilities = None
        self._uncertainties = None
        self._ids = None

    def run_inference(self, dataset_loader, trainer=None, force=False):
        """
        Run inference for this visit and save results.
        
        Args:
            dataset_loader: DatasetLoader instance
            trainer: Pre-loaded trainer (optional, for efficiency)
            force: Force re-run even if results exist
        """
        if not force and self.has_inference_results():
            print(f"Inference results already exist for visit {self.visit}")
            return
        
        if not self.weights_path:
            raise ValueError("weights_path required for running inference")
            
        if not self._inference_file:
            raise ValueError("Cannot determine inference file path without model hash")
        
        print(f"Running inference for visit {self.visit}...")
        
        # Create inference dataset for this visit
        from ML4transients.training.pytorch_dataset import PytorchDataset
        
        print(f"Creating inference dataset for visit {self.visit}...")
        inference_dataset = PytorchDataset.create_inference_dataset(dataset_loader, visit=self.visit)
        print(f"Created inference dataset for visit {self.visit} with {len(inference_dataset)} samples")
        
        # Create DataLoader
        inference_loader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
        
        print(f"Created lazy dataset for visit {self.visit} with {len(inference_dataset)} samples")
        print(f"Processing {len(inference_dataset)} samples in batches of 128...")
        
        # Import inference function
        from ML4transients.evaluation.inference import infer
        
        # Get diaSourceIds for saving
        dia_source_ids = inference_dataset.get_dia_source_ids()
        
        # Run inference with proper save_path - THIS IS THE KEY FIX
        results = infer(
            inference_loader=inference_loader,
            trainer=trainer,
            weights_path=self.weights_path,
            return_preds=True,
            compute_metrics=True,
            device=None,
            save_path=str(self._inference_file),  # Make sure to pass save_path!
            dia_source_ids=dia_source_ids,
            visit=self.visit,
            model_hash=self.model_hash,
            return_probabilities=None  # Auto-detect based on model type
        )
        
        if results:
            print(f"Inference completed for visit {self.visit}")
            # Clear cached data to force reload from saved file
            self._clear_cache()
        else:
            print(f"Warning: No inference results returned for visit {self.visit}")
            
        # Cleanup memory
        print("Cleaned up memory for dataset")
        del inference_dataset, inference_loader
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _clear_cache(self):
        """Clear cached data to force reload from file."""
        self._predictions = None
        self._labels = None
        self._probabilities = None
        self._uncertainties = None
        self._ids = None

    def has_inference_results(self) -> bool:
        """Check if inference results exist."""
        if not self._inference_file:
            return False
        return self._inference_file.exists()

    def _load_data(self):
        """Load inference data from HDF5 file."""
        if not self.has_inference_results():
            raise FileNotFoundError(f"No inference results found for visit {self.visit}")
        
        try:
            with h5py.File(self._inference_file, 'r') as f:
                self._predictions = f['predictions'][:]
                self._labels = f['labels'][:]
                
                # Load probabilities and uncertainties if available
                if 'probabilities' in f:
                    self._probabilities = f['probabilities'][:]
                if 'uncertainties' in f:
                    self._uncertainties = f['uncertainties'][:]
                    
                if 'diaSourceId' in f:
                    self._ids = f['diaSourceId'][:]
                else:
                    # Fallback: generate sequential IDs
                    self._ids = np.arange(len(self._predictions))
                    
        except Exception as e:
            print(f"Error loading inference results for visit {self.visit}: {e}")
            raise

    @property
    def predictions(self) -> np.ndarray:
        """Get predictions array."""
        if self._predictions is None:
            self._load_data()
        return self._predictions

    @property
    def labels(self) -> np.ndarray:
        """Get labels array."""
        if self._labels is None:
            self._load_data()
        return self._labels
        
    @property
    def probabilities(self) -> Optional[np.ndarray]:
        """Get probabilities array (if available)."""
        if self._probabilities is None and self.has_inference_results():
            self._load_data()
        return self._probabilities
        
    @property
    def uncertainties(self) -> Optional[np.ndarray]:
        """Get uncertainties array (if available)."""
        if self._uncertainties is None and self.has_inference_results():
            self._load_data()
        return self._uncertainties

    @property
    def ids(self) -> np.ndarray:
        """Get diaSourceId array."""
        if self._ids is None:
            self._load_data()
        return self._ids

    def get_results_by_id(self, dia_source_id: int) -> Optional[Dict]:
        """Get inference results for a specific diaSourceId."""
        try:
            idx = np.where(self.ids == dia_source_id)[0]
            if len(idx) == 0:
                return None
            
            idx = idx[0]
            result = {
                'prediction': self.predictions[idx],
                'label': self.labels[idx],
                'dia_source_id': dia_source_id
            }
            
            if self.probabilities is not None:
                result['probability'] = self.probabilities[idx]
            if self.uncertainties is not None:
                result['uncertainty'] = self.uncertainties[idx]
                
            return result
        except Exception as e:
            print(f"Error getting results for ID {dia_source_id}: {e}")
            return None

class LightCurveLoader:
    """Lightcurve loader with patch-based caching and cross-reference index."""
    
    def __init__(self, lightcurve_path: Path):
        self.lightcurve_path = Path(lightcurve_path)
        self._index = None
        self._diasource_index = None
        self._patch_cache = {}  # Cache loaded patch data
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 10  # Configurable cache limit
    
    @property
    def index(self) -> pd.DataFrame:
        """Lazy load the lightcurve index."""
        if self._index is None:
            index_file = self.lightcurve_path / "lightcurve_index.h5"
            if index_file.exists():
                self._index = pd.read_hdf(index_file, key="index")
                print(f"Loaded lightcurve index with {len(self._index)} objects")
            else:
                raise FileNotFoundError(f"Lightcurve index not found at {index_file}")
        return self._index
    
    @property
    def diasource_index(self) -> pd.DataFrame:
        """Lazy load the diaSourceId→patch index."""
        if self._diasource_index is None:
            index_file = self.lightcurve_path / "diasource_patch_index.h5"
            if index_file.exists():
                self._diasource_index = pd.read_hdf(index_file, key="diasource_index")
            else:
                self._diasource_index = pd.DataFrame()  # Empty fallback
        return self._diasource_index
    
    def find_patch_by_source_id(self, dia_source_id: int) -> Optional[str]:
        """Find which patch contains the given diaSourceId."""
        if self.diasource_index.empty:
            return None
        
        try:
            result = self.diasource_index.loc[self.diasource_index.index == dia_source_id]
            if len(result) > 0:
                return result.iloc[0]['patch_key']
        except (KeyError, IndexError):
            pass
        return None
    
    def get_lightcurve_by_source_id(self, dia_source_id: int) -> Optional[pd.DataFrame]:
        """
        Get full lightcurve for a diaObjectId given any diaSourceId from that lightcurve.
        This is the key function for your use case!
        """
        if self.diasource_index.empty:
            return None
        
        try:
            # Find the diaObjectId and patch for this diaSourceId
            source_info = self.diasource_index.loc[self.diasource_index.index == dia_source_id]
            if len(source_info) == 0:
                return None
            
            dia_object_id = source_info.iloc[0]['diaObjectId']
            patch_key = source_info.iloc[0]['patch_key']
            
            # Load the patch data (with caching)
            if patch_key in self._patch_cache:
                self._cache_hits += 1
                patch_data = self._patch_cache[patch_key]
            else:
                self._cache_misses += 1
                patch_data = self._load_patch_data(patch_key)
                if patch_data is None:
                    return None
                self._manage_cache(patch_key, patch_data)
            
            # Get all sources for this diaObjectId and sort by time
            lc_data = patch_data[patch_data['diaObjectId'] == dia_object_id].copy()
            
            if len(lc_data) > 0 and 'midpointMjdTai' in lc_data.columns:
                lc_data = lc_data.sort_values('midpointMjdTai').reset_index(drop=True)
            
            return lc_data if len(lc_data) > 0 else None
            
        except Exception as e:
            return None

    def get_lightcurve(self, dia_object_id: int) -> Optional[pd.DataFrame]:
        """
        Get lightcurve for a specific diaObjectId.
        Uses efficient patch lookup and caching.
        """
        # Find the patch containing this object
        patch_key = self.find_patch(dia_object_id)
        if not patch_key:
            return None
        
        # Check cache first
        if patch_key in self._patch_cache:
            self._cache_hits += 1
            patch_data = self._patch_cache[patch_key]
        else:
            self._cache_misses += 1
            patch_data = self._load_patch_data(patch_key)
            if patch_data is None:
                return None
            
            # Manage cache
            self._manage_cache(patch_key, patch_data)
        
        # Filter for the specific object and sort by time
        lc_data = patch_data[patch_data['diaObjectId'] == dia_object_id].copy()
        if len(lc_data) > 0 and 'midpointMjdTai' in lc_data.columns:
            lc_data = lc_data.sort_values('midpointMjdTai').reset_index(drop=True)
        
        return lc_data if len(lc_data) > 0 else None
    
    def get_multiple_lightcurves(self, dia_object_ids: List[int]) -> Dict[int, pd.DataFrame]:
        """
        Efficiently get lightcurves for multiple objects.
        Groups requests by patch to minimize I/O.
        """
        print(f"Getting lightcurves for {len(dia_object_ids)} objects...")
        
        # Group object IDs by patch efficiently
        patch_mapping = self.find_patches_for_objects(dia_object_ids)
        
        patch_groups = {}
        missing_objects = []
        
        for obj_id in dia_object_ids:
            patch_key = patch_mapping.get(obj_id)
            if patch_key:
                if patch_key not in patch_groups:
                    patch_groups[patch_key] = []
                patch_groups[patch_key].append(obj_id)
            else:
                missing_objects.append(obj_id)
        
        if missing_objects:
            print(f"Warning: {len(missing_objects)} objects not found in index")
        
        results = {}
        total_patches = len(patch_groups)
        
        # Process each patch once
        for i, (patch_key, obj_ids) in enumerate(patch_groups.items()):
            print(f"Processing patch {patch_key} ({i+1}/{total_patches}) with {len(obj_ids)} objects...")
            
            # Load patch data (with caching)
            if patch_key in self._patch_cache:
                self._cache_hits += 1
                patch_data = self._patch_cache[patch_key]
            else:
                self._cache_misses += 1
                patch_data = self._load_patch_data(patch_key)
                if patch_data is None:
                    continue
                
                # Manage cache
                self._manage_cache(patch_key, patch_data)
            
            # Extract lightcurves for all objects in this patch
            for obj_id in obj_ids:
                lc_data = patch_data[patch_data['diaObjectId'] == obj_id].copy()
                if len(lc_data) > 0:
                    # Sort by time if available
                    if 'midpointMjdTai' in lc_data.columns:
                        lc_data = lc_data.sort_values('midpointMjdTai').reset_index(drop=True)
                    results[obj_id] = lc_data
        
        print(f"Retrieved {len(results)} lightcurves from {total_patches} patches")
        return results
    
    def get_lightcurve_stats(self, dia_object_id: int) -> Optional[Dict]:
        """Get basic statistics for a lightcurve without loading full data."""
        lc = self.get_lightcurve(dia_object_id)
        if lc is None or len(lc) == 0:
            return None
        
        stats = {
            'num_points': len(lc),
            'time_span_days': None,
            'bands': [],
            'patch_key': self.find_patch(dia_object_id)
        }
        

        if 'midpointMjdTai' in lc.columns:
            time_span = lc['midpointMjdTai'].max() - lc['midpointMjdTai'].min()
            stats['time_span_days'] = time_span
        
        if 'band' in lc.columns:
            stats['bands'] = lc['band'].unique().tolist()
        
        return stats
    
    def list_available_patches(self) -> List[str]:
        """List all available patch files."""
        patch_files = list(self.lightcurve_path.glob("patch_*.h5"))
        return [f.stem.replace("patch_", "") for f in patch_files]
    
    def clear_cache(self):
        """Clear the patch cache to free memory."""
        self._patch_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def set_cache_size(self, max_size: int):
        """Set maximum cache size."""
        self._max_cache_size = max_size
        # Clear excess entries if needed
        while len(self._patch_cache) > max_size:
            oldest_key = next(iter(self._patch_cache))
            del self._patch_cache[oldest_key]
    
    @property
    def cache_stats(self):
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cached_patches': len(self._patch_cache),
            'cache_size_limit': self._max_cache_size
        }
    
    def get_all_source_ids_in_lightcurve(self, dia_source_id: int) -> List[int]:
        """Get all diaSourceIds that belong to the same lightcurve as the given diaSourceId."""
        if self.diasource_index.empty:
            return []
        
        try:
            # Get the diaObjectId for this source
            source_info = self.diasource_index.loc[self.diasource_index.index == dia_source_id]
            if len(source_info) == 0:
                return []
            
            dia_object_id = source_info.iloc[0]['diaObjectId']
            
            # Find all sources with the same diaObjectId
            all_sources_for_object = self.diasource_index[self.diasource_index['diaObjectId'] == dia_object_id]
            return all_sources_for_object.index.tolist()
            
        except Exception as e:
            return []

    def get_object_id_for_source(self, dia_source_id: int) -> Optional[int]:
        """Get diaObjectId for a given diaSourceId."""
        if self.diasource_index.empty:
            return None
        
        try:
            result = self.diasource_index.loc[self.diasource_index.index == dia_source_id]
            if len(result) > 0:
                return result.iloc[0]['diaObjectId']
        except (KeyError, IndexError):
            pass
        return None

    def find_patch(self, dia_object_id: int) -> Optional[str]:
        """Find which patch contains the given diaObjectId."""
        try:
            result = self.index.loc[self.index.index == dia_object_id]
            if len(result) > 0:
                return result.iloc[0]['patch_key']
        except (KeyError, IndexError):
            pass
        return None
    
    def find_patches_for_objects(self, dia_object_ids: List[int]) -> Dict[int, str]:
        """Efficiently find patches for multiple objects at once."""
        try:
            # Use pandas query for efficiency
            mask = self.index.index.isin(dia_object_ids)
            results = self.index.loc[mask]
            return dict(zip(results.index, results['patch_key']))
        except Exception as e:
            print(f"Error in batch patch lookup: {e}")
            # Fallback to individual lookups
            return {obj_id: self.find_patch(obj_id) for obj_id in dia_object_ids}
    
    def _load_patch_data(self, patch_key: str) -> Optional[pd.DataFrame]:
        """Load lightcurve data for a specific patch."""
        patch_file = self.lightcurve_path / f"patch_{patch_key}.h5"
        
        if patch_file.exists():
            try:
                data = pd.read_hdf(patch_file, key="lightcurves")
                return data
            except Exception as e:
                return None
        else:
            return None
    
    def _manage_cache(self, new_patch_key: str, new_data: pd.DataFrame):
        """Manage cache size by removing least recently used items."""
        if len(self._patch_cache) >= self._max_cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self._patch_cache))
            del self._patch_cache[oldest_key]
        
        self._patch_cache[new_patch_key] = new_data