import h5py
import pandas as pd
import numpy as np
import time
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
        # Try efficient row-based query first
        try:
            with h5py.File(self.file_path, 'r') as f:
                # Get the IDs to find the index
                ids = f['diaSourceId'][:]
                idx = np.where(ids == dia_source_id)[0]
                if len(idx) == 0:
                    return None
                
                # Read only the specific row from cutouts
                cutout = f['cutouts'][idx[0]]
                return cutout
        except Exception as e:
            # Fallback to property-based loading
            idx = np.where(self.ids == dia_source_id)[0]
            if len(idx) == 0:
                return None
            return self.data[idx[0]]

    def get_multiple_by_ids(self, dia_source_ids: List[int]) -> Dict[int, np.ndarray]:
        """Efficiently get multiple cutouts by diaSourceIds from this file.
        
        Parameters
        ----------
        dia_source_ids : List[int]
            List of diaSourceIds to retrieve
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping diaSourceId to cutout array
        """
        if not dia_source_ids:
            return {}
        
        start_time = time.time()
        results = {}
        try:
            file_start = time.time()
            with h5py.File(self.file_path, 'r') as f:
                file_open_time = time.time() - file_start
                
                # Get all IDs once
                ids_start = time.time()
                all_ids = f['diaSourceId'][:]
                ids_time = time.time() - ids_start
                
                # Find indices for all requested IDs
                index_start = time.time()
                indices = []
                id_to_idx = {}
                for target_id in dia_source_ids:
                    idx = np.where(all_ids == target_id)[0]
                    if len(idx) > 0:
                        indices.append(idx[0])
                        id_to_idx[target_id] = len(indices) - 1
                index_time = time.time() - index_start
                
                if indices:
                    # Read only the specific rows we need
                    read_start = time.time()
                    cutouts = f['cutouts'][indices]
                    read_time = time.time() - read_start
                    
                    # Map back to original IDs
                    map_start = time.time()
                    for target_id in dia_source_ids:
                        if target_id in id_to_idx:
                            results[target_id] = cutouts[id_to_idx[target_id]]
                    map_time = time.time() - map_start
                    
                    total_time = time.time() - start_time
                    if total_time > 0.1:  # Only log if operation takes significant time
                        print(f"      Cutout batch load ({len(dia_source_ids)} IDs): {total_time:.3f}s "
                              f"(open: {file_open_time:.3f}s, ids: {ids_time:.3f}s, index: {index_time:.3f}s, "
                              f"read: {read_time:.3f}s, map: {map_time:.3f}s)")
        
        except Exception as e:
            print(f"Error in batch cutout loading: {e}, falling back to individual loads")
            # Fallback to individual queries
            for dia_source_id in dia_source_ids:
                cutout = self.get_by_id(dia_source_id)
                if cutout is not None:
                    results[dia_source_id] = cutout
        
        return results

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
        self._is_table_format = None
    
    def _check_table_format(self):
        """Check if the HDF5 file uses table format (supports queries) or fixed format."""
        if self._is_table_format is None:
            try:
                with pd.HDFStore(self.file_path, 'r') as store:
                    # Try to get info about the features dataset
                    info = store.get_storer('features')
                    self._is_table_format = info is not None and hasattr(info, 'table')
            except:
                self._is_table_format = False
        return self._is_table_format
    
    @property
    def ids(self):
        """Get diaSourceIds without loading full data."""
        if self._ids is None:
            if self._check_table_format():
                with pd.HDFStore(self.file_path, 'r') as store:
                    self._ids = store.select_column('features', 'index')
            else:
                # For fixed format, we need to load the index
                with pd.HDFStore(self.file_path, 'r') as store:
                    self._ids = store['features'].index
        return self._ids
    
    @property
    def labels(self):
        """Get only labels without loading all features."""
        if self._labels is None:
            try:
                if self._check_table_format():
                    with pd.HDFStore(self.file_path, 'r') as store:
                        # Load only the is_injection column
                        self._labels = store.select('features', columns=['is_injection'])
                else:
                    # Fall back for fixed format
                    print(f"Warning: Loading full features for labels from {self.file_path}")
                    self._labels = self.data[['is_injection']]
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
        if self._check_table_format():
            with pd.HDFStore(self.file_path, 'r') as store:
                # Efficient query without loading full data
                return store.select('features', where=f'index == {dia_source_id}')
        else:
            # Fallback for fixed format - load all and filter
            if self._data is None:
                with pd.HDFStore(self.file_path, 'r') as store:
                    self._data = store['features']
            
            if dia_source_id in self._data.index:
                return self._data.loc[[dia_source_id]]
            else:
                return pd.DataFrame()  # Empty DataFrame if not found

    def get_multiple_by_ids(self, dia_source_ids: List[int]) -> Dict[int, pd.DataFrame]:
        """Efficiently get multiple features by diaSourceIds from this file.
        
        Parameters
        ----------
        dia_source_ids : List[int]
            List of diaSourceIds to retrieve
            
        Returns
        -------
        Dict[int, pd.DataFrame]
            Dictionary mapping diaSourceId to features DataFrame
        """
        if not dia_source_ids:
            return {}
        
        start_time = time.time()
        results = {}
        
        if self._check_table_format():
            try:
                # Create WHERE clause for multiple IDs
                query_start = time.time()
                id_list_str = ','.join(map(str, dia_source_ids))
                where_clause = f'index in [{id_list_str}]'
                query_prep_time = time.time() - query_start
                
                file_start = time.time()
                with pd.HDFStore(self.file_path, 'r') as store:
                    file_open_time = time.time() - file_start
                    
                    read_start = time.time()
                    batch_results = store.select('features', where=where_clause)
                    read_time = time.time() - read_start
                
                # Split back into individual DataFrames
                split_start = time.time()
                for dia_source_id in dia_source_ids:
                    if dia_source_id in batch_results.index:
                        results[dia_source_id] = batch_results.loc[[dia_source_id]]
                split_time = time.time() - split_start
                
                total_time = time.time() - start_time
                if total_time > 0.1:  # Only log if operation takes significant time
                    print(f"      Features batch load ({len(dia_source_ids)} IDs): {total_time:.3f}s "
                          f"(query_prep: {query_prep_time:.3f}s, open: {file_open_time:.3f}s, "
                          f"read: {read_time:.3f}s, split: {split_time:.3f}s)")
                
            except Exception as e:
                print(f"Error in batch feature loading: {e}, falling back to individual loads")
                # Fallback to individual queries
                for dia_source_id in dia_source_ids:
                    result = self.get_by_id(dia_source_id)
                    if not result.empty:
                        results[dia_source_id] = result
        else:
            # For fixed format, load once and filter multiple times
            load_start = time.time()
            if self._data is None:
                with pd.HDFStore(self.file_path, 'r') as store:
                    self._data = store['features']
            load_time = time.time() - load_start
            
            filter_start = time.time()
            for dia_source_id in dia_source_ids:
                if dia_source_id in self._data.index:
                    results[dia_source_id] = self._data.loc[[dia_source_id]]
            filter_time = time.time() - filter_start
            
            total_time = time.time() - start_time
            if total_time > 0.1:  # Only log if operation takes significant time
                print(f"      Features batch load fixed format ({len(dia_source_ids)} IDs): {total_time:.3f}s "
                      f"(load: {load_time:.3f}s, filter: {filter_time:.3f}s)")
        
        return results
    
    def get_object_id(self):
        """
        Get only diaObjectId without loading all features.
            
        Returns
        -------
        pd.DataFrame
            diaObjectId 
        """
        if self._check_table_format():
            with pd.HDFStore(self.file_path, 'r') as store:
                # Load only the diaObjectId column
                dia_obj = store.select('features', columns=['diaObjectId'])
        else:
            # Fallback for fixed format
            if self._data is None:
                with pd.HDFStore(self.file_path, 'r') as store:
                    self._data = store['features']
            dia_obj = self._data[['diaObjectId']]
            
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
    """Lightcurve loader with patch-based caching and cross-reference index.
    
    Provides efficient access to lightcurve data stored in patch-based
    HDF5 files with caching and cross-reference indices for fast lookups.
    
    Attributes:
        lightcurve_path: Path to lightcurve data directory
        _index: Cached lightcurve index (diaObjectId -> patch mapping)
        _diasource_index: Cached diaSourceId -> patch mapping
        _patch_cache: Cache for loaded patch data
        _cache_hits: Number of cache hits for performance tracking
        _cache_misses: Number of cache misses for performance tracking
        _max_cache_size: Maximum number of patches to keep in cache
    """
    
    def __init__(self, lightcurve_path: Path):
        """Initialize LightCurveLoader.
        
        Args:
            lightcurve_path: Path to directory containing lightcurve data files
        """
        self.lightcurve_path = Path(lightcurve_path)
        self._index = None
        self._diasource_index = None
        self._patch_cache = {}  # Cache loaded patch data
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 10  # Configurable cache limit
    
    @property
    def index(self) -> pd.DataFrame:
        """Lazy load the lightcurve index.
        
        Returns:
            pd.DataFrame: Index mapping diaObjectId to patch information
            
        Raises:
            FileNotFoundError: If lightcurve index file is not found
        """
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
        """Lazy load the diaSourceId→patch index.
        
        Returns:
            pd.DataFrame: Index mapping diaSourceId to patch and object information
        """
        if self._diasource_index is None:
            index_file = self.lightcurve_path / "diasource_patch_index.h5"
            if index_file.exists():
                self._diasource_index = pd.read_hdf(index_file, key="diasource_index")
            else:
                self._diasource_index = pd.DataFrame()  # Empty fallback
        return self._diasource_index
    
    def find_patch_by_source_id(self, dia_source_id: int) -> Optional[str]:
        """Find which patch contains the given diaSourceId.
        
        Args:
            dia_source_id: The diaSourceId to locate
            
        Returns:
            Optional[str]: Patch key if found, None otherwise
        """
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
        """Get full lightcurve for a diaObjectId given any diaSourceId from that lightcurve.
        
        Allows retrieving the complete lightcurve
        when you only know one diaSourceId from that lightcurve.
        
        Args:
            dia_source_id: Any diaSourceId from the desired lightcurve
            
        Returns:
            Optional[pd.DataFrame]: Complete lightcurve data sorted by time, or None if not found
        """
        if self.diasource_index.empty:
            return None
        
        start_time = time.time()
        try:
            # Find the diaObjectId and patch for this diaSourceId
            lookup_start = time.time()
            source_info = self.diasource_index.loc[self.diasource_index.index == dia_source_id]
            if len(source_info) == 0:
                return None
            
            dia_object_id = source_info.iloc[0]['diaObjectId']
            patch_key = source_info.iloc[0]['patch_key']
            lookup_time = time.time() - lookup_start
            
            # Load the patch data (with caching)
            cache_start = time.time()
            if patch_key in self._patch_cache:
                self._cache_hits += 1
                patch_data = self._patch_cache[patch_key]
                cache_time = time.time() - cache_start
                cache_type = "hit"
            else:
                self._cache_misses += 1
                patch_data = self._load_patch_data(patch_key)
                if patch_data is None:
                    return None
                self._manage_cache(patch_key, patch_data)
                cache_time = time.time() - cache_start
                cache_type = "miss/load"
            
            # Get all sources for this diaObjectId and sort by time
            filter_start = time.time()
            lc_data = patch_data[patch_data['diaObjectId'] == dia_object_id].copy()
            
            if len(lc_data) > 0 and 'midpointMjdTai' in lc_data.columns:
                lc_data = lc_data.sort_values('midpointMjdTai').reset_index(drop=True)
            filter_time = time.time() - filter_start
            
            total_time = time.time() - start_time
            lc_points = len(lc_data) if lc_data is not None else 0
            if total_time > 0.05:  # Only log if operation takes significant time
                print(f"    Lightcurve by source ID: {total_time:.3f}s (lookup: {lookup_time:.3f}s, "
                      f"cache {cache_type}: {cache_time:.3f}s, filter: {filter_time:.3f}s, points: {lc_points})")
            
            return lc_data if len(lc_data) > 0 else None
            
        except Exception as e:
            return None

    def get_lightcurve(self, dia_object_id: int) -> Optional[pd.DataFrame]:
        """Get lightcurve for a specific diaObjectId.
        
        Uses efficient patch lookup and caching for fast retrieval.
        
        Args:
            dia_object_id: The diaObjectId to retrieve lightcurve for
            
        Returns:
            Optional[pd.DataFrame]: Lightcurve data sorted by time, or None if not found
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
        """Efficiently get lightcurves for multiple objects.
        
        Groups requests by patch to minimize I/O operations.
        
        Args:
            dia_object_ids: List of diaObjectIds to retrieve
            
        Returns:
            Dict[int, pd.DataFrame]: Dictionary mapping diaObjectId to lightcurve DataFrame
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

    def get_all_source_ids_in_lightcurve(self, dia_source_id: int) -> List[int]:
        """Get all diaSourceIds that belong to the same lightcurve as the given diaSourceId.
        
        Args:
            dia_source_id: Any diaSourceId from the desired lightcurve
            
        Returns:
            List[int]: List of all diaSourceIds in the same lightcurve
        """
        if self.diasource_index.empty:
            return []
        
        start_time = time.time()
        try:
            # Get the diaObjectId for this source
            lookup_start = time.time()
            source_info = self.diasource_index.loc[self.diasource_index.index == dia_source_id]
            if len(source_info) == 0:
                return []
            
            dia_object_id = source_info.iloc[0]['diaObjectId']
            lookup_time = time.time() - lookup_start
            
            # Find all sources with the same diaObjectId
            search_start = time.time()
            all_sources_for_object = self.diasource_index[self.diasource_index['diaObjectId'] == dia_object_id]
            result = all_sources_for_object.index.tolist()
            search_time = time.time() - search_start
            
            total_time = time.time() - start_time
            if total_time > 0.05:  # Only log if operation takes significant time
                print(f"    Source ID lookup: {total_time:.3f}s (lookup: {lookup_time:.3f}s, search: {search_time:.3f}s, found: {len(result)} sources)")
            
            return result
            
        except Exception as e:
            return []

    def get_object_id_for_source(self, dia_source_id: int) -> Optional[int]:
        """Get diaObjectId for a given diaSourceId.
        
        Args:
            dia_source_id: The diaSourceId to look up
            
        Returns:
            Optional[int]: The corresponding diaObjectId, or None if not found
        """
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
        """Find which patch contains the given diaObjectId.
        
        Args:
            dia_object_id: The diaObjectId to locate
            
        Returns:
            Optional[str]: Patch key if found, None otherwise
        """
        try:
            result = self.index.loc[self.index.index == dia_object_id]
            if len(result) > 0:
                return result.iloc[0]['patch_key']
        except (KeyError, IndexError):
            pass
        return None
    
    def find_patches_for_objects(self, dia_object_ids: List[int]) -> Dict[int, str]:
        """Efficiently find patches for multiple objects at once.
        
        Args:
            dia_object_ids: List of diaObjectIds to find patches for
            
        Returns:
            Dict[int, str]: Dictionary mapping diaObjectId to patch key
        """
        try:
            # Use pandas query for efficiency
            mask = self.index.index.isin(dia_object_ids)
            results = self.index.loc[mask]
            return dict(zip(results.index, results['patch_key']))
        except Exception as e:
            print(f"Error in batch patch lookup: {e}")
            # Fallback to individual lookups
            return {obj_id: self.find_patch(obj_id) for obj_id in dia_object_ids}

    def get_lightcurve_stats(self, dia_object_id: int) -> Optional[Dict]:
        """Get basic statistics for a lightcurve without loading full data.
        
        Args:
            dia_object_id: The diaObjectId to get statistics for
            
        Returns:
            Optional[Dict]: Dictionary containing lightcurve statistics, or None if not found
        """
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
        """List all available patch files.
        
        Returns:
            List[str]: List of available patch identifiers
        """
        patch_files = list(self.lightcurve_path.glob("patch_*.h5"))
        return [f.stem.replace("patch_", "") for f in patch_files]
    
    def clear_cache(self):
        """Clear the patch cache to free memory."""
        self._patch_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def set_cache_size(self, max_size: int):
        """Set maximum cache size.
        
        Args:
            max_size: Maximum number of patches to keep in cache
        """
        self._max_cache_size = max_size
        # Clear excess entries if needed
        while len(self._patch_cache) > max_size:
            oldest_key = next(iter(self._patch_cache))
            del self._patch_cache[oldest_key]
    
    @property
    def cache_stats(self):
        """Get cache performance statistics.
        
        Returns:
            Dict: Dictionary containing cache performance metrics
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cached_patches': len(self._patch_cache),
            'cache_size_limit': self._max_cache_size
        }
    
    def _load_patch_data(self, patch_key: str) -> Optional[pd.DataFrame]:
        """Load lightcurve data for a specific patch.
        
        Args:
            patch_key: The patch identifier to load
            
        Returns:
            Optional[pd.DataFrame]: Patch data if successful, None otherwise
        """
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
        """Manage cache size by removing least recently used items.
        
        Args:
            new_patch_key: Key for the new patch data to cache
            new_data: The patch data to cache
        """
        if len(self._patch_cache) >= self._max_cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self._patch_cache))
            del self._patch_cache[oldest_key]
        
        self._patch_cache[new_patch_key] = new_data