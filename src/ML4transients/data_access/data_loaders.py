import h5py
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import yaml
import torch  
import matplotlib.pyplot as plt

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
                # If indices are not sorted, sort them and remap id_to_idx accordingly
                if indices:
                    sorted_indices = np.argsort(indices)
                    indices_sorted = [indices[i] for i in sorted_indices]
                    # Remap id_to_idx to match sorted indices
                    id_to_idx_sorted = {}
                    for orig_pos, sorted_pos in enumerate(sorted_indices):
                        target_id = list(id_to_idx.keys())[orig_pos]
                        id_to_idx_sorted[target_id] = sorted_pos
                    indices = indices_sorted
                    id_to_idx = id_to_idx_sorted
                # ------------------------------------------------
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
                    # Optional: log only for very slow operations
                    # if total_time > 1.0:
                    #     print(f"      Cutout batch load ({len(dia_source_ids)} IDs): {total_time:.3f}s")

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
                # Optional: log only for very slow operations
                # if total_time > 1.0:
                #     print(f"      Features batch load ({len(dia_source_ids)} IDs): {total_time:.3f}s")
                
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
            # Optional: log only for very slow operations
            # if total_time > 1.0:
            #     print(f"      Features batch load fixed format ({len(dia_source_ids)} IDs): {total_time:.3f}s")
        
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
                df = pd.read_hdf(index_file, key="index")
                if (df.index.name is None and 'diaObjectId' in df.columns and
                        (df.index.equals(df['diaObjectId']))):
                    df = df.set_index('diaObjectId')
                # handle if diaObjectId is a column and not the index
                elif 'diaObjectId' in df.columns and (df.index.name != 'diaObjectId'):
                    df = df.set_index('diaObjectId')
                self._index = df
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
            # Optional: log only for very slow operations
            # if total_time > 1.0:
            #     print(f"    Lightcurve by source ID: {total_time:.3f}s (points: {lc_points})")
            
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
            # Optional: log only for very slow operations
            # if total_time > 1.0:
            #     print(f"    Source ID lookup: {total_time:.3f}s (found: {len(result)} sources)")
            
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
            'patch_key': self.find_patch(dia_object_id),
            'mean_flux': None
        }
        
        if 'midpointMjdTai' in lc.columns:
            time_span = lc['midpointMjdTai'].max() - lc['midpointMjdTai'].min()
            stats['time_span_days'] = time_span
        
        if 'band' in lc.columns:
            stats['bands'] = lc['band'].unique().tolist()
        
        if 'psfFlux' in lc.columns:
            stats['mean_flux'] = lc['psfFlux'].mean()
        
        return stats
    
    def summarize_lightcurve_stats(self, stats_dict: Dict[int, Dict], plot: bool = True):
        """
        Summarize statistics for a set of lightcurves.

        Args:
            stats_dict: Output from get_multiple_lightcurve_stats (dict of diaObjectId -> stats)
            plot: If True, show summary plots

        Returns:
            summary: dict with summary statistics (mean, median, std, etc.)
        """
        if not stats_dict:
            print("No stats to summarize.")
            return {}

        import numpy as np
        import matplotlib.pyplot as plt

        num_points = [v['num_points'] for v in stats_dict.values()]
        time_spans = [v['time_span_days'] for v in stats_dict.values() if v['time_span_days'] is not None]
        bands = [b for v in stats_dict.values() for b in v['bands']]
        mean_fluxes = [v['mean_flux'] for v in stats_dict.values() if v.get('mean_flux') is not None]

        summary = {
            'num_lightcurves': len(stats_dict),
            'num_points': {
                'mean': float(np.mean(num_points)),
                'median': float(np.median(num_points)),
                'std': float(np.std(num_points)),
                'min': int(np.min(num_points)),
                'max': int(np.max(num_points)),
            },
            'time_span_days': {
                'mean': float(np.mean(time_spans)) if time_spans else None,
                'median': float(np.median(time_spans)) if time_spans else None,
                'std': float(np.std(time_spans)) if time_spans else None,
                'min': float(np.min(time_spans)) if time_spans else None,
                'max': float(np.max(time_spans)) if time_spans else None,
            },
            'mean_psf_flux': {
                'mean': float(np.mean(mean_fluxes)) if mean_fluxes else None,
                'median': float(np.median(mean_fluxes)) if mean_fluxes else None,
                'std': float(np.std(mean_fluxes)) if mean_fluxes else None,
                'min': float(np.min(mean_fluxes)) if mean_fluxes else None,
                'max': float(np.max(mean_fluxes)) if mean_fluxes else None,
            },
            'bands': {
                'unique': list(set(bands)),
                'counts': dict(pd.Series(bands).value_counts())
            }
        }

        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.flatten()
            
            axes[0].hist(num_points, bins=50, color='skyblue')
            axes[0].set_title('Number of Points per Lightcurve')
            axes[0].set_xlabel('Num Points')
            axes[0].set_ylabel('Count')

            if time_spans:
                axes[1].hist(time_spans, bins=50, color='orange')
                axes[1].set_title('Time Span (days) per Lightcurve')
                axes[1].set_xlabel('Time Span (days)')
                axes[1].set_ylabel('Count')
            else:
                axes[1].set_visible(False)

            if mean_fluxes:
                axes[2].hist(mean_fluxes, bins=50, color='purple')
                axes[2].set_title('Mean psfFlux per Lightcurve')
                axes[2].set_xlabel('Mean psfFlux')
                axes[2].set_ylabel('Count')
            else:
                axes[2].set_visible(False)

            if bands:
                pd.Series(bands).value_counts().plot(kind='bar', ax=axes[3], color='green')
                axes[3].set_title('Band Occurrences')
                axes[3].set_xlabel('Band')
                axes[3].set_ylabel('Count')
            else:
                axes[3].set_visible(False)

            plt.tight_layout()
            plt.show()

        return summary

    def summarize_multiple_lightcurves(
        self,
        dia_object_ids: List[int] = None,
        plot: bool = True
    ) -> Dict:
        """
        Efficiently compute and summarize statistics for multiple lightcurves.
        If dia_object_ids is None, computes stats for all objects in the dataset.
        Groups requests by patch to minimize I/O operations.

        Args:
            dia_object_ids: List of diaObjectIds to get statistics for (or None for all)
            plot: If True, show summary plots

        Returns:
            summary: dict with summary statistics (mean, median, std, etc.)
        """
        # --- Gather stats for all requested lightcurves ---
        if dia_object_ids is None:
            dia_object_ids = list(self.index.index)

        index_dtype = self.index.index.dtype
        try:
            dia_object_ids_cast = [index_dtype.type(obj_id) for obj_id in dia_object_ids]
        except Exception:
            dia_object_ids_cast = dia_object_ids

        # Filter out any dia_object_ids not present in the index
        index_objids = set(self.index.index)
        dia_object_ids_cast = [obj_id for obj_id in dia_object_ids_cast if obj_id in index_objids]
        if not dia_object_ids_cast:
            print("No matching diaObjectIds found in the index.")
            return {}

        patch_mapping = self.find_patches_for_objects(dia_object_ids_cast)
        patch_groups = {}
        for obj_id in dia_object_ids_cast:
            patch_key = patch_mapping.get(obj_id)
            if patch_key:
                patch_groups.setdefault(patch_key, []).append(obj_id)

        num_points = []
        time_spans = []
        bands = []
        mean_fluxes = []

        for patch_key, obj_ids in patch_groups.items():
            patch_file = self.lightcurve_path / f"patch_{patch_key}.h5"
            if not patch_file.exists():
                continue
            if patch_key in self._patch_cache:
                patch_data = self._patch_cache[patch_key]
                self._cache_hits += 1
            else:
                patch_data = self._load_patch_data(patch_key)
                if patch_data is None:
                    continue
                self._manage_cache(patch_key, patch_data)
                self._cache_misses += 1

            if patch_data.index.name == 'diaObjectId' and 'diaObjectId' not in patch_data.columns:
                patch_data = patch_data.reset_index()

            for obj_id in obj_ids:
                lc = patch_data[patch_data['diaObjectId'] == obj_id]
                if len(lc) == 0:
                    continue
                num_points.append(len(lc))
                if 'midpointMjdTai' in lc.columns:
                    ts = lc['midpointMjdTai'].max() - lc['midpointMjdTai'].min()
                    time_spans.append(ts)
                if 'band' in lc.columns:
                    bands.extend(lc['band'].unique().tolist())
                if 'psfFlux' in lc.columns:
                    mean_fluxes.append(lc['psfFlux'].mean())

        # --- Summarize ---
        summary = {
            'num_lightcurves': len(dia_object_ids_cast),
            'num_points': {
                'mean': float(np.mean(num_points)) if num_points else None,
                'median': float(np.median(num_points)) if num_points else None,
                'std': float(np.std(num_points)) if num_points else None,
                'min': int(np.min(num_points)) if num_points else None,
                'max': int(np.max(num_points)) if num_points else None,
            },
            'time_span_days': {
                'mean': float(np.mean(time_spans)) if time_spans else None,
                'median': float(np.median(time_spans)) if time_spans else None,
                'std': float(np.std(time_spans)) if time_spans else None,
                'min': float(np.min(time_spans)) if time_spans else None,
                'max': float(np.max(time_spans)) if time_spans else None,
            },
            'mean_psf_flux': {
                'mean': float(np.mean(mean_fluxes)) if mean_fluxes else None,
                'median': float(np.median(mean_fluxes)) if mean_fluxes else None,
                'std': float(np.std(mean_fluxes)) if mean_fluxes else None,
                'min': float(np.min(mean_fluxes)) if mean_fluxes else None,
                'max': float(np.max(mean_fluxes)) if mean_fluxes else None,
            },
            'bands': {
                'unique': list(set(bands)),
                'counts': dict(pd.Series(bands).value_counts()) if bands else {}
            }
        }

        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.flatten()
            
            if num_points:
                axes[0].hist(num_points, bins=50, color='skyblue')
                axes[0].set_title('Number of Points per Lightcurve')
                axes[0].set_xlabel('Num Points')
                axes[0].set_ylabel('Count')
            else:
                axes[0].set_visible(False)

            if time_spans:
                axes[1].hist(time_spans, bins=50, color='orange')
                axes[1].set_title('Time Span (days) per Lightcurve')
                axes[1].set_xlabel('Time Span (days)')
                axes[1].set_ylabel('Count')
            else:
                axes[1].set_visible(False)

            if mean_fluxes:
                axes[2].hist(mean_fluxes, bins=50, color='purple')
                axes[2].set_title('Mean psfFlux per Lightcurve')
                axes[2].set_xlabel('Mean psfFlux')
                axes[2].set_ylabel('Count')
            else:
                axes[2].set_visible(False)

            if bands:
                pd.Series(bands).value_counts().plot(kind='bar', ax=axes[3], color='green')
                axes[3].set_title('Band Occurrences')
                axes[3].set_xlabel('Band')
                axes[3].set_ylabel('Count')
            else:
                axes[3].set_visible(False)

            plt.tight_layout()
            plt.show()

        return summary

    def plot_lightcurve(
        self,
        dia_object_id: int,
        plot_cutouts: bool = False,
        cutout_loader=None,
        show: bool = True,
        figsize=(12, 5)
    ):
        """
        Visualize the light curve for a given diaObjectId.

        Parameters
        ----------
        dia_object_id : int
            The diaObjectId to plot.
        plot_cutouts : bool
            If True, plot cutouts for each epoch (requires cutout_loader).
        cutout_loader : CutoutLoader or None
            Loader to fetch cutouts (optional, required if plot_cutouts is True).
        show : bool
            Whether to show the plot immediately.
        figsize : tuple
            Figure size for the plots.
        """
        flux_col = "psfFlux"
        time_col = "midpointMjdTai"
        band_col = "band"

        lc = self.get_lightcurve(dia_object_id)
        if lc is None or len(lc) == 0:
            print(f"No lightcurve found for diaObjectId {dia_object_id}")
            return
        print(f"{len(lc)} sources in the light curves ")
        # Sort by time
        lc = lc.sort_values(time_col).reset_index(drop=True)

        # Plot 1: Full flux evolution
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax1, ax2 = axes

        # Plot by band if available
        bands = lc[band_col].unique() if band_col in lc.columns else [None]
        colors = plt.cm.tab10.colors

        for i, band in enumerate(bands):
            mask = (lc[band_col] == band) if band is not None else np.ones(len(lc), dtype=bool)
            label = f"Band {band}" if band is not None else "All"
            ax1.scatter(lc.loc[mask, time_col], lc.loc[mask, flux_col], label=label, color=colors[i % len(colors)], s=20)
        ax1.set_xlabel(time_col)
        ax1.set_ylabel(flux_col)
        ax1.set_title(f"Flux evolution for diaObjectId {dia_object_id}")
        if len(bands) > 1:
            ax1.legend()

        # Plot 2: Flux evolution relative to max brightness
        # Find max flux epoch (per band or overall)
        if len(bands) > 1:
            max_flux_idx = lc.groupby(band_col)[flux_col].idxmax()
            max_times = lc.loc[max_flux_idx, time_col].values
            # Use the earliest max as reference
            t_max = np.min(max_times)
        else:
            t_max = lc.loc[lc[flux_col].idxmax(), time_col]
        lc["t_from_max"] = lc[time_col] - t_max

        mask_window = (lc["t_from_max"] >= -30) & (lc["t_from_max"] <= 100)
        for i, band in enumerate(bands):
            mask = ((lc[band_col] == band) if band is not None else np.ones(len(lc), dtype=bool)) & mask_window
            label = f"Band {band}" if band is not None else "All"
            ax2.scatter(lc.loc[mask, "t_from_max"], lc.loc[mask, flux_col], label=label, color=colors[i % len(colors)], s=20)
        ax2.set_xlabel("Days from max brightness")
        ax2.set_ylabel(flux_col)
        ax2.set_title(f"Flux [-30,+100]d of max for diaObjectId {dia_object_id}")
        if len(bands) > 1:
            ax2.legend()

        plt.tight_layout()
        if show:
            plt.show()

        # Plot 3: Cutouts (optional)
        if plot_cutouts and cutout_loader is not None:
            # Get all diaSourceIds for this object
            if "diaSourceId" in lc.columns:
                dia_source_ids = lc["diaSourceId"].values
            elif lc.index.name == "diaSourceId":
                dia_source_ids = lc.index.values
            else:
                print("Cannot find diaSourceId column for cutout plotting.")
                return

            n = len(dia_source_ids)
            ncols = min(8, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))
            axes = np.array(axes).reshape(-1)
            for i, dsid in enumerate(dia_source_ids):
                cutout = cutout_loader.get_by_id(dsid)
                if cutout is not None:
                    ax = axes[i]
                    im = ax.imshow(cutout, cmap="gray", origin="lower")
                    ax.set_title(f"ID {dsid}")
                    ax.axis("off")
                else:
                    axes[i].set_visible(False)
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            plt.suptitle(f"Cutouts for diaObjectId {dia_object_id}")
            plt.tight_layout()
            if show:
                plt.show()

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

    def load_snn_inference(self, columns=None, snn_dataset_name="snn_inference"):
        """
        Efficiently load SNN inference results from all patch HDF5 files.
        Only loads specified columns for memory efficiency.
        Returns a concatenated DataFrame.
        """
        import glob

        patch_files = sorted(glob.glob(str(self.lightcurve_path / "patch_*.h5")))
        if not patch_files:
            print("No patch HDF5 files found.")
            return None

        dfs = []
        for f in patch_files:
            try:
                # Use pd.read_hdf with columns argument for table format
                df = pd.read_hdf(f, key=snn_dataset_name, columns=columns)
                dfs.append(df)
            except (KeyError, ValueError) as e:
                # KeyError if snn_inference not present, ValueError if not table format
                continue
            except Exception as e:
                # Log error but continue processing other files
                continue
        if not dfs:
            print("No valid SNN inference tables loaded from HDF5.")
            return None
        return pd.concat(dfs, ignore_index=True)

    @property
    def inference_snn(self):
        """
        Load and analyze SNN ensemble inference results for all patches from HDF5.
        Returns the concatenated DataFrame and shows summary/plots.
        Includes all lightcurves - those not processed by SNN have NaN values.
        """
        # Clear cache to ensure we get fresh data
        self.clear_cache()
        
        # Only load necessary columns for analysis
        columns = [
            "diaObjectId", "prob_class0_mean", "prob_class1_mean", "prob_class0_std",
            "prob_class1_std", "pred_class", "n_sources_at_inference"
        ]
        ensemble_df = self.load_snn_inference(columns=columns)
        if ensemble_df is None:
            print("No SNN inference data found in HDF5 files.")
            return None

        print(f"Loaded SNN inference data: {len(ensemble_df)} total lightcurves")
        
        # Separate processed vs unprocessed
        processed_mask = (ensemble_df['pred_class'] >= 0) & (~ensemble_df['prob_class1_mean'].isna())
        snn_processed = ensemble_df[processed_mask]
        not_processed = ensemble_df[~processed_mask]
        
        print(f"  SNN processed: {len(snn_processed)} lightcurves")
        print(f"  Not processed (failed filtering): {len(not_processed)} lightcurves")
        print(f"  Processing rate: {len(snn_processed)/len(ensemble_df)*100:.1f}%")

        if len(snn_processed) == 0:
            print("No lightcurves were successfully processed by SNN!")
            return ensemble_df

        # Show statistics about source counts used in SNN inference
        if 'n_sources_at_inference' in snn_processed.columns and not snn_processed['n_sources_at_inference'].isna().all():
            inference_counts = snn_processed['n_sources_at_inference'].dropna()
            print(f"\nSNN Inference Source Count Statistics:")
            print(f"  Mean sources per lightcurve: {inference_counts.mean():.1f}")
            print(f"  Min sources: {int(inference_counts.min())}")
            print(f"  Max sources: {int(inference_counts.max())}")
            
            # Check how many actually had < 10 sources during inference
            low_count = (inference_counts < 10).sum()
            if low_count > 0:
                print(f"  WARNING: {low_count} lightcurves had < 10 sources during SNN inference!")

        # VALIDATION: Check if SNN-processed lightcurves actually meet the filtering criteria
        print(f"\n{'='*60}")
        print(f"VALIDATING SNN FILTERING CONSISTENCY")
        print(f"{'='*60}")
        
        validation_issues = []
        sample_size = min(100, len(snn_processed))  # Check a sample to avoid overwhelming output
        print(f"Checking {sample_size} SNN-processed lightcurves against filtering criteria...")
        
        sample_objects = snn_processed.head(sample_size)['diaObjectId'].values
        for i, obj_id in enumerate(sample_objects):
            try:
                lc = self.get_lightcurve(int(obj_id))
                if lc is not None and len(lc) > 0:
                    # Check filtering criteria: 10+ points in [-30, +100] day window
                    flux_col = "psfFlux"
                    time_col = "midpointMjdTai"
                    
                    # Find max flux time
                    if len(lc) > 1:
                        max_flux_idx = lc[flux_col].idxmax()
                        t_max = lc.loc[max_flux_idx, time_col]
                    else:
                        t_max = lc[time_col].iloc[0]
                    
                    lc_copy = lc.copy()
                    lc_copy["t_from_max"] = lc_copy[time_col] - t_max
                    window_mask = (lc_copy["t_from_max"] >= -30) & (lc_copy["t_from_max"] <= 100)
                    n_in_window = window_mask.sum()
                    
                    if n_in_window < 10:
                        validation_issues.append({
                            'diaObjectId': obj_id,
                            'total_points': len(lc),
                            'points_in_window': n_in_window
                        })
                        
                else:
                    validation_issues.append({
                        'diaObjectId': obj_id,
                        'total_points': 0,
                        'points_in_window': 0,
                        'issue': 'No lightcurve data found'
                    })
                    
            except Exception as e:
                validation_issues.append({
                    'diaObjectId': obj_id,
                    'issue': f'Error loading lightcurve: {e}'
                })
        
        if len(validation_issues) > 0:
            print(f"WARNING: VALIDATION ISSUES FOUND: {len(validation_issues)}/{sample_size} objects")
            print(f"These objects have SNN results but don't meet filtering criteria:")
            for issue in validation_issues[:10]:  # Show first 10
                if 'issue' in issue:
                    print(f"  • {issue['diaObjectId']}: {issue['issue']}")
                else:
                    # Get SNN source count for comparison
                    snn_count = snn_processed[snn_processed['diaObjectId'] == issue['diaObjectId']]['n_sources_at_inference'].iloc[0]
                    if not pd.isna(snn_count):
                        print(f"  • {issue['diaObjectId']}: {issue['points_in_window']}/{issue['total_points']} pts in window, SNN used {int(snn_count)} pts")
                    else:
                        print(f"  • {issue['diaObjectId']}: {issue['points_in_window']}/{issue['total_points']} pts in window, SNN count unknown")
            if len(validation_issues) > 10:
                print(f"  ... and {len(validation_issues) - 10} more")
            print(f"\nRECOMMENDATION: Re-run the SNN pipeline to ensure consistent filtering!")
        else:
            print(f"Validation passed: All sampled objects meet filtering criteria")
        
        print(f"{'='*60}\n")

        # Continue analysis only with processed lightcurves
        preds_df = snn_processed.rename(columns={'prob_class0_mean': 'prob_class0', 'prob_class1_mean': 'prob_class1'})

        # Print summary
        print(f"\nSNN-processed objects: {len(snn_processed)}")
        for class_id, count in snn_processed['pred_class'].value_counts().sort_index().items():
            name = "Non-SN" if class_id == 0 else "Supernova"
            print(f"{name}: {count} ({count/len(snn_processed)*100:.1f}%)")

        print(f"\nUncertainty Statistics:")
        print(f"Mean SN probability uncertainty: {snn_processed['prob_class1_std'].mean():.3f}")
        print(f"High uncertainty objects (std > 0.1): {(snn_processed['prob_class1_std'] > 0.1).sum()}")

        sn_candidates = snn_processed[snn_processed['pred_class'] == 1]
        high_conf = sn_candidates[sn_candidates['prob_class1_mean'] > 0.7]
        low_uncertainty = high_conf[high_conf['prob_class1_std'] < 0.05]

        print(f"\nHigh-confidence SN (mean prob > 0.7): {len(high_conf)}")
        print(f"High-confidence + Low uncertainty (std < 0.05): {len(low_uncertainty)}")

        if len(high_conf) > 0:
            print("\nTop 5 candidates (by mean probability):")
            print("NOTE: Lightcurve statistics shown below are from the current HDF5 files,")
            print("      which may differ from the filtered data used for SNN inference.")
            top_candidates = high_conf.nlargest(500, 'prob_class1_mean')
            for i, (_, row) in enumerate(top_candidates.iterrows(), 1):
                # Get SNN inference count
                snn_count = row.get('n_sources_at_inference', np.nan)
                snn_info = f"SNN used {int(snn_count)} pts" if not pd.isna(snn_count) else "SNN count unknown"
                
                # Try to get current lightcurve info if possible
                try:
                    lc = self.get_lightcurve(int(row['diaObjectId']))
                    if lc is not None and len(lc) > 0:
                        total_points = len(lc)
                        # Compute window statistics
                        flux_col = "psfFlux"
                        time_col = "midpointMjdTai"
                        band_col = "band"
                        lc = lc.sort_values(time_col).reset_index(drop=True)
                        bands = lc[band_col].unique() if band_col in lc.columns else [None]
                        if len(bands) > 1:
                            max_flux_idx = lc.groupby(band_col)[flux_col].idxmax()
                            max_times = lc.loc[max_flux_idx, time_col].values
                            t_max = np.min(max_times)
                        else:
                            t_max = lc.loc[lc[flux_col].idxmax(), time_col]
                        lc["t_from_max"] = lc[time_col] - t_max
                        n_window = ((lc["t_from_max"] >= -30) & (lc["t_from_max"] <= 100)).sum()
                        lc_info = f"(current lc: {total_points} pts, in window: {n_window}, {snn_info})"
                    else:
                        lc_info = f"(no lc data, {snn_info})"
                except Exception:
                    lc_info = f"(lc data error, {snn_info})"
                print(f"  {i}. diaObjectId {str(int(row['diaObjectId']))}: {row['prob_class1_mean']:.3f} ± {row['prob_class1_std']:.3f} {lc_info}")
                if i >= 13:  # Limit output
                    break

        # Plotting
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Class distribution
        preds_df['pred_class'].value_counts().sort_index().plot(kind='bar', ax=axes[0,0], color=['orange', 'blue'])
        axes[0,0].set_title('Ensemble Classification Results')
        axes[0,0].set_xlabel('Class (0=Non-SN, 1=SN)')
        axes[0,0].tick_params(axis='x', rotation=0)

        # 2. Full probability distribution (both classes)
        non_sn_data = preds_df[preds_df['pred_class'] == 0]['prob_class1']
        sn_data = preds_df[preds_df['pred_class'] == 1]['prob_class1']
        prob_min = preds_df['prob_class1'].min()
        prob_max = preds_df['prob_class1'].max()
        if len(non_sn_data) > 0:
            axes[0,1].hist(non_sn_data, bins=30, alpha=0.7, color='orange', label=f'Non-SN ({len(non_sn_data)})')
        if len(sn_data) > 0:
            axes[0,1].hist(sn_data, bins=30, alpha=0.7, color='blue', label=f'SN ({len(sn_data)})')
        axes[0,1].set_xlabel('SN Probability (Mean)')
        axes[0,1].set_ylabel('Count')
        axes[0,1].set_title('Probability Distribution by Class')
        axes[0,1].legend()
        axes[0,1].set_xlim(prob_min-0.01, prob_max+0.01)

        # 3. Uncertainty distribution
        axes[0,2].hist(ensemble_df['prob_class1_std'], bins=30, alpha=0.7, color='green')
        axes[0,2].axvline(ensemble_df['prob_class1_std'].mean(), color='red', linestyle='--',
                          label=f'Mean: {ensemble_df["prob_class1_std"].mean():.3f}')
        axes[0,2].axvline(0.05, color='orange', linestyle=':', label='Low Uncertainty')
        axes[0,2].legend()
        axes[0,2].set_xlabel('SN Probability Std Dev')
        axes[0,2].set_ylabel('Count')
        axes[0,2].set_title('Ensemble Uncertainty Distribution')

        # 4. Confidence levels for SN candidates
        if len(sn_candidates) > 0:
            conf_counts = pd.cut(sn_candidates['prob_class1_mean'], bins=[0, 0.5, 0.7, 0.9, 1.0],
                                labels=['Low', 'Med', 'High', 'V.High']).value_counts()
            conf_counts.plot(kind='bar', ax=axes[1,0], color='skyblue')
        axes[1,0].set_title('SN Confidence Levels')
        axes[1,0].tick_params(axis='x', rotation=45)

        # 5. Uncertainty vs Probability scatter plot
        scatter = axes[1,1].scatter(ensemble_df['prob_class1_mean'], ensemble_df['prob_class1_std'],
                                   c=ensemble_df['pred_class'], cmap='RdYlBu', alpha=0.6, s=2)
        axes[1,1].axhline(0.05, color='orange', linestyle=':', alpha=0.7, label='Low Uncertainty')
        axes[1,1].axvline(0.7, color='red', linestyle='--', alpha=0.7, label='High Confidence')
        axes[1,1].set_xlabel('SN Probability (Mean)')
        axes[1,1].set_ylabel('SN Probability (Std Dev)')
        axes[1,1].set_title(f'Probability vs Uncertainty ({len(ensemble_df)} objects)')
        axes[1,1].legend()
        plt.colorbar(scatter, ax=axes[1,1], label='Predicted Class')

        # 6. Summary pie chart with uncertainty categories
        sizes, labels, colors = [], [], []
        non_sn = len(ensemble_df[ensemble_df['pred_class'] == 0])
        high_conf_low_unc = len(high_conf[high_conf['prob_class1_std'] < 0.05]) if len(high_conf) > 0 else 0
        high_conf_high_unc = len(high_conf[high_conf['prob_class1_std'] >= 0.05]) if len(high_conf) > 0 else 0
        low_conf_sn = len(sn_candidates) - len(high_conf) if len(sn_candidates) > 0 else 0

        if non_sn > 0:
            sizes.append(non_sn); labels.append('Non-SN'); colors.append('orange')
        if low_conf_sn > 0:
            sizes.append(low_conf_sn); labels.append('Low-Conf SN'); colors.append('lightblue')
        if high_conf_high_unc > 0:
            sizes.append(high_conf_high_unc); labels.append('High-Conf SN\n(High Unc)'); colors.append('lightcoral')
        if high_conf_low_unc > 0:
            sizes.append(high_conf_low_unc); labels.append('High-Conf SN\n(Low Unc)'); colors.append('darkblue')

        if sizes:
            axes[1,2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[1,2].set_title('Ensemble Summary with Uncertainty')

        plt.tight_layout()
        plt.show()

        print(f"\nEnsemble Summary:")
        print(f"SNN-processed objects: {len(ensemble_df)}")
        print(f"Total objects in dataset: {len(self.index)}")
        print(f"High-confidence SN candidates: {len(high_conf)}")
        print(f"High-conf + Low uncertainty: {high_conf_low_unc}")
        print(f"Mean uncertainty: {snn_processed['prob_class1_std'].mean():.3f}")
        print(f"Objects with high uncertainty (>0.1): {(snn_processed['prob_class1_std'] > 0.1).sum()}")

        return ensemble_df
   
    
    def get_high_conf_sn_sources(self, prob_threshold=0.7, std_threshold=0.05, snn_dataset_name="snn_inference"):
        """
        Retrieve DiaSourceIds for all sources that compose high-confidence SN candidates.
        Returns a set of DiaSourceIds (as integers).
        """
        columns = ["diaObjectId", "pred_class", "prob_class1_mean", "prob_class1_std"]
        df = self.load_snn_inference(columns=columns, snn_dataset_name=snn_dataset_name)
        if df is None:
            print("No SNN inference data found in HDF5 files.")
            return set()

        # Ensure correct dtypes - handle string diaObjectId properly
        df = df.copy()
        # Convert diaObjectId to int64 only if it's not already numeric
        if df["diaObjectId"].dtype == 'object':
            # Handle potential string representations
            df["diaObjectId"] = pd.to_numeric(df["diaObjectId"], errors='coerce').astype(np.int64)
        else:
            df["diaObjectId"] = df["diaObjectId"].astype(np.int64)
        df["pred_class"] = df["pred_class"].astype(int)
        df["prob_class1_mean"] = df["prob_class1_mean"].astype(float)
        df["prob_class1_std"] = df["prob_class1_std"].astype(float)

        # Select high-confidence SN candidates
        mask = (
            (df["pred_class"] == 1) &
            (df["prob_class1_mean"] > prob_threshold) &
            (df["prob_class1_std"] < std_threshold)
        )
        high_conf_objids = set(df.loc[mask, "diaObjectId"].tolist())

        print(f"Found {len(high_conf_objids)} high-confidence diaObjectId candidates.")

        if not high_conf_objids:
            print("No high-confidence SN candidates found after filtering.")
            return set()

        # Use diasource_index to map diaObjectId to diaSourceIds
        if self.diasource_index is None or self.diasource_index.empty:
            print("No diasource_index available.")
            return set()

        # Vectorized lookup: filter diasource_index once for all objids
        diasource_index = self.diasource_index
        # If not already, ensure diaObjectId is int64 for join
        diasource_index = diasource_index.copy()
        if diasource_index['diaObjectId'].dtype == 'object':
            diasource_index['diaObjectId'] = pd.to_numeric(diasource_index['diaObjectId'], errors='coerce').astype(np.int64)
        else:
            diasource_index['diaObjectId'] = diasource_index['diaObjectId'].astype(np.int64)
        mask = diasource_index['diaObjectId'].isin(high_conf_objids)
        high_conf_sources = diasource_index[mask].index.astype(np.int64)

        print(f"Total high-confidence diaSourceIds found: {len(high_conf_sources)}")
        return set(high_conf_sources)

    def save_high_conf_subset_dataset(
        self,
        out_dir: Path,
        prob_threshold: float = 0.7,
        std_threshold: float = 0.05,
        snn_dataset_name: str = "snn_inference"
    ):
        """
        Save a full mini-dataset (lightcurves, cutouts, features) for high-confidence SN sources,
        preserving the structure expected by DatasetLoader.
        Only saves data for lightcurves that have SNN inference results and meet confidence criteria.
        Skips visits (cutout/feature files) that are already saved in the output directory.
        """
        import shutil

        out_dir = Path(out_dir)
        out_lc_dir = out_dir / "lightcurves"
        out_cutout_dir = out_dir / "cutouts"
        out_feat_dir = out_dir / "features"
        out_lc_dir.mkdir(parents=True, exist_ok=True)
        out_cutout_dir.mkdir(parents=True, exist_ok=True)
        out_feat_dir.mkdir(parents=True, exist_ok=True)

        # 1. Get high-confidence diaSourceIds
        high_conf_ids = self.get_high_conf_sn_sources(
            prob_threshold=prob_threshold,
            std_threshold=std_threshold,
            snn_dataset_name=snn_dataset_name
        )
        high_conf_ids = set(high_conf_ids)
        print(f"Saving {len(high_conf_ids)} high-confidence SN sources to mini-dataset.")

        # 2. Save lightcurve index and diasource index for subset
        for index_file in ["lightcurve_index.h5", "diasource_patch_index.h5"]:
            src = self.lightcurve_path / index_file
            out_index = out_lc_dir / index_file
            if src.exists():
                if out_index.exists():
                    print(f"Skipping {index_file} (already exists in output)")
                    continue
                df = pd.read_hdf(src)
                if "diasource" in index_file:
                    # Filter to only high-confidence sources
                    df = df[df.index.isin(high_conf_ids)]
                else:
                    # Filter to only objects that have high-confidence sources
                    if not self.diasource_index.empty:
                        keep_objids = self.diasource_index.loc[
                            self.diasource_index.index.isin(high_conf_ids), 'diaObjectId'
                        ].unique()
                        df = df[df.index.isin(keep_objids)]
                    else:
                        print(f"Warning: No diasource_index available, keeping all objects in {index_file}")
                df.to_hdf(out_index, key=df.columns.name or "index")
                print(f"Saved filtered {index_file}: {len(df)} entries")

        # 3. Save patch files with only high-conf sources
        patch_files = list((self.lightcurve_path).glob("patch_*.h5"))
        for patch_file in patch_files:
            out_patch = out_lc_dir / patch_file.name
            if out_patch.exists():
                print(f"Skipping patch file {patch_file.name} (already exists in output)")
                continue
            
            # Load original lightcurves
            df = pd.read_hdf(patch_file, key="lightcurves")
            
            # Filter to only high-confidence sources
            if 'diaSourceId' in df.columns:
                df_filtered = df[df["diaSourceId"].isin(high_conf_ids)]
            elif df.index.name == "diaSourceId":
                df_filtered = df[df.index.isin(high_conf_ids)]
            elif df.index.name is None and df.index.dtype == np.int64:
                df_filtered = df[df.index.isin(high_conf_ids)]
            else:
                print(f"Warning: Could not find 'diaSourceId' in patch file: {patch_file}")
                continue
            
            if len(df_filtered) > 0:
                # Save filtered lightcurves
                df_filtered.to_hdf(out_patch, key="lightcurves")
                
                # Also copy SNN inference data if it exists
                try:
                    snn_df = pd.read_hdf(patch_file, key="snn_inference")
                    # Filter SNN data to only high-confidence objects
                    high_conf_objids = self.diasource_index.loc[
                        self.diasource_index.index.isin(high_conf_ids), 'diaObjectId'
                    ].unique()
                    snn_filtered = snn_df[snn_df['diaObjectId'].isin(high_conf_objids)]
                    if len(snn_filtered) > 0:
                        arr = snn_filtered.to_records(index=False)
                        with h5py.File(out_patch, "a") as h5f:
                            h5f.create_dataset("snn_inference", data=arr)
                        print(f"Saved {len(df_filtered)} lightcurve sources + {len(snn_filtered)} SNN results to {patch_file.name}")
                    else:
                        print(f"Saved {len(df_filtered)} lightcurve sources to {patch_file.name}")
                except (KeyError, ValueError):
                    print(f"Saved {len(df_filtered)} lightcurve sources to {patch_file.name} (no SNN data)")
            else:
                print(f"No high-confidence sources found in {patch_file.name}")

        # 4. Save cutouts for high-conf sources
        cutout_dir = self.lightcurve_path.parent / "cutouts"
        if cutout_dir.exists():
            for cutout_file in cutout_dir.glob("visit_*.h5"):
                out_cutout_file = out_cutout_dir / cutout_file.name
                if out_cutout_file.exists():
                    print(f"Skipping cutout file {cutout_file.name} (already exists in output)")
                    continue
                cloader = CutoutLoader(cutout_file)
                ids_in_file = set(cloader.ids) & high_conf_ids
                if ids_in_file:
                    cutouts = cloader.get_multiple_by_ids(list(ids_in_file))
                    if cutouts:
                        import h5py
                        with h5py.File(out_cutout_file, "w") as f:
                            f.create_dataset("diaSourceId", data=np.array(list(cutouts.keys()), dtype=np.int64))
                            f.create_dataset("cutouts", data=np.stack(list(cutouts.values())))
                        print(f"Saved {len(cutouts)} cutouts to {out_cutout_file}")

        # 5. Save features for high-conf sources
        feature_dir = self.lightcurve_path.parent / "features"
        if feature_dir.exists():
            for feature_file in feature_dir.glob("visit_*_features.h5"):
                out_feat_file = out_feat_dir / feature_file.name
                if out_feat_file.exists():
                    print(f"Skipping feature file {feature_file.name} (already exists in output)")
                    continue
                floader = FeatureLoader(feature_file)
                ids_in_file = set(floader.ids) & high_conf_ids
                if ids_in_file:
                    feats = floader.get_multiple_by_ids(list(ids_in_file))
                    if feats:
                        all_feat = pd.concat(feats.values())
                        all_feat.to_hdf(out_feat_file, key="features", format="table")
                        print(f"Saved {len(all_feat)} features to {out_feat_file}")

        print(f"Mini-dataset for high-confidence SN saved to {out_dir}")
        print(f"Dataset contains only sources from SNN-processed lightcurves that meet confidence criteria.")