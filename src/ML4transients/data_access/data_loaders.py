import h5py
import pandas as pd
import numpy as np
import time
import gc
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import yaml
import matplotlib.pyplot as plt

class CutoutLoader:
    """Lazy loader for cutout data.
    
    Provides efficient access to astronomical image cutouts stored in HDF5 format.
    Supports three types of cutouts: difference (default), coadd, and science.
    Data is loaded on-demand to minimize memory usage.
    """
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None  # Difference cutouts (default)
        self._coadd_data = None  # Coadd/template cutouts
        self._science_data = None  # Science/calexp cutouts
        self._ids = None
        self._available_types = None  # Cache which types are available
    
    @property
    def data(self):
        """Load difference cutouts (default behavior for backward compatibility)."""
        return self.get_data('diff')
    
    def get_data(self, cutout_type: str = 'diff'):
        """Load cutout data of specified type.
        
        Parameters
        ----------
        cutout_type : str, default 'diff'
            Type of cutout to load: 'diff', 'coadd', or 'science'
            
        Returns
        -------
        np.ndarray
            Array of cutouts of the specified type
        """
        cutout_type = cutout_type.lower()
        
        # Map to the correct cache and dataset name
        if cutout_type == 'diff':
            if self._data is None:
                self._data = self._load_cutouts_from_file('cutouts')
            return self._data
        elif cutout_type == 'coadd':
            if self._coadd_data is None:
                self._coadd_data = self._load_cutouts_from_file('coadd_cutouts')
            return self._coadd_data
        elif cutout_type == 'science':
            if self._science_data is None:
                self._science_data = self._load_cutouts_from_file('science_cutouts')
            return self._science_data
        else:
            raise ValueError(f"Invalid cutout_type: {cutout_type}. Must be 'diff', 'coadd', or 'science'")
    
    def _load_cutouts_from_file(self, dataset_name: str):
        """Load cutouts from specific dataset in HDF5 file.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset in the HDF5 file
            
        Returns
        -------
        np.ndarray or None
            Cutout data if available, None otherwise
        """
        try:
            with h5py.File(self.file_path, 'r') as f:
                if dataset_name in f:
                    return f[dataset_name][:]
                else:
                    print(f"Warning: Dataset '{dataset_name}' not found in {self.file_path}")
                    return None
        except Exception as e:
            print(f"Error loading {dataset_name} from {self.file_path}: {e}")
            return None
    
    def _check_available_types(self):
        """Check which cutout types are available in the file."""
        if self._available_types is not None:
            return self._available_types
        
        self._available_types = []
        try:
            with h5py.File(self.file_path, 'r') as f:
                if 'cutouts' in f:
                    self._available_types.append('diff')
                if 'coadd_cutouts' in f:
                    self._available_types.append('coadd')
                if 'science_cutouts' in f:
                    self._available_types.append('science')
        except Exception as e:
            print(f"Warning: Error checking available cutout types in {self.file_path}: {e}")
        
        return self._available_types
    
    @property
    def available_types(self):
        """Get list of available cutout types in this file."""
        return self._check_available_types()
    
    @property
    def ids(self):
        """Lazy load diaSourceIds."""
        if self._ids is None:
            with h5py.File(self.file_path, 'r') as f:
                self._ids = f['diaSourceId'][:]
        return self._ids
    
    def get_by_id(self, dia_source_id: int, cutout_type: str = 'diff'):
        """Get specific cutout by diaSourceId.
        
        Parameters
        ----------
        dia_source_id : int
            The diaSourceId to retrieve
        cutout_type : str, default 'diff'
            Type of cutout to retrieve: 'diff', 'coadd', or 'science'
            
        Returns
        -------
        np.ndarray or None
            Cutout array, or None if ID not found
        """
        cutout_type = cutout_type.lower()
        
        # Map cutout type to dataset name
        dataset_map = {
            'diff': 'cutouts',
            'coadd': 'coadd_cutouts',
            'science': 'science_cutouts'
        }
        
        if cutout_type not in dataset_map:
            raise ValueError(f"Invalid cutout_type: {cutout_type}. Must be 'diff', 'coadd', or 'science'")
        
        dataset_name = dataset_map[cutout_type]
        
        # Try efficient row-based query first
        try:
            with h5py.File(self.file_path, 'r') as f:
                if 'diaSourceId' not in f or dataset_name not in f:
                    return None
                
                ids = f['diaSourceId'][:]
                idx = np.where(ids == dia_source_id)[0]
                
                if len(idx) > 0:
                    return f[dataset_name][idx[0]]
                    
        except Exception as e:
            print(f"Error retrieving {cutout_type} cutout for ID {dia_source_id}: {e}")
            return None

    def get_multiple_by_ids(self, dia_source_ids: List[int], cutout_type: str = 'diff') -> Dict[int, np.ndarray]:
        """Efficiently get multiple cutouts by diaSourceIds from this file.
        
        Parameters
        ----------
        dia_source_ids : List[int]
            List of diaSourceIds to retrieve
        cutout_type : str, default 'diff'
            Type of cutout to retrieve: 'diff', 'coadd', or 'science'
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping diaSourceId to cutout array
        """
        if not dia_source_ids:
            return {}
        
        cutout_type = cutout_type.lower()
        
        # Map cutout type to dataset name
        dataset_map = {
            'diff': 'cutouts',
            'coadd': 'coadd_cutouts',
            'science': 'science_cutouts'
        }
        
        if cutout_type not in dataset_map:
            raise ValueError(f"Invalid cutout_type: {cutout_type}. Must be 'diff', 'coadd', or 'science'")
        
        dataset_name = dataset_map[cutout_type]

        start_time = time.time()
        results = {}
        try:
            with h5py.File(self.file_path, 'r') as f:
                if 'diaSourceId' not in f or dataset_name not in f:
                    print(f"Warning: Required datasets not found in {self.file_path}")
                    return results
                
                # Load all IDs from file
                all_ids = f['diaSourceId'][:]
                
                # Create a mapping from diaSourceId to index
                id_to_idx = {dia_id: idx for idx, dia_id in enumerate(all_ids)}
                
                # Find indices for requested IDs
                indices_to_load = []
                id_mapping = []  # Keep track of which ID corresponds to which index
                
                for dia_id in dia_source_ids:
                    if dia_id in id_to_idx:
                        idx = id_to_idx[dia_id]
                        indices_to_load.append(idx)
                        id_mapping.append(dia_id)
                
                if not indices_to_load:
                    return results
                
                # Sort indices for efficient sequential reading
                sorted_pairs = sorted(zip(indices_to_load, id_mapping))
                sorted_indices = [p[0] for p in sorted_pairs]
                sorted_ids = [p[1] for p in sorted_pairs]
                
                # Load cutouts in batches for efficiency
                batch_size = 100
                for i in range(0, len(sorted_indices), batch_size):
                    batch_indices = sorted_indices[i:i+batch_size]
                    batch_ids = sorted_ids[i:i+batch_size]
                    
                    # Load batch of cutouts
                    for idx, dia_id in zip(batch_indices, batch_ids):
                        cutout = f[dataset_name][idx]
                        results[dia_id] = cutout
                
                elapsed = time.time() - start_time
                print(f"Loaded {len(results)}/{len(dia_source_ids)} {cutout_type} cutouts from {self.file_path.name} in {elapsed:.3f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Error loading {cutout_type} cutouts from {self.file_path}: {e}")
            print(f"Partial results: {len(results)} cutouts loaded in {elapsed:.3f}s")

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
    
    def __init__(self, data_path: Path, visit: int, weights_path: str = None, mc_dropout: bool = False, mc_samples: int = 50):
        """
        Initialize InferenceLoader.
        
        Args:
            data_path: Path to data directory
            visit: Visit number
            weights_path: Path to model weights (optional, for running inference)
            mc_dropout: Whether MC Dropout is enabled (affects hash generation)
            mc_samples: Number of MC Dropout samples (affects hash generation)
        """
        self.data_path = Path(data_path)
        self.visit = visit
        self.weights_path = weights_path
        self.mc_dropout = mc_dropout
        self.mc_samples = mc_samples
        
        # Generate model hash from weights path if provided
        # Include MC Dropout config in hash for unique identification
        self.model_hash = None
        if weights_path:
            import hashlib
            # Base hash from weights path (for backward compatibility)
            hash_string = str(weights_path)
            # Append MC Dropout config to hash string if enabled
            if mc_dropout:
                hash_string += f"_mcd{mc_samples}"
            self.model_hash = hashlib.md5(hash_string.encode()).hexdigest()[:8]
        
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

    def run_inference(self, dataset_loader, trainer=None, force=False, mc_dropout=False, mc_samples=50):
        """
        Run inference for this visit and save results.
        
        Args:
            dataset_loader: DatasetLoader instance
            trainer: Pre-loaded trainer (optional, for efficiency)
            force: Force re-run even if results exist
            mc_dropout: Enable Monte Carlo Dropout for uncertainty estimation
            mc_samples: Number of MC Dropout forward passes
        """
        import torch
        
        if not force and self.has_inference_results():
            print(f"Inference results already exist for visit {self.visit}")
            return
        
        if not self.weights_path:
            raise ValueError("weights_path required for running inference")
            
        if not self._inference_file:
            raise ValueError("Cannot determine inference file path without model hash")
        
        print(f"Running inference for visit {self.visit}...")
        
        # Load config to get cutout_types for multi-channel support
        from ML4transients.utils import load_config
        config = load_config(f"{self.weights_path}/config.yaml")
        cutout_types = config.get('data', {}).get('cutout_types', ['diff'])
        print(f"Using cutout types from model config: {cutout_types}")
        
        # Create inference dataset for this visit
        from ML4transients.training.pytorch_dataset import PytorchDataset
        
        print(f"Creating inference dataset for visit {self.visit}...")
        inference_dataset = PytorchDataset.create_inference_dataset(
            dataset_loader, 
            visit=self.visit,
            cutout_types=cutout_types  # Pass cutout types from model config
        )
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
            return_probabilities=None,  # Auto-detect based on model type
            mc_dropout=mc_dropout,  # Pass MC Dropout flag
            mc_samples=mc_samples   # Pass number of MC samples
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
    """
    =====================================================================
    INDEXES MANAGEMENT
    =====================================================================
    """
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
                try:
                    # Try the expected key first
                    self._diasource_index = pd.read_hdf(index_file, key="diasource_index")
                except KeyError:
                    # Check what keys are available
                    try:
                        with pd.HDFStore(index_file, 'r') as store:
                            available_keys = list(store.keys())
                            print(f"Available keys in {index_file}: {available_keys}")
                            
                            # Try some common key names
                            possible_keys = ["index", "diasource", "sources", "/diasource_index"]
                            for key in possible_keys:
                                if key in available_keys:
                                    print(f"Using key '{key}' instead of 'diasource_index'")
                                    self._diasource_index = pd.read_hdf(index_file, key=key)
                                    break
                            else:
                                # If no suitable key found, try the first available key
                                if available_keys:
                                    first_key = available_keys[0]
                                    print(f"Using first available key '{first_key}'")
                                    self._diasource_index = pd.read_hdf(index_file, key=first_key)
                                else:
                                    print(f"No keys found in {index_file}")
                                    self._diasource_index = pd.DataFrame()  # Empty fallback
                    except Exception as e:
                        print(f"Error reading {index_file}: {e}")
                        self._diasource_index = pd.DataFrame()  # Empty fallback
            else:
                print(f"diasource_patch_index.h5 not found at {index_file}")
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
        
    def get_source_ids_for_objects(self, dia_object_ids: List[int]) -> Dict[int, List[int]]:
        """Get all diaSourceIds that belong to each specified diaObjectId.
        
        This function maps each diaObjectId to all the diaSourceIds (across all visits)
        that belong to that object's lightcurve.
        
        Args:
            dia_object_ids: List of diaObjectIds to map
            
        Returns:
            Dict[int, List[int]]: Dictionary mapping diaObjectId to list of diaSourceIds
        """
        print(f"Mapping {len(dia_object_ids)} diaObjectIds to their diaSourceIds...")
        
        # Convert to appropriate data types
        dia_object_ids = [int(obj_id) for obj_id in dia_object_ids]
        
        results = {}
        missing_objects = []
        
        # First, try using diasource_index if available
        print("Using diasource_index for mapping...")
        
        # Ensure proper data types
        diasource_index = self.diasource_index.copy()
        if diasource_index['diaObjectId'].dtype == 'object':
            diasource_index['diaObjectId'] = pd.to_numeric(diasource_index['diaObjectId'], errors='coerce').astype(np.int64)
        
        for obj_id in dia_object_ids:
            # Find all sources for this object
            sources_for_object = diasource_index[diasource_index['diaObjectId'] == obj_id]
            if len(sources_for_object) > 0:
                source_ids = sources_for_object.index.tolist()
                results[obj_id] = source_ids
            else:
                missing_objects.append(obj_id)
        
        return results
    
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
    """
    =====================================================================
    LIGHT CURVE STATS
    =====================================================================
    """
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


    """
    =====================================================================
    FILTERED LIGHTCURVES
    =====================================================================
    """

    def get_filtered_lightcurves_summary(self, plot: bool = True, filtered_dir: str = None):
        """
        Get comprehensive summary statistics of filtered lightcurves dataset.
        Works with the output of the SNR and extendedness filtering pipeline.
        
        Provides detailed statistics including:
        - Filtering pipeline statistics (kept/rejected counts)
        - Lightcurve characteristics (length, time span, flux distributions)
        - Band coverage statistics
        - Coordinate distributions
        
        Args:
            plot: If True, display summary plots
            filtered_dir: Path to filtered directory (default: auto-detect from pipeline output)
        
        Returns:
            Dict: Comprehensive summary statistics
        """
        import glob
        import json
        
        print(f"\n{'='*70}")
        print(f"FILTERED LIGHTCURVES DATASET SUMMARY")
        print(f"{'='*70}\n")
        
        # Auto-detect filtered directory if not provided
        if filtered_dir is None:
            # Check if we're already in the filtered directory
            if self.lightcurve_path.name == 'extendedness_filtered':
                search_path = self.lightcurve_path
            # Otherwise look for the standard filtered output location
            elif (self.lightcurve_path / 'extendedness_filtered').exists():
                search_path = self.lightcurve_path / 'extendedness_filtered'
            # Or check parent directory for extendedness_filtered
            elif (self.lightcurve_path.parent / 'lightcurves' / 'extendedness_filtered').exists():
                search_path = self.lightcurve_path.parent / 'lightcurves' / 'extendedness_filtered'
            else:
                search_path = self.lightcurve_path
        else:
            search_path = Path(filtered_dir)
        
        print(f"Searching for metadata in: {search_path}")
        
        # 1. Read filtering metadata
        metadata_files = list(search_path.glob("*_filter_metadata.json"))
        
        if not metadata_files:
            print("No filtering metadata found. Run the filtering pipeline first.")
            return None
        
        total_kept = 0
        total_rejected = 0
        rejected_point_host = 0
        rejected_low_flux_ratio = 0
        files_processed = 0
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    kept = metadata.get('kept_objects', 0)
                    rejected = metadata.get('rejected_objects', 0)
                    point_host = metadata.get('rejected_point_host', 0)
                    flux_ratio = metadata.get('rejected_low_flux_ratio', 0)
                    
                    total_kept += kept
                    total_rejected += rejected
                    rejected_point_host += point_host
                    rejected_low_flux_ratio += flux_ratio
                    files_processed += 1
            except Exception as e:
                print(f"Error reading {metadata_file}: {e}")
                continue
        
        print(f"FILTERING PIPELINE STATISTICS")
        print(f"   Patch files processed: {files_processed}")
        print(f"   Objects kept: {total_kept}")
        print(f"   Objects rejected: {total_rejected}")
        print(f"     |- Point source hosts: {rejected_point_host}")
        print(f"     |- Low flux ratio: {rejected_low_flux_ratio}")
        total_input = total_kept + total_rejected
        if total_input > 0:
            print(f"   Overall keep rate: {total_kept/total_input*100:.1f}%")
        print()
        
        # 2. Analyze actual lightcurve data from patch files
        print(f"ANALYZING LIGHTCURVE CHARACTERISTICS...")
        
        num_points_list = []
        time_spans = []
        mean_fluxes = []
        all_bands = []
        ra_coords = []
        dec_coords = []
        num_sources_total = 0
        
        # Look for patch files in the same directory as metadata
        patch_files = sorted(search_path.glob("patch_*.h5"))
        
        for patch_file in patch_files:
            try:
                # Skip empty files (all objects filtered out)
                if patch_file.stat().st_size < 2048:  # Less than 2KB means likely empty
                    continue
                
                # Read lightcurve data
                lc_data = pd.read_hdf(patch_file, key="lightcurves")
                
                if len(lc_data) == 0:
                    continue
                
                # Group by diaObjectId to get per-object statistics
                if 'diaObjectId' in lc_data.columns:
                    grouped = lc_data.groupby('diaObjectId')
                    
                    for obj_id, obj_data in grouped:
                        num_points_list.append(len(obj_data))
                        num_sources_total += len(obj_data)
                        
                        # Time span
                        if 'midpointMjdTai' in obj_data.columns:
                            time_span = obj_data['midpointMjdTai'].max() - obj_data['midpointMjdTai'].min()
                            time_spans.append(time_span)
                        
                        # Mean flux
                        if 'psfFlux' in obj_data.columns:
                            mean_fluxes.append(obj_data['psfFlux'].mean())
                        
                        # Bands
                        if 'band' in obj_data.columns:
                            all_bands.extend(obj_data['band'].unique().tolist())
                        
                        # Coordinates (first observation)
                        if 'ra' in obj_data.columns and 'dec' in obj_data.columns:
                            ra_coords.append(obj_data['ra'].iloc[0])
                            dec_coords.append(obj_data['dec'].iloc[0])
                
            except Exception as e:
                print(f"   Warning: Error reading {patch_file.name}: {e}")
                continue
        
        num_lightcurves = len(num_points_list)
        
        print(f"\nDATASET COMPOSITION")
        print(f"   Total lightcurves: {num_lightcurves}")
        print(f"   Total sources (observations): {num_sources_total}")
        print(f"   Average sources per lightcurve: {num_sources_total/num_lightcurves:.1f}" if num_lightcurves > 0 else "   N/A")
        print()
        
        # 3. Lightcurve length statistics
        if num_points_list:
            print(f"LIGHTCURVE LENGTH (number of observations)")
            print(f"   Mean: {np.mean(num_points_list):.1f}")
            print(f"   Median: {np.median(num_points_list):.1f}")
            print(f"   Std: {np.std(num_points_list):.1f}")
            print(f"   Min: {int(np.min(num_points_list))}")
            print(f"   Max: {int(np.max(num_points_list))}")
            print(f"   25th percentile: {np.percentile(num_points_list, 25):.1f}")
            print(f"   75th percentile: {np.percentile(num_points_list, 75):.1f}")
            print()
        
        # 4. Time span statistics
        if time_spans:
            print(f"TIME SPAN (days)")
            print(f"   Mean: {np.mean(time_spans):.1f} days")
            print(f"   Median: {np.median(time_spans):.1f} days")
            print(f"   Std: {np.std(time_spans):.1f} days")
            print(f"   Min: {np.min(time_spans):.1f} days")
            print(f"   Max: {np.max(time_spans):.1f} days")
            print()
        
        # 5. Flux statistics
        if mean_fluxes:
            print(f"MEAN PSF FLUX")
            print(f"   Mean: {np.mean(mean_fluxes):.2f}")
            print(f"   Median: {np.median(mean_fluxes):.2f}")
            print(f"   Std: {np.std(mean_fluxes):.2f}")
            print(f"   Min: {np.min(mean_fluxes):.2f}")
            print(f"   Max: {np.max(mean_fluxes):.2f}")
            print()
        
        # 6. Band coverage
        if all_bands:
            band_counts = pd.Series(all_bands).value_counts()
            print(f"BAND COVERAGE")
            print(f"   Unique bands: {sorted(set(all_bands))}")
            for band, count in band_counts.items():
                pct = count / len(all_bands) * 100
                print(f"   {band}: {count} observations ({pct:.1f}%)")
            print()
        
        # 7. Spatial coverage
        if ra_coords and dec_coords:
            print(f"SPATIAL COVERAGE")
            print(f"   RA range: [{np.min(ra_coords):.4f} deg, {np.max(ra_coords):.4f} deg]")
            print(f"   Dec range: [{np.min(dec_coords):.4f} deg, {np.max(dec_coords):.4f} deg]")
            print(f"   RA span: {np.max(ra_coords) - np.min(ra_coords):.4f} deg")
            print(f"   Dec span: {np.max(dec_coords) - np.min(dec_coords):.4f} deg")
            print()
        
        print(f"{'='*70}\n")
        
        # 8. Create comprehensive summary dictionary
        summary = {
            'filtering': {
                'files_processed': files_processed,
                'objects_kept': total_kept,
                'objects_rejected': total_rejected,
                'rejected_point_host': rejected_point_host,
                'rejected_low_flux_ratio': rejected_low_flux_ratio,
                'rejection_rate': total_rejected / (total_kept + total_rejected) if (total_kept + total_rejected) > 0 else 0
            },
            'dataset': {
                'num_lightcurves': num_lightcurves,
                'num_sources': num_sources_total,
                'avg_sources_per_lc': num_sources_total / num_lightcurves if num_lightcurves > 0 else 0
            },
            'lightcurve_length': {
                'mean': float(np.mean(num_points_list)) if num_points_list else None,
                'median': float(np.median(num_points_list)) if num_points_list else None,
                'std': float(np.std(num_points_list)) if num_points_list else None,
                'min': int(np.min(num_points_list)) if num_points_list else None,
                'max': int(np.max(num_points_list)) if num_points_list else None,
                'q25': float(np.percentile(num_points_list, 25)) if num_points_list else None,
                'q75': float(np.percentile(num_points_list, 75)) if num_points_list else None
            },
            'time_span_days': {
                'mean': float(np.mean(time_spans)) if time_spans else None,
                'median': float(np.median(time_spans)) if time_spans else None,
                'std': float(np.std(time_spans)) if time_spans else None,
                'min': float(np.min(time_spans)) if time_spans else None,
                'max': float(np.max(time_spans)) if time_spans else None
            },
            'mean_psf_flux': {
                'mean': float(np.mean(mean_fluxes)) if mean_fluxes else None,
                'median': float(np.median(mean_fluxes)) if mean_fluxes else None,
                'std': float(np.std(mean_fluxes)) if mean_fluxes else None,
                'min': float(np.min(mean_fluxes)) if mean_fluxes else None,
                'max': float(np.max(mean_fluxes)) if mean_fluxes else None
            },
            'bands': {
                'unique': sorted(set(all_bands)) if all_bands else [],
                'counts': dict(pd.Series(all_bands).value_counts()) if all_bands else {}
            },
            'spatial_coverage': {
                'ra_range': [float(np.min(ra_coords)), float(np.max(ra_coords))] if ra_coords else None,
                'dec_range': [float(np.min(dec_coords)), float(np.max(dec_coords))] if dec_coords else None,
                'ra_span': float(np.max(ra_coords) - np.min(ra_coords)) if ra_coords else None,
                'dec_span': float(np.max(dec_coords) - np.min(dec_coords)) if dec_coords else None
            }
        }
        
        # 9. Generate plots
        if plot and num_points_list:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            # Plot 1: Lightcurve length distribution
            axes[0].hist(num_points_list, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            axes[0].axvline(np.mean(num_points_list), color='red', linestyle='--', label=f'Mean: {np.mean(num_points_list):.1f}')
            axes[0].axvline(np.median(num_points_list), color='orange', linestyle='--', label=f'Median: {np.median(num_points_list):.1f}')
            axes[0].set_xlabel('Number of Observations')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Lightcurve Length Distribution')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # Plot 2: Time span distribution
            if time_spans:
                axes[1].hist(time_spans, bins=25, color='orange', edgecolor='black', alpha=0.7)
                axes[1].axvline(np.mean(time_spans), color='red', linestyle='--', label=f'Mean: {np.mean(time_spans):.1f}d')
                axes[1].set_xlabel('Time Span (days)')
                axes[1].set_ylabel('Count')
                axes[1].set_title('Time Span Distribution')
                axes[1].legend()
                axes[1].grid(alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'No time span data', ha='center', va='center')
                axes[1].set_title('Time Span Distribution')
            
            # Plot 3: Mean flux distribution
            if mean_fluxes:
                axes[2].hist(mean_fluxes, bins=50, color='purple', edgecolor='black', alpha=0.7)
                axes[2].set_xlabel('Mean PSF Flux')
                axes[2].set_ylabel('Count')
                axes[2].set_title('Mean Flux Distribution')
                axes[2].grid(alpha=0.3)
                axes[2].set_xscale('log')
            else:
                axes[2].text(0.5, 0.5, 'No flux data', ha='center', va='center')
                axes[2].set_title('Mean Flux Distribution')
            
            # Plot 4: Band coverage
            if all_bands:
                band_counts = pd.Series(all_bands).value_counts().sort_index()
                band_counts.plot(kind='bar', ax=axes[3], color='green', edgecolor='black', alpha=0.7)
                axes[3].set_xlabel('Band')
                axes[3].set_ylabel('Number of Observations')
                axes[3].set_title('Band Coverage')
                axes[3].tick_params(axis='x', rotation=0)
                axes[3].grid(alpha=0.3)
            else:
                axes[3].text(0.5, 0.5, 'No band data', ha='center', va='center')
                axes[3].set_title('Band Coverage')
            
            # Plot 5: Spatial distribution
            if ra_coords and dec_coords:
                scatter = axes[4].scatter(ra_coords, dec_coords, c=num_points_list, cmap='viridis', s=2, alpha=1)
                axes[4].set_xlabel('RA (degrees)')
                axes[4].set_ylabel('Dec (degrees)')
                axes[4].set_title('Spatial Distribution (colored by length)')
                plt.colorbar(scatter, ax=axes[4], label='Observations')
                axes[4].grid(alpha=0.3)
            else:
                axes[4].text(0.5, 0.5, 'No coordinate data', ha='center', va='center')
                axes[4].set_title('Spatial Distribution')
            
            # Plot 6: Filtering summary pie chart
            if total_kept > 0 or total_rejected > 0:
                sizes = [total_kept, rejected_point_host, rejected_low_flux_ratio]
                labels = ['Kept', 'Point Source Hosts', 'Low Flux Ratio']
                colors = ['green', 'orange', 'red']
                explode = (0.05, 0, 0)
                axes[5].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', explode=explode, startangle=90)
                axes[5].set_title('Filtering Results')
            else:
                axes[5].text(0.5, 0.5, 'No filtering data', ha='center', va='center')
                axes[5].set_title('Filtering Results')
            
            plt.suptitle(f'Filtered Lightcurves Dataset Summary ({num_lightcurves} objects, {num_sources_total} sources)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        return summary

    def get_filtered_object_ids(self):
        """
        Get list of all diaObjectIds that passed the filtering pipeline.
        
        Returns:
            List[int]: List of filtered diaObjectIds
        """
        if self.index is None or len(self.index) == 0:
            print("No lightcurve index found.")
            return []
        
        filtered_ids = list(self.index.index)
        print(f"Found {len(filtered_ids)} filtered objects in the dataset")
        return filtered_ids
    
    def get_filtered_source_ids(self):
        """
        Get list of all diaSourceIds that passed the filtering pipeline.
        
        Returns:
            List[int]: List of filtered diaSourceIds
        """
        if self.diasource_index is None or self.diasource_index.empty:
            print("No diasource index found.")
            return []
        
        filtered_source_ids = list(self.diasource_index.index)
        print(f"Found {len(filtered_source_ids)} filtered sources in the dataset")
        return filtered_source_ids

    def save_filtered_subset_dataset(
        self,
        out_dir: Path,
        filtered_source_dir: Path = None
    ):
        """
        Save a mini-dataset (lightcurves, cutouts, features) for filtered sources.
        Uses the output from the SNR and extendedness filtering pipeline.
        Now uses existing lightcurve loader methods to map diaObjectId -> diaSourceId.
        
        Args:
            out_dir: Output directory for the mini-dataset
            filtered_source_dir: Path to filtered lightcurves directory (default: auto-detect extendedness_filtered)
        
        Returns:
            None
        """
        import shutil
        import h5py
        
        # Auto-detect filtered directory if not provided (same logic as summary method)
        if filtered_source_dir is None:
            if self.lightcurve_path.name == 'extendedness_filtered':
                filtered_source_dir = self.lightcurve_path
            elif (self.lightcurve_path / 'extendedness_filtered').exists():
                filtered_source_dir = self.lightcurve_path / 'extendedness_filtered'
            elif (self.lightcurve_path.parent / 'lightcurves' / 'extendedness_filtered').exists():
                filtered_source_dir = self.lightcurve_path.parent / 'lightcurves' / 'extendedness_filtered'
            else:
                filtered_source_dir = self.lightcurve_path
        else:
            filtered_source_dir = Path(filtered_source_dir)
        
        print(f"Using filtered lightcurves from: {filtered_source_dir}")
        
        out_dir = Path(out_dir)
        out_lc_dir = out_dir / "lightcurves"
        out_cutout_dir = out_dir / "cutouts"
        out_feat_dir = out_dir / "features"
        out_lc_dir.mkdir(parents=True, exist_ok=True)
        out_cutout_dir.mkdir(parents=True, exist_ok=True)
        out_feat_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Copy filtered patch files and collect filtered diaObjectIds
        patch_files = list(filtered_source_dir.glob("patch_*.h5"))
        num_copied = 0
        num_skipped = 0
        filtered_objects = set()
        
        print(f"Reading {len(patch_files)} filtered lightcurve patch files...")
        for i, patch_file in enumerate(patch_files, 1):
            # Skip empty files
            if patch_file.stat().st_size < 2048:
                num_skipped += 1
                continue
            
            # Read patch to collect filtered diaObjectIds
            try:
                lc_df = pd.read_hdf(patch_file, key='lightcurves')
                if len(lc_df) > 0 and 'diaObjectId' in lc_df.columns:
                    filtered_objects.update(lc_df['diaObjectId'].unique())
            except Exception as e:
                print(f"  Warning: Could not read {patch_file.name}: {e}")
                continue
            
            # Copy the patch file
            out_patch = out_lc_dir / patch_file.name
            if not out_patch.exists():
                shutil.copy2(patch_file, out_patch)
                num_copied += 1
                if num_copied % 10 == 0:
                    print(f"  Processed {num_copied}/{len(patch_files)} patch files...")
        
        print(f"\nFiltered lightcurve summary:")
        print(f"  - Copied {num_copied} patch files (skipped {num_skipped} empty)")
        print(f"  - Found {len(filtered_objects)} unique diaObjectIds")

        # Step 2: Use the parent lightcurve loader to map diaObjectId -> diaSourceIds
        # This uses the existing get_source_ids_for_objects() method
        print(f"\nMapping {len(filtered_objects)} diaObjectIds to diaSourceIds using parent lightcurve loader...")
        parent_lc_path = self.lightcurve_path.parent / "lightcurves"
        
        # Create a temporary lightcurve loader for the parent (unfiltered) lightcurves
        parent_lc_loader = LightCurveLoader(parent_lc_path)
        
        # Get mapping of diaObjectId -> [diaSourceIds]
        object_to_sources = parent_lc_loader.get_source_ids_for_objects(list(filtered_objects))
        
        # Collect all diaSourceIds and group by visit
        all_source_ids = set()
        for obj_id, source_ids in object_to_sources.items():
            all_source_ids.update(source_ids)
        
        print(f"Found {len(all_source_ids)} diaSourceIds for {len(filtered_objects)} objects")
        
        # Step 3: Group source IDs by visit using the parent lightcurve's diasource_index
        print(f"\nGrouping diaSourceIds by visit...")
        visit_to_source_ids = {}  # visit -> set of diaSourceIds
        
        diasource_index = parent_lc_loader.diasource_index
        if not diasource_index.empty and 'visit' in diasource_index.columns:
            for source_id in all_source_ids:
                try:
                    visit_info = diasource_index.loc[source_id]
                    if isinstance(visit_info, pd.Series):
                        visit = int(visit_info['visit'])
                    else:  # DataFrame (multiple visits for same source)
                        visit = int(visit_info['visit'].iloc[0])
                    
                    if visit not in visit_to_source_ids:
                        visit_to_source_ids[visit] = set()
                    visit_to_source_ids[visit].add(source_id)
                except (KeyError, IndexError):
                    continue
            
            print(f"Mapped source IDs to {len(visit_to_source_ids)} visits")
        else:
            print("Error: diasource_index not available or missing 'visit' column")
            print("Cannot proceed with filtering - diasource_index is required")
            return

        # Step 4: Filter features using diaSourceId mappings
        feature_dir = self.lightcurve_path.parent / "features"
        if feature_dir.exists() and visit_to_source_ids:
            print(f"\nFiltering features for {len(visit_to_source_ids)} visits...")
            
            num_feat_visits = 0
            num_feat_samples = 0
            
            for visit, source_ids in visit_to_source_ids.items():
                feature_file = feature_dir / f"visit_{visit}_features.h5"
                if not feature_file.exists():
                    continue
                
                out_feat_file = out_feat_dir / f"visit_{visit}_features.h5"
                if out_feat_file.exists():
                    continue
                
                # Read and filter features for this visit
                try:
                    feat_df = pd.read_hdf(feature_file, key='features')
                    # Filter by diaSourceId (index of the dataframe)
                    filtered_df = feat_df[feat_df.index.isin(source_ids)]
                    
                    if len(filtered_df) > 0:
                        # Save filtered features
                        filtered_df.to_hdf(out_feat_file, key='features', mode='w', format='table')
                        num_feat_visits += 1
                        num_feat_samples += len(filtered_df)
                    
                    if num_feat_visits % 10 == 0:
                        print(f"  Processed {num_feat_visits} visits, {num_feat_samples} samples so far...")
                
                except Exception as e:
                    print(f"  Error filtering features for visit {visit}: {e}")
                    if out_feat_file.exists():
                        out_feat_file.unlink()
                    continue
            
            print(f"Filtered features: {num_feat_visits} visit files, {num_feat_samples} total samples")
        else:
            num_feat_visits = 0
            num_feat_samples = 0
            print(f"Warning: Feature directory not found or no source ID mappings")

        # Step 5: Filter cutouts using diaSourceId mappings
        cutout_dir = self.lightcurve_path.parent / "cutouts"
        if cutout_dir.exists() and visit_to_source_ids:
            print(f"\nFiltering cutouts using diaSourceId mappings...")
            
            num_cutout_visits = 0
            num_cutout_samples = 0
            
            for visit, source_ids in visit_to_source_ids.items():
                cutout_file = cutout_dir / f"visit_{visit}.h5"
                if not cutout_file.exists():
                    print(f"  Warning: Cutout file not found for visit {visit}")
                    continue
                
                out_cutout_file = out_cutout_dir / f"visit_{visit}.h5"
                if out_cutout_file.exists():
                    continue
                
                # Read and filter cutouts for this visit
                try:
                    with h5py.File(cutout_file, 'r') as f_in:
                        # Cutout files have diaSourceId
                        if 'diaSourceId' not in f_in:
                            print(f"  Warning: No diaSourceId in cutout file for visit {visit}")
                            continue
                        
                        dia_source_ids = f_in['diaSourceId'][:]
                        
                        # Find indices matching our filtered source IDs
                        mask = np.isin(dia_source_ids, list(source_ids))
                        indices = np.where(mask)[0]
                        
                        if len(indices) == 0:
                            print(f"  Warning: No matching cutouts for visit {visit}")
                            continue
                        
                        # Create filtered output file
                        with h5py.File(out_cutout_file, 'w') as f_out:
                            # Copy filtered data for each dataset
                            for key in f_in.keys():
                                data = f_in[key][:]
                                f_out.create_dataset(key, data=data[indices], compression='gzip')
                        
                        num_cutout_visits += 1
                        num_cutout_samples += len(indices)
                        
                        if num_cutout_visits % 10 == 0:
                            print(f"  Processed {num_cutout_visits}/{len(visit_to_source_ids)} visits, {num_cutout_samples} samples so far...")
                
                except Exception as e:
                    print(f"  Error filtering cutouts for visit {visit}: {e}")
                    if out_cutout_file.exists():
                        out_cutout_file.unlink()
                    continue
            
            print(f"Filtered cutouts: {num_cutout_visits} visit files, {num_cutout_samples} total samples")
        else:
            num_cutout_visits = 0
            num_cutout_samples = 0
            if not cutout_dir.exists():
                print(f"Warning: Cutout directory not found at {cutout_dir}")
            elif not visit_to_source_ids:
                print(f"Warning: No diaSourceId mappings available")

        # Step 6: Create a global index for the filtered dataset
        print(f"\nCreating global index for filtered dataset...")
        try:
            global_index = {}
            # Scan all cutout files in output directory
            for cutout_file in out_cutout_dir.glob("visit_*.h5"):
                try:
                    visit = int(cutout_file.stem.split('_')[1])
                    with h5py.File(cutout_file, 'r') as f:
                        if 'diaSourceId' in f:
                            dia_source_ids = f['diaSourceId'][:]
                            for dia_source_id in dia_source_ids:
                                global_index[int(dia_source_id)] = visit
                except Exception as e:
                    print(f"  Warning: Could not process {cutout_file.name} for index: {e}")
            
            # Save global index
            index_file = out_dir / "cutout_global_index.h5"
            with h5py.File(index_file, 'w') as f:
                if global_index:
                    ids = np.array(list(global_index.keys()), dtype=np.int64)
                    visits = np.array(list(global_index.values()), dtype=np.int32)
                    f.create_dataset('diaSourceId', data=ids, compression='gzip')
                    f.create_dataset('visit', data=visits, compression='gzip')
            
            print(f"Created global index with {len(global_index)} diaSourceId mappings")
        except Exception as e:
            print(f"Warning: Could not create global index: {e}")

        print(f"\n{'='*70}")
        print(f"FILTERED DATASET CREATED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Output directory: {out_dir}")
        print(f"Summary:")
        print(f"  - {num_copied} filtered lightcurve patch files")
        print(f"  - {len(filtered_objects)} unique diaObjectIds")
        print(f"  - {num_cutout_visits} cutout visit files ({num_cutout_samples} samples)")
        print(f"  - {num_feat_visits} feature visit files ({num_feat_samples} samples)")
        print(f"{'='*70}\n")
