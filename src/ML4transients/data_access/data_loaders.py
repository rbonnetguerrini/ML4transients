"""
Data loaders for transient detection dataset.

This module provides lazy-loading classes for accessing cutout images, tabular features, and model inference results stored in HDF5 format.
The loaders are designed to minimize memory usage through on-demand loading.
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import hashlib
import yaml
import torch  

class CutoutLoader:
    """Lazy loader for image cutouts stored in HDF5 format.
    
    Provides efficient access to image data with on-demand loading to minimize memory usage. Cutouts are small astronomical images centered on detected transient candidates.
    
    Parameters
    ----------
    file_path : Path
        Path to HDF5 file containing cutouts dataset
        
    Attributes
    ----------
    file_path : Path
        Path to the HDF5 file
    """
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
        self._ids = None
    
    @property
    def data(self) -> np.ndarray:
        """Get all cutout arrays (lazy loaded).
        
        Returns
        -------
        np.ndarray
            Array of cutout images with shape (n_samples, height, width, channels)
        """
        if self._data is None:
            with h5py.File(self.file_path, 'r') as f:
                self._data = f['cutouts'][:]
        return self._data
    
    @property
    def ids(self) -> np.ndarray:
        """Get all diaSourceIds (lazy loaded).
        
        Returns
        -------
        np.ndarray
            Array of diaSourceId values corresponding to cutouts
        """
        if self._ids is None:
            with h5py.File(self.file_path, 'r') as f:
                self._ids = f['diaSourceId'][:]
        return self._ids
    
    def get_by_id(self, dia_source_id: int) -> Optional[np.ndarray]:
        """Get specific cutout by diaSourceId.
        
        Parameters
        ----------
        dia_source_id : int
            The diaSourceId to retrieve
            
        Returns
        -------
        np.ndarray or None
            Cutout array if found, None otherwise
        """
        idx = np.where(self.ids == dia_source_id)[0]
        if len(idx) == 0:
            return None
        return self.data[idx[0]]

class FeatureLoader:
    """Lazy loader for tabular features stored in HDF5/Pandas format.
    
    Provides efficient access to features (magnitudes, colors, etc.)
    with column-wise and row-wise on-demand loading to minimize memory usage.
    
    Parameters
    ----------
    file_path : Path
        Path to HDF5 file containing features table
    """
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
        self._ids = None
        self._labels = None
    
    @property
    def ids(self) -> pd.Index:
        """Get diaSourceIds without loading full data.
        
        Returns
        -------
        pd.Index
            Index of diaSourceId values
        """
        if self._ids is None:
            with pd.HDFStore(self.file_path, 'r') as store:
                self._ids = store.select_column('features', 'index')
        return self._ids
    
    @property
    def labels(self) -> pd.DataFrame:
        """Get labels column without loading all features.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing only the is_injection column
        """
        if self._labels is None:
            try:
                with pd.HDFStore(self.file_path, 'r') as store:
                    self._labels = store.select('features', columns=['is_injection'])
            except (ValueError, TypeError):
                # Fallback for fixed format files
                print(f"Warning: Loading full features for labels from {self.file_path}")
                self._labels = self.data[['is_injection']]
        return self._labels
    
    def get_by_id(self, dia_source_id: int) -> pd.DataFrame:
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
            return store.select('features', where=f'index == {dia_source_id}')

class InferenceLoader:
    """Handles model inference results loading and execution.
    
    Manages inference results for a specific visit, including running new inference
    if results don't exist. Supports various model types (standard, ensemble, 
    co-teaching) with probabilistic outputs.
    
    Parameters
    ----------
    data_path : Path
        Path to data directory
    visit : int
        Visit number for this inference
    weights_path : str, optional
        Path to model weights directory (for running inference)
    """
    
    def __init__(self, data_path: Path, visit: int, weights_path: str = None):
        self.data_path = Path(data_path)
        self.visit = visit
        self.weights_path = weights_path
        
        # Generate model hash for file naming
        self.model_hash = None
        if weights_path:
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
        """Run model inference for this visit and save results.
        
        Parameters
        ----------
        dataset_loader : DatasetLoader
            Dataset loader instance for accessing data
        trainer : object, optional
            Pre-loaded trainer for efficiency (avoids reloading model)
        force : bool
            Force re-run even if results exist
            
        Raises
        ------
        ValueError
            If weights_path is required but not provided
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
        
        inference_dataset = PytorchDataset.create_inference_dataset(
            dataset_loader, visit=self.visit
        )
        print(f"Created inference dataset for visit {self.visit} with {len(inference_dataset)} samples")
        
        # Create DataLoader with optimized settings
        inference_loader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
        
        # Run inference
        from ML4transients.evaluation.inference import infer
        
        dia_source_ids = inference_dataset.get_dia_source_ids()
        
        results = infer(
            inference_loader=inference_loader,
            trainer=trainer,
            weights_path=self.weights_path,
            return_preds=True,
            compute_metrics=True,
            device=None,
            save_path=str(self._inference_file),
            dia_source_ids=dia_source_ids,
            visit=self.visit,
            model_hash=self.model_hash,
            return_probabilities=None  # Auto-detect based on model type
        )
        
        if results:
            print(f"Inference completed for visit {self.visit}")
            self._clear_cache()
        else:
            print(f"Warning: No inference results returned for visit {self.visit}")
            
        # Cleanup memory
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
        """Check if inference results file exists.
        
        Returns
        -------
        bool
            True if results file exists, False otherwise
        """
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
                
                # Load probabilistic outputs if available
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
        """Get binary predictions array (lazy loaded).
        
        Returns
        -------
        np.ndarray
            Binary predictions (0/1) for each sample
        """
        if self._predictions is None:
            self._load_data()
        return self._predictions

    @property
    def labels(self) -> np.ndarray:
        """Get true labels array (lazy loaded).
        
        Returns
        -------
        np.ndarray
            True binary labels (0/1) for each sample
        """
        if self._labels is None:
            self._load_data()
        return self._labels
        
    @property
    def probabilities(self) -> Optional[np.ndarray]:
        """Get prediction probabilities array (lazy loaded, if available).
        
        Returns
        -------
        np.ndarray or None
            Prediction probabilities [0,1] if available, None otherwise
        """
        if self._probabilities is None and self.has_inference_results():
            self._load_data()
        return self._probabilities
        
    @property
    def uncertainties(self) -> Optional[np.ndarray]:
        """Get prediction uncertainties array (lazy loaded, if available).
        
        Returns
        -------
        np.ndarray or None
            Prediction uncertainties if available, None otherwise
        """
        if self._uncertainties is None and self.has_inference_results():
            self._load_data()
        return self._uncertainties

    @property
    def ids(self) -> np.ndarray:
        """Get diaSourceId array (lazy loaded).
        
        Returns
        -------
        np.ndarray
            Array of diaSourceId values corresponding to predictions
        """
        if self._ids is None:
            self._load_data()
        return self._ids

    def get_results_by_id(self, dia_source_id: int) -> Optional[Dict]:
        """Get inference results for a specific diaSourceId.
        
        Parameters
        ----------
        dia_source_id : int
            The diaSourceId to retrieve results for
            
        Returns
        -------
        dict or None
            Dictionary containing prediction results if found, None otherwise
        """
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
    """Placeholder for future light curve data loading.
    
    Reserved for potential extension to time-series photometric data.
    
    Parameters
    ----------
    file_path : Path
        Path to light curve data file
    """
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
    
    @property
    def data(self):
        """Placeholder for light curve data access."""
        # TODO: Implement light curve loading when data format is defined
        pass