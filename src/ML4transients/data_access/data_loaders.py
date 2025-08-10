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
                # Now this works efficiently with table format
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
    """Placeholder for future lightcurve loading."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
    
    @property
    def data(self):
        #todo
        pass