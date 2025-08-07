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
    """Loader for inference results with automatic inference running capability.
    
    Manages loading of inference results from HDF5 files and can automatically
    run inference if results don't exist.
    """

    def __init__(self, data_path: Path, visit: int, weights_path: str = None):
        """
        Initialize InferenceLoader for a specific visit.
        
        Args:
            data_path: Path to the data directory
            visit: Visit number
            weights_path: Path to trained model weights directory
        """
        self.data_path = Path(data_path)
        self.visit = visit
        self.weights_path = weights_path
        self._inference_file = None
        self._predictions = None
        self._labels = None
        self._ids = None
        self._metrics = None
        
        if weights_path:
            self._check_inference_availability()
    
    def _get_model_hash(self):
        """Generate a hash for the model configuration to identify inference runs.
        
        Returns
        -------
        str or None
            8-character hash based on config and model modification time,
            or None if weights_path is invalid
        """
        if not self.weights_path:
            return None
            
        config_path = Path(self.weights_path) / "config.yaml"
        weights_file = Path(self.weights_path) / "model_best.pth"
        
        if not (config_path.exists() and weights_file.exists()):
            return None
        
        # Create hash from config and model file modification time
        with open(config_path, 'r') as f:
            config_str = f.read()
        
        weights_mtime = weights_file.stat().st_mtime
        hash_string = f"{config_str}_{weights_mtime}"
        
        return hashlib.md5(hash_string.encode()).hexdigest()[:8]
    
    def _get_inference_filename(self):
        """Generate inference results filename for this visit.
        
        Returns
        -------
        str or None
            Filename in format 'visit_{visit}_inference_{hash}.h5',
            or None if model hash cannot be generated
        """
        model_hash = self._get_model_hash()
        if not model_hash:
            return None
        return f"visit_{self.visit}_inference_{model_hash}.h5"
    
    def _check_inference_availability(self):
        """Check if inference results exist for the given model and visit.
        
        Returns
        -------
        bool
            True if inference file exists, False otherwise
        """
        inference_filename = self._get_inference_filename()
        if not inference_filename:
            return False
            
        inference_dir = self.data_path / "inference"
        self._inference_file = inference_dir / inference_filename
        
        return self._inference_file.exists()
    
    def has_inference_results(self):
        """Check if inference results are available."""
        # If _inference_file is already set (e.g., by get_inference_loader_by_hash), check that
        if self._inference_file:
            return self._inference_file.exists()
        
        # Otherwise, check using the model hash method (requires weights_path)
        if self.weights_path:
            inference_filename = self._get_inference_filename()
            if inference_filename:
                inference_dir = self.data_path / "inference"
                self._inference_file = inference_dir / inference_filename
                return self._inference_file.exists()
        
        return False
    
    @property
    def predictions(self):
        """Lazy load predictions."""
        if self._predictions is None and self.has_inference_results():
            with h5py.File(self._inference_file, 'r') as f:
                self._predictions = f['predictions'][:]
        return self._predictions
    
    @property
    def labels(self):
        """Lazy load true labels."""
        if self._labels is None and self.has_inference_results():
            with h5py.File(self._inference_file, 'r') as f:
                self._labels = f['labels'][:]
        return self._labels
    
    @property
    def ids(self):
        """Lazy load diaSourceIds."""
        if self._ids is None and self.has_inference_results():
            with h5py.File(self._inference_file, 'r') as f:
                self._ids = f['diaSourceId'][:]
        return self._ids
    
    @property
    def metrics(self):
        """Lazy load computed metrics."""
        if self._metrics is None and self.has_inference_results():
            with h5py.File(self._inference_file, 'r') as f:
                self._metrics = {
                    'accuracy': f.attrs.get('accuracy', None),
                    'confusion_matrix': f['confusion_matrix'][:] if 'confusion_matrix' in f else None
                }
        return self._metrics
    
    def get_results_by_id(self, dia_source_id: int):
        """Get prediction and label for specific diaSourceId.
        
        Parameters
        ----------
        dia_source_id : int
            The diaSourceId to retrieve results for
            
        Returns
        -------
        dict or None
            Dictionary with 'prediction', 'label', and 'dia_source_id' keys,
            or None if ID not found or no inference results available
        """
        if not self.has_inference_results():
            return None
            
        idx = np.where(self.ids == dia_source_id)[0]
        if len(idx) == 0:
            return None
            
        idx = idx[0]
        return {
            'prediction': self.predictions[idx],
            'label': self.labels[idx],
            'dia_source_id': dia_source_id
        }
    
    def run_inference(self, dataset_loader, trainer=None, force=False):
        """
        Run inference for this specific visit only.
        
        Args:
            dataset_loader: DatasetLoader instance
            trainer: Optional pre-loaded trainer (avoids reloading model)
            force: Whether to force re-running inference even if results exists
        """
        if self.has_inference_results() and not force:
            print(f"Inference results already exist for visit {self.visit}")
            return
        
        if not self.weights_path and trainer is None:
            raise ValueError("No weights path or trainer provided. Cannot run inference.")
        
        # Check if this visit has data
        if self.visit not in dataset_loader.visits:
            print(f"Warning: Visit {self.visit} not found in dataset")
            return
        
        if self.visit not in dataset_loader.cutouts or self.visit not in dataset_loader.features:
            print(f"Warning: Visit {self.visit} missing cutouts or features")
            return
        
        print(f"Running inference for visit {self.visit}...")
        
        # Import here to avoid circular imports
        from ML4transients.training.pytorch_dataset import PytorchDataset
        from ML4transients.evaluation.inference import infer
        from torch.utils.data import DataLoader
        
        # Create inference dataset with optimized DataLoader
        inference_dataset = PytorchDataset.create_inference_dataset(dataset_loader, visit=self.visit)
        
        # Use larger batch size and optimized settings for inference
        batch_size = 128  # Larger batch size for inference efficiency
        inference_dataloader = DataLoader(
            inference_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Disable multiprocessing to avoid memory issues
            pin_memory=False,  # Disable pin_memory to save GPU memory
            drop_last=False
        )
        
        # Ensure inference directory exists
        inference_dir = self.data_path / "inference"
        inference_dir.mkdir(exist_ok=True)
        
        # Get model hash for registration
        model_hash = self._get_model_hash()
        
        try:
            print(f"Processing {len(inference_dataset)} samples in batches of {batch_size}...")
            
            # Run inference
            results = infer(
                inference_loader=inference_dataloader,
                trainer=trainer,
                weights_path=self.weights_path if trainer is None else None,
                return_preds=True,
                compute_metrics=True,
                save_path=self._inference_file,
                dia_source_ids=inference_dataset.dia_source_ids,
                visit=self.visit,
                model_hash=model_hash
            )
            
            # Register the new inference file
            if model_hash:
                dataset_loader._register_inference_file(
                    data_path=self.data_path,
                    visit=self.visit,
                    model_hash=model_hash,
                    weights_path=self.weights_path,
                    metadata={
                        'accuracy': results.get('accuracy'),
                        'n_samples': len(results.get('y_pred', []))
                    }
                )
            
            print(f"Inference completed for visit {self.visit}")
            
        except Exception as e:
            print(f"Error during inference for visit {self.visit}: {e}")
            raise
            
        finally:
            # Always cleanup memory after inference
            if hasattr(inference_dataset, 'cleanup_memory'):
                inference_dataset.cleanup_memory()
            del inference_dataset, inference_dataloader
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Clear cached data to force reload
        self._predictions = None
        self._labels = None
        self._ids = None
        self._metrics = None
    
    def prompt_and_run_inference(self, dataset_loader):
        """Check for existing results and prompt user to run inference if needed.
        
        Parameters
        ----------
        dataset_loader : DatasetLoader
            Dataset loader instance
            
        Returns
        -------
        bool
            True if inference results are available after this call,
            False if inference failed or was declined
        """
        if self.has_inference_results():
            print(f"Found existing inference results: {self._inference_file}")
            return True
        
        if not self.weights_path:
            print("No weights path provided. Cannot run inference.")
            return False
        
        print(f"No inference results found for model: {self.weights_path}")
        response = input("Would you like to run inference now? (Y/n): ").lower().strip()
        
        if response in ['', 'y', 'yes']:
            try:
                self.run_inference(dataset_loader)
                return True
            except Exception as e:
                print(f"Error running inference: {e}")
                return False
        else:
            print("Inference not run. No results available.")
            return False

class LightCurveLoader:
    """Placeholder for future lightcurve loading."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
    
    @property
    def data(self):
        #todo
        pass