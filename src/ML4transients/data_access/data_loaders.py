import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import yaml

class CutoutLoader:
    """Lazy loader for cutout data."""
    
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
        """Get specific cutout by diaSourceId."""
        idx = np.where(self.ids == dia_source_id)[0]
        if len(idx) == 0:
            return None
        return self.data[idx[0]]

class FeatureLoader:
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
        """Get specific features by diaSourceId."""
        with pd.HDFStore(self.file_path, 'r') as store:
            # Efficient query without loading full data
            return store.select('features', where=f'index == {dia_source_id}')
            
class InferenceLoader:
    """Loader for inference results with automatic inference running capability."""
    
    def __init__(self, data_path: Path, weights_path: str = None):
        """
        Initialize InferenceLoader.
        
        Args:
            data_path: Path to the data directory
            weights_path: Path to trained model weights directory
        """
        self.data_path = Path(data_path)
        self.weights_path = weights_path
        self._inference_file = None
        self._predictions = None
        self._labels = None
        self._ids = None
        self._metrics = None
        
        if weights_path:
            self._check_inference_availability()
    
    def _get_model_hash(self):
        """Generate a hash for the model configuration to identify inference runs."""
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
        """Generate inference results filename."""
        model_hash = self._get_model_hash()
        if not model_hash:
            return None
        return f"inference_results_{model_hash}.h5"
    
    def _check_inference_availability(self):
        """Check if inference results exist for the given model."""
        inference_filename = self._get_inference_filename()
        if not inference_filename:
            return False
            
        inference_dir = self.data_path / "inference"
        self._inference_file = inference_dir / inference_filename
        
        return self._inference_file.exists()
    
    def has_inference_results(self):
        """Check if inference results are available."""
        return self._inference_file and self._inference_file.exists()
    
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
        """Get prediction and label for specific diaSourceId."""
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
    
    def run_inference(self, dataset_loader, force=False):
        """
        Run inference and save results.
        
        Args:
            dataset_loader: DatasetLoader instance
            force: Whether to force re-running inference even if results exist
        """
        if self.has_inference_results() and not force:
            print(f"Inference results already exist at {self._inference_file}")
            response = input("Do you want to re-run inference? (y/N): ").lower().strip()
            if response != 'y':
                return
        
        if not self.weights_path:
            raise ValueError("No weights path provided. Cannot run inference.")
        
        print("Running inference...")
        
        # Import here to avoid circular imports
        from ML4transients.training.pytorch_dataset import PytorchDataset
        from ML4transients.evaluation.inference import infer
        from torch.utils.data import DataLoader
        
        # Create inference dataset
        inference_dataset = PytorchDataset.create_inference_dataset(dataset_loader)
        inference_dataloader = DataLoader(inference_dataset, batch_size=32, shuffle=False)
        
        # Ensure inference directory exists
        inference_dir = self.data_path / "inference"
        inference_dir.mkdir(exist_ok=True)
        
        # Run inference with saving
        results = infer(
            inference_loader=inference_dataloader,
            weights_path=self.weights_path,
            return_preds=True,
            compute_metrics=True,
            save_path=self._inference_file,
            dia_source_ids=inference_dataset.dia_source_ids
        )
        
        # Clear cached data to force reload
        self._predictions = None
        self._labels = None
        self._ids = None
        self._metrics = None
    
    def prompt_and_run_inference(self, dataset_loader):
        """
        Check for existing results and prompt user to run inference if needed.
        
        Args:
            dataset_loader: DatasetLoader instance
            
        Returns:
            bool: True if inference results are available after this call
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