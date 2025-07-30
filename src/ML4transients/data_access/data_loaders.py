import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

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
    """Placeholder for future inference results loading."""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
        self._ids = None
        self._models = None
        self._labels = None
    
    @property
    def data(self):
        """Lazy load all inference results."""
        if self._data is None:
            with pd.HDFStore(self.file_path, 'r') as store:
                self._data = store.select('inference')
        return self._data
    
    @property
    def true_labels(self):
        """Get true labels without loading full data - FAST."""
        if self._true_labels is None:
            with pd.HDFStore(self.file_path, 'r') as store:
                # Single file access, no joins needed
                self._true_labels = store.select('inference', columns=['true_label'])
        return self._true_labels
    
    @property
    def models(self):
        """Get available model names."""
        if self._models is None:
            # Extract model names from column names (excluding metadata columns)
            with pd.HDFStore(self.file_path, 'r') as store:
                columns = store.get_storer('inference').table.colnames
                # Assuming model results are stored as {model_name}_pred, {model_name}_prob
                self._models = list(set([col.rsplit('_', 1)[0] for col in columns 
                                       if col.endswith(('_pred', '_prob'))]))
        return self._models

    def ids(self):
        """Get diaSourceIds without loading full data."""
        if self._ids is None:
            with pd.HDFStore(self.file_path, 'r') as store:
                self._ids = store.select_column('inference', 'index')
        return self._ids

class LightCurveLoader:
    """Placeholder for future lightcurve loading."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
    
    @property
    def data(self):
        #todo
        pass