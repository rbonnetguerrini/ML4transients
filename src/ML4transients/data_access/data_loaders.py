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

class LightCurveLoader:
    """Placeholder for future lightcurve loading."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
    
    @property
    def data(self):
        #todo
        pass

class InferenceLoader:
    """Placeholder for future inference results loading."""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
    
    @property
    def data(self):
        #todo
        pass