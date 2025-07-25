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
    """Lazy loader for feature data."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
        self._ids = None
    
    @property
    def data(self):
        """Lazy load features DataFrame."""
        if self._data is None:
            self._data = pd.read_hdf(self.file_path, key='features')
        return self._data
    @property
    def ids(self):
        """Lazy load diaSourceIds from the DataFrame index."""
        if self._ids is None:
            # Load just the index, not the full data
            with pd.HDFStore(self.file_path, 'r') as store:
                self._ids = self.data.index.values
        return self._ids

    def get_by_id(self, dia_source_id: int):
        """Get specific features by diaSourceId."""
        return self.data.loc[dia_source_id] if dia_source_id in self.data.index else None

class LightCurveLoader:
    """Placeholder for future lightcurve loading."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
    
    @property
    def data(self):
        # To be implemented when you have lightcurve data
        pass

class InferenceLoader:
    """Placeholder for future inference results loading."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
    
    @property
    def data(self):
        # To be implemented when you have inference results
        pass