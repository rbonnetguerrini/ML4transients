import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import warnings
from typing import List, Union, Optional
from pathlib import Path

# Add the ML4transients package to path
sys.path.append('/sps/lsst/users/rbonnetguerrini/ML4transients/src')
from ML4transients.data_access.dataset_loader import DatasetLoader

class PytorchDataset(Dataset):
    def __init__(self, 
                 data_source: Union[str, Path, List[Union[str, Path]], DatasetLoader],
                 visits: Optional[List[int]] = None,
                 train: bool = False, 
                 val: bool = False, 
                 test: bool = False, 
                 transform=None,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42):
        """
        PyTorch Dataset using DatasetLoader - loads all data into memory for fast training.
        
        Args:
            data_source: Path(s) to data directories OR DatasetLoader instance
            visits: List of specific visits to use (None for all available)
            train: If True, use training split
            test: If True, use test split
            transform: PyTorch transforms to apply
            test_size: Fraction of data to use for testing
            random_state: Random seed for train/test split
        """
        # Handle both path and DatasetLoader instance
        if isinstance(data_source, DatasetLoader):
            self.loader = data_source
        else:
            self.loader = DatasetLoader(data_source)
        
        self.transform = transform
        
        # Filter visits if specified
        available_visits = self.loader.visits
        if visits is not None:
            available_visits = [v for v in visits if v in available_visits]
        
        # Collect all data into memory
        all_cutouts = []
        all_features = []
        all_ids = []
        
        for visit in available_visits:
            if visit in self.loader.cutouts and visit in self.loader.features:
                cutouts = self.loader.get_all_cutouts(visit)
                features = self.loader.get_all_features(visit)
                
                # Match cutouts with features by diaSourceId
                cutout_ids = self.loader.cutouts[visit].ids
                feature_ids = features.index.values
                
                # Find common IDs
                common_ids = np.intersect1d(cutout_ids, feature_ids)
                
                for dia_id in common_ids:
                    cutout_idx = np.where(cutout_ids == dia_id)[0][0]
                    all_cutouts.append(cutouts[cutout_idx])
                    all_features.append(features.loc[dia_id])
                    all_ids.append(dia_id)
        
        # Convert to arrays
        self.images = np.array(all_cutouts)
        self.features_df = all_features
        self.dia_source_ids = np.array(all_ids)
        
        # Check if features are available
        if len(all_features) == 0:
            raise ValueError("No features provided. Dataset is empty.")

        # Create labels from features
        self.labels = np.array([f['is_injection'] for f in all_features])

        # Warn if only one class is present
        if np.all(self.labels == self.labels[0]):
            print(f"Only one class present, class {self.labels[0]}")

        # Train/val/test split
        if train or val or test:
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(self.images))
            
            # First split: train+val vs test
            trainval_idx, test_idx = train_test_split(
                indices, 
                test_size=test_size, 
                random_state=random_state,
                stratify=self.labels
            )
            
            # Second split: train vs val
            if val_size > 0:
                val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
                train_idx, val_idx = train_test_split(
                    trainval_idx,
                    test_size=val_size_adjusted,
                    random_state=random_state,
                    stratify=self.labels[trainval_idx]
                )
            else:
                train_idx = trainval_idx
                val_idx = []
            
            # Apply the split
            if train:
                self._apply_split(train_idx)
            elif val:
                self._apply_split(val_idx)
            elif test:
                self._apply_split(test_idx)
    
    def _apply_split(self, indices):
        """Helper to apply index split"""
        self.images = self.images[indices]
        self.labels = self.labels[indices]
        self.dia_source_ids = self.dia_source_ids[indices]
        self.features_df = [self.features_df[i] for i in indices]
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert image to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32).squeeze(-1)  # Remove channel dim: (H, W)
        image = image.unsqueeze(0)  # Add channel dim at front: (1, H, W)
        
        if self.transform:
            image = self.transform(image)

        # Ensure label is a PyTorch tensor
        label = torch.tensor(label, dtype=torch.float32)

        return image, label, idx

    def get_feature_by_idx(self, idx):
        """Get features for a specific index."""
        return self.features_df[idx]
    
    def get_dia_source_id(self, idx):
        """Get diaSourceId for a specific index."""
        return self.dia_source_ids[idx]
    
    def __repr__(self):
        return (f"PytorchDataset({len(self)} samples)\n"
                f"  Image shape: {self.images.shape}\n"
                f"  Labels: {np.sum(self.labels)} injected, {len(self.labels) - np.sum(self.labels)} real")
                