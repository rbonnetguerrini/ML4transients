import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import warnings
from typing import List, Union, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add the ML4transients package to path
sys.path.append('/sps/lsst/users/rbonnetguerrini/ML4transients/src')
from ML4transients.data_access.dataset_loader import DatasetLoader


class PytorchDataset(Dataset):
    @staticmethod
    def create_splits(data_source, test_size=0.2, val_size=0.1, random_state=42, **kwargs):
        """
        Create train/val/test splits efficiently - loads data only once.
        
        Args:
            data_source: DatasetLoader instance or path
            test_size: Fraction for test set
            val_size: Fraction for validation set  
            random_state: Random seed
            **kwargs: Additional arguments (visits, transform, etc.)
            
        Returns:
            dict: {'train': PytorchDataset, 'val': PytorchDataset, 'test': PytorchDataset}
        """
        if isinstance(data_source, DatasetLoader):
            loader = data_source
        else:
            loader = DatasetLoader(data_source)
        
        # Build sample index and labels (lightweight)
        sample_index = []
        labels = []
        
        visits = kwargs.get('visits', loader.visits)
        available_visits = [v for v in visits if v in loader.visits] if visits else loader.visits
        
        print("Building sample index...")
        for visit in available_visits:
            if visit in loader.cutouts and visit in loader.features:
                labels_df = loader.features[visit].labels
                cutout_ids = set(loader.cutouts[visit].ids)
                
                for dia_id in labels_df.index:
                    if dia_id in cutout_ids:
                        sample_index.append((visit, dia_id))
                        labels.append(labels_df.loc[dia_id, 'is_injection'])
        
        labels = np.array(labels)
        indices = np.arange(len(sample_index))
        
        if len(labels) == 0:
            raise ValueError("No samples found. Check your data.")
        
        print(f"Creating splits from {len(sample_index)} samples...")
        
        # Create splits
        trainval_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=val_size_adjusted, 
                random_state=random_state, stratify=labels[trainval_idx]
            )
        else:
            train_idx, val_idx = trainval_idx, []
        
        # Create split info
        split_info = {
            'sample_index': sample_index,
            'labels': labels,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
        
        # Create the three datasets
        train_dataset = PytorchDataset._from_split(loader, split_info, train_idx, **kwargs)
        val_dataset = PytorchDataset._from_split(loader, split_info, val_idx, **kwargs) if len(val_idx) > 0 else None
        test_dataset = PytorchDataset._from_split(loader, split_info, test_idx, **kwargs)
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'split_info': split_info  # Save for potential reuse
        }
    
    @staticmethod
    def _from_split(loader, split_info, indices, **kwargs):
        """Create dataset from pre-computed split."""
        return PytorchDataset(
            data_source=loader,
            _split_info=split_info,
            _indices=indices,
            **kwargs
        )

    @staticmethod
    def create_inference_dataset(data_source, visit=None, **kwargs):
        """
        Create dataset for inference - uses lazy loading by visit.
        
        Args:
            data_source: DatasetLoader instance or path
            visit: Optional specific visit number to process (for memory efficiency)
            **kwargs: Additional arguments (visits, transform, etc.)
            
        Returns:
            PytorchDataset: Dataset for inference (full or visit-specific)
        """
        if isinstance(data_source, DatasetLoader):
            loader = data_source
        else:
            loader = DatasetLoader(data_source)
        
        # If specific visit requested, process only that visit
        if visit is not None:
            if visit not in loader.visits:
                raise ValueError(f"Visit {visit} not found in dataset")
            
            if visit not in loader.cutouts or visit not in loader.features:
                raise ValueError(f"Visit {visit} missing cutouts or features")
            
            print(f"Creating inference dataset for visit {visit}...")
            available_visits = [visit]
        else:
            # Process all visits or specified visits
            visits = kwargs.get('visits', loader.visits)
            available_visits = [v for v in visits if v in loader.visits] if visits else loader.visits
            print("Building inference dataset index across visits...")
        
        # Build sample index
        sample_index = []
        labels = []
        
        for v in available_visits:
            if v in loader.cutouts and v in loader.features:
                labels_df = loader.features[v].labels
                cutout_ids = set(loader.cutouts[v].ids)
                
                for dia_id in labels_df.index:
                    if dia_id in cutout_ids:
                        sample_index.append((v, dia_id))
                        labels.append(labels_df.loc[dia_id, 'is_injection'])
        
        if len(sample_index) == 0:
            raise ValueError("No samples found. Check your data.")
            
        labels = np.array(labels)
        
        # Create inference dataset info
        inference_info = {
            'sample_index': sample_index,
            'labels': labels,
            'visits': available_visits,
            'single_visit': visit  # Track if this is single visit
        }
        
        if visit is not None:
            print(f"Created inference dataset for visit {visit} with {len(sample_index)} samples")
        else:
            print(f"Created inference dataset with {len(sample_index)} samples across {len(available_visits)} visits")
        
        return PytorchDataset._from_inference(loader, inference_info, **kwargs)

    @staticmethod
    def _from_inference(loader, inference_info, **kwargs):
        """Create inference dataset from pre-computed info."""
        return PytorchDataset(
            data_source=loader,
            _inference_info=inference_info,
            **kwargs
        )

    def __init__(self, 
                 data_source: Union[str, Path, List[Union[str, Path]], DatasetLoader],
                 visits: Optional[List[int]] = None,
                 train: bool = False, 
                 val: bool = False, 
                 test: bool = False, 
                 transform=None,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 _split_info: dict = None,  # Internal use
                 _indices: np.ndarray = None,  # Internal use
                 _inference_info: dict = None):  # Internal use for inference
        """
        PyTorch Dataset - can create full dataset or subset.
        For efficient usage, prefer:
        - PytorchDataset.create_splits() for training
        - PytorchDataset.create_inference_dataset() for inference
        - PytorchDataset.create_inference_dataset(visit=X) for visit-specific inference
        """
        # Handle both path and DatasetLoader instance
        if isinstance(data_source, DatasetLoader):
            self.loader = data_source
        else:
            self.loader = DatasetLoader(data_source)
        
        self.transform = transform
        
        # If using pre-computed split (efficient path)
        if _split_info is not None and _indices is not None:
            self._load_from_split(_split_info, _indices)
            return
        if _inference_info is not None:
            self._load_from_inference(_inference_info)
            return

        # Legacy path - load all data then split (less efficient)
        print("Loading all data then applying split (consider using create_splits() for efficiency)")
        
        # Filter visits if specified
        available_visits = self.loader.visits
        if visits is not None:
            available_visits = [v for v in visits if v in available_visits]
        
        # Collect all data into memory
        all_cutouts = []
        all_labels = []
        all_ids = []
        
        for visit in available_visits:
            if visit in self.loader.cutouts and visit in self.loader.features:
                cutouts = self.loader.get_all_cutouts(visit)
                labels_df = self.loader.features[visit].labels
                
                # Match cutouts with labels by diaSourceId
                cutout_ids = self.loader.cutouts[visit].ids
                label_ids = labels_df.index.values
                
                # Find common IDs
                common_ids = np.intersect1d(cutout_ids, label_ids)
                
                for dia_id in common_ids:
                    cutout_idx = np.where(cutout_ids == dia_id)[0][0]
                    all_cutouts.append(cutouts[cutout_idx])
                    all_labels.append(labels_df.loc[dia_id, 'is_injection'])
                    all_ids.append(dia_id)
        
        # Convert to arrays
        self.images = np.array(all_cutouts)
        self.labels = np.array(all_labels)
        self.dia_source_ids = np.array(all_ids)
        
        # Apply splits if requested
        if train or val or test:
            self._apply_legacy_split(train, val, test, test_size, val_size, random_state)

    def _load_from_inference(self, inference_info):
        """Load data for inference with lazy loading."""
        sample_index = inference_info['sample_index']
        all_labels = inference_info['labels']
        single_visit = inference_info.get('single_visit')
        
        # Store sample info instead of loading all data immediately
        self._sample_index = sample_index
        self.labels = all_labels
        self.dia_source_ids = np.array([dia_id for _, dia_id in sample_index])
        
        # Don't load images immediately - load on demand in __getitem__
        self.images = None
        self._loaded_images = {}  # Cache for loaded images
        
        if single_visit is not None:
            print(f"Created lazy dataset for visit {single_visit} with {len(sample_index)} samples")
        else:
            print(f"Created lazy dataset with {len(sample_index)} samples")

    def __getitem__(self, idx):
        # Load image on demand if using lazy loading
        if hasattr(self, '_sample_index') and self.images is None:
            if idx not in self._loaded_images:
                visit, dia_id = self._sample_index[idx]
                cutout = self.loader.get_cutout(visit, dia_id)
                self._loaded_images[idx] = cutout
            image = self._loaded_images[idx]
        else:
            image = self.images[idx]
        
        label = self.labels[idx]

        # Convert image to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32).squeeze(-1)
        image = image.unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)
        return image, label, idx

    def cleanup_memory(self):
        """Explicitly free memory used by this dataset."""
        if hasattr(self, 'images') and self.images is not None:
            del self.images
            self.images = None
        
        if hasattr(self, '_loaded_images'):
            self._loaded_images.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print(f"Cleaned up memory for dataset")

    def _load_from_split(self, split_info, indices):
        """Load only the data needed for this split."""
        sample_index = split_info['sample_index']
        all_labels = split_info['labels']
        
        # Get only the samples we need
        selected_samples = [sample_index[i] for i in indices]
        selected_labels = all_labels[indices]
        
        # Load only required cutouts
        print(f"Loading {len(selected_samples)} cutouts...")
        all_cutouts = []
        all_ids = []
        
        for visit, dia_id in selected_samples:
            cutout = self.loader.get_cutout(visit, dia_id)  # Load individual cutouts
            all_cutouts.append(cutout)
            all_ids.append(dia_id)
        
        self.images = np.array(all_cutouts)
        self.labels = selected_labels
        self.dia_source_ids = np.array(all_ids)
    
    def _apply_legacy_split(self, train, val, test, test_size, val_size, random_state):
        """Apply split to already loaded data (legacy method)."""
        if len(self.labels) == 0:
            raise ValueError("No features provided. Dataset is empty.")

        # Warn if only one class is present
        if np.all(self.labels == self.labels[0]):
            print(f"Only one class present, class {self.labels[0]}")

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
            val_size_adjusted = val_size / (1 - test_size)
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
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load image on demand if using lazy loading
        if hasattr(self, '_sample_index') and self.images is None:
            if idx not in self._loaded_images:
                visit, dia_id = self._sample_index[idx]
                cutout = self.loader.get_cutout(visit, dia_id)
                self._loaded_images[idx] = cutout
            image = self._loaded_images[idx]
        else:
            image = self.images[idx]
        
        label = self.labels[idx]

        # Convert image to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32).squeeze(-1)
        image = image.unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)
        return image, label, idx

    def get_feature_by_idx(self, idx):
        """Get features for a specific index - loads on demand."""
        dia_id = self.dia_source_ids[idx]
        visit = self.loader.find_visit(dia_id)
        if visit:
            return self.loader.get_features(visit, dia_id)
        return None
    
    def get_dia_source_id(self, idx):
        """Get diaSourceId for a specific index."""
        return self.dia_source_ids[idx]
    
    def __repr__(self):
        return (f"PytorchDataset({len(self)} samples)\n"
                f"  Image shape: {self.images.shape}\n"
                f"  Labels: {np.sum(self.labels)} injected, {len(self.labels) - np.sum(self.labels)} real")