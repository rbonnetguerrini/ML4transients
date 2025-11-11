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
            **kwargs: Additional arguments (visits, transform, cutout_types, etc.)
            
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
                 cutout_types: Optional[List[str]] = None,  # Parameter for multi-channel input
                 _split_info: dict = None,  # Internal use
                 _indices: np.ndarray = None,  # Internal use
                 _inference_info: dict = None):  # Internal use for inference
        """
        PyTorch Dataset - can create full dataset or subset.
        For efficient usage, prefer:
        - PytorchDataset.create_splits() for training
        - PytorchDataset.create_inference_dataset() for inference
        - PytorchDataset.create_inference_dataset(visit=X) for visit-specific inference
        
        Parameters
        ----------
        cutout_types : List[str], optional
            List of cutout types to stack as channels. Options: 'diff', 'coadd', 'science'
            If None, defaults to ['diff'] 
            Example: ['diff', 'coadd'] will create 2-channel input
        """
        # Handle both path and DatasetLoader instance
        if isinstance(data_source, DatasetLoader):
            self.loader = data_source
        else:
            self.loader = DatasetLoader(data_source)
        
        # Set cutout types (default to 'diff' only for backward compatibility)
        self.cutout_types = cutout_types if cutout_types is not None else ['diff']
        
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
        skipped = 0
        
        for visit in available_visits:
            if visit in self.loader.cutouts and visit in self.loader.features:
                # Get labels
                labels_df = self.loader.features[visit].labels
                cutout_ids = self.loader.cutouts[visit].ids
                label_ids = labels_df.index.values
                
                # Find common IDs
                common_ids = np.intersect1d(cutout_ids, label_ids)
                
                for dia_id in common_ids:
                    # Load and stack cutouts for each type
                    cutout = self._load_and_stack_cutouts(visit, dia_id)
                    if cutout is None:
                        # Skip samples with missing cutout types
                        skipped += 1
                        continue
                    all_cutouts.append(cutout)
                    all_labels.append(labels_df.loc[dia_id, 'is_injection'])
                    all_ids.append(dia_id)
        
        if skipped > 0:
            print(f"Skipped {skipped} samples due to missing cutout types")
        
        # Convert to arrays
        self.images = np.array(all_cutouts)
        self.labels = np.array(all_labels)
        self.dia_source_ids = np.array(all_ids)
        
        # Apply splits if requested
        if train or val or test:
            self._apply_legacy_split(train, val, test, test_size, val_size, random_state)

    def _load_from_inference(self, inference_info):
        """Load data for inference with lazy loading.
        
        Parameters
        ----------
        inference_info : dict
            Dictionary containing inference dataset information including:
            - sample_index: List of (visit, dia_id) tuples
            - labels: Array of true labels
            - single_visit: Optional visit number if single visit
        """
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

    def cleanup_memory(self):
        """Explicitly free memory used by this dataset.
        
        Deletes image arrays and loaded image cache, then forces garbage collection.
        Useful for managing memory in large-scale evaluations.
        """
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
        """Load only the data needed for this split.
        
        Parameters
        ----------
        split_info : dict
            Dictionary containing split information including sample_index and labels
        indices : np.ndarray
            Array of indices to select from the split
        """
        sample_index = split_info['sample_index']
        all_labels = split_info['labels']
        
        # Get only the samples we need
        selected_samples = [sample_index[i] for i in indices]
        selected_labels = all_labels[indices]
        
        # Load only required cutouts
        print(f"Loading {len(selected_samples)} cutouts with types {self.cutout_types}...")
        all_cutouts = []
        all_labels_kept = []
        all_ids = []
        skipped = 0
        
        for i, (visit, dia_id) in enumerate(selected_samples):
            cutout = self._load_and_stack_cutouts(visit, dia_id)
            if cutout is None:
                # Skip samples with missing cutout types
                skipped += 1
                continue
            all_cutouts.append(cutout)
            all_labels_kept.append(selected_labels[i])
            all_ids.append(dia_id)
        
        if skipped > 0:
            print(f"Skipped {skipped}/{len(selected_samples)} samples due to missing cutout types")
        
        self.images = np.array(all_cutouts)
        self.labels = np.array(all_labels_kept)
        self.dia_source_ids = np.array(all_ids)
    
    def _apply_legacy_split(self, train, val, test, test_size, val_size, random_state):
        """Apply split to already loaded data (legacy method).
        
        Parameters
        ----------
        train : bool
            Whether to create training split
        val : bool
            Whether to create validation split  
        test : bool
            Whether to create test split
        test_size : float
            Fraction of data for test set
        val_size : float
            Fraction of data for validation set
        random_state : int
            Random seed for reproducible splits
        """
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
        """Helper to apply index split.
        
        Parameters
        ----------
        indices : np.ndarray
            Array of indices to select for this split
        """
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
                # Load multiple cutout types and stack them
                cutout = self._load_and_stack_cutouts(visit, dia_id)
                if cutout is None:
                    # Skip this sample - raise an error to indicate data issue
                    raise ValueError(f"Missing cutout types for visit {visit}, dia_id {dia_id}. "
                                   f"Required types: {self.cutout_types}")
                self._loaded_images[idx] = cutout
            image = self._loaded_images[idx]
        else:
            image = self.images[idx]
        
        label = self.labels[idx]

        # Convert image to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32)
        
        # Handle multi-channel stacking
        if len(self.cutout_types) == 1:
            # Single channel squeeze last dim and add channel dim
            image = image.squeeze(-1).unsqueeze(0)
        else:
            # Multi-channel already stacked, just ensure correct shape
            # image should be [C, H, W] where C = number of cutout types
            if image.ndim == 3 and image.shape[-1] == len(self.cutout_types):
                # Shape is [H, W, C], transpose to [C, H, W]
                image = image.permute(2, 0, 1)
            elif image.ndim == 3 and image.shape[0] == len(self.cutout_types):
                # Already [C, H, W], keep as is
                pass
            else:
                # Fallback: squeeze and add channel
                image = image.squeeze(-1).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)
        return image, label, idx
    
    def _load_and_stack_cutouts(self, visit: int, dia_id: int) -> np.ndarray:
        """Load and stack multiple cutout types for a given source.
        
        Parameters
        ----------
        visit : int
            Visit number
        dia_id : int
            DIA source ID
            
        Returns
        -------
        np.ndarray or None
            Stacked cutouts with shape [H, W, C] where C is number of cutout types.
            Returns None if any cutout type is missing.
        """
        cutouts = []
        for cutout_type in self.cutout_types:
            cutout = self.loader.get_cutout(visit, dia_id, cutout_type=cutout_type)
            if cutout is None:
                # Discard sample completely if any cutout type is missing
                return None
            
            # Ensure cutout has shape [H, W, 1]
            if cutout.ndim == 2:
                cutout = cutout[..., np.newaxis]
            
            cutouts.append(cutout.squeeze(-1))  # Remove last dim before stacking
        
        # Stack along last axis to get [H, W, C]
        stacked = np.stack(cutouts, axis=-1)
        return stacked

    def get_feature_by_idx(self, idx):
        """Get features for a specific index - loads on demand.
        
        Parameters
        ----------
        idx : int
            Index of the sample
            
        Returns
        -------
        pd.DataFrame or None
            Features for the sample, or None if not found
        """
        dia_id = self.dia_source_ids[idx]
        visit = self.loader.find_visit(dia_id)
        if visit:
            return self.loader.get_features(visit, dia_id)
        return None
    
    def get_dia_source_id(self, idx):
        """Get diaSourceId for a specific index.
        
        Parameters
        ----------
        idx : int
            Index of the sample
            
        Returns
        -------
        int
            The diaSourceId for this sample
        """
        return self.dia_source_ids[idx]
    
    def get_dia_source_ids(self):
        """Get all diaSourceIds in the dataset.
        
        Returns
        -------
        np.ndarray
            Array of all diaSourceIds in the dataset
        """
        return self.dia_source_ids
    
    def __repr__(self):
        cutout_info = f"Cutout types: {self.cutout_types}" if hasattr(self, 'cutout_types') else ""
        
        if hasattr(self, '_sample_index') and self.images is None:
            return (f"PytorchDataset({len(self)} samples, lazy loading)\n"
                    f"  {cutout_info}\n"
                    f"  Labels: {np.sum(self.labels)} injected, {len(self.labels) - np.sum(self.labels)} real")
        else:
            return (f"PytorchDataset({len(self)} samples)\n"
                    f"  {cutout_info}\n"
                    f"  Image shape: {self.images.shape if self.images is not None else 'lazy'}\n"
                    f"  Labels: {np.sum(self.labels)} injected, {len(self.labels) - np.sum(self.labels)} real")