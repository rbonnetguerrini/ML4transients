import numpy as np
import pandas as pd
import torch
import torch.utils.data
import umap.umap_ as umap
from sklearn.model_selection import ParameterGrid
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import base64
import io
from PIL import Image
import hdbscan

from ML4transients.training import get_trainer
from ML4transients.utils import load_config

import warnings
warnings.filterwarnings("ignore", message="out of range integer may result in loss of precision", category=UserWarning)

def embeddable_image(image_array: np.ndarray) -> str:
    """Convert numpy array to embeddable base64 image string.
    
    Simple approach for grayscale astronomical images that creates
    base64-encoded PNG images suitable for embedding in HTML.
    
    Parameters
    ----------
    image_array : np.ndarray
        Input image array (2D or 3D)
        
    Returns
    -------
    str
        Base64-encoded image string with data URI prefix
    """
    # Handle different image shapes - ensure we get a 2D grayscale array
    if len(image_array.shape) == 3:
        # If 3D, take the middle channel (astronomical images often have 3 identical channels)
        if image_array.shape[2] == 3:
            img_data = image_array[:, :, 1]  # Take middle channel
        elif image_array.shape[0] == 3:
            img_data = image_array[1, :, :]  # Channel first format
        else:
            img_data = image_array.squeeze()  # Remove singleton dimensions
    else:
        img_data = image_array
    
    # Ensure 2D array
    if len(img_data.shape) > 2:
        img_data = img_data.squeeze()
    
    # Simple normalization approach - just like what probably works in your utils
    # Assume the data is already preprocessed and just needs basic scaling
    
    # Convert to 0-255 range for PIL
    img_min = img_data.min()
    img_max = img_data.max()
    
    if img_max > img_min:
        # Scale to 0-255
        img_normalized = ((img_data - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        # If all values are the same, create a mid-gray image
        img_normalized = np.full_like(img_data, 128, dtype=np.uint8)
    
    # Convert to PIL Image - explicitly grayscale mode
    img = Image.fromarray(img_normalized, mode='L')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

class UMAPInterpreter:
    """Encapsulates feature extraction + UMAP embedding + optional clustering.

    Workflow
    --------
    1. extract_features()
    2. fit_umap()
    3. (optional) cluster_high_dimensional_features()
    4. create_interpretability_dataframe()
    """
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        """
        Initialize UMAP interpreter with a trained model.
        
        Args:
            model_path: Path to trained model directory
            device: Device to run model on
        """
        self.model_path = Path(model_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        config = load_config(self.model_path / "config.yaml")
        self.trainer = get_trainer(config["training"]["trainer_type"], config["training"])
        state_dict = torch.load(self.model_path / "model_best.pth", map_location=self.device)
        self.trainer.model.load_state_dict(state_dict)
        self.trainer.model.to(self.device)
        self.trainer.model.eval()
        
        self.umap_reducer = None
        self.features = None
        self.umap_embedding = None
        self.high_dim_clusters = None
        self.sample_indices = None  # Add this line
    
        # For hook-based feature extraction
        self.hook_features = None
        self.hook_handle = None
    
    def _get_layer_by_name(self, layer_name: str):
        """Get a layer by name from the model.
        
        Parameters
        ----------
        layer_name : str
            Name of the layer ('conv1', 'conv2', 'fc1', 'output', 'flatten')
            
        Returns
        -------
        torch.nn.Module or None
            The requested layer, or None for special cases
            
        Raises
        ------
        ValueError
            If layer name is not recognized
        """
        # Map layer names to actual layers for CustomCNN
        layer_mapping = {
            'conv1': self.trainer.model.conv1,
            'conv2': self.trainer.model.conv2, 
            'fc1': self.trainer.model.fc1,
            'output': None  # Special case - use final model output
        }
        
        if layer_name == 'flatten':
            # For flatten, we'll hook after conv2 and manually flatten
            return self.trainer.model.conv2
        
        if layer_name in layer_mapping:
            return layer_mapping[layer_name]
        else:
            available_layers = list(layer_mapping.keys()) + ['flatten']
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available_layers}")
    
    def _hook_fn(self, module, input, output):
        """Hook function to capture layer activations.
        
        Parameters
        ----------
        module : torch.nn.Module
            The module being hooked
        input : tuple
            Input tensors to the module
        output : torch.Tensor
            Output tensor from the module
        """
        self.hook_features = output.detach()
    
    def extract_features(self, data_loader, layer_name: str = "fc1",
                    max_samples: Optional[int] = None, random_state: int = 42) -> np.ndarray:
        """Extract flattened activations from a target layer.

        Parameters
        ----------
        data_loader : DataLoader
            Source of input tensors.
        layer_name : str
            Layer identifier ('conv1','conv2','fc1','flatten','output').
        max_samples : int, optional
            If set, random subset used for UMAP (sampling after full collection).
        random_state : int
            Seed for reproducible subsampling.

        Returns
        -------
        np.ndarray
            Feature matrix (n_selected, n_features).

        Notes
        -----
        Current implementation collects all features before subsampling.
        For extremely large datasets, a streaming reservoir approach could
        replace this to reduce peak memory usage.
        """
        print(f"Extracting features from layer: {layer_name}")
        if max_samples is not None:
            print(f"Will sample up to {max_samples} samples for UMAP computation")
        
        # First pass: extract all features
        features = []
        total_batches = len(data_loader)
        
        # Use hooks to extract from intermediate layers
        target_layer = self._get_layer_by_name(layer_name)
        
        # Register hook
        self.hook_handle = target_layer.register_forward_hook(self._hook_fn)
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(data_loader):
                    if batch_idx % 10 == 0:  # Progress every 10 batches
                        print(f"Processing batch {batch_idx+1}/{total_batches}")
                    
                    images, *_ = batch
                    images = images.to(self.device)
                    
                    # Forward pass (hook will capture the features)
                    _ = self.trainer.model(images)
                    
                    # Process the captured features
                    if layer_name == 'flatten':
                        # Apply the same processing as in the model's forward pass
                        feat = self.hook_features
                        # Apply pooling and dropout if we're at conv2
                        if hasattr(self.trainer.model, 'pool2'):
                            feat = self.trainer.model.pool2(feat)
                        if hasattr(self.trainer.model, 'dropout2'):
                            feat = self.trainer.model.dropout2(feat)
                        feat_flat = feat.view(feat.size(0), -1)
                    else:
                        feat_flat = self.hook_features.view(self.hook_features.size(0), -1)
                    
                    features.append(feat_flat.cpu().numpy())
                    
                    # Clear GPU memory after each batch
                    del images, feat_flat
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            # Always remove the hook
            if self.hook_handle:
                self.hook_handle.remove()
                self.hook_handle = None
        
        # Stack all features
        all_features = np.vstack(features)
        print(f"Total extracted features shape: {all_features.shape}")
        
        # Apply random sampling if max_samples is specified
        if max_samples is not None and len(all_features) > max_samples:
            print(f"Randomly sampling {max_samples} from {len(all_features)} total samples for UMAP computation (random_state={random_state})")
            
            np.random.seed(random_state)
            indices = np.random.choice(len(all_features), size=max_samples, replace=False)
            indices = np.sort(indices)  # Sort to maintain some order
            
            self.features = all_features[indices]
            self.sample_indices = indices  # Store indices for later use
            print(f"Final features shape for UMAP computation: {self.features.shape}")
        else:
            print(f"Using all {len(all_features)} samples (no sampling needed)")
            self.features = all_features
            self.sample_indices = np.arange(len(all_features))
        
        # Force garbage collection after feature extraction
        import gc
        gc.collect()
        
        return self.features
    
    def optimize_umap_parameters(self, param_grid: Dict = None, 
                                n_samples: int = 1000, random_state: int = 42) -> Dict:
        """Grid-search heuristic for approximate UMAP parameter selection.

        Scoring metric: mean std-dev of embedding axes (spread proxy).
        """
        if self.features is None:
            raise ValueError("No features extracted. Call extract_features first.")
        
        if param_grid is None:
            param_grid = {
                'n_neighbors': [5, 10, 15, 20, 30],
                'min_dist': [0.01, 0.1, 0.3, 0.5],
                'n_components': [2]  # Keep 2D for visualization
            }
        
        print("Optimizing UMAP parameters...")
        print(f"Parameter grid: {param_grid}")
        
        # Subsample features for faster optimization
        if len(self.features) > n_samples:
            indices = np.random.RandomState(random_state).choice(
                len(self.features), n_samples, replace=False
            )
            features_subset = self.features[indices]
        else:
            features_subset = self.features
            
        print(f"Using {len(features_subset)} samples for optimization")
        
        best_score = -np.inf
        best_params = None
        
        param_combinations = list(ParameterGrid(param_grid))
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        for i, params in enumerate(param_combinations):
            print(f"Testing {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Fit UMAP with current parameters
                reducer = umap.UMAP(
                    n_neighbors=params['n_neighbors'],
                    min_dist=params['min_dist'], 
                    n_components=params['n_components'],
                    random_state=random_state,
                    verbose=False
                )
                
                embedding = reducer.fit_transform(features_subset)
                
                # Score based on embedding quality - use spread of points as a simple metric
                # Higher spread generally indicates better separation
                embedding_std = np.std(embedding, axis=0).mean()
                score = embedding_std
                
                print(f"  Embedding spread score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        if best_params is None:
            print("No valid parameter combination found, using defaults")
            best_params = {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 2}
        else:
            print(f"\nBest parameters found:")
            print(f"  Parameters: {best_params}")
            print(f"  Embedding spread score: {best_score:.4f}")
        
        return best_params

    def fit_umap(self, n_neighbors: int = 15, min_dist: float = 0.1, 
                 n_components: int = 2, random_state: int = 42,
                 optimize_params: bool = False) -> np.ndarray:
        """Fit UMAP on previously extracted features (with optional param search)."""
        if self.features is None:
            raise ValueError("No features extracted. Call extract_features first.")
        
        # Print number of samples being used for UMAP
        print(f"Building UMAP embedding with {len(self.features)} samples")
        print(f"Feature dimensionality: {self.features.shape[1]}D -> {n_components}D")
        
        if optimize_params:
            print("Optimizing UMAP parameters first...")
            try:
                best_params = self.optimize_umap_parameters(random_state=random_state)
                n_neighbors = best_params['n_neighbors']
                min_dist = best_params['min_dist']
                n_components = best_params['n_components']
                print(f"Using optimized parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")
            except Exception as e:
                print(f"UMAP optimization failed: {e}")
                print("Using default parameters instead")
        
        print("Fitting UMAP embedding...")
        print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}")
        
        try:
            self.umap_reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=random_state,
                verbose=True
            )
            
            self.umap_embedding = self.umap_reducer.fit_transform(self.features)
            print(f"UMAP embedding completed. Final embedding shape: {self.umap_embedding.shape}")
            return self.umap_embedding
            
        except Exception as e:
            print(f"UMAP fitting failed: {e}")
            raise

    def cluster_high_dimensional_features(self, min_cluster_size: int = 10, 
                                        min_samples: int = 5) -> np.ndarray:
        """
        Perform HDBSCAN clustering on high-dimensional features (before UMAP).
        
        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
            
        Returns:
            Cluster labels for each sample
        """
        if self.features is None:
            raise ValueError("No features extracted. Call extract_features first.")
        
        print(f"Performing HDBSCAN clustering on high-dimensional features...")
        print(f"Feature space shape: {self.features.shape}")
        print(f"Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            
            self.high_dim_clusters = clusterer.fit_predict(self.features)
            n_clusters = len(set(self.high_dim_clusters)) - (1 if -1 in self.high_dim_clusters else 0)
            n_noise = list(self.high_dim_clusters).count(-1)
            
            print(f"HDBSCAN clustering completed.")
            print(f"Found {n_clusters} clusters with {n_noise} noise points")
            
            return self.high_dim_clusters
            
        except Exception as e:
            print(f"HDBSCAN clustering failed: {e}")
            # Return all points as noise if clustering fails
            self.high_dim_clusters = np.full(len(self.features), -1)
            print("Assigning all points as noise due to clustering failure")
            return self.high_dim_clusters

    def create_interpretability_dataframe(self, predictions: np.ndarray, labels: np.ndarray,
                                data_loader, sample_indices: Optional[np.ndarray] = None,
                                additional_features: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """Build DataFrame joining UMAP output + classification + hover images."""
        if self.umap_embedding is None:
            raise ValueError("No UMAP embedding. Call fit_umap first.")
        
        # Use stored sample indices if not provided
        if sample_indices is None:
            if self.sample_indices is not None:
                sample_indices = self.sample_indices
                print(f"Using stored sample indices ({len(sample_indices)} samples)")
            else:
                sample_indices = np.arange(len(predictions))
                print(f"No sample indices available, using all {len(sample_indices)} samples")
    
        # Create base DataFrame with UMAP coordinates
        df = pd.DataFrame({
            'umap_x': self.umap_embedding[:, 0],
            'umap_y': self.umap_embedding[:, 1],
            'prediction': predictions[sample_indices],
            'true_label': labels[sample_indices],
            'sample_index': sample_indices
        })
        
        # Add correct/incorrect classification
        df['correct'] = df['prediction'] == df['true_label']
    
        # Add classification categories
        df['class_type'] = 'True Negative'
        df.loc[(df['true_label'] == 1) & (df['prediction'] == 1), 'class_type'] = 'True Positive'
        df.loc[(df['true_label'] == 0) & (df['prediction'] == 1), 'class_type'] = 'False Positive'
        df.loc[(df['true_label'] == 1) & (df['prediction'] == 0), 'class_type'] = 'False Negative'
        
        # Add high-dimensional clusters if available
        if self.high_dim_clusters is not None:
            cluster_strings = np.where(self.high_dim_clusters == -1, "Noise", self.high_dim_clusters.astype(str))
            df['high_dim_cluster'] = cluster_strings
        
        # Add embeddable images
        print("Creating embeddable images for hover tooltips...")
        images = []
        current_idx = 0
        
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 10 == 0 and batch_idx > 0:
                print(f"Processed {current_idx} images...")
            
            batch_images, *_ = batch
            batch_size = batch_images.shape[0]
            
            for i in range(batch_size):
                if current_idx < len(sample_indices):
                    # Convert tensor to numpy if needed
                    if hasattr(batch_images, 'cpu'):
                        img_array = batch_images[i].cpu().numpy()
                    else:
                        img_array = batch_images[i]
                    
                    # Convert to embeddable format
                    embeddable_img = embeddable_image(img_array)
                    images.append(embeddable_img)
                    current_idx += 1
                else:
                    break
            
            if current_idx >= len(sample_indices):
                break
        
        df['image'] = images
        print(f"Created {len(images)} embeddable images")
        
        if additional_features:
            for key, values in additional_features.items():
                df[key] = values[sample_indices]
        
        return df