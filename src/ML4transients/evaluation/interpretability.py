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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import joblib

from ML4transients.training import get_trainer
from ML4transients.utils import load_config

import warnings
warnings.filterwarnings("ignore", message="out of range integer may result in loss of precision", category=UserWarning)

def embeddable_image(image_array: np.ndarray, cmap: str = 'gray') -> str:
    """Convert numpy array to embeddable base64 image string with colormap.
    
    Simple approach for grayscale astronomical images that creates
    base64-encoded PNG images suitable for embedding in HTML.
    Preserves negative values when using diverging colormaps.
    
    Parameters
    ----------
    image_array : np.ndarray
        Input image array (2D or 3D)
    cmap : str
        Matplotlib colormap name (default: 'RdYlGn')
        For diverging colormaps, negative values are preserved
        
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
    
    # Get colormap
    colormap = cm.get_cmap(cmap)
    
    img_min = img_data.min()
    img_max = img_data.max()
    
    if img_max > img_min:
        if img_min < 0 and img_max > 0:
            # For data with negative values, center around zero
            abs_max = max(abs(img_min), abs(img_max))
            # Normalize to [-1, 1] range, then shift to [0, 1] for colormap
            img_normalized = img_data / abs_max  # Now in [-1, 1]
            img_normalized = (img_normalized + 1) / 2  # Now in [0, 1] with 0.5 = zero
        else:
            # Standard normalization for all-positive or all-negative data
            img_normalized = (img_data - img_min) / (img_max - img_min)
    else:
        # If all values are the same, create a mid-range image
        img_normalized = np.full_like(img_data, 0.5, dtype=np.float32)
    
    # Apply colormap to convert to RGB
    img_rgb = colormap(img_normalized)  # Returns RGBA values in 0-1 range
    
    # Convert to 0-255 range and remove alpha channel
    img_rgb_255 = (img_rgb[:, :, :3] * 255).astype(np.uint8)
    
    # Convert to PIL Image - RGB mode
    img = Image.fromarray(img_rgb_255, mode='RGB')
    
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
        trainer_type = config["training"]["trainer_type"]
        self.trainer = get_trainer(trainer_type, config["training"])
        
        # Load models based on trainer type
        if trainer_type == "ensemble":
            num_models = config["training"]["num_models"]
            print(f"Loading ensemble model with {num_models} models...")
            for i in range(num_models):
                model_path = self.model_path / f"ensemble_model_{i}_best.pth"
                state_dict = torch.load(model_path, map_location=self.device)
                self.trainer.models[i].load_state_dict(state_dict)
                self.trainer.models[i].to(self.device)
                self.trainer.models[i].eval()
        elif trainer_type == "coteaching":
            print("Loading co-teaching model...")
            state_dict1 = torch.load(self.model_path / "model1_best.pth", map_location=self.device)
            state_dict2 = torch.load(self.model_path / "model2_best.pth", map_location=self.device)
            self.trainer.model1.load_state_dict(state_dict1)
            self.trainer.model2.load_state_dict(state_dict2)
            self.trainer.model1.to(self.device)
            self.trainer.model2.to(self.device)
            self.trainer.model1.eval()
            self.trainer.model2.eval()
        else:
            print("Loading standard model...")
            state_dict = torch.load(self.model_path / "model_best.pth", map_location=self.device)
            self.trainer.model.load_state_dict(state_dict)
            self.trainer.model.to(self.device)
            self.trainer.model.eval()
        
        self.trainer_type = trainer_type
        self.umap_reducer = None
        self.features = None
        self.umap_embedding = None
        self.high_dim_clusters = None
        self.sample_indices = None
    
        # For hook-based feature extraction
        self.hook_features = None
        self.hook_handle = None

    def _get_model_for_layer_extraction(self, layer_name: str):
        """Get the appropriate model for layer extraction based on trainer type."""
        if hasattr(self.trainer, 'models'):  # Ensemble
            return self.trainer.models[0]  # Use first model for feature extraction
        elif hasattr(self.trainer, 'model1'):  # CoTeaching
            return self.trainer.model1  # Use first model for feature extraction
        else:  # Standard
            return self.trainer.model

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
        model = self._get_model_for_layer_extraction(layer_name)
        
        # Handle CustomCNN's dynamic conv_blocks structure
        if hasattr(model, 'conv_blocks'):
            # CustomCNN uses ModuleList for conv blocks
            if layer_name == 'conv1':
                if len(model.conv_blocks) >= 1:
                    return model.conv_blocks[0]
                else:
                    raise ValueError("Model doesn't have conv1 layer")
            elif layer_name == 'conv2':
                if len(model.conv_blocks) >= 2:
                    return model.conv_blocks[1]
                else:
                    raise ValueError("Model doesn't have conv2 layer")
            elif layer_name == 'flatten':
                # For flatten, use the last conv block
                if len(model.conv_blocks) >= 1:
                    return model.conv_blocks[-1]
                else:
                    raise ValueError("Model doesn't have any conv blocks")
            elif layer_name == 'fc1':
                if hasattr(model, 'fc1'):
                    return model.fc1
                else:
                    raise ValueError("Model doesn't have fc1 layer")
            elif layer_name == 'output':
                return None  # Special case - use final model output
            else:
                available_layers = ['conv1', 'conv2', 'fc1', 'output', 'flatten']
                # Filter available layers based on actual model structure
                actual_layers = ['output', 'fc1']  # These are always available
                if len(model.conv_blocks) >= 1:
                    actual_layers.extend(['conv1', 'flatten'])
                if len(model.conv_blocks) >= 2:
                    actual_layers.append('conv2')
                raise ValueError(f"Layer '{layer_name}' not found. Available layers: {actual_layers}")
        else:
            # Fallback for models with different structure
            layer_mapping = {
                'conv1': getattr(model, 'conv1', None),
                'conv2': getattr(model, 'conv2', None),
                'fc1': getattr(model, 'fc1', None),
                'output': None  # Special case - use final model output
            }
            
            if layer_name == 'flatten':
                # Try conv2 first, then conv1
                return getattr(model, 'conv2', getattr(model, 'conv1', None))
            
            if layer_name in layer_mapping and layer_mapping[layer_name] is not None:
                return layer_mapping[layer_name]
            else:
                available_layers = [k for k, v in layer_mapping.items() if v is not None] + ['flatten']
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
        print(f"Extracting features from layer: {layer_name} using {self.trainer_type} model")
        if max_samples is not None:
            print(f"Will sample up to {max_samples} samples for UMAP computation")
        
        # First pass: extract all features
        features = []
        total_batches = len(data_loader)
        
        # Use the appropriate model for feature extraction
        model = self._get_model_for_layer_extraction(layer_name)
        
        # Handle special case for output layer
        if layer_name == 'output':
            # Extract final model outputs directly
            with torch.no_grad():
                for batch_idx, batch in enumerate(data_loader):
                    if batch_idx % 10 == 0:  # Progress every 10 batches
                        print(f"Processing batch {batch_idx+1}/{total_batches}")
                    
                    images, *_ = batch
                    images = images.to(self.device)
                    
                    # Get model output
                    outputs = model(images)
                    features.append(outputs.cpu().numpy())
                    
                    # Clear GPU memory after each batch
                    del images, outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
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
                        _ = model(images)
                        
                        # Process the captured features
                        if layer_name == 'flatten':
                            # For flatten, just flatten the hook features
                            feat_flat = self.hook_features.view(self.hook_features.size(0), -1)
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

    def extract_prediction_uncertainties(self, data_loader, max_samples: Optional[int] = None, 
                                      random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Extract prediction probabilities and uncertainties for all model types.
        
        Parameters
        ----------
        data_loader : DataLoader
            Source of input tensors.
        max_samples : int, optional
            If set, random subset used.
        random_state : int
            Seed for reproducible subsampling.
            
        Returns
        -------
        tuple
            (probabilities, uncertainties) arrays
            For standard models, uncertainties will be distance from decision boundary
        """
        print(f"Extracting prediction probabilities and uncertainties from {self.trainer_type} model...")
        
        probabilities = []
        uncertainties = []
        total_batches = len(data_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx+1}/{total_batches}")
                
                images, *_ = batch
                images = images.to(self.device)
                
                if hasattr(self.trainer, 'models'):  # Ensemble
                    individual_probs = []
                    for model in self.trainer.models:
                        outputs = model(images)
                        probs = torch.sigmoid(outputs.squeeze())
                        individual_probs.append(probs)
                    
                    # Stack all model probabilities
                    ensemble_probs = torch.stack(individual_probs)  # [num_models, batch_size]
                    
                    # Mean probability and uncertainty
                    mean_probs = ensemble_probs.mean(dim=0)
                    uncertainty = ensemble_probs.std(dim=0)  # Standard deviation as uncertainty
                    
                elif hasattr(self.trainer, 'model1'):  # CoTeaching
                    outputs1 = self.trainer.model1(images)
                    outputs2 = self.trainer.model2(images)
                    probs1 = torch.sigmoid(outputs1.squeeze())
                    probs2 = torch.sigmoid(outputs2.squeeze())
                    
                    # Average the two models
                    mean_probs = (probs1 + probs2) / 2
                    uncertainty = torch.abs(probs1 - probs2)  # Disagreement as uncertainty
                    
                else:  # Standard model
                    outputs = self.trainer.model(images)
                    mean_probs = torch.sigmoid(outputs.squeeze())
                    # For standard models, use distance from decision boundary as uncertainty proxy
                    uncertainty = torch.abs(mean_probs - 0.5)
                
                probabilities.append(mean_probs.cpu().numpy())
                uncertainties.append(uncertainty.cpu().numpy())
                
                # Clear GPU memory after each batch
                del images
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate all results
        all_probabilities = np.concatenate(probabilities)
        all_uncertainties = np.concatenate(uncertainties)
        
        # Apply sampling if needed (use same indices as features if available)
        if max_samples is not None and len(all_probabilities) > max_samples:
            if self.sample_indices is not None:
                # Use the same indices as feature extraction
                indices = self.sample_indices
            else:
                np.random.seed(random_state)
                indices = np.random.choice(len(all_probabilities), size=max_samples, replace=False)
                indices = np.sort(indices)
            
            all_probabilities = all_probabilities[indices]
            all_uncertainties = all_uncertainties[indices]
        
        uncertainty_type = "ensemble std" if hasattr(self.trainer, 'models') else \
                          "model disagreement" if hasattr(self.trainer, 'model1') else \
                          "decision boundary distance"
        print(f"Extracted {len(all_probabilities)} prediction probabilities and uncertainties ({uncertainty_type})")
        return all_probabilities, all_uncertainties

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
                                additional_features: Optional[Dict[str, np.ndarray]] = None,
                                probabilities: Optional[np.ndarray] = None,
                                uncertainties: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Build DataFrame joining UMAP output + classification + hover images + uncertainties."""
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
        
        # Check if sample_indices are compatible with the provided predictions/labels
        # This handles the case where predictions/labels are filtered (e.g., by object IDs)
        # but sample_indices still reference the original unfiltered dataset
        max_index = np.max(sample_indices) if len(sample_indices) > 0 else -1
        
        if len(predictions) < len(sample_indices) or max_index >= len(predictions):
            print(f"Warning: Sample indices mismatch detected!")
            print(f"  - UMAP samples: {len(sample_indices)} (max index: {max_index})")
            print(f"  - Available predictions: {len(predictions)}")
            print(f"  - Available labels: {len(labels)}")
            
            # Strategy 1: If we have the right number of samples but wrong indices, remap to sequential
            if len(predictions) == len(self.umap_embedding):
                print("  - Predictions/labels match UMAP embedding size: using sequential indices")
                sample_indices = np.arange(len(predictions))
            
            # Strategy 2: If we have more UMAP embeddings than predictions, truncate UMAP
            elif len(self.umap_embedding) > len(predictions):
                print(f"  - Truncating UMAP embedding from {len(self.umap_embedding)} to {len(predictions)} samples")
                self.umap_embedding = self.umap_embedding[:len(predictions)]
                sample_indices = np.arange(len(predictions))
                print(f"  - New UMAP embedding shape: {self.umap_embedding.shape}")
                
            # Strategy 3: If UMAP embedding matches sample_indices length, remap indices
            elif len(self.umap_embedding) == len(sample_indices):
                print(f"  - UMAP embedding matches sample count: remapping indices sequentially")
                sample_indices = np.arange(len(self.umap_embedding))
                print(f"  - Using indices 0 to {len(sample_indices)-1} for {len(self.umap_embedding)} UMAP points")
                
                # Verify predictions/labels have enough samples
                if len(predictions) < len(sample_indices):
                    print(f"  - Warning: Still not enough predictions ({len(predictions)}) for UMAP samples ({len(sample_indices)})")
                    print(f"  - Truncating to minimum: {min(len(predictions), len(sample_indices))}")
                    min_samples = min(len(predictions), len(sample_indices))
                    self.umap_embedding = self.umap_embedding[:min_samples]
                    sample_indices = np.arange(min_samples)
                    
            else:
                print("  - Error: Cannot resolve sample indices mismatch")
                print(f"    UMAP embedding shape: {self.umap_embedding.shape}")
                print(f"    Sample indices length: {len(sample_indices)}")
                print(f"    Predictions length: {len(predictions)}")
                raise ValueError(
                    f"Cannot align sample indices (count: {len(sample_indices)}, max: {max_index}) "
                    f"with predictions array (size: {len(predictions)}). "
                    f"UMAP embedding has {len(self.umap_embedding)} samples. "
                    f"This usually happens when filtering by object IDs - "
                    f"the feature extraction and inference use different filtered datasets."
                )
    
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
        
        # OPTIMIZATION: Use pre-computed probabilities/uncertainties if available
        print("Adding prediction probabilities and uncertainties...")
        
        if probabilities is not None and uncertainties is not None:
            print("Using pre-computed probabilities and uncertainties from inference results")
            # Use the already computed values, applying sampling if needed
            # Check if probabilities/uncertainties arrays match sample_indices requirements
            if len(probabilities) >= len(sample_indices) and np.max(sample_indices) < len(probabilities):
                df['prediction_probability'] = probabilities[sample_indices]
                df['prediction_uncertainty'] = uncertainties[sample_indices]
            else:
                # If there's a size mismatch, use direct alignment
                print(f"Warning: Probability array size mismatch. Using direct alignment.")
                print(f"  - Probabilities length: {len(probabilities)}")
                print(f"  - Sample indices length: {len(sample_indices)}, max: {np.max(sample_indices)}")
                if len(probabilities) == len(self.umap_embedding):
                    df['prediction_probability'] = probabilities
                    df['prediction_uncertainty'] = uncertainties
                else:
                    print("Cannot align probabilities with UMAP embedding. Falling back to computation.")
                    probabilities = None
                    uncertainties = None
            
            if probabilities is not None:  # Check if we still have valid probabilities
                uncertainty_type = "ensemble std" if hasattr(self.trainer, 'models') else \
                                  "model disagreement" if hasattr(self.trainer, 'model1') else \
                                  "decision boundary distance"
        
        if probabilities is None or uncertainties is None:
            # Fall back to computing uncertainties only if not available
            if hasattr(self.trainer, 'models') or hasattr(self.trainer, 'model1'):
                print("Computing uncertainties for ensemble/coteaching model (not pre-computed)...")
                computed_probs, computed_uncert = self._extract_uncertainties_efficient(data_loader, sample_indices)
                df['prediction_probability'] = computed_probs
                df['prediction_uncertainty'] = computed_uncert
                uncertainty_type = "ensemble std" if hasattr(self.trainer, 'models') else "model disagreement"
            else:
                print("Creating uncertainty proxy for standard model...")
                # For standard models, create uncertainty proxy from predictions
                if probabilities is not None:
                    probs_subset = probabilities[sample_indices]
                else:
                    # Use predictions as probability proxy
                    probs_subset = predictions[sample_indices].astype(float)
                
                df['prediction_probability'] = probs_subset
                df['prediction_uncertainty'] = np.abs(probs_subset - 0.5)  # Distance from decision boundary
                uncertainty_type = "decision boundary distance"
        
        print(f"Added prediction probabilities and uncertainties ({uncertainty_type})")
        
        # OPTIMIZATION: Efficient image processing - only process sampled indices
        print("Creating embeddable images for hover tooltips...")
        images = self._create_embeddable_images_efficient(data_loader, sample_indices)
        df['image'] = images
        print(f"Created {len(images)} embeddable images")
        
        if additional_features:
            for key, values in additional_features.items():
                df[key] = values[sample_indices]
        
        return df

    def _extract_uncertainties_efficient(self, data_loader, sample_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Efficiently extract uncertainties only for sampled indices."""
        print(f"Efficiently extracting uncertainties for {len(sample_indices)} samples...")
        
        probabilities = []
        uncertainties = []
        
        # Create a set for O(1) lookup
        indices_set = set(sample_indices)
        current_idx = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx % 20 == 0:  # Reduce print frequency
                    print(f"Processing batch {batch_idx}/{len(data_loader)}")
                
                images, *_ = batch
                batch_size = images.shape[0]
                
                # Check if any indices in this batch are needed
                batch_indices = list(range(current_idx, current_idx + batch_size))
                needed_indices = [i for i in batch_indices if i in indices_set]
                
                if not needed_indices:
                    current_idx += batch_size
                    continue
                
                # Only process if we need data from this batch
                images = images.to(self.device)
                
                if hasattr(self.trainer, 'models'):  # Ensemble
                    individual_probs = []
                    for model in self.trainer.models:
                        outputs = model(images)
                        probs = torch.sigmoid(outputs.squeeze())
                        individual_probs.append(probs)
                    
                    ensemble_probs = torch.stack(individual_probs)
                    mean_probs = ensemble_probs.mean(dim=0)
                    uncertainty = ensemble_probs.std(dim=0)
                    
                elif hasattr(self.trainer, 'model1'):  # CoTeaching
                    outputs1 = self.trainer.model1(images)
                    outputs2 = self.trainer.model2(images)
                    probs1 = torch.sigmoid(outputs1.squeeze())
                    probs2 = torch.sigmoid(outputs2.squeeze())
                    
                    mean_probs = (probs1 + probs2) / 2
                    uncertainty = torch.abs(probs1 - probs2)
                
                # Extract only needed samples from this batch
                batch_probs = mean_probs.cpu().numpy()
                batch_uncert = uncertainty.cpu().numpy()
                
                for i, global_idx in enumerate(batch_indices):
                    if global_idx in indices_set:
                        probabilities.append(batch_probs[i])
                        uncertainties.append(batch_uncert[i])
                
                current_idx += batch_size
                
                # Clear GPU memory
                del images
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return np.array(probabilities), np.array(uncertainties)

    def _create_embeddable_images_efficient(self, data_loader, sample_indices: np.ndarray) -> List[str]:
        """Efficiently create embeddable images only for sampled indices."""
        print(f"Efficiently creating images for {len(sample_indices)} samples...")
        
        # Create mapping for quick lookup
        indices_set = set(sample_indices)
        index_to_position = {idx: pos for pos, idx in enumerate(sample_indices)}
        images = [''] * len(sample_indices) # Pre-allocate
        
        current_idx = 0
        processed_count = 0
        
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 20 == 0 and processed_count > 0:  # Reduce print frequency
                print(f"Processed {processed_count}/{len(sample_indices)} images...")
            
            batch_images, *_ = batch
            batch_size = batch_images.shape[0]
            
            # Check if any indices in this batch are needed
            batch_indices = list(range(current_idx, current_idx + batch_size))
            needed_indices = [i for i in batch_indices if i in indices_set]
            
            if not needed_indices:
                current_idx += batch_size
                continue
            
            # Process only needed images from this batch
            for i, global_idx in enumerate(batch_indices):
                if global_idx in indices_set:
                    # Convert tensor to numpy if needed
                    if hasattr(batch_images, 'cpu'):
                        img_array = batch_images[i].cpu().numpy()
                    else:
                        img_array = batch_images[i]
                    
                    # Convert to embeddable format with RdYlGn colormap (preserves negatives)
                    embeddable_img = embeddable_image(img_array, cmap='gray')
                    position = index_to_position[global_idx]
                    images[position] = embeddable_img
                    processed_count += 1
                    
                    # Early exit if we've processed all needed images
                    if processed_count >= len(sample_indices):
                        return images
            
            current_idx += batch_size
        
        return images

    def save_umap(self, path: str):
        """Save the fitted UMAP reducer to disk."""
        if self.umap_reducer is None:
            raise ValueError("UMAP reducer not fitted yet.")
        joblib.dump(self.umap_reducer, path)
        print(f"UMAP reducer saved to {path}")

    def load_umap(self, path: str):
        """Load a fitted UMAP reducer from disk."""
        self.umap_reducer = joblib.load(path)
        print(f"UMAP reducer loaded from {path}")

    def transform_with_umap(self, features: np.ndarray) -> np.ndarray:
        """Transform features using the loaded/fitted UMAP reducer."""
        if self.umap_reducer is None:
            raise ValueError("UMAP reducer not loaded/fitted.")
        embedding = self.umap_reducer.transform(features)
        print(f"Transformed {features.shape[0]} samples with loaded UMAP.")
        return embedding