import numpy as np
import pandas as pd
import torch
import umap.umap_ as umap
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import base64
import io
from PIL import Image

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
    """Class for UMAP-based model interpretability analysis (computation only).
    
    Provides tools for extracting features from neural network layers,
    applying UMAP dimensionality reduction, and creating interpretable
    visualizations of model behavior.
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
    
    def extract_features(self, data_loader, layer_name: str = "fc1") -> np.ndarray:
        """
        Extract features from specified layer of the model.
        
        Args:
            data_loader: DataLoader with input data
            layer_name: Which layer to extract features from
            
        Returns:
            Extracted features as numpy array
        """
        print(f"Extracting features from layer: {layer_name}")
        
        features = []
        total_batches = len(data_loader)
        
        # Handle special cases
        if layer_name == "output":
            # Extract from final model output
            with torch.no_grad():
                for batch_idx, batch in enumerate(data_loader):
                    if batch_idx % 10 == 0:  # Progress every 10 batches
                        print(f"Processing batch {batch_idx+1}/{total_batches}")
                    
                    images, *_ = batch
                    images = images.to(self.device)
                    feat = self.trainer.model(images)
                    feat_flat = feat.view(feat.size(0), -1)
                    features.append(feat_flat.cpu().numpy())
                    
                    # Clear GPU memory after each batch
                    del images, feat, feat_flat
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
        
        self.features = np.vstack(features)
        print(f"Extracted features shape: {self.features.shape}")
        
        # Force garbage collection after feature extraction
        import gc
        gc.collect()
        
        return self.features
    
    def fit_umap(self, n_neighbors: int = 15, min_dist: float = 0.1, 
                 n_components: int = 2, random_state: int = 42) -> np.ndarray:
        """
        Fit UMAP on extracted features.
        
        Args:
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter  
            n_components: Number of UMAP dimensions
            random_state: Random seed
            
        Returns:
            UMAP embedding
        """
        if self.features is None:
            raise ValueError("No features extracted. Call extract_features first.")
        
        self.umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state
        )
        
        self.umap_embedding = self.umap_reducer.fit_transform(self.features)
        return self.umap_embedding
    
    def cluster_umap(self, n_components_range: Tuple[int, int] = (5, 15)) -> np.ndarray:
        """Perform Gaussian Mixture clustering on UMAP embedding.
        
        Uses grid search to find optimal GMM parameters based on BIC score.
        
        Parameters
        ----------
        n_components_range : tuple of int, default=(5, 15)
            Range of number of components to try for GMM
            
        Returns
        -------
        np.ndarray
            Cluster labels for each sample
            
        Raises
        ------
        ValueError
            If no UMAP embedding exists
        """
        if self.umap_embedding is None:
            raise ValueError("No UMAP embedding. Call fit_umap first.")
        
        # Grid search for best GMM parameters
        param_grid = {
            "n_components": range(n_components_range[0], n_components_range[1] + 1),
            "covariance_type": ["spherical", "tied", "diag", "full"],
        }
        
        def gmm_bic_score(estimator, X):
            return -estimator.bic(X)
        
        grid_search = GridSearchCV(
            GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
        )
        grid_search.fit(self.umap_embedding)
        
        # Fit best GMM
        best_gmm = GaussianMixture(
            n_components=grid_search.best_params_["n_components"],
            covariance_type=grid_search.best_params_["covariance_type"]
        )
        best_gmm.fit(self.umap_embedding)
        
        return best_gmm.predict(self.umap_embedding)

    def create_interpretability_dataframe(self, predictions: np.ndarray, labels: np.ndarray,
                                    data_loader, sample_indices: Optional[np.ndarray] = None,
                                    additional_features: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """
        Create interpretability DataFrame with UMAP coordinates and additional information.
        
        Args:
            predictions: Model predictions
            labels: True labels
            data_loader: DataLoader with input data
            sample_indices: Indices of samples to include (optional)
            additional_features: Additional features to include (e.g., SNR)
            
        Returns:
            DataFrame with UMAP coordinates and additional information
        """
        if self.umap_embedding is None:
            raise ValueError("No UMAP embedding. Call fit_umap first.")
        
        # Sample indices if not provided
        if sample_indices is None:
            sample_indices = np.arange(len(predictions))
        
        # Create base DataFrame with UMAP coordinates
        df = pd.DataFrame({
            'umap_x': self.umap_embedding[:, 0],
            'umap_y': self.umap_embedding[:, 1],
            'prediction': predictions,
            'true_label': labels,
            'correct': predictions == labels
        }, index=sample_indices)
        
        # Add classification categories
        df['class_type'] = 'True Negative'
        df.loc[(df['true_label'] == 1) & (df['prediction'] == 1), 'class_type'] = 'True Positive'
        df.loc[(df['true_label'] == 0) & (df['prediction'] == 1), 'class_type'] = 'False Positive'
        df.loc[(df['true_label'] == 1) & (df['prediction'] == 0), 'class_type'] = 'False Negative'
        
        # Add embeddable images if we can extract them from data_loader
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
    
    def add_clustering_to_dataframe(self, df: pd.DataFrame, 
                                  n_components_range: Tuple[int, int] = (5, 15)) -> pd.DataFrame:
        """Add clustering information to the DataFrame using Gaussian Mixture Model.
        
        Performs GMM clustering on the UMAP embedding and adds cluster labels
        to the DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with UMAP coordinates
        n_components_range : tuple of int, default=(5, 15)
            Range of components to try for GMM optimization
            
        Returns
        -------
        pd.DataFrame
            DataFrame with additional 'cluster' column containing string labels
            
        Raises
        ------
        ValueError
            If no UMAP embedding exists
        """
        if self.umap_embedding is None:
            raise ValueError("No UMAP embedding. Call fit_umap first.")
        
        # Grid search for best GMM parameters
        param_grid = {
            "n_components": range(n_components_range[0], n_components_range[1] + 1),
            "covariance_type": ["spherical", "tied", "diag", "full"],
        }
        
        def gmm_bic_score(estimator, X):
            return -estimator.bic(X)
        
        grid_search = GridSearchCV(
            GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
        )
        grid_search.fit(self.umap_embedding)
        
        # Fit best GMM
        best_gmm = GaussianMixture(
            n_components=grid_search.best_params_["n_components"],
            covariance_type=grid_search.best_params_["covariance_type"]
        )
        best_gmm.fit(self.umap_embedding)
        
        # Predict clusters
        df['cluster'] = best_gmm.predict(self.umap_embedding).astype(str)
        
        return df