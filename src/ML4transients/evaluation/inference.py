import torch
import numpy as np
import h5py
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
from ML4transients.training import get_trainer
from ML4transients.utils import load_config

def infer(inference_loader, trainer=None, weights_path=None, return_preds=True, compute_metrics=True, device=None, save_path=None, dia_source_ids=None, visit=None):
    """
    Run inference using a trained model on a dataset with minimal memory usage.

    Args:
        trainer: An object with .model and .device.
        inference_loader: A PyTorch DataLoader providing (images, labels, ...) tuples.
        return_preds (bool): If True, returns predictions and labels.
        compute_metrics (bool): If True, computes accuracy and confusion matrix.
        device: Device to run inference on.
        save_path: Optional path to save inference results.
        dia_source_ids: Array of diaSourceIds corresponding to the inference data.
        visit: Optional visit number for logging purposes.

    Returns:
        Optionally returns predictions, true labels, accuracy, and confusion matrix.
    """
    visit_str = f" for visit {visit}" if visit is not None else ""
    print(f"Starting inference{visit_str}...")

    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model only if trainer not provided (avoid redundant loading)
    if trainer is None:
        if weights_path is None:
            raise ValueError("Please provide either a trainer instance or a weights_path.")
        
        print(f"Loading model from {weights_path}...")
        config = load_config(f"{weights_path}/config.yaml")
        trainer = get_trainer(config["training"]["trainer_type"], config["training"])
        state_dict = torch.load(f"{weights_path}/model_best.pth", map_location=device)
        trainer.model.load_state_dict(state_dict)
        trainer.model.to(device)
        trainer.model.eval()
    else:
        print(f"Using cached model{visit_str}...")

    model = trainer.model
    # Ensure model is in eval mode and on correct device
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(inference_loader):
                if batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx}/{len(inference_loader)}{visit_str}")
                    
                    # Monitor GPU memory if available
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                        if memory_used > 8:  # Warning if using more than 8GB
                            print(f"Warning: GPU memory usage: {memory_used:.1f}GB")
                
                images, labels, *_ = batch
                images = images.to(device)
                outputs = model(images)
                preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float().cpu().numpy()
                labels = labels.cpu().numpy()

                if return_preds:
                    all_preds.append(preds)
                    all_labels.append(labels)
                
                # Clear batch from GPU memory immediately
                del images, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        # Ensure cleanup even if error occurs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {}
    if return_preds:
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        results["y_pred"] = y_pred
        results["y_true"] = y_true

        if compute_metrics:
            accuracy = accuracy_score(y_true, y_pred.astype(int))
            conf_matrix = confusion_matrix(y_true, y_pred.astype(int))

            results["accuracy"] = accuracy
            results["confusion_matrix"] = conf_matrix/len(y_true)

        # Save results if requested
        if save_path is not None:
            save_inference_results(results, save_path, weights_path, dia_source_ids)
        
        if visit is not None:
            print(f"Inference for visit {visit} completed. Accuracy: {results.get('accuracy', 'N/A'):.4f}")

    return results if return_preds else None

def save_inference_results(results, save_path, weights_path=None, dia_source_ids=None, visit=None, model_hash=None):
    """
    Save inference results to HDF5 file and register in the dataset.
    
    Args:
        results: Dictionary containing inference results
        save_path: Path where to save the results
        weights_path: Path to the model weights (for metadata)
        dia_source_ids: Array of diaSourceIds corresponding to the results
        visit: Visit number for registry
        model_hash: Model hash for registry
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('predictions', data=results['y_pred'])
        f.create_dataset('labels', data=results['y_true'])
        
        if dia_source_ids is not None:
            f.create_dataset('diaSourceId', data=dia_source_ids)
        
        if 'accuracy' in results:
            f.attrs['accuracy'] = results['accuracy']
        
        if 'confusion_matrix' in results:
            f.create_dataset('confusion_matrix', data=results['confusion_matrix'])
        
        # Save metadata
        if weights_path:
            f.attrs['weights_path'] = str(weights_path)
        
        if visit is not None:
            f.attrs['visit'] = visit
            
        if model_hash:
            f.attrs['model_hash'] = model_hash
        
        # Save dataset size for validation
        f.attrs['n_samples'] = len(results['y_pred'])
    
    print(f"Inference results saved to {save_path}")