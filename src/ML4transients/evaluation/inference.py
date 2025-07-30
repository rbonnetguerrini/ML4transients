import torch
import numpy as np
import h5py
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
from ML4transients.training import get_trainer
from ML4transients.utils import load_config

def infer(inference_loader, trainer=None, weights_path=None, return_preds=True, compute_metrics=True, device=None, save_path=None, dia_source_ids=None):
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

    Returns:
        Optionally returns predictions, true labels, accuracy, and confusion matrix.
    """

    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if trainer is None and weights_path is None:
        raise ValueError("Please provide either a trainer instance or a dir for trained weights.")
    
    if trainer is None:
        config = load_config(f"{weights_path}/config.yaml")
        trainer = get_trainer(config["training"]["trainer_type"], config)
        state_dict = torch.load(f"{weights_path}/model_best.pth", map_location=device)
        trainer.model.load_state_dict(state_dict)

    model = trainer.model.to(device)
    model.eval()
    model.to(trainer.device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, *_ in inference_loader:
            images = images.to(trainer.device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float().cpu().numpy()
            labels = labels.cpu().numpy()

            if return_preds:
                all_preds.append(preds)
                all_labels.append(labels)

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

    return results if return_preds else None

def save_inference_results(results, save_path, weights_path=None, dia_source_ids=None):
    """
    Save inference results to HDF5 file.
    
    Args:
        results: Dictionary containing inference results
        save_path: Path where to save the results
        weights_path: Path to the model weights (for metadata)
        dia_source_ids: Array of diaSourceIds corresponding to the results
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
    
    print(f"Inference results saved to {save_path}")