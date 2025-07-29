import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from ML4transients.training import get_trainer
from ML4transients.utils import load_config

def infer(inference_loader, trainer= None, weights_path= None,  return_preds=True, compute_metrics=True, device=None):
    """
    Run inference using a trained model on a dataset with minimal memory usage.

    Args:
        trainer: An object with .model and .device.
        inference_loader: A PyTorch DataLoader providing (images, labels, ...) tuples.
        return_preds (bool): If True, returns predictions and labels.
        compute_metrics (bool): If True, computes accuracy and confusion matrix.

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

    return results if return_preds else None
