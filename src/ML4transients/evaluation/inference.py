import torch
import numpy as np
import h5py
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
from ML4transients.training import get_trainer
from ML4transients.utils import load_config

def infer(inference_loader, trainer=None, weights_path=None, return_preds=True, compute_metrics=True, device=None, save_path=None, dia_source_ids=None, visit=None, model_hash=None, return_probabilities=None, mc_dropout=False, mc_samples=50):
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
        model_hash: Optional model hash for saving in results metadata.
        return_probabilities (bool): If True, also returns prediction probabilities and uncertainty.
                                   If None, auto-detects based on model type.
        mc_dropout (bool): If True, enables Monte Carlo Dropout for standard models.
        mc_samples (int): Number of forward passes for MC Dropout (default: 50).

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
        trainer_type = config["training"]["trainer_type"]
        trainer = get_trainer(trainer_type, config["training"])
        print(trainer_type)
        if trainer_type == "ensemble":
            num_models = config["training"]["num_models"]
            print(f"Ensemble model, loading {num_models} models.")
            for i in range(num_models):
                model_path = f"{weights_path}/ensemble_model_{i}_best.pth"   
                print(f"Loading model {i} from {model_path}")
                state_dict = torch.load(model_path, map_location=device)
                trainer.models[i].load_state_dict(state_dict)
                trainer.models[i].to(device)
                trainer.models[i].eval()
        elif (trainer_type == "coteaching" or 
                trainer_type == "coteaching_asym" or 
                trainer_type == "stochastic_coteaching"):
            # Load both models for co-teaching
            state_dict1 = torch.load(f"{weights_path}/model1_best.pth", map_location=device)
            state_dict2 = torch.load(f"{weights_path}/model2_best.pth", map_location=device)
            trainer.model1.load_state_dict(state_dict1)
            trainer.model2.load_state_dict(state_dict2)
            trainer.model1.to(device)
            trainer.model2.to(device)
            trainer.model1.eval()
            trainer.model2.eval()
        else:
            # Standard single model
            state_dict = torch.load(f"{weights_path}/model_best.pth", map_location=device)
            trainer.model.load_state_dict(state_dict)
            trainer.model.to(device)
            trainer.model.eval()
    else:
        print(f"Using cached model{visit_str}...")

    # Auto-detect if we should return probabilities based on trainer type
    if return_probabilities is None:
        if hasattr(trainer, 'models') or hasattr(trainer, 'model1'):
            return_probabilities = True  # Ensemble or CoTeaching - extract uncertainties
            if mc_dropout and hasattr(trainer, 'model1'):
                print(f"Auto-detected CoTeaching model with MC Dropout ({mc_samples} samples per model) - will extract probabilities and uncertainties")
            else:
                print("Auto-detected ensemble/coteaching model - will extract probabilities and uncertainties")
        elif mc_dropout:
            return_probabilities = True  # MC Dropout enabled - extract uncertainties
            print(f"MC Dropout enabled with {mc_samples} samples - will extract probabilities and uncertainties")
        else:
            return_probabilities = False  # Standard model

    # Set all models to eval mode and correct device
    if hasattr(trainer, 'models'):  # Ensemble
        for model in trainer.models:
            model.eval()
            model.to(device)
    elif hasattr(trainer, 'model1'):  # CoTeaching
        trainer.model1.eval()
        trainer.model1.to(device)
        trainer.model2.eval()
        trainer.model2.to(device)
    else:  # Standard
        trainer.model.eval()
        trainer.model.to(device)

    all_preds = []
    all_labels = []
    all_probabilities = []
    all_uncertainties = []

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
                
                # Get predictions based on trainer type
                if hasattr(trainer, 'models'):  # Ensemble
                    individual_probs = []
                    for model in trainer.models:
                        outputs = model(images)
                        probs = torch.sigmoid(outputs.squeeze())
                        individual_probs.append(probs)
                    
                    # Stack all model probabilities
                    ensemble_probs = torch.stack(individual_probs)  # [num_models, batch_size]
                    
                    # Mean probability and uncertainty
                    mean_probs = ensemble_probs.mean(dim=0)
                    uncertainty = ensemble_probs.std(dim=0)  # Standard deviation as uncertainty
                    
                    preds = (mean_probs > 0.5).float().cpu().numpy()
                    probs = mean_probs.cpu().numpy()
                    uncert = uncertainty.cpu().numpy()
                    
                elif hasattr(trainer, 'model1'):  # CoTeaching
                    if mc_dropout and mc_samples > 1:
                        # MC Dropout for CoTeaching: N=2 models, M=mc_samples per model
                        # Combined uncertainty from both model diversity and dropout sampling
                        trainer.model1.enable_mc_dropout()
                        trainer.model2.enable_mc_dropout()
                        
                        all_predictions = []
                        for _ in range(mc_samples):
                            # Forward pass through both models with different dropout masks
                            outputs1 = trainer.model1(images)
                            outputs2 = trainer.model2(images)
                            probs1 = torch.sigmoid(outputs1.squeeze())
                            probs2 = torch.sigmoid(outputs2.squeeze())
                            # Collect predictions from both models
                            all_predictions.append(probs1)
                            all_predictions.append(probs2)
                        
                        # Stack all N*M predictions (2 models × mc_samples)
                        all_probs = torch.stack(all_predictions)  # [2*mc_samples, batch_size]
                        mean_probs = all_probs.mean(dim=0)
                        uncertainty = all_probs.std(dim=0)  # Variance across all samples
                        
                        preds = (mean_probs > 0.5).float().cpu().numpy()
                        probs = mean_probs.cpu().numpy()
                        uncert = uncertainty.cpu().numpy()
                        
                        # Restore normal eval mode
                        trainer.model1.disable_mc_dropout()
                        trainer.model2.disable_mc_dropout()
                    else:
                        # Standard CoTeaching without MC Dropout
                        outputs1 = trainer.model1(images)
                        outputs2 = trainer.model2(images)
                        probs1 = torch.sigmoid(outputs1.squeeze())
                        probs2 = torch.sigmoid(outputs2.squeeze())
                        
                        # Average the two models
                        mean_probs = (probs1 + probs2) / 2
                        uncertainty = torch.abs(probs1 - probs2)  # Disagreement as uncertainty
                        
                        preds = (mean_probs > 0.5).float().cpu().numpy()
                        probs = mean_probs.cpu().numpy()
                        uncert = uncertainty.cpu().numpy()
                    
                else:  # Standard
                    if mc_dropout and mc_samples > 1:
                        # Monte Carlo Dropout: Multiple forward passes with dropout enabled
                        trainer.model.enable_mc_dropout()
                        
                        mc_predictions = []
                        for _ in range(mc_samples):
                            outputs = trainer.model(images)
                            mc_probs = torch.sigmoid(outputs.squeeze())
                            mc_predictions.append(mc_probs)
                        
                        # Stack and compute statistics
                        mc_probs_stacked = torch.stack(mc_predictions)  # [mc_samples, batch_size]
                        mean_probs = mc_probs_stacked.mean(dim=0)
                        uncertainty = mc_probs_stacked.std(dim=0)  # Standard deviation as uncertainty
                        
                        preds = (mean_probs > 0.5).float().cpu().numpy()
                        probs = mean_probs.cpu().numpy()
                        uncert = uncertainty.cpu().numpy()
                        
                        # Restore normal eval mode
                        trainer.model.disable_mc_dropout()
                    else:
                        # Standard single forward pass
                        outputs = trainer.model(images)
                        probs = torch.sigmoid(outputs.squeeze())
                        preds = (probs > 0.5).float().cpu().numpy()
                        probs = probs.cpu().numpy()
                        uncert = np.zeros_like(probs)  # No uncertainty for single model
                
                labels = labels.cpu().numpy()

                if return_preds:
                    all_preds.append(preds)
                    all_labels.append(labels)
                    if return_probabilities:
                        all_probabilities.append(probs)
                        all_uncertainties.append(uncert)
                
                # Clear batch from GPU memory immediately
                del images
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
        
        if return_probabilities:
            results["y_prob"] = np.concatenate(all_probabilities)
            results["uncertainty"] = np.concatenate(all_uncertainties)
            print(f"Extracted probabilities and uncertainties (uncertainty mean: {results['uncertainty'].mean():.4f})")

        if compute_metrics:
            accuracy = accuracy_score(y_true, y_pred.astype(int))
            conf_matrix = confusion_matrix(y_true, y_pred.astype(int))

            results["accuracy"] = accuracy
            results["confusion_matrix"] = conf_matrix/len(y_true)

        # Save results if requested
        if save_path is not None:
            print(f"Saving inference results to {save_path}...")
            save_inference_results(results, save_path, weights_path, dia_source_ids, visit, model_hash)
        
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
        
        # Save probabilities and uncertainties if available
        if 'y_prob' in results:
            f.create_dataset('probabilities', data=results['y_prob'])
        if 'uncertainty' in results:
            f.create_dataset('uncertainties', data=results['uncertainty'])
        
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