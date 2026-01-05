import os
import torch
from torch.utils.data import DataLoader
import yaml
import argparse
import sys
from pathlib import Path
import copy
import numpy as np

# Add the ML4transients package to path
sys.path.append('/sps/lsst/users/rbonnetguerrini/ML4transients/src')

from ML4transients.training.pytorch_dataset import PytorchDataset
from ML4transients.training.trainers import get_trainer
from ML4transients.data_access.dataset_loader import DatasetLoader
from ML4transients.evaluation import infer

from ML4transients.utils import load_config

import optuna


def create_data_loaders(config):
    """Create train/val/test data loaders"""
    data_path = config['data']['path']
    visits = config['data'].get('visits', None)
    batch_size = config['training']['batch_size']
    cutout_types = config['data'].get('cutout_types', ['diff'])  # Default to diff only
    
    # Create splits efficiently
    splits = PytorchDataset.create_splits(
        data_source=data_path,
        visits=visits,
        cutout_types=cutout_types,  # Pass cutout types to dataset
        test_size=config['data'].get('test_size', 0.2),
        val_size=config['data'].get('val_size', 0.1),
        random_state=config.get('random_state', 42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        splits['train'], 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    test_loader = DataLoader(
        splits['test'], 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    val_loader = None
    if splits['val'] is not None:
        val_loader = DataLoader(
            splits['val'], 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=config['training'].get('num_workers', 4)
        )
    
    return train_loader, val_loader, test_loader

def create_data_loaders_from_datasets(datasets, batch_size, num_workers):
    """Create loaders from pre-created dataset splits."""
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers) if datasets['val'] else None
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def set_nested(config_dict, dotted_key, value):
    """Set nested config value with dotted path (e.g., model_params.filters_1)."""
    parts = dotted_key.split('.')
    d = config_dict
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = value

def suggest_param(trial, name, spec):
    """Suggest a single parameter based on spec."""
    if 'choices' in spec:
        return trial.suggest_categorical(name, spec['choices'])
    
    ptype = spec.get('type', 'float')
    
    # Ensure numeric values are properly converted
    try:
        if ptype == 'int':
            low = int(spec['low'])
            high = int(spec['high'])
            if 'step' in spec:
                step = int(spec['step'])
                return trial.suggest_int(name, low, high, step=step)
            return trial.suggest_int(name, low, high)
        
        elif ptype == 'float':
            low = float(spec['low'])
            high = float(spec['high'])
            log_scale = spec.get('log', False)
            
            if log_scale:
                if low <= 0:
                    raise ValueError(f"For log scale, low must be > 0, got {low}")
                return trial.suggest_float(name, low, high, log=True)
            
            if 'step' in spec:
                step = float(spec['step'])
                return trial.suggest_float(name, low, high, step=step)
            
            return trial.suggest_float(name, low, high)
        
        else:
            raise ValueError(f"Unsupported param type: {ptype}")
            
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error parsing parameter '{name}' with spec {spec}: {e}")

def run_bayesian_optimization(config):
    if optuna is None:
        raise ImportError("optuna not installed. Install it or disable bayes_search.")
    bs_cfg = config['training']['bayes_search']
    direction = bs_cfg.get('direction', 'minimize')
    n_trials = bs_cfg.get('n_trials', 20)
    max_epochs = bs_cfg.get('max_epochs', config['training']['epochs'])
    prune = bs_cfg.get('prune', False)
    cutout_types = config['data'].get('cutout_types', ['diff'])  # Get cutout types
    hpo_train_fraction = bs_cfg.get('hpo_train_fraction', 1.0)  # Default: use full training set

    # Precompute dataset splits once
    print("Preparing datasets for Bayesian optimization...")
    splits = PytorchDataset.create_splits(
        data_source=config['data']['path'],
        visits=config['data'].get('visits'),
        cutout_types=cutout_types,  # Pass cutout types
        test_size=config['data'].get('test_size', 0.2),
        val_size=config['data'].get('val_size', 0.1),
        random_state=config.get('random_state', 42)
    )
    
    # Apply stratified subsampling to training set if requested
    if hpo_train_fraction < 1.0:
        original_train_size = len(splits['train'])
        train_dataset = splits['train']
        
        # Get labels efficiently from split_info if available
        if 'split_info' in splits and splits['split_info'] is not None:
            # Access labels from the saved split_info
            split_info = splits['split_info']
            train_indices = split_info['train_idx']
            all_labels = split_info['labels']
            train_labels = all_labels[train_indices]
            print(f"Using cached labels from split_info")
        else:
            # Fallback: iterate through dataset (slower)
            print(f"Warning: No split_info found, iterating through dataset to get labels...")
            train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
        
        # Subsample indices with stratification
        from sklearn.model_selection import train_test_split
        subset_indices, _ = train_test_split(
            np.arange(len(train_dataset)),
            train_size=hpo_train_fraction,
            stratify=train_labels,
            random_state=config.get('random_state', 42)
        )
        
        # Create subsampled dataset using Subset
        from torch.utils.data import Subset
        splits['train'] = Subset(train_dataset, subset_indices)
        
        print(f"Reduced training set from {original_train_size} to {len(splits['train'])} samples "
              f"({hpo_train_fraction*100:.0f}%) for HPO")
        print(f"Val set: {len(splits['val'])} samples (unchanged)")
        print(f"Test set: {len(splits['test'])} samples (unchanged)")
    
    datasets = {'train': splits['train'], 'val': splits['val'], 'test': splits['test']}

    def objective(trial):
        trial_config = copy.deepcopy(config)
        # Override epochs per trial
        trial_config['training']['epochs'] = max_epochs

        # Apply parameter suggestions
        for dotted_key, spec in bs_cfg.get('params', {}).items():
            val = suggest_param(trial, dotted_key, spec)
            set_nested(trial_config['training'], dotted_key, val)

        # Expand base_filters into geometric progression (F, 2F, 4F) if present
        model_params = trial_config['training'].get('model_params', {})        
        if 'base_filters' in model_params:
            F = model_params['base_filters']
            model_params['filters_1'] = F
            model_params['filters_2'] = 2 * F
            model_params['filters_3'] = 4 * F
            print(f"Expanded base_filters={F} to filters: ({F}, {2*F}, {4*F})")
            # Remove base_filters so it's not passed to model __init__
            del model_params['base_filters']
        
        # Expand base_dropout into scaled rates (0.5*DR, 0.5*DR, DR) if present
        if 'base_dropout' in model_params:
            DR = model_params['base_dropout']
            model_params['dropout_1'] = 0.5 * DR
            model_params['dropout_2'] = 0.5 * DR
            model_params['dropout_3'] = DR
            print(f"Expanded base_dropout={DR:.3f} to dropouts: ({0.5*DR:.3f}, {0.5*DR:.3f}, {DR:.3f})")
            # Remove base_dropout so it's not passed to model __init__
            del model_params['base_dropout']
        

        batch_size = trial_config['training'].get('batch_size', config['training']['batch_size'])
        num_workers = trial_config['training'].get('num_workers', 4)
        train_loader, val_loader, test_loader = create_data_loaders_from_datasets(
            datasets, batch_size=batch_size, num_workers=num_workers
        )

        try:
            trainer = get_trainer(trial_config['training']['trainer_type'], trial_config['training'])
            best_metric = trainer.fit(train_loader, val_loader, test_loader)
        except RuntimeError as e:
            # Handle training failures (e.g., too many rejected batches in co-teaching)
            error_msg = str(e)
            if 'rejected' in error_msg.lower() or 'batch' in error_msg.lower():
                print(f"Trial {trial.number} failed due to training instability: {error_msg}")
                # Return a very bad metric so Optuna knows this configuration is poor
                if direction == 'minimize':
                    return float('inf')
                else:
                    return float('-inf')
            else:
                # Re-raise unexpected errors
                raise

        # Get pruning metric (loss or accuracy for stable early stopping)
        prune_metric = bs_cfg.get('prune_metric', 'loss')  # Default to loss for pruning
        if prune_metric == 'loss':
            pruning_value = best_metric  # This is the loss from trainer.fit()
        elif prune_metric == 'accuracy':
            # Get accuracy from validation if available
            if val_loader is not None:
                val_metrics = trainer.evaluate(val_loader)
                pruning_value = 1.0 - val_metrics.get('accuracy', 0.0)  # Convert to minimization
            else:
                pruning_value = best_metric
        else:
            pruning_value = best_metric
        
        # Report for pruning (use stable metric like loss/accuracy)
        if prune:
            epoch = trainer.best_epoch if trainer.best_epoch >= 0 else trial.number
            trial.report(pruning_value, step=epoch)
            print(f"Trial {trial.number} - Pruning metric ({prune_metric}): {pruning_value:.4f} at epoch {epoch}")
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()
        
        # Compute final optimization metric (can be different from pruning metric)
        monitor_metric = bs_cfg.get('monitor', 'loss')
        if monitor_metric == 'fnr' and val_loader is not None:
            print(f"Computing FNR on validation set for trial {trial.number}...")
            val_results = infer(val_loader, trainer=trainer, return_preds=True, compute_metrics=True)
            
            # Extract confusion matrix and compute FNR
            tn, fp, fn, tp = val_results['confusion_matrix'].ravel()
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            print(f"Trial {trial.number} - Final FNR: {fnr:.4f}")
            metric_to_return = fnr
        else:
            metric_to_return = best_metric

        return metric_to_return

    # Create pruner for early stopping of unpromising trials
    if prune:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Don't prune the first 5 trials
            n_warmup_steps=20,   # Wait at least 20 epochs before pruning
            interval_steps=10    # Check every 10 epochs
        )
        print(f"Using MedianPruner for early stopping (startup_trials=5, warmup=20 epochs)")
    else:
        pruner = optuna.pruners.NopPruner()
    
    study = optuna.create_study(direction=direction, pruner=pruner)
    print(f"Starting Bayesian optimization for {n_trials} trials (direction={direction})")
    print(f"Pruning: {'enabled' if prune else 'disabled'} | Monitor metric: {bs_cfg.get('monitor', 'loss')} | Prune metric: {bs_cfg.get('prune_metric', 'loss')}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Bayesian optimization completed.")
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    # Save best params
    output_dir = config['training'].get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(output_dir, "bayes_best_params.yaml")
    with open(best_path, 'w') as f:
        yaml.dump({'best_value': study.best_value, 'best_params': study.best_params}, f)
    print(f"Saved best params to {best_path}")
    return study

def main():
    parser = argparse.ArgumentParser(description='Train transient bogus classifier')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--experiment-name', type=str, required=False, default=None, 
                        help='Name of the training (optional, defaults to experiment_name in config)')
    parser.add_argument('--hpo', action='store_true', help='Run Bayesian hyperparameter optimization instead of normal training')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
        
        # Override experiment name if specified via command line
        if args.experiment_name:
            config['training']['experiment_name'] = args.experiment_name
            print(f"Experiment name overridden to: {args.experiment_name}")
        
        # Ensure experiment_name exists in config
        if 'experiment_name' not in config['training']:
            config['training']['experiment_name'] = 'default_experiment'
            print("Warning: No experiment_name in config, using 'default_experiment'")
            
        # Create output directory (facilitating inference and weight loading)
        output_dir = config['training'].get('output_dir', 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Save a copy of the config in the output directory
        config_copy_path = os.path.join(output_dir, 'config.yaml')
        with open(config_copy_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Config saved to: {config_copy_path}")
        
        # Determine training mode
        run_hpo = args.hpo or config['training'].get('bayes_search', {}).get('enabled', False)
        
        if run_hpo:
            print("=== BAYESIAN HYPERPARAMETER OPTIMIZATION MODE ===")
            print(f"Using trainer: {config['training']['trainer_type']}")
            run_bayesian_optimization(config)
            return
        else:
            print("=== NORMAL TRAINING MODE ===")
            print(f"Using trainer: {config['training']['trainer_type']}")
        
        # Print TensorBoard info
        if config['training'].get('use_tensorboard', True):
            log_dir = config['training'].get('tensorboard_log_dir', 'runs')
            exp_name = config['training'].get('experiment_name', 'experiment')
            print(f"TensorBoard will log to: {log_dir}/{exp_name}")
            print(f"To view logs, run: tensorboard --logdir={log_dir}")
        
        # Determine number of input channels from cutout_types
        cutout_types = config['data'].get('cutout_types', ['diff'])
        in_channels = len(cutout_types)
        print(f"Using {in_channels} input channel(s): {cutout_types}")
        
        # Expand base_filters into geometric progression (F, 2F, 4F) if present
        model_params = config['training'].get('model_params', {})
        if 'base_filters' in model_params:
            F = model_params['base_filters']
            model_params['filters_1'] = F
            model_params['filters_2'] = 2 * F
            model_params['filters_3'] = 4 * F
            print(f"Expanded base_filters={F} to filters: ({F}, {2*F}, {4*F})")
            # Remove base_filters so it's not passed to model __init__
            del model_params['base_filters']
        
        # Expand base_dropout into scaled rates (0.5*DR, 0.5*DR, DR) if present
        if 'base_dropout' in model_params:
            DR = model_params['base_dropout']
            model_params['dropout_1'] = 0.5 * DR
            model_params['dropout_2'] = 0.5 * DR
            model_params['dropout_3'] = DR
            print(f"Expanded base_dropout={DR:.3f} to dropouts: ({0.5*DR:.3f}, {0.5*DR:.3f}, {DR:.3f})")
            # Remove base_dropout so it's not passed to model __init__
            del model_params['base_dropout']
        
        # Set in_channels in model_params if not already set
        if 'in_channels' not in config['training'].get('model_params', {}):
            if 'model_params' not in config['training']:
                config['training']['model_params'] = {}
            config['training']['model_params']['in_channels'] = in_channels
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        if val_loader:
            print(f"Val samples: {len(val_loader.dataset)}")
        
        # Create trainer
        trainer = get_trainer(config['training']['trainer_type'], config['training'])
        
        # Train model
        print("Starting training...")
        best_metric = trainer.fit(train_loader, val_loader, test_loader)
        
        monitor_key = config['training'].get('early_stopping', {}).get('monitor', 'accuracy')
        print(f"Training completed. Best {monitor_key}: {best_metric:.4f}")

        if test_loader:  
            print("Running inference on test set...")
            test_results = infer(test_loader, trainer=trainer, return_preds=True, compute_metrics=True)
            print(f"Accuracy on the test set: {test_results['accuracy']}")
            print(f"TN on the test set: {test_results['confusion_matrix'][0][0]}")  
            print(f"TP on the test set: {test_results['confusion_matrix'][1][1]}")  
            print(f"FN on the test set: {test_results['confusion_matrix'][1][0]}")
            print(f"TP on the test set: {test_results['confusion_matrix'][1][1]}")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()