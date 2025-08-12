import os
import torch
from torch.utils.data import DataLoader
import yaml
import argparse
import sys
from pathlib import Path
import copy

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
    
    # Create splits efficiently
    splits = PytorchDataset.create_splits(
        data_source=data_path,
        visits=visits,
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

    # Precompute dataset splits once
    print("Preparing datasets for Bayesian optimization...")
    splits = PytorchDataset.create_splits(
        data_source=config['data']['path'],
        visits=config['data'].get('visits'),
        test_size=config['data'].get('test_size', 0.2),
        val_size=config['data'].get('val_size', 0.1),
        random_state=config.get('random_state', 42)
    )
    datasets = {'train': splits['train'], 'val': splits['val'], 'test': splits['test']}

    def objective(trial):
        trial_config = copy.deepcopy(config)
        # Override epochs per trial
        trial_config['training']['epochs'] = max_epochs

        # Apply parameter suggestions
        for dotted_key, spec in bs_cfg.get('params', {}).items():
            val = suggest_param(trial, dotted_key, spec)
            set_nested(trial_config['training'], dotted_key, val)

        batch_size = trial_config['training'].get('batch_size', config['training']['batch_size'])
        num_workers = trial_config['training'].get('num_workers', 4)
        train_loader, val_loader, test_loader = create_data_loaders_from_datasets(
            datasets, batch_size=batch_size, num_workers=num_workers
        )

        trainer = get_trainer(trial_config['training']['trainer_type'], trial_config['training'])
        best_metric = trainer.fit(train_loader, val_loader, test_loader)

        # Report for pruning (uses monitored metric; assumed lower is better if minimizing loss)
        if prune:
            trial.report(best_metric, step=trainer.best_epoch if trainer.best_epoch >= 0 else trial.number)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_metric

    study = optuna.create_study(direction=direction)
    print(f"Starting Bayesian optimization for {n_trials} trials (direction={direction})")
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
    parser.add_argument('--experiment-name', type=str, required=True, help='Name of the training')
    parser.add_argument('--hpo', action='store_true', help='Run Bayesian hyperparameter optimization instead of normal training')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
        
        # Override experiment name if specified
        if args.experiment_name:
            config['training']['experiment_name'] = args.experiment_name
            
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
            print(f"TP on the test set: {test_results['confusion_matrix'][0][0]}")
            print(f"FP on the test set: {test_results['confusion_matrix'][0][1]}")
            print(f"FN on the test set: {test_results['confusion_matrix'][1][0]}")
            print(f"TN on the test set: {test_results['confusion_matrix'][1][1]}")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()