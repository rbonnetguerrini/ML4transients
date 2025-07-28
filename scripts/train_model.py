import torch
from torch.utils.data import DataLoader
import yaml
import argparse
import sys
from pathlib import Path

# Add the ML4transients package to path
sys.path.append('/sps/lsst/users/rbonnetguerrini/ML4transients/src')

from ML4transients.training.pytorch_dataset import PytorchDataset
from ML4transients.training.trainers import get_trainer
from ML4transients.data_access.dataset_loader import DatasetLoader

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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

def main():
    parser = argparse.ArgumentParser(description='Train transient detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--trainer', type=str, choices=['standard', 'coteaching', 'ensemble'], 
                       default='standard', help='Type of trainer to use')
    parser.add_argument('--experiment-name', type=str, help='Experiment name for TensorBoard')
    parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override trainer type if specified
    if args.trainer:
        config['training']['trainer_type'] = args.trainer
    
    # Override experiment name if specified
    if args.experiment_name:
        config['training']['experiment_name'] = args.experiment_name
    
    # Disable TensorBoard if requested
    if args.no_tensorboard:
        config['training']['use_tensorboard'] = False
    
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
    best_acc = trainer.fit(train_loader, test_loader, val_loader)
    
    print(f"Training completed. Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()