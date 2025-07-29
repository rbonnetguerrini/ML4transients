import torch
from torch.utils.data import DataLoader
import yaml
import argparse
import sys
from pathlib import Path

# Add the ML4transients package to path
sys.path.append('/sps/lsst/users/rbonnetguerrini/ML4transients/src')

from ML4transients.training.pytorch_dataset import PytorchDataset

from ML4transients.data_access.dataset_loader import DatasetLoader

from ML4transients.training.trainers import get_trainer

from utils import load_config

def main():

    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')

    config_inference = load_config(args.config)
    config_training = load_config(args.config)

    # Create trainer
    trainer = get_trainer(config['training']['trainer_type'], config['training'])



if __name__ == "__main__":
    main()