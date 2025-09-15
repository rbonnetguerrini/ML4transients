#!/bin/bash
#SBATCH --job-name=transient_training
#SBATCH --output=logs/trainings/training_%j.out
#SBATCH --error=logs/trainings/training_%j.err
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G

# Exit on any error
set -e

# Create logs directory if it doesn't exist
mkdir -p logs/trainings

echo "=== SLURM JOB STARTED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

source /usr/share/Modules/init/bash
module load Programming_Languages/anaconda/3.11
conda activate env_ML

# Get arguments
CONFIG_FILE=${1:-"configs/training/standard_training.yaml"}
EXPERIMENT_NAME=${2:-"training_${SLURM_JOB_ID}"}
USE_HPO=${3:-""}

if [[ "$USE_HPO" == "--hpo" ]]; then
    echo "=== Bayesian Optimization Job ==="
else
    echo "=== Normal Training Job ==="
fi

echo "Config: $CONFIG_FILE"
echo "Experiment: $EXPERIMENT_NAME"
echo "================================="

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Change to project directory
cd /sps/lsst/users/rbonnetguerrini/ML4transients

# Run training
python scripts/training/train_model.py \
    --config "$CONFIG_FILE" \
    --experiment-name "$EXPERIMENT_NAME" \
    $USE_HPO

echo "Training completed successfully on $(hostname)"