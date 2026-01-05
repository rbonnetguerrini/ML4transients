#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --partition=gpu_v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

# Exit on any error
set -e

# Get config file first (before SLURM directives are parsed)
CONFIG_FILE=${1:-"configs/training/standard_training.yaml"}

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract experiment name from YAML config
EXPERIMENT_NAME=$(grep -E '^\s*experiment_name:' "$CONFIG_FILE" | sed -E 's/.*experiment_name:\s*"?([^"]+)"?.*/\1/' | tr -d '"')

# Fallback if experiment_name not found in YAML
if [[ -z "$EXPERIMENT_NAME" ]]; then
    EXPERIMENT_NAME="training_${SLURM_JOB_ID}"
    echo "Warning: No experiment_name found in config, using default: $EXPERIMENT_NAME"
fi

# Set SLURM job name and log files dynamically
export SLURM_JOB_NAME="$EXPERIMENT_NAME"

# Create logs directory if it doesn't exist
mkdir -p logs/trainings/${EXPERIMENT_NAME}

# Redirect output to experiment-specific log files
exec > >(tee "logs/trainings/${EXPERIMENT_NAME}/${SLURM_JOB_ID}.out")
exec 2> >(tee "logs/trainings/${EXPERIMENT_NAME}/${SLURM_JOB_ID}.err" >&2)

echo "=== SLURM JOB STARTED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

source /usr/share/Modules/init/bash
module load Programming_Languages/anaconda/3.11
conda activate env_ML

# Get remaining arguments
USE_HPO=${2:-""}

if [[ "$USE_HPO" == "--hpo" ]]; then
    echo "=== Bayesian Optimization Job ==="
else
    echo "=== Normal Training Job ==="
fi

echo "Config: $CONFIG_FILE"
echo "Experiment: $EXPERIMENT_NAME"
echo "================================="

# Change to project directory
cd /sps/lsst/users/rbonnetguerrini/ML4transients

# Run training
python scripts/training/train_model.py \
    --config "$CONFIG_FILE" \
    $USE_HPO

echo "Training completed successfully on $(hostname)"