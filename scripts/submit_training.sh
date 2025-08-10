#!/bin/bash
#SBATCH --job-name=transient_bayes
#SBATCH --output=logs/trainings/training_%j.out
#SBATCH --error=logs/trainings/training_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G

# Create logs directory if it doesn't exist
mkdir -p logs/trainings

source /usr/share/Modules/init/bash
module load Programming_Languages/anaconda/3.11
conda activate env_ML

# Get arguments
CONFIG_FILE=${1:-"configs/standard_training.yaml"}
EXPERIMENT_NAME=${2:-"bayes_opt_${SLURM_JOB_ID}"}
USE_HPO=${3:-"--hpo"}

echo "=== Bayesian Optimization Job ==="
echo "Config: $CONFIG_FILE"
echo "Experiment: $EXPERIMENT_NAME"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================="

# Run training with Bayesian optimization
cd /sps/lsst/users/rbonnetguerrini/ML4transients
python scripts/train_model.py \
    --config "$CONFIG_FILE" \
    --experiment-name "$EXPERIMENT_NAME" \
    $USE_HPO

echo "Training completed on $(hostname)"