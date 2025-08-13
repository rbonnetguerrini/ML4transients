#!/bin/bash
#SBATCH --job-name=bayes_opt
#SBATCH --output=logs/bayes_opt/bayes_%j.out
#SBATCH --error=logs/bayes_opt/bayes_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Create logs directory
mkdir -p logs/bayes_opt

source /usr/share/Modules/init/bash
module load Programming_Languages/anaconda/3.11
conda activate env_ML

cd /sps/lsst/users/rbonnetguerrini/ML4transients

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: sbatch submit_bayes_opt.sh <config_file> [experiment_name]"
    exit 1
fi

CONFIG_FILE=$1
EXPERIMENT_NAME=${2:-"bayes_opt_${SLURM_JOB_ID}"}

echo "=== Bayesian Optimization Job ==="
echo "Config: $CONFIG_FILE"
echo "Experiment: $EXPERIMENT_NAME"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================="

# Run Bayesian optimization
python scripts/train_model.py \
    --config "$CONFIG_FILE" \
    --experiment-name "$EXPERIMENT_NAME" \
    --hpo

echo "Bayesian optimization completed on $(hostname)"
