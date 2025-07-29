#!/bin/bash
#SBATCH --job-name=transient_ensemble
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G

# Create logs directory if it doesn't exist
mkdir -p logs/trainings

source /usr/share/Modules/init/bash
module load Programming_Languages/anaconda/3.11
conda activate env_ML

# Get a random port for TensorBoard
TB_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Print connection info
echo "=== TensorBoard Connection Info ==="
echo "Node: $(hostname)"
echo "Port: $TB_PORT"
echo "Job ID: $SLURM_JOB_ID"
echo "==================================="

# Start TensorBoard in background
tensorboard --logdir=runs --host=0.0.0.0 --port=$TB_PORT &
TB_PID=$!

# Run training
cd /sps/lsst/users/rbonnetguerrini/ML4transients
python scripts/train_model.py \
    --config configs/standard_training.yaml \
    --experiment-name "gpu_training_${SLURM_JOB_ID}"

echo "Training completed on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"

# Keep TensorBoard running for a while after training
echo "Training completed. TensorBoard still running for 30 minutes..."
sleep 1800

# Kill TensorBoard
kill $TB_PID