#!/bin/bash
#SBATCH --job-name=cutout_batches
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/batch_%A_%a.out
#SBATCH --error=logs/batch_%A_%a.err

# Get config base name from argument
CONFIG_BASE=${1:-"configs_cutout"}

# Setup environment
source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2024_30/loadLSST.bash
setup lsst_distrib

# Project root is one level up from scripts/
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project directory and set PYTHONPATH
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Create logs directory
mkdir -p "logs/$CONFIG_BASE"

# Get batch number (zero-padded)
BATCH_NUM=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
CONFIG_FILE="configs/batches/batch_${BATCH_NUM}_${CONFIG_BASE}.yaml"

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running batch $BATCH_NUM: $CONFIG_FILE"
echo "Started at: $(date)"

if [ -f "$CONFIG_FILE" ]; then
    # Run the cutout processing
    python run_cutout.py $CONFIG_FILE
    exit_code=$?
    
    echo "Finished at: $(date)"
    echo "Exit code: $exit_code"
    
    if [ $exit_code -eq 0 ]; then
        echo "Batch $BATCH_NUM completed successfully"
    else
        echo "Batch $BATCH_NUM failed with exit code $exit_code"
    fi
    
    exit $exit_code
else
    echo "ERROR: Config file $CONFIG_FILE not found"
    exit 1
fi