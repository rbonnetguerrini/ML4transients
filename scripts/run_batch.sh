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

# Use hardcoded project root (since we know where we are)
PROJECT_ROOT="/sps/lsst/users/rbonnetguerrini/ML4transients"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Create logs directory
mkdir -p logs

# Get batch config file
BATCH_NUM=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
CONFIG_FILE="configs/batches/batch_${BATCH_NUM}_${CONFIG_BASE}.yaml"

echo "Running batch $BATCH_NUM: $CONFIG_FILE"
echo "Started at: $(date)"

# Check if config exists and run
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file $CONFIG_FILE not found" >&2
    exit 1
fi

# Run the script
python scripts/run_cutout.py "$CONFIG_FILE"
exit $exit_code