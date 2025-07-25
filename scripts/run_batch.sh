#!/bin/bash
#SBATCH --job-name=cutout_batches
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/batch_%A_%a.out

# Get config base name from argument
CONFIG_BASE=${1:-"configs_cutout"}

# Setup environment
source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2024_30/loadLSST.bash
setup lsst_distrib

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project directory and set PYTHONPATH
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Create logs directory
mkdir -p "logs"

# Get batch number and files
BATCH_NUM=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
CONFIG_FILE="configs/batches/batch_${BATCH_NUM}_${CONFIG_BASE}.yaml"
ERROR_LOG="logs/batch_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.ERROR"

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Project root: $PROJECT_ROOT"
echo "Running batch $BATCH_NUM: $CONFIG_FILE"
echo "Started at: $(date)"

if [ -f "$CONFIG_FILE" ]; then
    # Use process substitution to capture stderr in real-time
    {
        python run_cutout.py "$CONFIG_FILE" 2> >(
            # This runs in parallel and writes errors immediately
            while IFS= read -r line; do
                echo "$line" >&2  # Show error in terminal
                echo "$(date '+%Y-%m-%d %H:%M:%S'): $line" >> "$ERROR_LOG"  # Log with timestamp
            done
        )
        exit_code=$?
    }
    
    echo "Finished at: $(date)"
    echo "Exit code: $exit_code"
    
    if [ $exit_code -eq 0 ]; then
        echo "Batch $BATCH_NUM completed successfully"
        # Remove error log if it exists but job succeeded (might be warnings only)
        if [ -f "$ERROR_LOG" ] && [ ! -s "$ERROR_LOG" ]; then
            rm -f "$ERROR_LOG"
        fi
    else
        echo "Batch $BATCH_NUM failed with exit code $exit_code"
        if [ -f "$ERROR_LOG" ]; then
            echo "Error details available in: $ERROR_LOG"
        else
            echo "No detailed error output captured"
        fi
    fi
    
    exit $exit_code
else
    echo "ERROR: Config file $CONFIG_FILE not found"
    # Create error log immediately
    {
        echo "$(date '+%Y-%m-%d %H:%M:%S'): ERROR: Config file not found"
        echo "Config file: $CONFIG_FILE"
        echo "Searched in: $PROJECT_ROOT/$CONFIG_FILE"
    } > "$ERROR_LOG"
    echo "Error logged to: $ERROR_LOG"
    exit 1
fi