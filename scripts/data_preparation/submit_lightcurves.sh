#!/bin/bash
#SBATCH --job-name=extract_lightcurves
#SBATCH --time=08:00:00
#SBATCH --partition=hpc
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/data_preparation/lightcurves_%j.out
#SBATCH --error=logs/data_preparation/lightcurves_%j.err

# Usage: sbatch scripts/data_preparation/submit_lightcurves.sh <config_file>

CONFIG_FILE=${1:-"configs/data_preparation/configs_cutout.yaml"}

echo "=========================================="
echo "Lightcurve Extraction Job"
echo "Config: $CONFIG_FILE"
echo "Started: $(date)"
echo "=========================================="

# Setup environment
source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2024_30/loadLSST.bash
setup lsst_distrib

# Use hardcoded project root
PROJECT_ROOT="/sps/lsst/users/rbonnetguerrini/ML4transients"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Create logs directory
mkdir -p logs/data_preparation

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file $CONFIG_FILE not found" >&2
    exit 1
fi

# Run lightcurve extraction
python -u scripts/data_preparation/extract_lightcurves_post_batch.py "$CONFIG_FILE"
exit_code=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
