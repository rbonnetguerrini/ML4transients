#!/bin/bash

# Usage: ./scripts/data_preparation/submit_collection.sh configs/configs_cutout.yaml [batch_size]

CONFIG_FILE=${1:-"configs/data_preparation/configs_cutout.yaml"}
BATCH_SIZE=${2:-50}

echo "Submitting collection processing job..."
echo "Config: $CONFIG_FILE"
echo "Batch size: $BATCH_SIZE visits per job"

# Setup environment for batch creation
source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2024_30/loadLSST.bash
setup lsst_distrib

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Create batch configs
echo "Creating batch configurations..."
python scripts/data_preparation/create_batch_jobs.py $CONFIG_FILE $BATCH_SIZE

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create batch configs"
    exit 1
fi

# Count number of batches
NUM_BATCHES=$(ls configs/data_preparation/batches/batch_*_$(basename $CONFIG_FILE .yaml).yaml 2>/dev/null | wc -l)

if [ $NUM_BATCHES -eq 0 ]; then
    echo "ERROR: No batch configs created"
    exit 1
fi

echo "Found $NUM_BATCHES batches"

# Submit job array
CONFIG_BASE=$(basename $CONFIG_FILE .yaml)
SLURM_JOB_ID=$(sbatch --array=0-$((NUM_BATCHES-1)) --parsable "$PROJECT_ROOT/scripts/data_preparation/run_batch.sh" $CONFIG_BASE)

echo "Submitted SLURM job array: $SLURM_JOB_ID"
echo "Array indices: 0-$((NUM_BATCHES-1))"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo "  sacct -j $SLURM_JOB_ID"
echo ""
echo "Check logs in: logs/batch_${SLURM_JOB_ID}_*.{out,err}"
echo ""
echo "After all jobs complete, run:"
echo "  python scripts/data_preparation/create_global_index_post_batch.py $CONFIG_FILE"