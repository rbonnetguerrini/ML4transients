#!/bin/bash

# SLURM job submission script for injection catalog generation using job arrays

# Configuration
CONFIG_FILE="configs/injection/injection_all_bands.yaml"
SCRIPT_PATH="scripts/injection/gen_injection_catalogue.py"
BANDS=("u" "g" "r" "i" "z" "y")

# SLURM parameters
PARTITION="htc"
NODES=1
NTASKS=1
CPUS_PER_TASK=1
MEMORY="8G"
TIME="02:00:00"
OUTPUT_DIR="logs/injection"

# Create output directory for SLURM logs
mkdir -p $OUTPUT_DIR

# Calculate array size (number of bands)
NUM_BANDS=${#BANDS[@]}
ARRAY_RANGE="0-$((NUM_BANDS-1))"

echo "Submitting injection catalog generation job array..."
echo "Config: $CONFIG_FILE"
echo "Bands: ${BANDS[*]}"
echo "Array range: $ARRAY_RANGE"
echo ""

# Submit job array
SLURM_JOB_ID=$(sbatch --array=$ARRAY_RANGE --parsable \
    --job-name="injection_catalog" \
    --partition=$PARTITION \
    --nodes=$NODES \
    --ntasks=$NTASKS \
    --cpus-per-task=$CPUS_PER_TASK \
    --mem=$MEMORY \
    --time=$TIME \
    --output="${OUTPUT_DIR}/injection_%A_%a.out" \
    --error="${OUTPUT_DIR}/injection_%A_%a.err" \
    --export=ALL \
    --wrap="
        # Setup LSST environment
        source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2024_30/loadLSST.bash
        setup lsst_distrib
        
        # Change to project directory
        cd /sps/lsst/users/rbonnetguerrini/ML4transients
        
        # Add src to Python path
        export PYTHONPATH=/sps/lsst/users/rbonnetguerrini/ML4transients/src:\$PYTHONPATH
        
        # Get band from array index
        BANDS=(${BANDS[*]})
        BAND=\${BANDS[\$SLURM_ARRAY_TASK_ID]}
        
        echo \"Processing band: \$BAND (array task \$SLURM_ARRAY_TASK_ID)\"
        python $SCRIPT_PATH --config $CONFIG_FILE --band \$BAND
    ")

echo "Submitted SLURM job array: $SLURM_JOB_ID"
echo "Array indices: $ARRAY_RANGE"
echo ""
echo "Job IDs will be:"
for i in $(seq 0 $((NUM_BANDS-1))); do
    echo "  Band ${BANDS[$i]}: ${SLURM_JOB_ID}_${i}"
done
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo "  sacct -j $SLURM_JOB_ID"
echo ""
echo "Check logs in: ${OUTPUT_DIR}/injection_${SLURM_JOB_ID}_*.{out,err}"