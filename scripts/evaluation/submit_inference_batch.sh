#!/bin/bash
#./scripts/evaluation/submit_inference_batch.sh     --data-path /sps/lsst/groups/transients/HSC/fouchez/raphael/data/UDEEP_coadd_v2     --weights-path /sps/lsst/groups/transients/HSC/fouchez/raphael/training/ensemble_50_multichannel_v2     --visits-per-job 10
DATA_PATH="/sps/lsst/groups/transients/HSC/fouchez/raphael/data/rc2_coadd_v2"
WEIGHTS_PATH="/sps/lsst/groups/transients/HSC/fouchez/raphael/training/coadd_coteaching_optimized"
VISITS_PER_JOB=10  # Number of visits each job should process
NUM_JOBS=""  # Number of parallel jobs to submit (if empty, will create enough to cover all visits)
FORCE_RERUN=false  # Set to true to force re-running inference on visits with existing results

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --weights-path)
            WEIGHTS_PATH="$2"
            shift 2
            ;;
        --visits-per-job)
            VISITS_PER_JOB="$2"
            shift 2
            ;;
        --num-jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --force)
            FORCE_RERUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--data-path PATH] [--weights-path PATH] [--visits-per-job N] [--num-jobs N] [--force]"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$DATA_PATH" ] || [ -z "$WEIGHTS_PATH" ]; then
    echo "Error: DATA_PATH and WEIGHTS_PATH are required"
    echo "Usage: $0 --data-path PATH --weights-path PATH [--visits-per-job N] [--num-jobs N] [--force]"
    exit 1
fi

echo "Submitting inference batch job..."
echo "Data path: $DATA_PATH"
echo "Weights path: $WEIGHTS_PATH"
echo "Visits per job: $VISITS_PER_JOB"
echo "Number of jobs: ${NUM_JOBS:-auto (based on total visits)}"
echo "Force rerun: $FORCE_RERUN"

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Extract run name from weights path for organizing logs
RUN_NAME=$(basename "$WEIGHTS_PATH")

# Create logs directory
LOGS_DIR="$PROJECT_ROOT/logs/inference/$RUN_NAME"
mkdir -p "$LOGS_DIR"

# Create a temporary Python script to discover visits
DISCOVER_SCRIPT=$(mktemp)
cat > "$DISCOVER_SCRIPT" << 'EOFPYTHON'
import sys
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append('/sps/lsst/users/rbonnetguerrini/ML4transients/src')

# Redirect both stdout and stderr during import to suppress all messages
import io

old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

try:
    from ML4transients.data_access.dataset_loader import DatasetLoader
    
    data_path = sys.argv[1]
    
    loader = DatasetLoader(data_path)
    visits = loader.visits
    
finally:
    # Restore stdout and stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr

# Print visits one per line (only output)
for visit in visits:
    print(visit)
EOFPYTHON

# Discover ALL visits using Python
ALL_VISITS=($(python "$DISCOVER_SCRIPT" "$DATA_PATH" 2>/dev/null))
rm -f "$DISCOVER_SCRIPT"

if [ ${#ALL_VISITS[@]} -eq 0 ]; then
    echo "Error: No visits found in dataset"
    exit 1
fi

TOTAL_VISITS=${#ALL_VISITS[@]}
echo "Found $TOTAL_VISITS total visits in dataset"

# Calculate number of jobs needed
if [ -z "$NUM_JOBS" ]; then
    NUM_JOBS=$(( (TOTAL_VISITS + VISITS_PER_JOB - 1) / VISITS_PER_JOB ))
    echo "Auto-calculated: $NUM_JOBS jobs needed to process all visits ($VISITS_PER_JOB visits per job)"
fi

echo "Submitting $NUM_JOBS jobs, each processing up to $VISITS_PER_JOB visits"

# Create the SLURM job script
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=__LOGS_DIR__/inference_%A_%a.out
#SBATCH --error=__LOGS_DIR__/inference_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1

# Get arguments
DATA_PATH="$1"
WEIGHTS_PATH="$2"
PROJECT_ROOT="$3"
FORCE_RERUN="$4"
VISITS_PER_JOB="$5"
TOTAL_VISITS="$6"
shift 6
ALL_VISITS=("$@")

# Calculate which visits this job should process
START_IDX=$((SLURM_ARRAY_TASK_ID * VISITS_PER_JOB))
END_IDX=$((START_IDX + VISITS_PER_JOB))
if [ $END_IDX -gt $TOTAL_VISITS ]; then
    END_IDX=$TOTAL_VISITS
fi

# Extract visits for this job
MY_VISITS=("${ALL_VISITS[@]:$START_IDX:$VISITS_PER_JOB}")

echo "=========================================="
echo "Job array task ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Processing visits $START_IDX to $((END_IDX-1)) (${#MY_VISITS[@]} visits)"
echo "Visits: ${MY_VISITS[@]}"
echo "=========================================="

# Setup Python environment
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Activate conda environment
conda activate env_ML

# Convert visit array to comma-separated string
VISIT_LIST=$(IFS=,; echo "${MY_VISITS[*]}")

echo ""
echo "Running inference on ${#MY_VISITS[@]} visits: $VISIT_LIST"

# Build the command to run inference on these visits
CMD="python -u $PROJECT_ROOT/scripts/evaluation/run_inference.py \
    --dataset-path \"$DATA_PATH\" \
    --weights-path \"$WEIGHTS_PATH\" \
    --visits \"$VISIT_LIST\""

if [ "$FORCE_RERUN" = "true" ]; then
    CMD="$CMD --force"
fi

echo "Running: $CMD"
echo ""

# Run inference
eval $CMD

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "Successfully completed inference for ${#MY_VISITS[@]} visits"
    exit 0
else
    echo ""
    echo "Failed inference (exit code: $exit_code)"
    exit 1
fi
EOF

# Replace placeholders
sed -i "s|__LOGS_DIR__|$LOGS_DIR|g" "$TEMP_SCRIPT"

chmod +x "$TEMP_SCRIPT"

# Submit job array
FORCE_ARG=$([ "$FORCE_RERUN" = true ] && echo "true" || echo "false")

SLURM_JOB_ID=$(sbatch --array=0-$((NUM_JOBS-1)) --parsable "$TEMP_SCRIPT" \
    "$DATA_PATH" "$WEIGHTS_PATH" "$PROJECT_ROOT" "$FORCE_ARG" "$VISITS_PER_JOB" "$TOTAL_VISITS" "${ALL_VISITS[@]}")

echo ""
echo "=========================================="
echo "Submitted SLURM job array: $SLURM_JOB_ID"
echo "Array indices: 0-$((NUM_JOBS-1))"
echo "Total jobs: $NUM_JOBS"
echo "Total visits: $TOTAL_VISITS"
echo "Visits per job: $VISITS_PER_JOB"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo "  sacct -j $SLURM_JOB_ID"
echo ""
echo "Check logs in: $LOGS_DIR/"
echo ""

# Cleanup temporary script after a delay
(sleep 120 && rm -f "$TEMP_SCRIPT") &
