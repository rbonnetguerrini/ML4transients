#!/bin/bash

# Configuration - Edit these parameters directly in the script
INPUT_DIR="/sps/lsst/groups/transients/HSC/fouchez/raphael/data/UDEEP/lightcurves"
SNR_FILTERED_DIR="${INPUT_DIR}/snr_filtered"
EXTENDEDNESS_FILTERED_DIR="${INPUT_DIR}/extendedness_filtered"
REPO="/sps/lsst/groups/transients/HSC/fouchez/RC2_repo/butler.yaml"
COLLECTION="run/ssp_ud_cosmos/step5_new"

# Filtering parameters
MIN_NIGHTS=4
SNR_THRESHOLD=5.0
BAND="i"
MATCH_RADIUS=1.0

# SLURM parameters
TIME_LIMIT="01:00:00"
MEMORY="8G"
CPUS=2

# Allow command line arguments to override defaults
if [ $# -ge 1 ]; then
    INPUT_DIR="$1"
    shift
fi

echo "Submitting SNN filtering batch jobs..."
echo "Input directory: $INPUT_DIR"
echo "SNR filtered output: $SNR_FILTERED_DIR"
echo "Extendedness filtered output: $EXTENDEDNESS_FILTERED_DIR"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Create output directories
mkdir -p "$SNR_FILTERED_DIR"
mkdir -p "$EXTENDEDNESS_FILTERED_DIR"
mkdir -p "$PROJECT_ROOT/logs/snn_filtering"

# Find all patch files
PATCH_FILES=($(find "$INPUT_DIR" -maxdepth 1 -name "patch_*.h5" -type f | sort))

if [ ${#PATCH_FILES[@]} -eq 0 ]; then
    echo "ERROR: No patch files found in $INPUT_DIR"
    exit 1
fi

NUM_PATCHES=${#PATCH_FILES[@]}
echo "Found $NUM_PATCHES patch files to process"

# Create a temporary script for the SLURM job array
TEMP_SCRIPT=$(mktemp)

cat > "$TEMP_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --job-name=snn_filter
#SBATCH --output=__LOG_DIR__/snn_filter_%A_%a.out
#SBATCH --error=__LOG_DIR__/snn_filter_%A_%a.err
#SBATCH --time=__TIME_LIMIT__
#SBATCH --mem=__MEMORY__
#SBATCH --cpus-per-task=__CPUS__
#SBATCH --partition=hpc

# Load LSST environment
source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2024_30/loadLSST.bash
setup lsst_distrib

# Get parameters from command line
INPUT_DIR="$1"
SNR_FILTERED_DIR="$2"
EXTENDEDNESS_FILTERED_DIR="$3"
REPO="$4"
COLLECTION="$5"
MIN_NIGHTS="$6"
SNR_THRESHOLD="$7"
BAND="$8"
MATCH_RADIUS="$9"
PROJECT_ROOT="${10}"
SCRIPT_DIR="${11}"
shift 11
PATCH_FILES=("$@")

# Get the patch file for this array task
INPUT_FILE="${PATCH_FILES[$SLURM_ARRAY_TASK_ID]}"
FILENAME=$(basename "$INPUT_FILE")
PATCH_NAME="${FILENAME%.h5}"

echo "========================================"
echo "Processing: $FILENAME"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Started: $(date)"
echo "========================================"

# Setup Python path
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Step 1: SNR and Quality Filtering
SNR_OUTPUT="$SNR_FILTERED_DIR/$FILENAME"
echo ""
echo "=== Step 1: SNR and Quality Filtering ==="
echo "Input: $INPUT_FILE"
echo "Output: $SNR_OUTPUT"

python "$SCRIPT_DIR/filter_snr_and_quality.py" \
    --input "$INPUT_FILE" \
    --output "$SNR_OUTPUT" \
    --min-nights "$MIN_NIGHTS" \
    --snr-threshold "$SNR_THRESHOLD"

if [ $? -ne 0 ]; then
    echo "ERROR: SNR filtering failed for $FILENAME"
    exit 1
fi

# Check metadata
SNR_METADATA="${SNR_OUTPUT%.h5}_snr_filter_metadata.json"
if [ -f "$SNR_METADATA" ]; then
    echo "SNR filtering statistics:"
    python -c "import json; d=json.load(open('$SNR_METADATA')); print(f\"  Initial objects: {d.get('initial_objects', 0)}\"); print(f\"  Final objects: {d.get('final_objects', 0)}\"); print(f\"  Discarded (SNR): {d.get('discarded_snr_filter', 0)}\"); print(f\"  Discarded (window): {d.get('discarded_window_filter', 0)}\"); print(f\"  Discarded (min nights): {d.get('discarded_min_nights', 0)}\"); print(f\"  Discarded (negative flux): {d.get('discarded_negative_flux', 0)}\")"
fi

# Check if SNR output file is empty or very small
filesize=$(stat -c%s "$SNR_OUTPUT" 2>/dev/null || stat -f%z "$SNR_OUTPUT" 2>/dev/null)
if [ "$filesize" -lt 2048 ]; then
    echo ""
    echo "=== Step 1 Result: All objects filtered out ==="
    echo "Creating empty marker for Step 2 output"
    
    # Create empty marker in extendedness output
    EXTENDEDNESS_OUTPUT="$EXTENDEDNESS_FILTERED_DIR/$FILENAME"
    touch "$EXTENDEDNESS_OUTPUT"
    
    # Create metadata for tracking
    EXTENDEDNESS_METADATA="${EXTENDEDNESS_OUTPUT%.h5}_filter_metadata.json"
    echo "{\"total_objects\": 0, \"kept_objects\": 0, \"rejected_objects\": 0, \"note\": \"Empty from SNR filtering\"}" > "$EXTENDEDNESS_METADATA"
    
    echo "========================================"
    echo "Completed: $FILENAME (no objects passed Step 1)"
    echo "Finished: $(date)"
    echo "========================================"
    exit 0
fi

# Step 2: Host Galaxy Extendedness Filtering
EXTENDEDNESS_OUTPUT="$EXTENDEDNESS_FILTERED_DIR/$FILENAME"
echo ""
echo "=== Step 2: Host Galaxy Extendedness Filtering ==="
echo "Input: $SNR_OUTPUT"
echo "Output: $EXTENDEDNESS_OUTPUT"

python "$SCRIPT_DIR/filter_host_extendedness.py" \
    --input "$SNR_OUTPUT" \
    --output "$EXTENDEDNESS_OUTPUT" \
    --repo "$REPO" \
    --collection "$COLLECTION" \
    --band "$BAND" \
    --match-radius "$MATCH_RADIUS"

if [ $? -ne 0 ]; then
    echo "ERROR: Extendedness filtering failed for $FILENAME"
    exit 1
fi

# Check metadata
EXTENDEDNESS_METADATA="${EXTENDEDNESS_OUTPUT%.h5}_filter_metadata.json"
if [ -f "$EXTENDEDNESS_METADATA" ]; then
    echo "Extendedness filtering statistics:"
    python -c "import json; d=json.load(open('$EXTENDEDNESS_METADATA')); print(f\"  Input objects: {d.get('total_objects', 0)}\"); print(f\"  Kept objects: {d.get('kept_objects', 0)}\"); print(f\"  Rejected (point hosts): {d.get('rejected_point_host', 0)}\"); print(f\"  Rejected (flux ratio): {d.get('rejected_low_flux_ratio', 0)}\")"
fi

echo "========================================"
echo "Successfully completed: $FILENAME"
echo "Finished: $(date)"
echo "========================================"
exit 0
EOF

# Replace placeholders in template
sed -i "s|__LOG_DIR__|$PROJECT_ROOT/logs/snn_filtering|g" "$TEMP_SCRIPT"
sed -i "s|__TIME_LIMIT__|$TIME_LIMIT|g" "$TEMP_SCRIPT"
sed -i "s|__MEMORY__|$MEMORY|g" "$TEMP_SCRIPT"
sed -i "s|__CPUS__|$CPUS|g" "$TEMP_SCRIPT"

chmod +x "$TEMP_SCRIPT"

# Submit job array
echo ""
echo "Submitting SLURM job array for $NUM_PATCHES patches..."
SLURM_JOB_ID=$(sbatch --array=0-$((NUM_PATCHES-1)) --parsable "$TEMP_SCRIPT" \
    "$INPUT_DIR" "$SNR_FILTERED_DIR" "$EXTENDEDNESS_FILTERED_DIR" \
    "$REPO" "$COLLECTION" "$MIN_NIGHTS" "$SNR_THRESHOLD" "$BAND" "$MATCH_RADIUS" \
    "$PROJECT_ROOT" "$SCRIPT_DIR" "${PATCH_FILES[@]}")

echo "Submitted SLURM job array: $SLURM_JOB_ID"
echo "Array indices: 0-$((NUM_PATCHES-1))"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo "  sacct -j $SLURM_JOB_ID"
echo ""
echo "Check logs in: $PROJECT_ROOT/logs/snn_filtering/snn_filter_${SLURM_JOB_ID}_*.{out,err}"

# Create summary script that runs after all filtering jobs complete
SUMMARY_SCRIPT=$(mktemp)
cat > "$SUMMARY_SCRIPT" << 'SUMMARY_EOF'
#!/bin/bash
#SBATCH --job-name=snn_summary
#SBATCH --dependency=afterany:__JOB_ID__
#SBATCH --output=__LOG_DIR__/snn_summary_%j.out
#SBATCH --error=__LOG_DIR__/snn_summary_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1

echo "========================================"
echo "Filtering Pipeline Summary"
echo "Started: $(date)"
echo "========================================"

SNR_FILTERED_DIR="$1"
EXTENDEDNESS_FILTERED_DIR="$2"
LOG_FILE="${EXTENDEDNESS_FILTERED_DIR}/pipeline_summary.log"

# Initialize counters
TOTAL_OBJECTS=0
SNR_KEPT=0
SNR_FILTER_DISCARDED=0
WINDOW_DISCARDED=0
MINOBS_DISCARDED=0
NEGFLUX_DISCARDED=0
EXTENDEDNESS_KEPT=0
EXTENDEDNESS_REJECTED=0
POINT_HOST_REJECTED=0
FLUX_RATIO_REJECTED=0
FILES_PROCESSED=0

echo "=== Collecting statistics from Step 1 (SNR filtering) ==="
for metadata_file in "$SNR_FILTERED_DIR"/*_snr_filter_metadata.json; do
    if [ -f "$metadata_file" ]; then
        FILES_PROCESSED=$((FILES_PROCESSED + 1))
        initial=$(python -c "import json; d=json.load(open('$metadata_file')); print(d.get('initial_objects', 0))" 2>/dev/null || echo "0")
        final=$(python -c "import json; d=json.load(open('$metadata_file')); print(d.get('final_objects', 0))" 2>/dev/null || echo "0")
        snr_disc=$(python -c "import json; d=json.load(open('$metadata_file')); print(d.get('discarded_snr_filter', 0))" 2>/dev/null || echo "0")
        win_disc=$(python -c "import json; d=json.load(open('$metadata_file')); print(d.get('discarded_window_filter', 0))" 2>/dev/null || echo "0")
        nights_disc=$(python -c "import json; d=json.load(open('$metadata_file')); print(d.get('discarded_min_nights', 0))" 2>/dev/null || echo "0")
        neg_disc=$(python -c "import json; d=json.load(open('$metadata_file')); print(d.get('discarded_negative_flux', 0))" 2>/dev/null || echo "0")
        
        TOTAL_OBJECTS=$((TOTAL_OBJECTS + initial))
        SNR_KEPT=$((SNR_KEPT + final))
        SNR_FILTER_DISCARDED=$((SNR_FILTER_DISCARDED + snr_disc))
        WINDOW_DISCARDED=$((WINDOW_DISCARDED + win_disc))
        MINOBS_DISCARDED=$((MINOBS_DISCARDED + nights_disc))
        NEGFLUX_DISCARDED=$((NEGFLUX_DISCARDED + neg_disc))
    fi
done

echo "=== Collecting statistics from Step 2 (Extendedness filtering) ==="
for metadata_file in "$EXTENDEDNESS_FILTERED_DIR"/*_filter_metadata.json; do
    if [ -f "$metadata_file" ]; then
        kept=$(python -c "import json; d=json.load(open('$metadata_file')); print(d.get('kept_objects', 0))" 2>/dev/null || echo "0")
        rejected=$(python -c "import json; d=json.load(open('$metadata_file')); print(d.get('rejected_objects', 0))" 2>/dev/null || echo "0")
        point_rejected=$(python -c "import json; d=json.load(open('$metadata_file')); print(d.get('rejected_point_host', 0))" 2>/dev/null || echo "0")
        flux_rejected=$(python -c "import json; d=json.load(open('$metadata_file')); print(d.get('rejected_low_flux_ratio', 0))" 2>/dev/null || echo "0")
        
        EXTENDEDNESS_KEPT=$((EXTENDEDNESS_KEPT + kept))
        EXTENDEDNESS_REJECTED=$((EXTENDEDNESS_REJECTED + rejected))
        POINT_HOST_REJECTED=$((POINT_HOST_REJECTED + point_rejected))
        FLUX_RATIO_REJECTED=$((FLUX_RATIO_REJECTED + flux_rejected))
    fi
done

SNR_TOTAL_DISCARDED=$((SNR_FILTER_DISCARDED + WINDOW_DISCARDED + MINOBS_DISCARDED + NEGFLUX_DISCARDED))
TOTAL_REJECTED=$((SNR_TOTAL_DISCARDED + EXTENDEDNESS_REJECTED))

# Write summary to log file
echo "=== Filtering Pipeline Summary ===" > "$LOG_FILE"
echo "Generated: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "Files processed: $FILES_PROCESSED" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "Step 1 - SNR and Quality Filters:" >> "$LOG_FILE"
echo "  Initial objects: $TOTAL_OBJECTS" >> "$LOG_FILE"
echo "  Rejected (SNR < 5.0): $SNR_FILTER_DISCARDED" >> "$LOG_FILE"
echo "  Rejected (time window): $WINDOW_DISCARDED" >> "$LOG_FILE"
echo "  Rejected (min nights): $MINOBS_DISCARDED" >> "$LOG_FILE"
echo "  Rejected (negative avg flux): $NEGFLUX_DISCARDED" >> "$LOG_FILE"
echo "  Total rejected: $SNR_TOTAL_DISCARDED" >> "$LOG_FILE"
echo "  After Step 1: $SNR_KEPT" >> "$LOG_FILE"
if [ $TOTAL_OBJECTS -gt 0 ]; then
    STEP1_KEEP_PCT=$(python -c "print(f'{100.0 * $SNR_KEPT / $TOTAL_OBJECTS:.2f}')")
    echo "  Keep rate: ${STEP1_KEEP_PCT}%" >> "$LOG_FILE"
fi
echo "" >> "$LOG_FILE"
echo "Step 2 - Host Galaxy Extendedness Filter:" >> "$LOG_FILE"
echo "  Input from Step 1: $SNR_KEPT" >> "$LOG_FILE"
echo "  Rejected (point source hosts): $POINT_HOST_REJECTED" >> "$LOG_FILE"
echo "  Rejected (low flux ratio): $FLUX_RATIO_REJECTED" >> "$LOG_FILE"
echo "  Total rejected: $EXTENDEDNESS_REJECTED" >> "$LOG_FILE"
echo "  After Step 2: $EXTENDEDNESS_KEPT" >> "$LOG_FILE"
if [ $SNR_KEPT -gt 0 ]; then
    STEP2_REJECT_PCT=$(python -c "print(f'{100.0 * $EXTENDEDNESS_REJECTED / $SNR_KEPT:.2f}')")
    echo "  Rejection rate: ${STEP2_REJECT_PCT}%" >> "$LOG_FILE"
fi
echo "" >> "$LOG_FILE"
echo "Overall Statistics:" >> "$LOG_FILE"
echo "  Total input objects: $TOTAL_OBJECTS" >> "$LOG_FILE"
echo "  Total rejected: $TOTAL_REJECTED" >> "$LOG_FILE"
echo "     SNR filter: $SNR_FILTER_DISCARDED" >> "$LOG_FILE"
echo "     Time window: $WINDOW_DISCARDED" >> "$LOG_FILE"
echo "     Min nights: $MINOBS_DISCARDED" >> "$LOG_FILE"
echo "     Negative flux: $NEGFLUX_DISCARDED" >> "$LOG_FILE"
echo "     Point source hosts: $POINT_HOST_REJECTED" >> "$LOG_FILE"
echo "     Low flux ratio: $FLUX_RATIO_REJECTED" >> "$LOG_FILE"
echo "  Final filtered objects: $EXTENDEDNESS_KEPT" >> "$LOG_FILE"
if [ $TOTAL_OBJECTS -gt 0 ]; then
    OVERALL_KEEP_PCT=$(python -c "print(f'{100.0 * $EXTENDEDNESS_KEPT / $TOTAL_OBJECTS:.2f}')")
    echo "  Overall keep rate: ${OVERALL_KEEP_PCT}%" >> "$LOG_FILE"
fi
echo "" >> "$LOG_FILE"
echo "Output directories:" >> "$LOG_FILE"
echo "  SNR filtered: $SNR_FILTERED_DIR" >> "$LOG_FILE"
echo "  Extendedness filtered: $EXTENDEDNESS_FILTERED_DIR" >> "$LOG_FILE"

echo ""
echo "========================================"
echo "Summary Statistics:"
cat "$LOG_FILE"
echo "========================================"
echo "Summary saved to: $LOG_FILE"
echo "Finished: $(date)"
SUMMARY_EOF

# Replace placeholders
sed -i "s|__JOB_ID__|$SLURM_JOB_ID|g" "$SUMMARY_SCRIPT"
sed -i "s|__LOG_DIR__|$PROJECT_ROOT/logs/snn_filtering|g" "$SUMMARY_SCRIPT"
chmod +x "$SUMMARY_SCRIPT"

# Submit summary job
SUMMARY_JOB_ID=$(sbatch --parsable "$SUMMARY_SCRIPT" "$SNR_FILTERED_DIR" "$EXTENDEDNESS_FILTERED_DIR")

echo ""
echo "Summary job submitted: $SUMMARY_JOB_ID (will run after all filtering jobs complete)"
echo "Summary will be saved to: $EXTENDEDNESS_FILTERED_DIR/pipeline_summary.log"

# Clean up temporary scripts after delay
(sleep 120 && rm -f "$TEMP_SCRIPT" "$SUMMARY_SCRIPT") &

echo ""
echo "Batch submission complete!"
