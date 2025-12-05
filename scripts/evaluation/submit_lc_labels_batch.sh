#!/bin/bash
conda activate env_ML
# Configuration - Edit these parameters directly in the script
DATA_PATH="/sps/lsst/groups/transients/HSC/fouchez/raphael/data/UDEEP_coadd_v2"
WEIGHTS_PATH="/sps/lsst/groups/transients/HSC/fouchez/raphael/training/ensemble_50_multichannel_v2"
RUN_NAME="UDEEP_rand_LC"
OUTPUT_PATH="../../saved/lc_labels/$RUN_NAME"

# List of diaObjectIds to process - Edit this list
DIA_OBJECT_IDS=(3496020365816102913
3496020365816102914
3496020365816102915
3496020365816102916
3496020365816102917
3496020365816102918
3496020365816102919
3496020365816102922
3496020365816102923
3496020365816102925
3496020365816102927
3496020365816102930
3496020365816102932
3496020365816102933
3496020365816102934
3496020365816102935
3496020365816102936
3496020365816102938
3496020365816102939
3496020365816102942
3496020365816102943
3496020365816102945
3496020365816102946
3496020365816102947
3496020365816102948
3496020365816102950
3496020365816102952
)

# Allow command line arguments to override the defaults
if [ $# -ge 3 ]; then
    DATA_PATH="$1"
    WEIGHTS_PATH="$2"
    OUTPUT_PATH="$3"
    shift 3
    if [ $# -gt 0 ]; then
        DIA_OBJECT_IDS=("$@")
    fi
fi

echo "Submitting lc_labels visualization batch job..."
echo "Data path: $DATA_PATH"
echo "Weights path: $WEIGHTS_PATH"
echo "Output path: $OUTPUT_PATH"
echo "DiaObjectIds: ${DIA_OBJECT_IDS[@]}"

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Convert OUTPUT_PATH to absolute path if it's relative
if [[ "$OUTPUT_PATH" == ../* ]]; then
    OUTPUT_PATH="$PROJECT_ROOT/$(echo "$OUTPUT_PATH" | sed 's|^../../||')"
fi

# Create logs and output directories
mkdir -p "$PROJECT_ROOT/logs/lc_labels/$RUN_NAME" "$OUTPUT_PATH"

# Create a temporary script for the SLURM job
TEMP_SCRIPT=$(mktemp)

# Create comma-separated list of all IDs for navigation
ALL_IDS_STR=$(IFS=,; echo "${DIA_OBJECT_IDS[*]}")

cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=lc_labels_viz
#SBATCH --output=$PROJECT_ROOT/logs/lc_labels/$RUN_NAME/lc_labels_%A_%a.out
#SBATCH --error=$PROJECT_ROOT/logs/lc_labels/$RUN_NAME/lc_labels_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

DATA_PATH="\$1"
WEIGHTS_PATH="\$2"  
OUTPUT_PATH="\$3"
PROJECT_ROOT="\$4"
ALL_IDS="\$5"
shift 5
DIA_OBJECT_IDS=("\$@")

# Get the diaObjectId for this array task
DIA_OBJECT_ID=\${DIA_OBJECT_IDS[\$SLURM_ARRAY_TASK_ID]}

echo "Processing diaObjectId: \$DIA_OBJECT_ID"

# Setup Python path
export PYTHONPATH="\$PROJECT_ROOT/src:\$PYTHONPATH"

# Run the lc_labels visualization with navigation support
python "\$PROJECT_ROOT/src/ML4transients/evaluation/lc_labels.py" \\
    "\$DIA_OBJECT_ID" \\
    --data-path "\$DATA_PATH" \\
    --weights-path "\$WEIGHTS_PATH" \\
    --output "\$OUTPUT_PATH/lc_labels_\${DIA_OBJECT_ID}.html" \\
    --all-ids "\$ALL_IDS"

exit_code=\$?
echo "Exit code: \$exit_code for diaObjectId: \$DIA_OBJECT_ID"
exit \$exit_code
EOF

chmod +x "$TEMP_SCRIPT"

# Submit job array
NUM_IDS=${#DIA_OBJECT_IDS[@]}
SLURM_JOB_ID=$(sbatch --array=0-$((NUM_IDS-1)) --parsable "$TEMP_SCRIPT" "$DATA_PATH" "$WEIGHTS_PATH" "$OUTPUT_PATH" "$PROJECT_ROOT" "$ALL_IDS_STR" "${DIA_OBJECT_IDS[@]}")

echo "Submitted SLURM job array: $SLURM_JOB_ID"
echo "Array indices: 0-$((NUM_IDS-1))"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo "  sacct -j $SLURM_JOB_ID"
echo ""
echo "Check logs in: $PROJECT_ROOT/logs/lc_labels/$RUN_NAME/lc_labels_${SLURM_JOB_ID}_*.{out,err}"

# Submit a follow-up job to create index after all lc_labels jobs complete
INDEX_SCRIPT=$(mktemp)
cat > "$INDEX_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --job-name=lc_labels_index
#SBATCH --dependency=afterany:__JOB_ID__
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

OUTPUT_PATH="$1"
SCRIPT_DIR="$2"

sleep 10
python "$SCRIPT_DIR/create_lightcurve_index.py" "$OUTPUT_PATH"
EOF

sed -i "s/__JOB_ID__/$SLURM_JOB_ID/g" "$INDEX_SCRIPT"
chmod +x "$INDEX_SCRIPT"

INDEX_JOB_ID=$(sbatch --parsable "$INDEX_SCRIPT" "$OUTPUT_PATH" "$SCRIPT_DIR")

echo "Index creation job submitted: $INDEX_JOB_ID (will run after lc_labels jobs complete)"

(sleep 60 && rm -f "$TEMP_SCRIPT" "$INDEX_SCRIPT") &
