#!/bin/bash
conda activate env_ML
# Configuration - Edit these parameters directly in the script
DATA_PATH="/sps/lsst/groups/transients/HSC/fouchez/raphael/data/UDEEP_norm_high_conf_smallv4"
WEIGHTS_PATH="/sps/lsst/groups/transients/HSC/fouchez/raphael/training/coteaching_optimized"
RUN_NAME="highconfv4"
OUTPUT_PATH="../../saved/lc/$RUN_NAME"

# List of diaObjectIds to process - Edit this list
DIA_OBJECT_IDS=(
    3495848842002204206
3496046754095217818
3495901618560353722
3495963191211497507
3495848842002204206
3496002773630105195
3496011569723114382
3495853240048707626
3496007171676607251
3495910414653359589
3496024763862652160
3496007171676607177
3495967589258005494
3496033559955712391
3495985181444047538
3495853240048704850
3496007171676616368
3495949997071971839
3495971987304539225
3495848842002209632
3496099530653305638
3495976385351016718
3496099530653325498
3496011569723113981
3495967589257999199
3496051152141716680
3495853240048704323
3495963191211497851
3495941200978940809
3496064346281240533
3496029161909182824
3496086336513796240
3495897220513822327
3495985181444047294
3495910414653355215
3496051152141716619
3495906016606839094
3496020365816150213
3495892822467318088
3495906016606838787
3495910414653356246

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

echo "Submitting lightcurve visualization batch job..."
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
mkdir -p "$PROJECT_ROOT/logs/lc/$RUN_NAME" "$OUTPUT_PATH"

# Create a temporary script for the SLURM job
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=lightcurve_viz
#SBATCH --output=$PROJECT_ROOT/logs/lc/$RUN_NAME/lightcurve_%A_%a.out
#SBATCH --error=$PROJECT_ROOT/logs/lc/$RUN_NAME/lightcurve_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

DATA_PATH="\$1"
WEIGHTS_PATH="\$2"  
OUTPUT_PATH="\$3"
PROJECT_ROOT="\$4"
shift 4
DIA_OBJECT_IDS=("\$@")

# Get the diaObjectId for this array task
DIA_OBJECT_ID=\${DIA_OBJECT_IDS[\$SLURM_ARRAY_TASK_ID]}

echo "Processing diaObjectId: \$DIA_OBJECT_ID"

# Setup Python path
export PYTHONPATH="\$PROJECT_ROOT/src:\$PYTHONPATH"

# Run the lightcurve visualization
python "\$PROJECT_ROOT/src/ML4transients/evaluation/lightcurve_visualization.py" \\
    "\$DIA_OBJECT_ID" \\
    --data-path "\$DATA_PATH" \\
    --weights-path "\$WEIGHTS_PATH" \\
    --output "\$OUTPUT_PATH/lightcurve_\${DIA_OBJECT_ID}.html"

exit_code=\$?
echo "Exit code: \$exit_code for diaObjectId: \$DIA_OBJECT_ID"
exit \$exit_code
EOF

chmod +x "$TEMP_SCRIPT"

# Submit job array
NUM_IDS=${#DIA_OBJECT_IDS[@]}
SLURM_JOB_ID=$(sbatch --array=0-$((NUM_IDS-1)) --parsable "$TEMP_SCRIPT" "$DATA_PATH" "$WEIGHTS_PATH" "$OUTPUT_PATH" "$PROJECT_ROOT" "${DIA_OBJECT_IDS[@]}")

echo "Submitted SLURM job array: $SLURM_JOB_ID"
echo "Array indices: 0-$((NUM_IDS-1))"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo "  sacct -j $SLURM_JOB_ID"
echo ""
echo "Check logs in: $PROJECT_ROOT/logs/lc/$RUN_NAME/lightcurve_${SLURM_JOB_ID}_*.{out,err}"

# Submit a follow-up job to create index after all lightcurve jobs complete
INDEX_SCRIPT=$(mktemp)
cat > "$INDEX_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --job-name=lightcurve_index
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

echo "Index creation job submitted: $INDEX_JOB_ID (will run after lightcurve jobs complete)"

(sleep 60 && rm -f "$TEMP_SCRIPT" "$INDEX_SCRIPT") &