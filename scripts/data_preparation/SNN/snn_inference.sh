#!/bin/bash

# --- Step 1: LSST environment for host extendedness filtering ---
source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2024_30/loadLSST.bash
setup lsst_distrib

export PYTHONPATH=/sps/lsst/users/rbonnetguerrini/ML4transients/src:$PYTHONPATH

INPUT_DIR="/sps/lsst/groups/transients/HSC/fouchez/raphael/data/UDEEP_coadd/lightcurves"
SNR_FILTERED_DIR="${INPUT_DIR}/snr_filtered"
EXTENDEDNESS_FILTERED_DIR="${INPUT_DIR}/extendedness_filtered"
REPO="/sps/lsst/groups/transients/HSC/fouchez/RC2_repo/butler.yaml"
COLLECTION="run/ssp_ud_cosmos/step5_new"
LOG_FILE="${EXTENDEDNESS_FILTERED_DIR}/pipeline_summary.log"

mkdir -p "$SNR_FILTERED_DIR"
mkdir -p "$EXTENDEDNESS_FILTERED_DIR"

# Initialize log file
echo "=== Lightcurve Filtering Pipeline Summary ===" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

echo "=== Step 1: SNR and Quality Filtering ==="
echo "=== Step 1: SNR and Quality Filtering ===" >> "$LOG_FILE"
TOTAL_OBJECTS=0
SNR_KEPT=0
SNR_TOTAL_DISCARDED=0
SNR_FILTER_DISCARDED=0
WINDOW_DISCARDED=0
MINOBS_DISCARDED=0
NEGFLUX_DISCARDED=0
FILES_PROCESSED=0

# Check if SNR filtering has already been completed
SNR_FILES_EXIST=$(find "$SNR_FILTERED_DIR" -name "patch_*.h5" 2>/dev/null | wc -l)
if [ "$SNR_FILES_EXIST" -gt 0 ]; then
    echo "Step 1 already completed - found $SNR_FILES_EXIST filtered files in $SNR_FILTERED_DIR"
    echo "Skipping SNR filtering and reading existing metadata..."
    echo "Step 1 skipped - using existing filtered files" >> "$LOG_FILE"
    
    # Read metadata from existing files to populate statistics
    for metadata_file in "$SNR_FILTERED_DIR"/*_snr_filter_metadata.json; do
        if [ -f "$metadata_file" ]; then
            FILES_PROCESSED=$((FILES_PROCESSED + 1))
            initial=$(python -c "import json; print(json.load(open('$metadata_file'))['initial_objects'])")
            final=$(python -c "import json; print(json.load(open('$metadata_file'))['final_objects'])")
            snr_disc=$(python -c "import json; print(json.load(open('$metadata_file'))['discarded_snr_filter'])")
            win_disc=$(python -c "import json; print(json.load(open('$metadata_file'))['discarded_window_filter'])")
            obs_disc=$(python -c "import json; print(json.load(open('$metadata_file'))['discarded_minobs_filter'])")
            neg_disc=$(python -c "import json; print(json.load(open('$metadata_file'))['discarded_negative_flux'])")
            
            TOTAL_OBJECTS=$((TOTAL_OBJECTS + initial))
            SNR_KEPT=$((SNR_KEPT + final))
            SNR_FILTER_DISCARDED=$((SNR_FILTER_DISCARDED + snr_disc))
            WINDOW_DISCARDED=$((WINDOW_DISCARDED + win_disc))
            MINOBS_DISCARDED=$((MINOBS_DISCARDED + obs_disc))
            NEGFLUX_DISCARDED=$((NEGFLUX_DISCARDED + neg_disc))
        fi
    done
else
    # Process each lightcurve file with SNR filter
    for input_file in "$INPUT_DIR"/patch_*.h5; do
        if [ -f "$input_file" ]; then
            filename=$(basename "$input_file")
            output_file="$SNR_FILTERED_DIR/$filename"
            
            echo "Processing $filename..."
            python /sps/lsst/users/rbonnetguerrini/ML4transients/scripts/data_preparation/SNN/filter_snr_and_quality.py \
                --input "$input_file" \
                --output "$output_file" \
                --min-obs 10 \
                --snr-threshold 5.0 \
                --high-snr-threshold 3.0
            
            if [ $? -ne 0 ]; then
                echo "SNR filtering failed for $filename."
                echo "ERROR: SNR filtering failed for $filename" >> "$LOG_FILE"
                exit 1
            fi
            
            # Read metadata and accumulate totals
            metadata_file="${output_file%.h5}_snr_filter_metadata.json"
            if [ -f "$metadata_file" ]; then
                FILES_PROCESSED=$((FILES_PROCESSED + 1))
                initial=$(python -c "import json; print(json.load(open('$metadata_file'))['initial_objects'])")
                final=$(python -c "import json; print(json.load(open('$metadata_file'))['final_objects'])")
                snr_disc=$(python -c "import json; print(json.load(open('$metadata_file'))['discarded_snr_filter'])")
                win_disc=$(python -c "import json; print(json.load(open('$metadata_file'))['discarded_window_filter'])")
                obs_disc=$(python -c "import json; print(json.load(open('$metadata_file'))['discarded_minobs_filter'])")
                neg_disc=$(python -c "import json; print(json.load(open('$metadata_file'))['discarded_negative_flux'])")
                
                TOTAL_OBJECTS=$((TOTAL_OBJECTS + initial))
                SNR_KEPT=$((SNR_KEPT + final))
                SNR_FILTER_DISCARDED=$((SNR_FILTER_DISCARDED + snr_disc))
                WINDOW_DISCARDED=$((WINDOW_DISCARDED + win_disc))
                MINOBS_DISCARDED=$((MINOBS_DISCARDED + obs_disc))
                NEGFLUX_DISCARDED=$((NEGFLUX_DISCARDED + neg_disc))
            fi
        fi
    done
fi

SNR_TOTAL_DISCARDED=$((SNR_FILTER_DISCARDED + WINDOW_DISCARDED + MINOBS_DISCARDED + NEGFLUX_DISCARDED))

# Write Step 1 summary to log
echo "Files processed: $FILES_PROCESSED" >> "$LOG_FILE"
echo "Total diaObjects: $TOTAL_OBJECTS" >> "$LOG_FILE"
echo "Discarded (SNR < 5.0): $SNR_FILTER_DISCARDED" >> "$LOG_FILE"
echo "Discarded (time window): $WINDOW_DISCARDED" >> "$LOG_FILE"
echo "Discarded (min observations): $MINOBS_DISCARDED" >> "$LOG_FILE"
echo "Discarded (negative avg flux): $NEGFLUX_DISCARDED" >> "$LOG_FILE"
echo "Total discarded in Step 1: $SNR_TOTAL_DISCARDED" >> "$LOG_FILE"
echo "After Step 1: $SNR_KEPT" >> "$LOG_FILE"
if [ $TOTAL_OBJECTS -gt 0 ]; then
    STEP1_KEEP_PCT=$(python -c "print(f'{100.0 * $SNR_KEPT / $TOTAL_OBJECTS:.2f}')")
    echo "Keep rate: ${STEP1_KEEP_PCT}%" >> "$LOG_FILE"
fi
echo "" >> "$LOG_FILE"

echo "=== Step 2: Host Galaxy Extendedness Filtering ==="
echo "=== Step 2: Host Galaxy Extendedness Filtering ===" >> "$LOG_FILE"
EXTENDEDNESS_KEPT=0
EXTENDEDNESS_REJECTED=0
POINT_HOST_REJECTED=0
FLUX_RATIO_REJECTED=0
FILES_PROCESSED=0

# Process each SNR-filtered file with extendedness filter
for input_file in "$SNR_FILTERED_DIR"/patch_*.h5; do
    if [ -f "$input_file" ]; then
        # Skip empty marker files (size < 2KB means likely empty)
        filesize=$(stat -c%s "$input_file" 2>/dev/null || stat -f%z "$input_file" 2>/dev/null)
        if [ "$filesize" -lt 2048 ]; then
            filename=$(basename "$input_file")
            echo "Skipping $filename - empty file (all objects filtered in Step 1)"
            # Create corresponding empty marker in output
            output_file="$EXTENDEDNESS_FILTERED_DIR/$filename"
            touch "$output_file"
            # Create metadata for tracking
            metadata_file="${output_file%.h5}_filter_metadata.json"
            echo "{\"total_objects\": 0, \"kept_objects\": 0, \"rejected_objects\": 0, \"note\": \"Empty from previous step\"}" > "$metadata_file"
            continue
        fi
        
        filename=$(basename "$input_file")
        output_file="$EXTENDEDNESS_FILTERED_DIR/$filename"
        
        echo "Processing $filename..."
        python /sps/lsst/users/rbonnetguerrini/ML4transients/scripts/data_preparation/SNN/filter_host_extendedness.py \
            --input "$input_file" \
            --output "$output_file" \
            --repo "$REPO" \
            --collection "$COLLECTION" \
            --band i \
            --match-radius 1.0
        
        if [ $? -ne 0 ]; then
            echo "Host extendedness filtering failed for $filename."
            echo "ERROR: Extendedness filtering failed for $filename" >> "$LOG_FILE"
            exit 1
        fi
        
        # Read metadata and accumulate totals
        metadata_file="${output_file%.h5}_filter_metadata.json"
        if [ -f "$metadata_file" ]; then
            FILES_PROCESSED=$((FILES_PROCESSED + 1))
            kept=$(python -c "import json; print(json.load(open('$metadata_file'))['kept_objects'])")
            rejected=$(python -c "import json; print(json.load(open('$metadata_file'))['rejected_objects'])")
            point_rejected=$(python -c "import json; print(json.load(open('$metadata_file')).get('rejected_point_host', 0))")
            flux_rejected=$(python -c "import json; print(json.load(open('$metadata_file')).get('rejected_low_flux_ratio', 0))")
            EXTENDEDNESS_KEPT=$((EXTENDEDNESS_KEPT + kept))
            EXTENDEDNESS_REJECTED=$((EXTENDEDNESS_REJECTED + rejected))
            POINT_HOST_REJECTED=$((POINT_HOST_REJECTED + point_rejected))
            FLUX_RATIO_REJECTED=$((FLUX_RATIO_REJECTED + flux_rejected))
        fi
    fi
done

# Write Step 2 summary to log
echo "Files processed: $FILES_PROCESSED" >> "$LOG_FILE"
echo "Input from Step 1: $SNR_KEPT" >> "$LOG_FILE"
echo "Rejected (point source hosts): $POINT_HOST_REJECTED" >> "$LOG_FILE"
echo "Rejected (low flux ratio): $FLUX_RATIO_REJECTED" >> "$LOG_FILE"
echo "Total rejected in Step 2: $EXTENDEDNESS_REJECTED" >> "$LOG_FILE"
echo "After Step 2: $EXTENDEDNESS_KEPT" >> "$LOG_FILE"
if [ $SNR_KEPT -gt 0 ]; then
    STEP2_REJECT_PCT=$(python -c "print(f'{100.0 * $EXTENDEDNESS_REJECTED / $SNR_KEPT:.2f}')")
    echo "Rejection rate: ${STEP2_REJECT_PCT}%" >> "$LOG_FILE"
fi
echo "" >> "$LOG_FILE"

# Write completion summary
echo "=== Pipeline Completion ===" >> "$LOG_FILE"
echo "Completed: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "=== Complete Filtering Pipeline Summary ===" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "Step 1 - SNR and Quality Filters:" >> "$LOG_FILE"
echo "  Initial objects: $TOTAL_OBJECTS" >> "$LOG_FILE"
echo "  Rejected (SNR < 5.0): $SNR_FILTER_DISCARDED" >> "$LOG_FILE"
echo "  Rejected (time window): $WINDOW_DISCARDED" >> "$LOG_FILE"
echo "  Rejected (min observations): $MINOBS_DISCARDED" >> "$LOG_FILE"
echo "  Rejected (negative avg flux): $NEGFLUX_DISCARDED" >> "$LOG_FILE"
echo "  After Step 1: $SNR_KEPT" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "Step 2 - Host Galaxy Extendedness Filter:" >> "$LOG_FILE"
echo "  Input from Step 1: $SNR_KEPT" >> "$LOG_FILE"
echo "  Rejected (point source hosts): $POINT_HOST_REJECTED" >> "$LOG_FILE"
echo "  Rejected (low flux ratio): $FLUX_RATIO_REJECTED" >> "$LOG_FILE"
echo "  Total rejected: $EXTENDEDNESS_REJECTED" >> "$LOG_FILE"
echo "  After Step 2: $EXTENDEDNESS_KEPT" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "Overall Statistics:" >> "$LOG_FILE"
echo "  Total input objects: $TOTAL_OBJECTS" >> "$LOG_FILE"
TOTAL_REJECTED=$((SNR_TOTAL_DISCARDED + EXTENDEDNESS_REJECTED))
echo "  Total rejected: $TOTAL_REJECTED" >> "$LOG_FILE"
echo "    └─ SNR filter: $SNR_FILTER_DISCARDED" >> "$LOG_FILE"
echo "    └─ Time window: $WINDOW_DISCARDED" >> "$LOG_FILE"
echo "    └─ Min observations: $MINOBS_DISCARDED" >> "$LOG_FILE"
echo "    └─ Negative flux: $NEGFLUX_DISCARDED" >> "$LOG_FILE"
echo "    └─ Point source hosts: $POINT_HOST_REJECTED" >> "$LOG_FILE"
echo "    └─ Low flux ratio: $FLUX_RATIO_REJECTED" >> "$LOG_FILE"
echo "  Final filtered objects: $EXTENDEDNESS_KEPT" >> "$LOG_FILE"
if [ $TOTAL_OBJECTS -gt 0 ]; then
    OVERALL_KEEP_PCT=$(python -c "print(f'{100.0 * $EXTENDEDNESS_KEPT / $TOTAL_OBJECTS:.2f}')")
    echo "  Overall keep rate: ${OVERALL_KEEP_PCT}%" >> "$LOG_FILE"
fi
echo "" >> "$LOG_FILE"
echo "Filtered lightcurves saved in: $EXTENDEDNESS_FILTERED_DIR" >> "$LOG_FILE"

echo "=== Pipeline complete ==="
echo ""
echo "Pipeline summary saved to: $LOG_FILE"
echo "Filtered lightcurves saved in: $EXTENDEDNESS_FILTERED_DIR"
cat "$LOG_FILE"
