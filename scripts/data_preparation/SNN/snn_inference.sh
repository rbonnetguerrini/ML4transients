#!/bin/bash

# --- Step 1: LSST environment for CSV conversion ---
source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2024_30/loadLSST.bash
setup lsst_distrib

export PYTHONPATH=/sps/lsst/users/rbonnetguerrini/ML4transients/src:$PYTHONPATH

INPUT_DIR="/sps/lsst/groups/transients/HSC/fouchez/raphael/data/UDEEP/lightcurves"
CSV_DIR="${INPUT_DIR}/csv"

# Clean up any old debug files to prevent conflicts
echo "=== Cleaning old debug files ==="
rm -f "${CSV_DIR}"/*debug_counts*.csv
rm -f "${CSV_DIR}"/*filtered_ids*.txt
rm -f "${CSV_DIR}"/*inference_counts*.csv

echo "=== Running CSV conversion ==="
python /sps/lsst/users/rbonnetguerrini/ML4transients/scripts/data_preparation/SNN/convert_lc_for_snn.py "$INPUT_DIR" "$CSV_DIR"
if [ $? -ne 0 ]; then
    echo "CSV conversion failed."
    exit 1
fi

# --- Step 2: SuperNNova inference in snn_env ---
echo "=== Running SuperNNova inference ==="
conda activate snn_env
INFER_OUT="${CSV_DIR}/snn_results"
mkdir -p "$INFER_OUT"
python /sps/lsst/users/rbonnetguerrini/ML4transients/scripts/data_preparation/SNN/infer_snn.py "$CSV_DIR" "$INFER_OUT"
if [ $? -ne 0 ]; then
    echo "Inference failed."
    exit 1
fi
conda deactivate

# --- Step 3: Save inference results to HDF5 with validation (LSST env) ---
echo "=== Saving inference results to HDF5 with validation ==="
source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2024_30/loadLSST.bash
setup lsst_distrib
export PYTHONPATH=/sps/lsst/users/rbonnetguerrini/ML4transients/src:$PYTHONPATH

python /sps/lsst/users/rbonnetguerrini/ML4transients/scripts/data_preparation/SNN/save_inference_snn.py "$INFER_OUT" "$INPUT_DIR" --csv_dir "$CSV_DIR"

echo "=== Pipeline complete ==="
