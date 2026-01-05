#!/bin/bash
#SBATCH --job-name=eval_ensemble
#SBATCH --output=logs/evaluation/eval_%j.out
#SBATCH --error=logs/evaluation/eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=hpc

# Activate conda environment
source ~/.bashrc
PROJECT_ROOT="/sps/lsst/users/rbonnetguerrini/ML4transients"
cd "$PROJECT_ROOT"
conda activate env_ML
#--object-ids-file saved/object_ID/coadd_transients.txt \
# Run evaluation
python scripts/evaluation/run_evaluation.py \
  --config configs/evaluation/evaluation_config.yaml \
  --data-path /sps/lsst/groups/transients/HSC/fouchez/raphael/data/rc2_89570 \
  --weights-path /sps/lsst/groups/transients/HSC/fouchez/raphael/training/multichannel_coteaching_asym_89570 \
  --output-dir saved/test_eval/multichannel_coteaching_asym_MC_filtered_89570 \
  --umap-save-path saved/umap_weight/multichannel_coteaching_asym_MC_filtered_89570 \
  --interpretability \
  --mc-dropout \
  --optimize-umap \
  --run-inference \
  --show-all-cutouts

echo "Evaluation completed!"
