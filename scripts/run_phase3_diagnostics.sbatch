#!/bin/bash -l
#SBATCH --job-name=phase3_diagnostics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#SBATCH --array=0-7
#SBATCH --output=diagnostics.%A_%a.out
#SBATCH --error=diagnostics.%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate
VARIANTS=(anatomy_snn anatomy_analog)
MODES=(none zero_events reverse_events no_gate)
VARIANT="${VARIANTS[$((SLURM_ARRAY_TASK_ID / 4))]}"
MODE="${MODES[$((SLURM_ARRAY_TASK_ID % 4))]}"
python -u scripts/evaluate_segmenter.py \
  --checkpoint "runs/phase2_${VARIANT}_s1/best_model.pt" \
  --data-root ../data/amos22 --cache-dir ../cache \
  --output-dir "runs/phase3_diagnostics/$VARIANT/$MODE" \
  --diagnostic-mode "$MODE" \
  --keep-largest 2 --min-component-voxels 30 --device cuda
