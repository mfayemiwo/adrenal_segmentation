#!/bin/bash -l
#SBATCH --job-name=amos_ts_overlap
#SBATCH --partition=k2-medpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=amos_ts_overlap.%j.out
#SBATCH --error=amos_ts_overlap.%j.err

# Read-only duplicate SCREEN. Original CTs, reference labels, and all nnU-Net
# checkpoints/predictions are preserved; results are candidate pairs for review.
# Cached fingerprints allow resubmission without rereading finished CTs.
# First use optional OVERLAP_LIMIT=10 to run a small pilot.
set -Eeuo pipefail
ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -c 'import numpy, scipy, nibabel; print("CT overlap Python prerequisites OK", flush=True)'

OUT="runs/ct_overlap_screen"
EXTRA=()
if [[ "${OVERLAP_LIMIT:-0}" -gt 0 ]]; then
  OUT="runs/ct_overlap_screen_pilot"
  EXTRA+=(--limit "$OVERLAP_LIMIT")
fi

python -u scripts/screen_amos_totalseg_ct_overlap.py \
  --amos-root ../data/amos22 \
  --totalseg-root ../data/totalsegmentator_v3/extracted/Totalsegmentator_dataset_v300 \
  --audit runs/totalsegmentator_v3_audit/case_audit.csv \
  --output-dir "$OUT" \
  "${EXTRA[@]}"

echo "SUCCESS: read-only CT overlap screening; manually review flagged cases."
