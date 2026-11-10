#!/bin/bash -l
#SBATCH --job-name=tsv3_adrenal_qc
#SBATCH --partition=k2-medpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=tsv3_adrenal_qc.%j.out
#SBATCH --error=tsv3_adrenal_qc.%j.err

# CPU-only. Pre-register a metadata-stratified annotation-review sample and
# render reference-only CT+adrenal overlays (never model outputs/accuracy).
# Can safely re-submit after wall limit: skips existing plan and complete PNGs.
set -Eeuo pipefail
ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -c "import numpy, nibabel, matplotlib; print('QC dependencies OK', flush=True)"

PLAN="runs/totalsegmentator_v3_qc_plan"
if [[ ! -s "$PLAN/plan_summary.json" ]]; then
  python -u scripts/plan_totalseg_v3_adrenal_qc.py \
    --audit runs/totalsegmentator_v3_audit/case_audit.csv \
    --train-review-n 60 \
    --test-review-n 15 \
    --seed 20261110 \
    --out-dir "$PLAN"
fi

python -u scripts/render_totalseg_v3_adrenal_qc.py \
  --plan "$PLAN/training_label_review_sample.csv" \
  --out-dir "$PLAN/montages_train" --resume

python -u scripts/render_totalseg_v3_adrenal_qc.py \
  --plan "$PLAN/external_test_label_review_sample.csv" \
  --out-dir "$PLAN/montages_external_test" --resume

echo "SUCCESS: review plan and reference-only CT montages created."
echo "Reviewers must manually complete status, laterality, boundary and completeness columns."
echo "Do not upload clinical images to a public repository."
