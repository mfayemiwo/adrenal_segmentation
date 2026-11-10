#!/bin/bash -l
#SBATCH --job-name=tsv3_external_prep
#SBATCH --partition=k2-medpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=tsv3_external_prep.%j.out
#SBATCH --error=tsv3_external_prep.%j.err

# Create symlinked external ONLY CTs and combined bilateral GT labels, but do
# not infer or train. Does not modify original TotalSegmentator or AMOS.
set -Eeuo pipefail
ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

[[ -s runs/totalsegmentator_v3_qc_plan/plan_summary.json ]] || {
  echo "ERROR: first execute the blinded QC planning job" >&2; exit 2;
}
python -u scripts/prepare_totalseg_external_adrenal_test.py \
  --dataset-root ../data/totalsegmentator_v3/extracted/Totalsegmentator_dataset_v300 \
  --inventory runs/totalsegmentator_v3_qc_plan/external_test_bilateral_inventory.csv \
  --output nnunet_raw/ExternalTotalSegV3AdrenalTest_v1 \
  --resume

echo "SUCCESS: external-only 76-case source cohort PREPARED; not yet reviewed or scored."
