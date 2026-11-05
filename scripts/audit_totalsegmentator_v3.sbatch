#!/bin/bash -l
#SBATCH --job-name=tsv3_adrenal_audit
#SBATCH --partition=k2-medpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=tsv3_adrenal_audit.%j.out
#SBATCH --error=tsv3_adrenal_audit.%j.err

# Safely scan compressed CT headers and gland/kidney masks in the 1,939-case
# TotalSegmentator v3 dataset. Read-only to all source NIfTI and AMOS data.
# The output is only an audit, NEVER an automatic train/val split merge.
set -Eeuo pipefail
ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"

module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

DATASET="${TOTALSEG_V3_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/data/totalsegmentator_v3/extracted/Totalsegmentator_dataset_v300}"
if [[ ! -d "$DATASET" || ! -f "$DATASET/meta.csv" ]]; then
    echo "ERROR: TotalSegmentator v3 root missing: $DATASET" >&2
    exit 2
fi

python - <<'PY'
import sys, numpy, nibabel
print("Python:", sys.executable, flush=True)
print("NumPy:", numpy.__version__, "NiBabel:", nibabel.__version__, flush=True)
PY

python -u scripts/audit_totalsegmentator_v3.py \
  --dataset "$DATASET" \
  --output-dir runs/totalsegmentator_v3_audit

echo "SUCCESS: reports generated; no training, inference or source-file edits."
