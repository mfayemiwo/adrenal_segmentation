#!/bin/bash -l
#SBATCH --job-name=dataset702_cpu_prep
#SBATCH --partition=k2-medpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=dataset702_cpu_prep.%j.out
#SBATCH --error=dataset702_cpu_prep.%j.err

# CPU-only: freeze development splits and prepare all 1253 AMOS/TS imagesTr.
# No GPU, no training, no official TS external test cases, and no use of
# original 100 AMOS test images. Can re-submit after walltime: checks frozen
# manifest & resumes compatible already processed CTs/masks in Dataset702.
# Original Dataset701 and all source NIfTI files are NEVER modified.
set -Eeuo pipefail

ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PLAN="runs/expanded_dataset702/plan.json"
DATASET="nnunet_raw/Dataset702_AMOSTotalSegAdrenalCT"

python - <<'PY'
import nibabel, SimpleITK
print("Preflight: nibabel and SimpleITK imports work",flush=True)
PY

echo "== Kelvin2 project storage =="
df -h "$ROOT"
echo "== Dataset702 frozen development plan =="
if [[ ! -s "$PLAN" ]]; then
  python -u scripts/freeze_expanded_amos_tsv3_manifest.py \
    --amos nnunet_raw/Dataset701_AMOSAdrenalCT \
    --totalseg ../data/totalsegmentator_v3/extracted/Totalsegmentator_dataset_v300 \
    --candidates runs/totalsegmentator_v3_audit/candidates_not_train_ready.csv \
    --audit runs/totalsegmentator_v3_audit/case_audit.csv \
    --external nnunet_raw/ExternalTotalSegV3AdrenalTest_v1/external_test_manifest.json \
    --out "$PLAN" --seed 20261116 --ts-validation-fraction 0.15
else
  echo "Existing frozen split manifest retained without changes: $PLAN"
fi

echo "== Dataset702 preparation (CPU, resumable) =="
python -u scripts/prepare_expanded_amos_tsv3_dataset702.py \
  --manifest "$PLAN" --out "$DATASET" --resume

echo "== Dataset702 strict result check =="
python - "$PLAN" "$DATASET" <<'PY'
import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
ds=Path(sys.argv[2])
expected=set(m["fold0_train_ids"])|set(m["fold0_val_ids"])
imgs={p.name.removesuffix("_0000.nii.gz") for p in (ds/"imagesTr").glob("*_0000.nii.gz")}
labs={p.name.removesuffix(".nii.gz") for p in (ds/"labelsTr").glob("*.nii.gz")}
assert len(expected)==1253 and expected==imgs==labs, "Missing or extra Dataset702 CT/label"
assert len(m["fold0_train_ids"])==1095 and len(m["fold0_val_ids"])==158
assert all(not (ds/folder).exists() for folder in ("imagesTs","labelsTs"))
print("SUCCESS: Dataset702 has 1253 development cases; 1095 fold0 train and 158 fold0 val.")
print("Clinical label quality is unreviewed; this is exploratory training data.")
PY
