#!/bin/bash -l
#SBATCH --job-name=dataset702_preprocess
#SBATCH --partition=k2-medpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=dataset702_preprocess.%j.out
#SBATCH --error=dataset702_preprocess.%j.err

# CPU ONLY: nnU-Net train-data preprocessing for Dataset702, then freeze one
# 1095-train/158-validation fold. Never trains a model, touches Dataset701,
# or preprocesses the 109 official TS test / 100 AMOS heldout CTs.
# DO NOT schedule until dataset702_cpu_prep job completes successfully.
set -Eeuo pipefail
ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

export nnUNet_raw="$ROOT/nnunet_raw"
export nnUNet_preprocessed="$ROOT/nnunet_preprocessed"
export nnUNet_results="$ROOT/nnunet_results"
export nnUNet_compile=F

DATASET="$nnUNet_raw/Dataset702_AMOSTotalSegAdrenalCT"
PREPROC="$nnUNet_preprocessed/Dataset702_AMOSTotalSegAdrenalCT"
[[ -s "$DATASET/dataset.json" && -s "$DATASET/geometry_manifest.json" && \
   -s "$ROOT/runs/expanded_dataset702/plan.json" ]] || {
   echo "STOP: frozen Dataset702 is not fully prepared yet." >&2
   exit 2
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
echo "Storage available:"
df -h "$ROOT"

if python - "$ROOT/runs/expanded_dataset702/plan.json" "$PREPROC" <<'PY'
import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
pre=Path(sys.argv[2])
expected=set(p["fold0_train_ids"])|set(p["fold0_val_ids"])
data=pre/"nnUNetPlans_3d_fullres"
complete=(pre/"nnUNetPlans.json").is_file() and all(
   (data/(case+".npz")).is_file() or (data/(case+".b2nd")).is_file()
   for case in expected
)
raise SystemExit(0 if complete else 1)
PY
then
  echo "All 1253 nnU-Net fullres cases already preprocessed. Skipping."
else
  echo "Preprocessing 1253 training/development CTs (may require substantial disk and CPU)."
  nnUNetv2_plan_and_preprocess -d 702 --verify_dataset_integrity -c 3d_fullres
fi

python -u scripts/install_dataset702_frozen_split.py \
  --plan runs/expanded_dataset702/plan.json \
  --nnunet-raw nnunet_raw --nnunet-preprocessed nnunet_preprocessed

echo "SUCCESS: Dataset702 nnU-Net fullres ready with immutable fold0."
echo "Next stage is separately approved GPU training; no training in this job."
