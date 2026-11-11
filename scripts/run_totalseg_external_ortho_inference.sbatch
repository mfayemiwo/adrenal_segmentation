#!/bin/bash -l
#SBATCH --job-name=nnunet_tsv3_geometry
#SBATCH --partition=k2-gpu-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:mi300x:1
#SBATCH --output=nnunet_tsv3_geometry.%j.out
#SBATCH --error=nnunet_tsv3_geometry.%j.err

# REPLACEMENT for failed external NIfTI read. Infer on the same 76 CTs,
# fixing sheared/invalid NIfTI headers only where needed. Original 76 labels
# never train or influence inference preparation. Restore predictions to
# ORIGINAL CT grid (nearest-neighbour) BEFORE calculating metrics.
# Strictly exploratory: source annotations were not manually reviewed.
set -Eeuo pipefail
ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

CACHE_BASE="$ROOT/.hpc_cache/nnunet"
mkdir -p "$CACHE_BASE"
export TRITON_HOME="$CACHE_BASE"
export TRITON_CACHE_DIR="$CACHE_BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor"
export XDG_CACHE_HOME="$CACHE_BASE/xdg"
export TORCH_HOME="$CACHE_BASE/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_BASE/torch_extensions"
export MIOPEN_USER_DB_PATH="$CACHE_BASE/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_BASE/miopen_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" \
         "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
         "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
export nnUNet_compile=F
export nnUNet_raw="$ROOT/nnunet_raw"
export nnUNet_preprocessed="$ROOT/nnunet_preprocessed"
export nnUNet_results="$ROOT/nnunet_results"

ORIGINAL="$ROOT/nnunet_raw/ExternalTotalSegV3AdrenalTest_v1"
PREPARED="$ROOT/nnunet_raw/ExternalTotalSegV3AdrenalOrtho_v1"
PRED="$ROOT/runs/nnunet3d_external_tsv3_ortho_predictions_fold0"
RESTORED="$ROOT/runs/nnunet3d_external_tsv3_native_predictions_fold0"
METRICS="$ROOT/runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.csv"
MODEL="$nnUNet_results/Dataset701_AMOSAdrenalCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

[[ -s "$MODEL" && -s "$PREPARED/geometry_manifest.json" && -s "$ORIGINAL/external_test_manifest.json" ]] || {
  echo "STOP: original AMOS checkpoint / frozen external manifest / completed geometry-prep job missing." >&2
  exit 2
}

python - "$ORIGINAL" "$PREPARED" "$MODEL" <<'PY'
import hashlib,json,sys
from pathlib import Path
orig, prepared, model = map(Path, sys.argv[1:])
m=json.loads((orig/"external_test_manifest.json").read_text())
g=json.loads((prepared/"geometry_manifest.json").read_text())
h=hashlib.sha256((orig/"external_test_manifest.json").read_bytes()).hexdigest()
assert g["original_manifest_sha256"]==h, "CT geometry prep does not match frozen external cohort"
assert g["status"]=="COMPLETE" and g["case_count"]==76 and len(g["cases"])==76
assert m["source_partition"]=="test" and len(m["case_ids"])==76
assert all((prepared/"imagesTs"/(cid+"_0000.nii.gz")).is_file() for cid in m["case_ids"])
assert model.is_file()
print("PASS: frozen original 76-case cohort and prepared ITK-readable CTs.",flush=True)
PY

mkdir -p "$PRED"
if python - "$ORIGINAL" "$PRED" <<'PY'
import json,sys
from pathlib import Path
original,pred = map(Path,sys.argv[1:])
ids=json.loads((original/"external_test_manifest.json").read_text())["case_ids"]
names={x.name for x in pred.glob("*.nii.gz")}
expected={cid+".nii.gz" for cid in ids}
raise SystemExit(0 if names==expected and all((pred/x).stat().st_size>0 for x in expected) else 1)
PY
then
  echo "All 76 orthogonal-input CT predictions exist; skip inference."
else
  echo "Predicting 76 CTs with original AMOS-only fold0 CNN. No training."
  nnUNetv2_predict -i "$PREPARED/imagesTs" -o "$PRED" \
    -d 701 -c 3d_fullres -f 0 -device cuda
fi

python -u scripts/repair_totalseg_external_ct_geometry.py restore \
  --original "$ORIGINAL" \
  --prepared "$PREPARED" \
  --predictions "$PRED" \
  --restored "$RESTORED" \
  --resume

python -u scripts/evaluate_totalseg_external_adrenal.py \
  --dataset-dir "$ORIGINAL" --pred-dir "$RESTORED" --output "$METRICS"

echo "SUCCESS: 76 full-volume exploratory external CTs scored on ORIGINAL reference grid."
echo "Predictions stored; full result: ${METRICS%.csv}.summary.json"
