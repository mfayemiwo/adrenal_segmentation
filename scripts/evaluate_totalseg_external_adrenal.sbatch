#!/bin/bash -l
#SBATCH --job-name=nnunet_external_tsv3
#SBATCH --partition=k2-gpu-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:mi300x:1
#SBATCH --output=nnunet_external_tsv3.%j.out
#SBATCH --error=nnunet_external_tsv3.%j.err

# Later stage ONLY: run original AMOS-trained fold-0 nnU-Net on independent
# TotalSegmentator official test cases. No new training, model selection,
# postprocessing, or inference-energy comparison.
#
# Must explicitly record that blinded annotation/provenance QC was undertaken
# before submitting: sbatch --export=ALL,EXTERNAL_TEST_QC_SIGNOFF=1 ...
# This flag is a human signoff, NOT automatic certification of label quality.
set -Eeuo pipefail
ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
if [[ "${EXTERNAL_TEST_QC_SIGNOFF:-0}" != "1" ]]; then
  echo "STOP: complete blinded external annotation QA, document decisions and " >&2
  echo "then explicitly set EXTERNAL_TEST_QC_SIGNOFF=1 to authorize heldout evaluation." >&2
  exit 2
fi

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

DATA="$ROOT/nnunet_raw/ExternalTotalSegV3AdrenalTest_v1"
PRED="$ROOT/runs/nnunet3d_external_tsv3_predictions_fold0"
METRICS="$ROOT/runs/nnunet3d_external_tsv3_metrics_fold0.csv"
MODEL="$nnUNet_results/Dataset701_AMOSAdrenalCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

[[ -s "$MODEL" && -s "$DATA/external_test_manifest.json" ]] || {
  echo "ERROR: existing AMOS-trained fold-0 weights or frozen external manifest missing" >&2
  exit 2
}
python - "$DATA/external_test_manifest.json" <<'PY'
import json,sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text())
if len(manifest.get("case_ids",[]))!=76 or manifest.get("source_partition")!="test":
    raise SystemExit("External cohort is not the expected 76 official test cases")
print("Frozen 76-case official external test cohort loaded.", flush=True)
PY

mkdir -p "$PRED"
if python - "$DATA" "$PRED" <<'PY'
import json,sys
from pathlib import Path
data, pred = map(Path, sys.argv[1:])
ids = json.loads((data/"external_test_manifest.json").read_text())["case_ids"]
complete = len(ids)==76 and all((pred/(cid+".nii.gz")).is_file()
                               and (pred/(cid+".nii.gz")).stat().st_size>0
                               for cid in ids)
raise SystemExit(0 if complete else 1)
PY
then
  echo "All 76 existing external predictions present; skip inference."
else
  echo "Predicting from ORIGINAL AMOS-trained nnU-Net 3d_fullres fold0; NO training."
  nnUNetv2_predict -i "$DATA/imagesTs" -o "$PRED" \
    -d 701 -c 3d_fullres -f 0 -device cuda
fi

python -u scripts/evaluate_totalseg_external_adrenal.py \
  --dataset-dir "$DATA" --pred-dir "$PRED" --output "$METRICS"

echo "SUCCESS: external benchmark scored; results remain preliminary until QA signoff is filed."
echo "Summary: ${METRICS%.csv}.summary.json"
