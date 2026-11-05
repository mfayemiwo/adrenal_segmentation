#!/bin/bash -l
#SBATCH --job-name=adrenal_postproc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=nnunet_postproc.%j.out
#SBATCH --error=nnunet_postproc.%j.err

# CPU-only work: reuse EXISTING .venv, saved nnU-Net NIfTI masks and evaluation.
# Choose a CPU partition at submission, e.g. sbatch -p <actual_cpu_partition>
# scripts/run_nnunet_adrenal_postprocessing.sbatch. No MI300X requested.
# Both postprocessing variants use original full-volume predictions and never
# overwrite the raw saved masks. Holdout comparisons are EXPLORATORY only.
set -Eeuo pipefail

ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
module load apps/python3/3.12.4/gcc-14.1.0
source .venv/bin/activate

python - <<'PY'
import sys
import numpy, scipy, nibabel
print("Python:", sys.executable, flush=True)
print("numpy:", numpy.__version__, "scipy:", scipy.__version__,
      "nibabel:", nibabel.__version__, flush=True)
PY

INPUT_DIR="runs/nnunet3d_holdout_predictions_fold0"
DATASET_DIR="nnunet_raw/Dataset701_AMOSAdrenalCT"
if [[ ! -d "$INPUT_DIR" || ! -d "$DATASET_DIR/labelsTs" ]]; then
    echo "ERROR: saved nnU-Net predictions or heldout labels directory missing" >&2
    exit 2
fi

# Require exact patient-level correspondence before any processing.
python - "$INPUT_DIR" "$DATASET_DIR/labelsTs" <<'PY'
import sys
from pathlib import Path
pred = {p.name for p in Path(sys.argv[1]).glob("*.nii.gz")}
truth = {p.name for p in Path(sys.argv[2]).glob("*.nii.gz")}
if not pred or pred != truth:
    raise SystemExit(
        f"ERROR: predictions do not match test labels: "
        f"n_pred={len(pred)} n_gt={len(truth)}, "
        f"missing={sorted(truth-pred)[:5]}, extra={sorted(pred-truth)[:5]}"
    )
print(f"Validated {len(pred)} saved full-volume predictions.", flush=True)
PY

echo "[AUDIT] Investigate lowest-Dice patients, including any zero-overlap gland"
python -u scripts/audit_nnunet_adrenal_failures.py --worst 5

echo "[1/2] 3D largest-component cleanup (each gland separately)"
python -u scripts/postprocess_nnunet_adrenal.py \
  --input-dir "$INPUT_DIR" \
  --output-dir runs/nnunet3d_holdout_predictions_cc_largest

echo "[2/2] Retain any secondary component >=15% size of largest gland component"
python -u scripts/postprocess_nnunet_adrenal.py \
  --input-dir "$INPUT_DIR" \
  --output-dir runs/nnunet3d_holdout_predictions_cc_relative15 \
  --mode relative --min-relative-size 0.15

echo "Both masks-only postprocessing variants completed."
echo "Raw input predictions unchanged: $INPUT_DIR"
echo "Inspect audit CSVs in runs/nnunet3d_holdout_predictions_cc_*/postprocessing_audit.csv"

# This optional evaluation compares already-observed AMOS holdout masks:
# it is an EXPLORATORY sensitivity analysis, not unbiased method selection.
# The filenames deliberately do not end _foldN, so the existing evaluator's
# optional energy benchmark hook is not invoked.
if [[ "${EVAL_HOLDOUT_EXPLORATORY:-0}" == "1" ]]; then
    echo "WARNING: held-out postprocessing comparisons are exploratory."
    for mode in largest relative15; do
        python -u scripts/evaluate_nnunet_adrenal.py \
          --pred-dir "runs/nnunet3d_holdout_predictions_cc_$mode" \
          --dataset-dir "$DATASET_DIR" \
          --output "runs/nnunet3d_holdout_metrics_cc_$mode.csv"
    done
fi

echo "SUCCESS: CPU-only postprocessing job completed."
