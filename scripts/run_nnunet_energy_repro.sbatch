#!/bin/bash -l
#SBATCH --job-name=nnunet_energy_clean
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#SBATCH --output=nnunet_energy_clean.%j.out
#SBATCH --error=nnunet_energy_clean.%j.err

# Strict, standalone energy measurement of EXISTING 3D nnU-Net checkpoint.
# No retraining, no modification to original prediction NIfTI or metrics.
# One repeat first is recommended: sbatch --export=ALL,ENERGY_REPEATS=1 ...
# Later run 3 repeats to the SAME output folder, which resumes valid trials.
# Do not benchmark while personally running I/O-heavy audit on shared storage.
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

FOLD_DIR="$nnUNet_results/Dataset701_AMOSAdrenalCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0"
METRICS="$ROOT/runs/nnunet3d_holdout_metrics_fold0.csv"
DATASET="$nnUNet_raw/Dataset701_AMOSAdrenalCT"
OUT="$ROOT/runs/energy_nnunet_fullvolume_repro"
REPEATS="${ENERGY_REPEATS:-1}"

if [[ ! -s "$FOLD_DIR/checkpoint_final.pth" ]]; then
    echo "ERROR: missing final trained checkpoint: $FOLD_DIR/checkpoint_final.pth" >&2
    exit 2
fi
if [[ ! -s "$METRICS" || ! -d "$DATASET/imagesTs" ]]; then
    echo "ERROR: original holdout metrics or images absent" >&2
    exit 2
fi
if ! [[ "$REPEATS" =~ ^[1-3]$ ]]; then
    echo "ERROR: ENERGY_REPEATS must be 1, 2 or 3" >&2
    exit 2
fi

echo "===== POWER TELEMETRY PRECHECK ====="
python -u scripts/measure_gpu_energy.py \
  --probe --output-dir "$OUT"

echo "===== CONTROLLED ORIGINAL FOLD-0 FULL-VOLUME ENERGY ====="
python -u scripts/run_nnunet_energy_suite.py \
  --dataset-dir "$DATASET" \
  --metrics "$METRICS" \
  --fold 0 \
  --out-dir "$OUT" \
  --repeats "$REPEATS" \
  --min-remaining-minutes 150

echo "===== FINAL BENCHMARK STATUS ====="
python - "$OUT/nnunet_energy_summary.json" "$REPEATS" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
print(json.dumps(report, indent=2), flush=True)
expected = int(sys.argv[2])
if report.get("status") != "complete" or report.get("trials_completed") != expected:
    raise SystemExit(
        f"Benchmark incomplete: {report.get('status')}; "
        f"{report.get('trials_completed')} trials vs {expected} requested"
    )
if report.get("gpu_joules_per_ct", {}).get("mean") is None:
    raise SystemExit("No valid GPU joules-per-CT value")
print("SUCCESS: actual MI300X full-volume GPU energy measured.", flush=True)
PY
