#!/bin/bash -l
#SBATCH --job-name=nnunet702_expanded
#SBATCH --partition=k2-gpu-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:mi300x:1
#SBATCH --output=nnunet702_expanded.%j.out
#SBATCH --error=nnunet702_expanded.%j.err

# **EXPLORATORY** expanded-data CNN training after separate Dataset702
# development-only data preparation and CPU preprocessing. 1095 fold0 train,
# 158 fold0 validation. Never uses original AMOS 100 heldout or any official
# TotalSegmentator test. Resubmittable at SLURM walltime if latest checkpoint.
# Do not infer on the 76 external test cases during training/model selection.
set -Eeuo pipefail
ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"

module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate
export nnUNet_raw="$ROOT/nnunet_raw"
export nnUNet_preprocessed="$ROOT/nnunet_preprocessed"
export nnUNet_results="$ROOT/nnunet_results"
export nnUNet_compile=F

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
   "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$MIOPEN_USER_DB_PATH" \
   "$MIOPEN_CUSTOM_CACHE_DIR"

# No checkpoint or split must be overwritten if this is the original AMOS
# Dataset701; Dataset702 is a new fullres nnU-Net model namespace.
FOLD_DIR="$nnUNet_results/Dataset702_AMOSTotalSegAdrenalCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0"
SPLIT="$nnUNet_preprocessed/Dataset702_AMOSTotalSegAdrenalCT/splits_final.json"
[[ -s "$SPLIT" ]] || { echo "STOP: development validation split missing" >&2; exit 2; }
python - "$SPLIT" <<'PY'
import json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
assert len(x)>=1 and len(x[0]["train"])==1095 and len(x[0]["val"])==158
assert len(set(x[0]["train"]) & set(x[0]["val"]))==0
assert all(c.startswith(("am_", "tsv3_")) for c in x[0]["train"]+x[0]["val"])
assert not any(c.startswith("am_") for c in x[0]["val"])
print("PASS: Dataset702 frozen 1095-train/158-development-validation split.", flush=True)
PY

echo "Git commit: $(git rev-parse HEAD)"
echo "SLURM job: ${SLURM_JOB_ID:-manual}"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "AMD ROCm GPU is not accessible"
print("ROCm torch",torch.__version__,"GPU",torch.cuda.get_device_name(0),flush=True)
PY

if [[ -s "$FOLD_DIR/checkpoint_final.pth" || -s "$FOLD_DIR/checkpoint_final.safetensors" ]]; then
  echo "Existing Dataset702 fold0 training complete; skip/reuse final checkpoint."
  exit 0
fi
CONTINUE=()
if [[ -s "$FOLD_DIR/checkpoint_latest.pth" || -s "$FOLD_DIR/checkpoint_latest.safetensors" ]]; then
  CONTINUE+=(--c)
  echo "Resuming Dataset702 fold0 training from latest checkpoint."
fi

nnUNetv2_train 702 3d_fullres 0 -device cuda "${CONTINUE[@]}"

if [[ ! -s "$FOLD_DIR/checkpoint_final.pth" && ! -s "$FOLD_DIR/checkpoint_final.safetensors" ]]; then
  echo "STOP: trainer exited without final checkpoint. Resubmit same job to resume."
  exit 3
fi
echo "SUCCESS: expanded-data nnU-Net Dataset702 fold0 finished."
echo "Do not evaluate on external test until the model comparison protocol is frozen."
