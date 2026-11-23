#!/bin/bash -l
#SBATCH --job-name=nnunet702_resencl_sd
#SBATCH --partition=k2-gpu-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:mi300x:1
#SBATCH --output=nnunet702_resencl_sd.%j.out
#SBATCH --error=nnunet702_resencl_sd.%j.err

# C2b: Dataset702, SAME data, SAME frozen 1095/158 split, SAME 1.5 mm
# preprocessing as C2. Two changes, both aimed at the small-gland tail:
#   1. nnU-Net residual-encoder "L" preset (nnUNetPlannerResEncL), the
#      nnU-Net authors' recommended configuration.
#   2. Dice loss computed per patch, not pooled over the batch
#      (batch_dice=false), so a small gland weighs as much as a large one.
#
# No re-preprocessing: the ResEnc planner reuses the existing
# nnUNetPlans_3d_fullres data (nnunetv2 2.8.1 ResEncUNetPlanner.
# generate_data_identifier). The script checks that before training.
# Nothing existing is overwritten: new plans file, new results folder.
#
# ResEnc L is heavier than the default network: expect more than one 3-day
# allocation. If the job exits with status 3, resubmit it unchanged - it
# resumes from checkpoint_latest.
set -Eeuo pipefail
ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

CACHE_BASE="$ROOT/.hpc_cache/nnunet"
export TRITON_HOME="$CACHE_BASE" TRITON_CACHE_DIR="$CACHE_BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor" XDG_CACHE_HOME="$CACHE_BASE/xdg"
export TORCH_HOME="$CACHE_BASE/torch" TORCH_EXTENSIONS_DIR="$CACHE_BASE/torch_extensions"
export MIOPEN_USER_DB_PATH="$CACHE_BASE/miopen_db" MIOPEN_CUSTOM_CACHE_DIR="$CACHE_BASE/miopen_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" \
         "$TORCH_EXTENSIONS_DIR" "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
export nnUNet_compile=F
export nnUNet_raw="$ROOT/nnunet_raw"
export nnUNet_preprocessed="$ROOT/nnunet_preprocessed"
export nnUNet_results="$ROOT/nnunet_results"

PRE="$nnUNet_preprocessed/Dataset702_AMOSTotalSegAdrenalCT"
BASE_PLANS="$PRE/nnUNetResEncUNetLPlans.json"
PLANS_NAME="nnUNetResEncUNetLPlans_sampleDice"
PLANS="$PRE/$PLANS_NAME.json"
FOLD_DIR="$nnUNet_results/Dataset702_AMOSTotalSegAdrenalCT/nnUNetTrainer__${PLANS_NAME}__3d_fullres/fold_0"

echo "Git commit: $(git rev-parse HEAD) | SLURM job: ${SLURM_JOB_ID:-manual}"
python - "$PRE/splits_final.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
assert len(x[0]["train"]) == 1095 and len(x[0]["val"]) == 158
assert not set(x[0]["train"]) & set(x[0]["val"])
print("PASS: same frozen 1095/158 split as C2.", flush=True)
PY

# 1. ResEnc L plans (planning only - never preprocessing)
if [[ ! -s "$BASE_PLANS" ]]; then
  echo "Planning ResEnc L (writes $BASE_PLANS only)"
  nnUNetv2_plan_experiment -d 702 -pl nnUNetPlannerResEncL
fi

# 2. Derived plans: per-sample Dice, everything else identical
if [[ ! -s "$PLANS" ]]; then
  python - "$BASE_PLANS" "$PLANS" "$PLANS_NAME" <<'PY'
import json, sys
src, dst, name = sys.argv[1:]
p = json.load(open(src))
p["plans_name"] = name
p["configurations"]["3d_fullres"]["batch_dice"] = False
json.dump(p, open(dst, "w"), indent=4)
print(f"wrote {dst}: 3d_fullres batch_dice=False", flush=True)
PY
fi

# 3. Guards: reuse of existing preprocessed data, same spacing as C2
python - "$PLANS" "$PRE" <<'PY'
import json, sys
from pathlib import Path
plans, pre = json.load(open(sys.argv[1])), Path(sys.argv[2])
c = plans["configurations"]["3d_fullres"]
assert c["data_identifier"] == "nnUNetPlans_3d_fullres", f"would need new preprocessing: {c['data_identifier']}"
assert [round(s, 4) for s in c["spacing"]] == [1.5, 1.5, 1.5], c["spacing"]
assert c["batch_dice"] is False
n = len(list((pre / "nnUNetPlans_3d_fullres").glob("*.pkl")))
assert n == 1253, f"expected 1253 preprocessed cases, found {n}"
print(f"PASS: reuses nnUNetPlans_3d_fullres ({n} cases) | spacing {c['spacing']} | patch {c['patch_size']} "
      f"| batch size {c['batch_size']} | batch_dice False", flush=True)
PY

python - <<'PY'
import torch
assert torch.cuda.is_available(), "AMD ROCm GPU is not accessible"
print("ROCm torch", torch.__version__, "GPU", torch.cuda.get_device_name(0), flush=True)
PY

if [[ -s "$FOLD_DIR/checkpoint_final.pth" ]]; then
  echo "C2b already finished: $FOLD_DIR/checkpoint_final.pth"; exit 0
fi
CONTINUE=()
[[ -s "$FOLD_DIR/checkpoint_latest.pth" ]] && { CONTINUE+=(--c); echo "Resuming C2b from checkpoint_latest."; }

nnUNetv2_train 702 3d_fullres 0 -p "$PLANS_NAME" -device cuda "${CONTINUE[@]}"

if [[ ! -s "$FOLD_DIR/checkpoint_final.pth" ]]; then
  echo "STOP: no final checkpoint yet - resubmit this script unchanged to resume."; exit 3
fi
sha256sum "$FOLD_DIR/checkpoint_final.pth"
echo "SUCCESS: C2b (ResEnc L + per-sample Dice) finished. Its validation/summary.json is on the"
echo "same 158 development cases as C2 (0.8527) - a paired comparison."
