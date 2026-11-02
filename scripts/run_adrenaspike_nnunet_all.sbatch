#!/bin/bash -l
#SBATCH --job-name=nnunet3d_amos_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#SBATCH --output=nnunet3d_all.%j.out
#SBATCH --error=nnunet3d_all.%j.err

# Single-allocation workflow: prepare -> preprocess -> train -> predict -> score.
# Re-submit THIS SAME sbatch file if training exceeds one allocation.
# Training is skipped only after an nnU-Net checkpoint_final exists.
# Prepared/preprocessed data and complete predictions are also reused.
set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

export nnUNet_raw="${NNUNET_RAW:-$PWD/nnunet_raw}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED:-$PWD/nnunet_preprocessed}"
export nnUNet_results="${NNUNET_RESULTS:-$PWD/nnunet_results}"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" runs

ID="${DATASET_ID:-701}"
FOLD="${FOLD:-0}"
DATA_ROOT="${DATA_ROOT:-../data/amos22}"
DATASET_NAME="Dataset$(printf '%03d' "$ID")_AMOSAdrenalCT"
DATASET="$nnUNet_raw/$DATASET_NAME"
PREPROCESSED="$nnUNet_preprocessed/$DATASET_NAME"
FOLD_DIR="$nnUNet_results/$DATASET_NAME/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_$FOLD"
PRED_DIR="$PWD/runs/nnunet3d_holdout_predictions_fold$FOLD"
METRICS="$PWD/runs/nnunet3d_holdout_metrics_fold$FOLD.csv"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
note() { echo "[$(stamp)] $*"; }
trap 'rc=$?; echo "[$(date "+%Y-%m-%d %H:%M:%S")] EXIT status=$rc; inspect nnunet3d_all.*.out/.err if nonzero" >&2' EXIT

note "Single-allocation pipeline started: job=${SLURM_JOB_ID:-manual} dataset=$ID fold=$FOLD"
note "Time limit requested: 72h; nnUNet training may need another submission"
note "Raw: $nnUNet_raw | Preprocessed: $nnUNet_preprocessed | Results: $nnUNet_results"
for program in python nnUNetv2_plan_and_preprocess nnUNetv2_train nnUNetv2_predict; do
  command -v "$program" >/dev/null || { note "ERROR: required command missing: $program"; exit 2; }
done
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("No ROCm GPU visible in allocation")
print("PyTorch:", torch.__version__, "HIP:", torch.version.hip,
      "GPU:", torch.cuda.get_device_name(0), flush=True)
PY

# 1. Prepare once. The conversion script writes the split manifest last.
# If a previous preparation stopped mid-way, --force repairs the partial folder.
if [[ -s "$DATASET/dataset.json" && -s "$DATASET/split_manifest.json" ]]; then
  note "Stage 1/5: prepared dataset already present; skipping"
else
  note "Stage 1/5: preparing full-scan CT and adrenal labels"
  PREP_ARGS=()
  if [[ -d "$DATASET" ]]; then PREP_ARGS+=(--force); fi
  python -u scripts/prepare_nnunet_adrenal.py \
    --data-root "$DATA_ROOT" --output-root "$nnUNet_raw" \
    --dataset-id "$ID" "${PREP_ARGS[@]}"
fi

# 2. Verify *all* training cases were preprocessed, not just one partial
# npz. Both the older .npz and newer .b2nd nnU-Net case formats are supported.
preprocessing_complete() {
  [[ -s "$PREPROCESSED/nnUNetPlans.json" ]] || return 1
  python - "$DATASET/split_manifest.json" "$PREPROCESSED/nnUNetPlans_3d_fullres" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
names = manifest.get("train_cases", [])
if not names or not root.is_dir():
    raise SystemExit(1)
missing = [name for name in names if not any(
    (root / (name + ext)).is_file() for ext in (".npz", ".b2nd")
)]
raise SystemExit(1 if missing else 0)
PY
}
if preprocessing_complete; then
  note "Stage 2/5: all training cases already preprocessed; skipping"
else
  note "Stage 2/5: nnU-Net fingerprint, planning, and preprocessing"
  nnUNetv2_plan_and_preprocess -d "$ID" --verify_dataset_integrity -c 3d_fullres
  preprocessing_complete || { note "ERROR: preprocessing incomplete"; exit 5; }
fi

# 3. Train the selected fold. Official nnU-Net resumes using --c when
# checkpoint_latest is available; no final checkpoint means not yet finished.
if [[ -s "$FOLD_DIR/checkpoint_final.pth" ||
      -s "$FOLD_DIR/checkpoint_final.safetensors" ]]; then
  note "Stage 3/5: fold=$FOLD final checkpoint exists; skipping training"
else
  note "Stage 3/5: training 3D full-resolution nnU-Net fold=$FOLD"
  TRAIN_ARGS=()
  if [[ -s "$FOLD_DIR/checkpoint_latest.pth" ||
        -s "$FOLD_DIR/checkpoint_latest.safetensors" ]]; then
    TRAIN_ARGS+=(--c)
    note "Resuming from checkpoint_latest"
  else
    note "No checkpoint_latest found: starting training from beginning"
  fi
  nnUNetv2_train "$ID" 3d_fullres "$FOLD" -device cuda "${TRAIN_ARGS[@]}"
fi

if [[ ! -s "$FOLD_DIR/checkpoint_final.pth" &&
      ! -s "$FOLD_DIR/checkpoint_final.safetensors" ]]; then
  note "Final checkpoint not found; training incomplete. Re-submit same script to resume."
  exit 3
fi

# 4. Predict *all* held-out CTs and only skip if every output is present.
shopt -s nullglob
INPUTS=( "$DATASET"/imagesTs/*_0000.nii.gz )
if (( ${#INPUTS[@]} == 0 )); then
  note "ERROR: no heldout CT volumes found in $DATASET/imagesTs"
  exit 4
fi
PRED_COMPLETE=1
for image in "${INPUTS[@]}"; do
  case_name="$(basename "$image" _0000.nii.gz)"
  if [[ ! -s "$PRED_DIR/$case_name.nii.gz" ]]; then
    PRED_COMPLETE=0
    break
  fi
done
if (( PRED_COMPLETE == 1 )); then
  note "Stage 4/5: predictions already complete for ${#INPUTS[@]} cases; skipping"
else
  note "Stage 4/5: running heldout CT inference"
  mkdir -p "$PRED_DIR"
  PRED_ARGS=()
  if [[ "${SAVE_PROBABILITIES:-0}" == "1" ]]; then
    PRED_ARGS+=(--save_probabilities)
  fi
  nnUNetv2_predict -i "$DATASET/imagesTs" -o "$PRED_DIR" \
    -d "$ID" -c 3d_fullres -f "$FOLD" -device cuda "${PRED_ARGS[@]}"
fi

# 5. Recompute paired per-case metrics after every successful prediction stage.
note "Stage 5/5: evaluating full-volume per-gland Dice, HD95, NSD"
python -u scripts/evaluate_nnunet_adrenal.py \
  --pred-dir "$PRED_DIR" --dataset-dir "$DATASET" --output "$METRICS"
note "SUCCESS. Fold: $FOLD | Metrics: $METRICS | Predictions: $PRED_DIR"
