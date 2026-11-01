#!/bin/bash -l
#SBATCH --job-name=anatomy_phase2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=anatomy.%A_%a.out
#SBATCH --error=anatomy.%A_%a.err
#SBATCH --mem=96G
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#SBATCH --array=0-1

# Initial 2-arm Phase 2: anatomy_snn vs anatomy_analog; both seed 1.
# SMOKE=1 sbatch scripts/run_anatomy_phase2.sbatch to verify first.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

: "${BASE_CKPT:=runs/sb_positives/best_model.pt}"
: "${DATA_ROOT:=../data/amos22}"
: "${CACHE_DIR:=../cache}"
VARIANTS=(anatomy_snn anatomy_analog)
VARIANT=${VARIANTS[${SLURM_ARRAY_TASK_ID}]}
RUN="phase2_${VARIANT}_s1"
EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN="${RUN}_smoke"
  EXTRA+=(--smoke-test)
fi

python - <<'PY'
import torch
assert torch.cuda.is_available(), "No MI300X GPU is visible"
print("Using:", torch.cuda.get_device_name(0), flush=True)
PY

python -u scripts/train_adrenal_segmenter.py \
  --data-root "$DATA_ROOT" --cache-dir "$CACHE_DIR" \
  --model-variant "$VARIANT" \
  --base-checkpoint "$BASE_CKPT" --freeze-base \
  --fixed-neuron-params --snn-readout membrane \
  --seed 1 --run-name "$RUN" \
  --max-epochs 60 --patience 12 --warmup-epochs 3 \
  --lr 1e-3 --event-threshold 0.05 "${EXTRA[@]}"

if [[ "${SMOKE:-0}" != "1" ]]; then
  python -u scripts/evaluate_segmenter.py \
    --checkpoint "runs/$RUN/best_model.pt" \
    --data-root "$DATA_ROOT" --cache-dir "$CACHE_DIR" \
    --output-dir "runs/$RUN/full_volume" \
    --keep-largest 2 --min-component-voxels 30 --device cuda
fi
