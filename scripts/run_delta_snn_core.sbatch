#!/bin/bash -l
#SBATCH --job-name=adrenal_delta_snn
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=./out.%A_%a
#SBATCH --error=./err.%A_%a
#SBATCH --mem=96G
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#SBATCH --array=0-14

# Core publication experiment: 3 variants x 5 training seeds.
#
# The BASE_CKPT should be the strongest defensible Stage-B baseline selected
# BEFORE these refiner experiments. The trainer inherits its preprocessing,
# geometry, sampling protocol and patient split automatically.
#
# Example:
#   sbatch --export=ALL,BASE_CKPT=runs/sb_positives/best_model.pt \
#       scripts/run_delta_snn_core.sbatch
#
# Arms:
#   delta_snn    proposed: inter-slice delta events + LIF temporal refinement
#   raw_snn      control: raw slice stream + same LIF temporal refinement
#   delta_analog control: delta events + continuous leaky temporal refinement
#
# The baseline itself is not retrained in this array. Every arm starts from the
# exact same baseline checkpoint and freezes it. This asks a clean question:
# can a small temporal residual module correct errors left by the baseline?

set -euo pipefail

module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

: "${BASE_CKPT:?Set BASE_CKPT to the selected baseline best_model.pt}"
DATA_ROOT="${DATA_ROOT:-../data/amos22}"
CACHE_DIR="${CACHE_DIR:-../cache}"

python - <<'PY'
import sys, torch
print(f"torch {torch.__version__} | hip {torch.version.hip} | cuda {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("No GPU visible - aborting instead of training on CPU.")
print(f"GPU {torch.cuda.get_device_name(0)}")
PY

VARIANTS=(delta_snn raw_snn delta_analog)
SEEDS=(1 2 3 4 5)

TASK=${SLURM_ARRAY_TASK_ID}
VINDEX=$(( TASK / 5 ))
SINDEX=$(( TASK % 5 ))
VARIANT=${VARIANTS[$VINDEX]}
SEED=${SEEDS[$SINDEX]}
RUN="delta_refiner_${VARIANT}_s${SEED}"

echo "======================================================================"
echo "run       $RUN"
echo "variant   $VARIANT"
echo "seed      $SEED"
echo "baseline  $BASE_CKPT"
echo "======================================================================"

python -u scripts/train_adrenal_segmenter.py \
    --data-root "$DATA_ROOT" \
    --cache-dir "$CACHE_DIR" \
    --model-variant "$VARIANT" \
    --base-checkpoint "$BASE_CKPT" \
    --freeze-base \
    --seed "$SEED" \
    --run-name "$RUN" \
    --max-epochs 80 \
    --patience 15 \
    --warmup-epochs 3 \
    --lr 1e-3 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --event-threshold 0.05 \
    --fixed-neuron-params \
    --snn-readout membrane

python -u scripts/evaluate_segmenter.py \
    --checkpoint "runs/$RUN/best_model.pt" \
    --data-root "$DATA_ROOT" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "runs/$RUN/full_volume" \
    --keep-largest 2 \
    --min-component-voxels 30

echo
echo "Finished $RUN"
echo "  training:    runs/$RUN/metrics.csv"
echo "  full volume: runs/$RUN/full_volume/per_case.csv"
echo "  report:      runs/$RUN/full_volume/evaluation_report.txt"
