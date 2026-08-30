#!/bin/bash -l
#SBATCH --job-name=adrenal_stageA
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=./out.%j
#SBATCH --error=./err.%j
#SBATCH --mem=96G
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#
# Stage A (slice gate) - all three ablation arms, one after another.
#
#   sbatch scripts/train_slice_gate.sbatch
#   tail -f runs/gate_snn/progress.txt
#
# The arms differ by ONE flag each. Everything else - sampler, augmentation,
# schedule, seed, metric - is identical, which is the whole point: the paper's
# claim is that the spiking gate beats the recurrent baseline given the same
# windows, and that is only measurable if nothing else varies.
#
# As with Stage B there is deliberately no `pip install` here: compute nodes
# often have no network, it burns GPU allocation, and PyPI's torch is a CUDA
# build that silently replaces the ROCm one.

set -euo pipefail

module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

python - <<'PY'
import sys
import torch
print(f"torch      {torch.__version__}")
print(f"hip        {torch.version.hip}")
print(f"cuda_avail {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("\nNo GPU visible - aborting instead of training on CPU.\n")
print(f"GPU        {torch.cuda.get_device_name(0)}")
PY

COMMON="--data-root ../data/amos22 --cache-dir ../cache --num-workers 16"

# Arm E - the contribution: k consecutive slices as k SNN time steps.
python -u scripts/train_slice_gate.py $COMMON --model snn --run-name gate_snn

# Arm D - the mandatory baseline. --lstm-hidden 32 matches parameter counts to
# within ~2%; at the default 128 the LSTM carries ~46% more parameters and any
# result is confounded by capacity.
python -u scripts/train_slice_gate.py $COMMON --model cnn_lstm --lstm-hidden 32 \
    --run-name gate_cnnlstm

# Arm G - the control that tests the sequence claim itself: same spiking model,
# same capacity, but the centre slice repeated k times instead of k different
# slices. If arm E does not beat this, the temporal axis is carrying no
# anatomical information and the framing must change.
python -u scripts/train_slice_gate.py $COMMON --model snn --static-repeat \
    --run-name gate_snn_static

echo
echo "=== slice reduction at target retention (the number that matters) ==="
for r in gate_snn gate_cnnlstm gate_snn_static; do
    python - "$r" <<'PY'
import sys, torch
from pathlib import Path
run = sys.argv[1]
p = Path("runs") / run / "best_model.pt"
if not p.exists():
    print(f"{run:<18} (no checkpoint)"); raise SystemExit
c = torch.load(p, map_location="cpu", weights_only=False)
print(f"{run:<18} epoch {c['epoch']:>3}  PR-AUC L {c['prauc']['left_present']:.4f} "
      f"R {c['prauc']['right_present']:.4f}  retention {c['any_retention']:.3f}  "
      f"slices removed {c['slice_reduction']:.1%}")
PY
done
