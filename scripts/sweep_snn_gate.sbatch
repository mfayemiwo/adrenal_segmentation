#!/bin/bash -l
#SBATCH --job-name=adrenal_snn_sweep
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=./out.%j
#SBATCH --error=./err.%j
#SBATCH --mem=96G
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#
# Giving the spiking gate its fairest shot, and testing the sequence claim.
#
#   sbatch scripts/sweep_snn_gate.sbatch
#   tail -f runs/snn_group/progress.txt
#
# WHAT THIS IS AND IS NOT AIMED AT
#
# scripts/probe_order_sensitivity.py established that the architecture CAN use
# slice order - arm E as run scores 99%+ on a task where order is the only
# signal. So arm E == arm G is not a representational defect, and there is no
# point sweeping the readout hoping to "unlock" the sequence.
#
# The problems that remain are ordinary ones: the spiking arms overfit badly
# (train 0.166 / val 0.326, against the CNN-LSTM's 0.185 / 0.193) and lose by
# 0.121 PR-AUC. This sweep targets generalisation and optimisation, which is
# where the evidence points.
#
# Baseline for comparison is the existing runs/gate_snn (PR-AUC 0.761).
# ~13 min per run, so the whole sweep is roughly 2 hours.

set -euo pipefail

module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

python - <<'PY'
import sys, torch
print(f"torch {torch.__version__} | hip {torch.version.hip} | cuda {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("\nNo GPU visible - aborting instead of training on CPU.\n")
print(f"GPU {torch.cuda.get_device_name(0)}")
PY

COMMON="--data-root ../data/amos22 --cache-dir ../cache --num-workers 16 --model snn"

# --- generalisation: remove the BatchNorm-across-time-steps train/eval mismatch
python -u scripts/train_slice_gate.py $COMMON --norm group --run-name snn_group
python -u scripts/train_slice_gate.py $COMMON --norm step  --run-name snn_stepnorm

# --- optimisation: let the network choose its own time constant, and soften
#     the surrogate so gradients survive k time steps
python -u scripts/train_slice_gate.py $COMMON --learn-beta --learn-threshold \
    --run-name snn_learnlif
python -u scripts/train_slice_gate.py $COMMON --surrogate atan --surrogate-slope 2.0 \
    --run-name snn_atan

# --- regularisation: the spiking arms are the ones overfitting
python -u scripts/train_slice_gate.py $COMMON --weight-decay 1e-3 --run-name snn_wd

# --- the analog readout. Included for completeness, NOT because the probe
#     suggests it will help. Note it makes the network a hybrid rather than a
#     pure SNN, which weakens any neuromorphic-energy argument.
python -u scripts/train_slice_gate.py $COMMON --readout integrator --run-name snn_integrator

# --- everything the above justifies, combined
python -u scripts/train_slice_gate.py $COMMON --norm group --learn-beta --learn-threshold \
    --surrogate atan --surrogate-slope 2.0 --weight-decay 1e-3 --run-name snn_combined

# --- the experiment that tests the CLAIM rather than the model. If the
#     craniocaudal sequence carries information, a longer window should help
#     arm E more than arm G, which gains only redundancy from extra steps.
python -u scripts/train_slice_gate.py $COMMON --slice-window 9 --run-name snn_k9
python -u scripts/train_slice_gate.py $COMMON --slice-window 9 --static-repeat \
    --run-name snn_k9_static

echo
echo "=== sweep results (baseline: runs/gate_snn) ==="
python - <<'PY'
import torch
from pathlib import Path
runs = ["gate_snn", "gate_cnnlstm", "gate_snn_static", "snn_group", "snn_stepnorm",
        "snn_learnlif", "snn_atan", "snn_wd", "snn_integrator", "snn_combined",
        "snn_k9", "snn_k9_static"]
print(f"{'run':<18}{'epoch':>6}{'PR-AUC':>9}{'retention':>11}{'removed':>9}")
for r in runs:
    p = Path("runs") / r / "best_model.pt"
    if not p.exists():
        print(f"{r:<18}{'(none)':>6}")
        continue
    c = torch.load(p, map_location="cpu", weights_only=False)
    pr = sum(c["prauc"].values()) / len(c["prauc"])
    print(f"{r:<18}{c['epoch']:>6}{pr:>9.4f}{c['any_retention']:>11.3f}"
          f"{c['slice_reduction']:>9.1%}")
PY
