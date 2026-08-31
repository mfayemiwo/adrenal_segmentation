#!/bin/bash -l
#SBATCH --job-name=adrenal_lstm_sweep
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
# The matched sweep for the CNN-LSTM baseline.
#
#   sbatch scripts/sweep_cnnlstm_gate.sbatch
#   tail -f runs/lstm_group/progress.txt
#
# WHY THIS EXISTS
#
# After scripts/sweep_snn_gate.sbatch the spiking gate had been given nine
# configurations and this baseline exactly one. That flipped the comparison
# from being biased AGAINST the SNN (the capacity confound, fixed earlier) to
# being biased in its favour: the SNN's 0.761 -> 0.841 would then be measured
# against an untuned opponent, and a difference in results would be a
# difference in tuning effort rather than in architecture.
#
# This gives the baseline the SAME BUDGET - nine configurations, mirroring the
# nine the SNN received, one knob at a time and then the justified combination.
# Both models keep capacity matched to within 2% in every configuration
# (train_slice_gate.py prints both counts and warns otherwise).
#
# The pairing, SNN arm -> baseline arm:
#   snn_group      -> lstm_group        same normaliser, same reason
#   snn_stepnorm   -> lstm_stepnorm     same
#   snn_learnlif   -> lstm_layers2      more recurrent capacity
#   snn_integrator -> lstm_meanread     order-preserving vs order-reducing readout
#   snn_atan       -> lstm_lr           an optimisation knob with no direct analogue
#   snn_wd         -> lstm_wd           identical
#   snn_combined   -> lstm_combined     the justified combination
#   snn_k9         -> lstm_k9           does more context help the baseline too?
#   snn_k9_static  -> lstm_k9_static    and does the baseline gain from true order
#                                       where the SNN does not? That comparison is
#                                       interesting in its own right.
#
# ~9 min per run for the baseline, so roughly 1.5 hours.

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

# --lstm-hidden 32 keeps parameter counts matched to the spiking gate. At the
# default 128 the baseline carries 46% more and no result is interpretable.
COMMON="--data-root ../data/amos22 --cache-dir ../cache --num-workers 16 \
        --model cnn_lstm --lstm-hidden 32"

# --- normalisation: the change that gave the SNN most of its gain
python -u scripts/train_slice_gate.py $COMMON --norm group --run-name lstm_group
python -u scripts/train_slice_gate.py $COMMON --norm step  --run-name lstm_stepnorm

# --- recurrent capacity, and the readout
python -u scripts/train_slice_gate.py $COMMON --lstm-layers 2 --run-name lstm_layers2
python -u scripts/train_slice_gate.py $COMMON --lstm-readout mean --run-name lstm_meanread

# --- optimisation and regularisation
python -u scripts/train_slice_gate.py $COMMON --lr 1.5e-4 --run-name lstm_lr
python -u scripts/train_slice_gate.py $COMMON --weight-decay 1e-3 --run-name lstm_wd

# --- the justified combination
python -u scripts/train_slice_gate.py $COMMON --norm group --lstm-layers 2 \
    --weight-decay 1e-3 --run-name lstm_combined

# --- longer window, and its control
python -u scripts/train_slice_gate.py $COMMON --slice-window 9 --run-name lstm_k9
python -u scripts/train_slice_gate.py $COMMON --slice-window 9 --static-repeat \
    --run-name lstm_k9_static

echo
echo "=== BOTH SWEEPS, SIDE BY SIDE ==="
python - <<'PY'
import torch
from pathlib import Path

GROUPS = [
    ("baselines", ["gate_snn", "gate_cnnlstm", "gate_snn_static"]),
    ("snn sweep", ["snn_group", "snn_stepnorm", "snn_learnlif", "snn_atan", "snn_wd",
                   "snn_integrator", "snn_combined", "snn_k9", "snn_k9_static"]),
    ("cnn_lstm sweep", ["lstm_group", "lstm_stepnorm", "lstm_layers2", "lstm_meanread",
                        "lstm_lr", "lstm_wd", "lstm_combined", "lstm_k9", "lstm_k9_static"]),
]

def row(name):
    p = Path("runs") / name / "best_model.pt"
    if not p.exists():
        return f"{name:<18}{'(no checkpoint)':>40}"
    c = torch.load(p, map_location="cpu", weights_only=False)
    pr = sum(c["prauc"].values()) / len(c["prauc"])
    return (f"{name:<18}{c['epoch']:>6}{pr:>10.4f}{c['any_retention']:>11.3f}"
            f"{c['slice_reduction']:>10.1%}")

print(f"{'run':<18}{'epoch':>6}{'PR-AUC':>10}{'retention':>11}{'removed':>10}")
for title, names in GROUPS:
    print(f"\n-- {title} " + "-" * (55 - len(title)))
    for n in names:
        print(row(n))

print("\nRead the two sweeps against each other, not the best of one against the")
print("single baseline run. Each model has now had nine attempts; the fair")
print("comparison is best-of-nine against best-of-nine, and both figures are")
print("optimistic by an unknown amount because the selection used validation data.")
PY
