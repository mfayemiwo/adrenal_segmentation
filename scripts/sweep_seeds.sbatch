#!/bin/bash -l
#SBATCH --job-name=adrenal_seeds
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=18:00:00
#SBATCH --output=./out.%j
#SBATCH --error=./err.%j
#SBATCH --mem=96G
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#
# Seed replication: the run that turns the Stage A findings into results.
#
#   sbatch scripts/sweep_seeds.sbatch
#   python scripts/aggregate_seeds.py          # any time, on whatever has finished
#
# WHY
#
# Every Stage A number so far is n=1. Two of the three conclusions are claims
# about DIFFERENCES between configurations, and a difference from one run each
# is not a measurement:
#
#   arm E vs arm G          0.7607 vs 0.7572  (+0.0035) - "the sequence adds nothing"
#   snn_comb vs lstm_comb   0.8407 vs 0.8827  (-0.0420) - "the SNN loses"
#
# The first is a null result, and a null from n=1 is an absence of evidence
# rather than evidence of absence. It needs a variance estimate more than
# anything else in the project.
#
# Five configurations x five seeds = 25 runs, roughly 5 hours.
#
# WHAT THE SEED CHANGES
#
# --seed drives weight initialisation, the negative-slice sampler and the
# augmentation stream. It does NOT change the train/validation split, which is
# fixed by AMOS22's own imagesVa/. So this measures TRAINING variance with the
# data held constant - which is the right quantity for comparing two
# architectures on one cohort. Split variance is a separate question and would
# need cross-validation.

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

COMMON="--data-root ../data/amos22 --cache-dir ../cache --num-workers 16"

for SEED in 1 2 3 4 5; do
  echo "############################## seed $SEED ##############################"

  # arm E - the spiking gate as originally specified
  python -u scripts/train_slice_gate.py $COMMON --model snn --seed $SEED \
      --run-name "seed_snn_s${SEED}"

  # arm G - the static-repeat control. The E-vs-G pair is the sequence claim.
  python -u scripts/train_slice_gate.py $COMMON --model snn --static-repeat --seed $SEED \
      --run-name "seed_snn_static_s${SEED}"

  # the tuned spiking gate
  python -u scripts/train_slice_gate.py $COMMON --model snn --norm group \
      --learn-beta --learn-threshold --surrogate atan --surrogate-slope 2.0 \
      --weight-decay 1e-3 --seed $SEED --run-name "seed_snn_combined_s${SEED}"

  # arm D - the matched baseline
  python -u scripts/train_slice_gate.py $COMMON --model cnn_lstm --lstm-hidden 32 \
      --seed $SEED --run-name "seed_cnnlstm_s${SEED}"

  # the tuned baseline
  python -u scripts/train_slice_gate.py $COMMON --model cnn_lstm --lstm-hidden 32 \
      --norm group --lstm-layers 2 --weight-decay 1e-3 --seed $SEED \
      --run-name "seed_lstm_combined_s${SEED}"
done

echo
python scripts/aggregate_seeds.py
