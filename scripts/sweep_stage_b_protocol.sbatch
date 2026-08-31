#!/bin/bash -l
#SBATCH --job-name=adrenal_stageB_protocol
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
# Every identified protocol difference from the 2025 pipeline, one at a time.
#
#   sbatch scripts/sweep_stage_b_protocol.sbatch
#   tail -f runs/sb_positives/progress.txt
#
# WHY ONE AT A TIME
#
# Our 0.748 / 0.747 is measured against the same dataset with the same metric,
# so the deficit is real. Five things differ between the two pipelines. Changing
# several together would improve or fail to improve the number without saying
# which change was responsible, which is the position we are already in.
#
#   sb_baseline    a clean same-session reference. run5 was trained under an
#                  older code path, so it is not a safe control for these.
#   sb_positives   train on gland-bearing slices only. Their paper used 1,749
#                  of 26,069 slices and reports that gland-free slices "did not
#                  enhance the training process". We train at 0.7 negatives per
#                  positive. LARGEST identified difference.
#   sb_hu          HU window (10, 60) instead of (-135, 215).
#   sb_clahe       CLAHE on top of the tight window - their full preprocessing.
#   sb_iou         Jaccard loss instead of Dice. IoU = Dice/(2-Dice), so IoU
#                  penalises false positives more sharply, which is not
#                  obviously irrelevant at ~0.3% foreground.
#   sb_scratch     no ImageNet initialisation. The reference paper never states
#                  whether its backbones were pretrained, and our own claim that
#                  scratch "does not work" has never actually been measured on
#                  this pipeline. Cheap to settle.
#
# CACHE COST. --hu-window and --clahe change the cache key, so sb_hu and
# sb_clahe each rebuild the volume cache from the NIfTIs before training. Budget
# an extra hour or so for sb_clahe in particular: adaptive histogram
# equalisation runs per slice over the whole volume. The runs are ordered so
# that everything sharing the existing cache finishes first - if the job is cut
# short you lose the expensive arms, not the cheap ones.
#
# TTA is not here. It needs no training: it is a flag on evaluate_segmenter.py
# against the existing run5 checkpoint, and the tail of this script runs it.

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

# ---- arms that reuse the existing cache (fast) ----
python -u scripts/train_adrenal_segmenter.py $COMMON --run-name sb_baseline
python -u scripts/train_adrenal_segmenter.py $COMMON --negative-ratio 0 --run-name sb_positives
python -u scripts/train_adrenal_segmenter.py $COMMON --loss iou_focal  --run-name sb_iou
python -u scripts/train_adrenal_segmenter.py $COMMON --encoder-weights none --run-name sb_scratch

# ---- arms that rebuild the cache (slow; keep last) ----
python -u scripts/train_adrenal_segmenter.py $COMMON --hu-window 10 60 --run-name sb_hu
python -u scripts/train_adrenal_segmenter.py $COMMON --hu-window 10 60 --clahe --run-name sb_clahe

echo
echo "=== full-volume evaluation of every arm (mean per-case Dice) ==="
for R in sb_baseline sb_positives sb_iou sb_scratch sb_hu sb_clahe; do
    if [ -f "runs/$R/best_model.pt" ]; then
        echo "--- $R ---"
        python scripts/evaluate_segmenter.py --checkpoint "runs/$R/best_model.pt" \
            --data-root ../data/amos22 --cache-dir ../cache \
            --keep-largest 2 --output-dir "runs/$R/eval" 2>&1 | grep -E "^ (raw|postprocessed)"
    fi
done

echo
echo "=== test-time augmentation on run5 (no training required) ==="
for T in rot flip all; do
    echo "--- tta=$T ---"
    python scripts/evaluate_segmenter.py --checkpoint runs/run5/best_model.pt \
        --data-root ../data/amos22 --cache-dir ../cache --tta $T --keep-largest 2 \
        --output-dir "runs/run5/eval_tta_$T" 2>&1 | grep -E "^ (raw|postprocessed)"
done

echo
echo "Reference: run5 = 0.746 / 0.745 raw, 0.748 / 0.747 post-processed."
echo "2025 paper on AMOS: TransUNet 0.88 / 0.83 raw, nnU-Net (2D) 0.85 / 0.82."
