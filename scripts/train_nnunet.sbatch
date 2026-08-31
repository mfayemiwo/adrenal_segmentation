#!/bin/bash -l
#SBATCH --job-name=adrenal_nnunet
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=./out.%j
#SBATCH --error=./err.%j
#SBATCH --mem=128G
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#
# nnU-Net 3D full-res on AMOS adrenals - the standard anchor, and a diagnostic.
#
# If it reaches ~0.85 while our 2.5D pipeline sits at 0.746, the deficit is in
# our pipeline and we know how much is recoverable. If it lands near 0.78, the
# published figures are what needs explaining. Either way the answer changes
# what we do next, which is why this is worth a day.
#
# BEFORE SUBMITTING - one-off setup on the login node, with the venv active:
#
#   1. Install nnU-Net WITHOUT letting it touch the ROCm torch:
#        python -m pip install nnunetv2
#        python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#      That MUST still print 2.6.0+rocm6.3.3 and True. nnunetv2 requires
#      torch>=2.1, which the ROCm build already satisfies, so pip should leave
#      it alone - but verify, because a silent swap to a +cuXXX wheel costs a
#      whole allocation before the symptom shows. If it did swap:
#        python -m pip --no-cache-dir install --force-reinstall \
#            torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
#            -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.3/
#
#   2. Build the dataset (a few minutes; images are symlinked, labels rewritten):
#        python scripts/prepare_nnunet_dataset.py \
#            --data-root ../data/amos22 --out ../nnunet/nnUNet_raw
#
#      Read its summary. If it warns that cases have no adrenal voxels, stop -
#      the label ids are wrong and everything downstream will be silently empty.

set -euo pipefail

module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

# nnU-Net is configured entirely through these three directories.
export nnUNet_raw="$(cd .. && pwd)/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$(cd .. && pwd)/nnunet/nnUNet_preprocessed"
export nnUNet_results="$(cd .. && pwd)/nnunet/nnUNet_results"
mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"

DATASET_ID=501
CONFIG=3d_fullres
FOLD=0                 # single fold. Set to "0 1 2 3 4" for the full ensemble
                       # (5x the time; do that only once fold 0 looks sane).

python - <<'PY'
import sys, torch
print(f"torch {torch.__version__} | hip {torch.version.hip} | cuda {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("\nNo GPU visible - aborting instead of training on CPU.\n")
if "rocm" not in torch.__version__:
    sys.exit(f"\ntorch is {torch.__version__}, not a ROCm build - installing nnunetv2 "
             "replaced it. Reinstall per the header of this script.\n")
print(f"GPU {torch.cuda.get_device_name(0)}")
PY

echo "=== planning and preprocessing ==="
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity

echo "=== training ($CONFIG, fold $FOLD) ==="
for F in $FOLD; do
    nnUNetv2_train $DATASET_ID $CONFIG $F
done

echo "=== predicting on the held-out split ==="
RAW="$nnUNet_raw/Dataset$(printf %03d $DATASET_ID)_AdrenalAMOS"
PRED="$(cd .. && pwd)/nnunet/predictions"
mkdir -p "$PRED"
nnUNetv2_predict -i "$RAW/imagesTs" -o "$PRED" \
    -d $DATASET_ID -c $CONFIG -f $FOLD

echo "=== scoring with THIS project's metric ==="
# Deliberately not nnUNetv2_evaluate_folder: it reports Dice its own way, and
# the entire point of this run is a comparison that is like-for-like.
python scripts/evaluate_nnunet.py --predictions "$PRED" --labels "$RAW/labelsTs"
