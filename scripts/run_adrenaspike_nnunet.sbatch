#!/bin/bash -l
#SBATCH --job-name=adrenaspike3d
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#SBATCH --output=adrenaspike3d.%j.out
#SBATCH --error=adrenaspike3d.%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate
export nnUNet_raw="${NNUNET_RAW:-$PWD/nnunet_raw}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED:-$PWD/nnunet_preprocessed}"
export nnUNet_results="${NNUNET_RESULTS:-$PWD/nnunet_results}"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
STAGE="${NN_STAGE:-prepare}"
ID="${DATASET_ID:-701}"
FOLD="${FOLD:-0}"
DATA_ROOT="${DATA_ROOT:-../data/amos22}"
DATASET="${nnUNet_raw}/Dataset$(printf '%03d' "$ID")_AMOSAdrenalCT"

echo "Stage=$STAGE Dataset=$ID Fold=$FOLD"
if [[ "$STAGE" == "prepare" ]]; then
    python scripts/prepare_nnunet_adrenal.py --data-root "$DATA_ROOT" \
        --output-root "$nnUNet_raw" --dataset-id "$ID"
elif [[ "$STAGE" == "preprocess" ]]; then
    command -v nnUNetv2_plan_and_preprocess >/dev/null || {
        echo "nnU-Net v2 missing in .venv; install nnunetv2 without replacing ROCm torch" >&2; exit 2;
    }
    nnUNetv2_plan_and_preprocess -d "$ID" --verify_dataset_integrity -c 3d_fullres
elif [[ "$STAGE" == "train" ]]; then
    command -v nnUNetv2_train >/dev/null || { echo "nnU-Net v2 is not installed"; exit 2; }
    python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
    if [[ "${RESUME:-0}" == "1" ]]; then
        nnUNetv2_train "$ID" 3d_fullres "$FOLD" --npz -device cuda --c
    else
        nnUNetv2_train "$ID" 3d_fullres "$FOLD" --npz -device cuda
    fi
elif [[ "$STAGE" == "predict" ]]; then
    command -v nnUNetv2_predict >/dev/null || { echo "nnU-Net v2 is not installed"; exit 2; }
    python -c "import torch; assert torch.cuda.is_available()"
    nnUNetv2_predict -i "$DATASET/imagesTs" \
        -o "runs/nnunet3d_holdout_predictions" -d "$ID" -c 3d_fullres \
        -f "$FOLD" --save_probabilities -device cuda
else
    echo "Unknown NN_STAGE: $STAGE; choose prepare, preprocess, train, predict" >&2
    exit 2
fi
