#!/bin/bash -l
#SBATCH --job-name=adrenaspike_roi
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#SBATCH --array=0-1
#SBATCH --output=adrenaspike_roi.%A_%a.out
#SBATCH --error=adrenaspike_roi.%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
VARIANTS=(snn analog)
VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"
SEED="${SEED:-1}"
RUN="adrenaspike3d_${VARIANT}_s${SEED}"
STAGE="${STAGE:-train}"
SMOKE="${SMOKE:-0}"
OPTS=()
if [[ "$SMOKE" == "1" ]];then
    RUN="${RUN}_smoke"
    OPTS+=(--smoke-test)
fi
if [[ "$STAGE" == "train" ]];then
    python -u scripts/train_adrenaspike3d.py --variant "$VARIANT" \
        --seed "$SEED" --run-name "$RUN" "${OPTS[@]}"
elif [[ "$STAGE" == "evaluate" ]];then
    if [[ "$SMOKE" == "1" ]];then MAX=(--max-cases 3);else MAX=();fi
    python -u scripts/evaluate_adrenaspike3d.py \
        --checkpoint "runs/$RUN/best_model.pt" \
        --output-dir "runs/$RUN/full_volume" "${MAX[@]}"
else
    echo "STAGE must be train or evaluate" >&2; exit 2
fi
