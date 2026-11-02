#!/bin/bash -l
#SBATCH --job-name=adrenaspike_inputs
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#SBATCH --output=adrenaspike_inputs.%j.out
#SBATCH --error=adrenaspike_inputs.%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate
STAGE="${STAGE:-cache}"
EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]];then EXTRA+=(--max-cases 3);fi
if [[ "$STAGE" == "cache" ]];then
    python -c "import torch; assert torch.cuda.is_available()"
    python -u scripts/cache_adrenaspike3d_inputs.py \
        --data-root "${DATA_ROOT:-../data/amos22}" \
        --checkpoint "${BASE_CKPT:-runs/sb_positives/best_model.pt}" "${EXTRA[@]}"
elif [[ "$STAGE" == "tiles" ]];then
    python -u scripts/prepare_adrenaspike_tiles.py "${EXTRA[@]}"
else
    echo "STAGE must be cache or tiles" >&2; exit 2
fi
