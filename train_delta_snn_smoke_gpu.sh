#!/bin/bash -l
#SBATCH --job-name=delta_snn_smoke
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=delta_smoke.%j.out
#SBATCH --error=delta_smoke.%j.err
#SBATCH --mem=48G
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1

set -euo pipefail

module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0

source .venv/bin/activate

echo "SLURM job: ${SLURM_JOB_ID:-unknown} | Host: $(hostname)"
python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "| ROCm:", torch.version.hip, flush=True)
print("GPU available:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("No GPU available: verify SLURM GPU allocation and ROCm environment.")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
PY

python -u scripts/train_adrenal_segmenter.py \
    --data-root ../data/amos22 \
    --cache-dir ../cache \
    --model-variant delta_snn \
    --base-checkpoint runs/sb_positives/best_model.pt \
    --freeze-base \
    --fixed-neuron-params \
    --seed 1 \
    --run-name delta_snn_smoke_gpu \
    --smoke-test
