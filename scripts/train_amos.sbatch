#!/bin/bash -l
#SBATCH --job-name=adrenal_stageB
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=./out.%j
#SBATCH --error=./err.%j
#SBATCH --mem=64G
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#
# Stage B (2.5D segmenter) training on the AMD MI300X partition.
#
#   sbatch scripts/train_amos.sbatch
#   squeue -u $USER
#   tail -f runs/run1/train.log      # live even while out.%j is still buffered
#
# NOTE: there is deliberately no `pip install` here. Set the environment up
# once on the login node (see README "Installation"). Installing inside the job
# wastes GPU allocation, fails on nodes without network, and - because PyPI's
# torch is a CUDA build - can silently replace the ROCm torch with one that
# cannot see the GPU.

set -euo pipefail

module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

# Fail fast rather than spending the whole allocation training on CPU.
python - <<'PY'
import sys
import torch
print(f"python     {sys.executable}")
print(f"torch      {torch.__version__}")
print(f"hip        {torch.version.hip}")
print(f"cuda_avail {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit(
        "\nNo GPU visible to this job - aborting instead of training on CPU.\n"
        "A torch version ending in +cuXXX means the CUDA build was installed over\n"
        "the ROCm one; reinstall per the README with the venv activated.\n"
    )
print(f"GPU        {torch.cuda.get_device_name(0)}")
PY

# -u so stdout reaches out.%j immediately instead of in 4KB blocks.
python -u scripts/train_adrenal_segmenter.py \
    --data-root ../data/amos22 \
    --run-name run1 \
    --num-workers 8

# To continue a run the wall-clock limit cut short, append:
#     --resume runs/run1/last_model.pt
