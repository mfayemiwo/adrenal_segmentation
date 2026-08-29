#!/bin/bash -l

#SBATCH --job-name=adrenal_stageB
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=70:00:00
#SBATCH --output=./out.%j
#SBATCH --error=./err.%j
#SBATCH --mem=96G
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1


module load amd-rocm/rocm-6.3.3
 
module load apps/python3/3.12.4/gcc-14.1.0
 
module load compilers/gcc/14.1.0

module load libs/gcc/14.1.0

source .venv/bin/activate

python -u scripts/train_adrenal_segmenter.py \
    --data-root ../data/amos22 --cache-dir ../cache \
    --run-name run5 --batch-size 24 --lr 2e-4 --warmup-epochs 8 --num-workers 16

