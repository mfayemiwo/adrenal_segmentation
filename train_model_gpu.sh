#!/bin/bash -l

#SBATCH --job-name=LLAMA_Min
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00

###SBATCH --time=72:00:00

#SBATCH --output=./out.%j
#SBATCH --error=./err.%j
#SBATCH --mem=20G

###SBATCH -p k2-gpu-a100
###SBATCH -p k2-gpu-h100
###SBATCH -p k2-gpu-a100mig

#SBATCH --partition=k2-gpu
#SBATCH --gres=gpu:mi300x:1

###SBATCH --gres gpu:a100:1
###SBATCH --gres gpu:h100:1
###SBATCH --gres gpu:h100:1
###SBATCH --gres gpu:2g.20gb:1
###SBATCH --gres gpu:3g.40gb:1

###SBATCH --gres=gpu:v100:1
###SBATCH --gres=gpu:i1100:1
###SBATCH --gres=gpu:mi300x:1
 
###module load libs/nvidia-cuda/12.8.0/bin

module load amd-rocm/rocm-6.3.3
 
module load apps/python3/3.12.4/gcc-14.1.0
 
module load compilers/gcc/14.1.0

module load libs/gcc/14.1.0

source .venv/bin/activate

### module load amd-rocm/rocm-6.3.3
### module load intel/oneapi/hpc-toolkit/2025.0.0/gcc-14.1.0

###module load python3/3.10.5/gcc-9.3.0
###export PYTHONUSERBASE=/mnt/scratch2/users/$USER/IntelGPU/gridware

### export PATH=/mnt/scratch2/users/jsanchez/IntelGPU/gridware/bin:$PATH
### export TMPDIR=/tmp/users/$USER

python train_adrenal_segmenter.py


