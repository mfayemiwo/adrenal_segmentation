#!/bin/bash -l
#SBATCH --job-name=adrenal_energy_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#SBATCH --output=adrenal_energy_probe.%j.out
#SBATCH --error=adrenal_energy_probe.%j.err

# Short, SAFE GPU telemetry preflight; does not load model or rerun inference.
# A successful probe reports working GPU power/energy telemetry and a
# trustworthy mapping between HIP logical GPU 0 and physical AMD-SMI GPU.
set -Eeuo pipefail

ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

echo "===== GPU TELEMETRY PREFLIGHT ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname)"
echo "PYTHON=$(command -v python)"
echo "ROCM_PATH=${ROCM_PATH:-unset} ROCM_HOME=${ROCM_HOME:-unset}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "amd-smi binary: $(command -v amd-smi || echo absent)"
if [[ -n "${ROCM_PATH:-}" ]]; then
  ls -ld "$ROCM_PATH/share/amd_smi" "$ROCM_PATH/share/amd_smi/amdsmi" 2>/dev/null || true
fi

echo "===== ROCm DEVICE ====="
python - <<'PY'
import torch
print("torch", torch.__version__, "HIP", torch.version.hip, flush=True)
if not torch.cuda.is_available():
    raise SystemExit("No MI300X visible in SLURM allocation")
print("GPU 0:", torch.cuda.get_device_name(0), flush=True)
PY

echo "===== AMD-SMI BINDING, POWER, ENERGY AND BDF ====="
python -u scripts/measure_gpu_energy.py \
  --probe --output-dir runs/energy_telemetry_preflight

echo "SUCCESS: AMD-SMI binding and telemetry available; now plan controlled GPU inference energy trials."
