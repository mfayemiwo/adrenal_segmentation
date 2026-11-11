#!/bin/bash -l
#SBATCH --job-name=tsv3_external_geometry
#SBATCH --partition=k2-medpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=tsv3_external_geometry.%j.out
#SBATCH --error=tsv3_external_geometry.%j.err

# CPU-only, resumable spatial-header preflight and CT-only orthogonalisation.
# Original 76 NIfTI CTs and labels remain untouched. Never reads the adrenal
# masks to prepare the inputs. Supports orthonormal SimpleITK/nnU-Net reader.
set -Eeuo pipefail

ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u scripts/repair_totalseg_external_ct_geometry.py prepare \
  --original nnunet_raw/ExternalTotalSegV3AdrenalTest_v1 \
  --prepared nnunet_raw/ExternalTotalSegV3AdrenalOrtho_v1 \
  --resume

python - <<'PY'
import json
from pathlib import Path
p = Path("nnunet_raw/ExternalTotalSegV3AdrenalOrtho_v1/geometry_manifest.json")
x = json.loads(p.read_text())
assert x["status"] == "COMPLETE" and x["case_count"] == 76
assert len(x["cases"]) == 76
print(f"PASS: full 76-case CT-only geometry preflight; "
      f"{x['regridded_count']} CTs orthogonalised; original labels untouched.", flush=True)
PY

echo "SUCCESS: all 76 CT inputs now suitable for staged nnU-Net inference."
