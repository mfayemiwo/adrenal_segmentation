#!/bin/bash -l
#SBATCH --job-name=dataset702_preprocess
#SBATCH --partition=k2-medpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=dataset702_preprocess.%j.out
#SBATCH --error=dataset702_preprocess.%j.err

# CPU-only, TRUE checkpointed and resumable Dataset702 nnU-Net 3d_fullres
# preprocessor. Unlike the standard nnUNetv2_preprocess/plan_and_preprocess
# CLI, our casewise wrapper does NOT delete previously preprocessed cases.
# The installed nnU-Net run_case_save is still used unchanged. All files are
# under /mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation.
# If exit code is 3, this job checkpointed safely before its 24h time limit.
# Resubmit THE SAME script; completed cases are not recomputed or erased.
set -Eeuo pipefail
ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
START_EPOCH="$(date +%s)"
# Hold back four hours from the 24h allocation as a safety margin.
# This deadline includes fingerprinting/planning, not just per-case images.
DEADLINE_EPOCH="$((START_EPOCH + 20 * 3600))"

module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

export nnUNet_raw="$ROOT/nnunet_raw"
export nnUNet_preprocessed="$ROOT/nnunet_preprocessed"
export nnUNet_results="$ROOT/nnunet_results"
export nnUNet_compile=F

# Let nnU-Net's 4 simultaneous full-3D case workers each have CPU resources;
# avoid 4 x 16 nested BLAS/ITK threads and large memory amplification.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

DATASET="$nnUNet_raw/Dataset702_AMOSTotalSegAdrenalCT"
PREPROC="$nnUNet_preprocessed/Dataset702_AMOSTotalSegAdrenalCT"
[[ -s "$DATASET/dataset.json" && -s "$DATASET/geometry_manifest.json" && \
   -s "$ROOT/runs/expanded_dataset702/plan.json" ]] || {
   echo "STOP: frozen Dataset702 is not fully prepared." >&2; exit 2;
}

for command in nnUNetv2_plan_and_preprocess python; do
  command -v "$command" >/dev/null || {
    echo "STOP: missing command $command" >&2; exit 2;
  }
done
echo "Dataset702 CPU preprocessing job: ${SLURM_JOB_ID:-manual}"
echo "Git code version: $(git rev-parse HEAD)"
echo "Project root: $ROOT"
df -h "$ROOT"
echo "Configured 24h SLURM limit; checkpoint deadline before walltime."
echo "CPU allocation: 16 cores, 192G RAM. Per-case workers: ${NN702_WORKERS:-4}."
echo "Do not request MI300X GPU for this unchanged CPU nnU-Net preprocessing path."

# Stage 1: Fingerprint + planning only. **Never** invoke nnUNetv2_preprocess
# or nnUNetv2_plan_and_preprocess without --no_pp on an in-progress dataset:
# official DefaultPreprocessor.run() would delete all prior case outputs.
if [[ -s "$PREPROC/dataset_fingerprint.json" && -s "$PREPROC/nnUNetPlans.json" && \
      -s "$PREPROC/dataset.json" ]]; then
  echo "Stage 1/4: fingerprint/plans already present; preserving unchanged."
else
  echo "Stage 1/4: first-time fingerprint and 3D fullres plan; NO preprocessing."
  nnUNetv2_plan_and_preprocess \
    -d 702 -c 3d_fullres --verify_dataset_integrity \
    --no_pp --no_pbar -npfp 4
  [[ -s "$PREPROC/dataset_fingerprint.json" && \
     -s "$PREPROC/nnUNetPlans.json" && -s "$PREPROC/dataset.json" ]] || {
    echo "STOP: dataset plan/fingerprint incomplete. Rerun same job." >&2
    exit 3
  }
fi

# Stage 2: Process missing case files only. A staged case is considered
# complete ONLY after data, segmentation and its .pkl metadata are committed.
# If killed, remaining cases are detected on next submission. Status 3 is
# expected for an allocation that ended before the entire cohort was done.
echo "Stage 2/4: resumable nnU-Net installed-preprocessor case execution."
python -u scripts/resume_nnunet702_preprocessing.py \
  --nnunet-raw nnunet_raw \
  --nnunet-preprocessed nnunet_preprocessed \
  --frozen-plan runs/expanded_dataset702/plan.json \
  --workers "${NN702_WORKERS:-4}" \
  --batch-factor 2 \
  --stop-at-epoch "$DEADLINE_EPOCH"

# Stage 3: Original GT backup, identical labels and source partitions.
echo "Stage 3/4: verify and atomically copy original full CT gland labels."
python -u scripts/finalize_nnunet702_preprocessing.py \
  --nnunet-raw nnunet_raw \
  --nnunet-preprocessed nnunet_preprocessed \
  --frozen-plan runs/expanded_dataset702/plan.json

# Newer nnU-Net v2 stores foreground sampling locations separately.
# Older installations can store equivalent sampling positions in the per-case
# .pkl metadata. Never run full nnUNet preprocessing here.
if command -v nnUNetv2_extract_sampling_locations >/dev/null; then
  echo "Stage 3b: nnU-Net foreground sampling-location store (segmentations only)."
  nnUNetv2_extract_sampling_locations -d 702 -c 3d_fullres -np 4 --no_pbar
else
  echo "Sampling-location CLI not installed; checking legacy nnU-Net metadata."
  python - "$PREPROC" <<'PY'
import pickle,sys
from pathlib import Path
folder=Path(sys.argv[1])/"nnUNetPlans_3d_fullres"
pkls=sorted(folder.glob("*.pkl"))
if len(pkls)!=1253:
    raise SystemExit("Missing per-case properties for legacy nnU-Net")
for p in pkls:
    with p.open("rb") as fd:
        d=pickle.load(fd)
    if "class_locations" not in d:
        raise SystemExit(
            "Installed nnU-Net has neither foreground-location CLI nor "
            "legacy class_locations in pkl; cannot safely train."
        )
print("PASS: all 1253 legacy case properties contain class_locations.",flush=True)
PY
fi

# Stage 4: Frozen validation split; existing conflicting fold never overwritten.
echo "Stage 4/4: install immutable Dataset702 train/val split."
python -u scripts/install_dataset702_frozen_split.py \
  --plan runs/expanded_dataset702/plan.json \
  --nnunet-raw nnunet_raw \
  --nnunet-preprocessed nnunet_preprocessed

echo "SUCCESS: all 1253 Dataset702 fullres cases preprocessed, original GT copied,"
echo "foreground sampling locations ready, and 1095/158 fold0 installed."
echo "NEXT: train_expanded_amos_tsv3_dataset702.sbatch (separate MI300X job)."
