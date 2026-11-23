#!/bin/bash -l
#SBATCH --job-name=c0_c2_ensemble
#SBATCH --partition=k2-gpu-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:mi300x:1
#SBATCH --output=c0_c2_ensemble.%j.out
#SBATCH --error=c0_c2_ensemble.%j.err

# INFERENCE ONLY - no training, no tuning. Three cohorts x three predictors:
#
#   cohorts:    dev158  = Dataset702 fold-0 internal validation (development)
#               amos100 = Dataset701 imagesTs/labelsTs (held-out AMOS)
#               ts76    = frozen TotalSegmentator v3 external test (unreviewed)
#   predictors: C0  = Dataset701 3d_fullres fold 0, checkpoint_final
#               C2  = Dataset702 3d_fullres fold 0, checkpoint_final
#               ENS = C0 + C2 probability average, EQUAL weights, fixed here in
#                     advance and never tuned.
#
# Probabilities are saved for every run so that probability-level
# postprocessing can be chosen on dev158 ONLY (scripts/
# sweep_dev_probability_postprocessing.py) and then applied frozen to the
# test cohorts.
#
# Resumable: every stage is skipped when its outputs are already complete, so
# resubmit unchanged after a walltime stop.
#
# DISK: probability .npz files are large (rough guide 100-200 GB in total for
# all 9 runs). Check your quota before submitting.
set -Eeuo pipefail
ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

CACHE_BASE="$ROOT/.hpc_cache/nnunet"
export TRITON_HOME="$CACHE_BASE" TRITON_CACHE_DIR="$CACHE_BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor" XDG_CACHE_HOME="$CACHE_BASE/xdg"
export TORCH_HOME="$CACHE_BASE/torch" TORCH_EXTENSIONS_DIR="$CACHE_BASE/torch_extensions"
export MIOPEN_USER_DB_PATH="$CACHE_BASE/miopen_db" MIOPEN_CUSTOM_CACHE_DIR="$CACHE_BASE/miopen_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" \
         "$TORCH_EXTENSIONS_DIR" "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
export nnUNet_compile=F
export nnUNet_raw="$ROOT/nnunet_raw"
export nnUNet_preprocessed="$ROOT/nnunet_preprocessed"
export nnUNet_results="$ROOT/nnunet_results"

D701="$nnUNet_raw/Dataset701_AMOSAdrenalCT"
D702="$nnUNet_raw/Dataset702_AMOSTotalSegAdrenalCT"
SPLIT="$nnUNet_preprocessed/Dataset702_AMOSTotalSegAdrenalCT/splits_final.json"
EXT_ORIG="$nnUNet_raw/ExternalTotalSegV3AdrenalTest_v1"
EXT_PREP="$nnUNet_raw/ExternalTotalSegV3AdrenalOrtho_v1"
C0_CHK="$nnUNet_results/Dataset701_AMOSAdrenalCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
C2_CHK="$nnUNet_results/Dataset702_AMOSTotalSegAdrenalCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
C2_SHA_EXPECTED=0c9b3008789923d7abb74e7fff90d0cd94cc45be875a07c06bc6b8c6139f93ac
OUT="$ROOT/runs/c0_c2_ensemble"
mkdir -p "$OUT"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
note() { echo "[$(stamp)] $*"; }
note "Git commit: $(git rev-parse HEAD) | SLURM job: ${SLURM_JOB_ID:-manual}"

for f in "$C0_CHK" "$C2_CHK" "$SPLIT" "$D701/labelsTs" "$EXT_PREP/geometry_manifest.json" \
         "$EXT_ORIG/external_test_manifest.json"; do
  [[ -e "$f" ]] || { echo "STOP: missing $f" >&2; exit 2; }
done
C2_SHA=$(sha256sum "$C2_CHK" | cut -d' ' -f1)
[[ "$C2_SHA" == "$C2_SHA_EXPECTED" ]] || { echo "STOP: C2 checkpoint hash $C2_SHA is not the recorded C2 model" >&2; exit 2; }
note "C0 checkpoint sha256 $(sha256sum "$C0_CHK" | cut -d' ' -f1)"
note "C2 checkpoint sha256 $C2_SHA (matches record)"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "AMD ROCm GPU is not accessible"
print("ROCm torch", torch.__version__, "GPU", torch.cuda.get_device_name(0), flush=True)
PY

# ---- dev158 inputs: symlinks to the 158 frozen fold-0 validation cases ------
DEV="$OUT/dev158_inputs"
python - "$SPLIT" "$D702" "$DEV" <<'PY'
import json, sys
from pathlib import Path
split, d702, dev = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
val = json.loads(split.read_text())[0]["val"]
assert len(val) == 158 and all(c.startswith("tsv3_") for c in val), "unexpected fold-0 validation split"
for sub in ("imagesTs", "labelsTs"):
    (dev / sub).mkdir(parents=True, exist_ok=True)
for cid in val:
    for src, dst in ((d702 / "imagesTr" / f"{cid}_0000.nii.gz", dev / "imagesTs" / f"{cid}_0000.nii.gz"),
                     (d702 / "labelsTr" / f"{cid}.nii.gz", dev / "labelsTs" / f"{cid}.nii.gz")):
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.is_symlink() or dst.exists():
            assert dst.resolve() == src.resolve(), f"stale link {dst}"
        else:
            dst.symlink_to(src.resolve())
print("PASS: dev158 inputs linked (158 images, 158 labels).", flush=True)
PY

complete() {  # complete <pred_dir> <input_dir>: every input has a .nii.gz AND a .npz
  local pred="$1" inp="$2" img cid
  for img in "$inp"/*_0000.nii.gz; do
    cid=$(basename "$img" _0000.nii.gz)
    [[ -s "$pred/$cid.nii.gz" && -s "$pred/$cid.npz" ]] || return 1
  done
}

predict() {  # predict <label> <dataset_id> <input_dir> <pred_dir>
  local label="$1" id="$2" inp="$3" pred="$4"
  if complete "$pred" "$inp"; then
    note "$label: predictions complete, skipping"
  else
    note "$label: predicting (nnU-Net default sliding window + mirroring, checkpoint_final)"
    mkdir -p "$pred"
    nnUNetv2_predict -i "$inp" -o "$pred" -d "$id" -c 3d_fullres -f 0 \
      -chk checkpoint_final.pth -device cuda --save_probabilities --continue_prediction
  fi
}

ensemble() {  # ensemble <label> <c0_dir> <c2_dir> <out_dir> <input_dir>
  local label="$1" a="$2" b="$3" o="$4" inp="$5"
  if complete "$o" "$inp"; then
    note "$label: ensemble complete, skipping"
  else
    note "$label: averaging C0 and C2 probabilities with equal weights"
    rm -rf "$o"
    nnUNetv2_ensemble -i "$a" "$b" -o "$o" -np 8 --save_npz
  fi
}

# ---- dev158 (development) ----------------------------------------------------
predict "dev158 C0" 701 "$DEV/imagesTs" "$OUT/dev158_C0"
predict "dev158 C2" 702 "$DEV/imagesTs" "$OUT/dev158_C2"
ensemble "dev158 ENS" "$OUT/dev158_C0" "$OUT/dev158_C2" "$OUT/dev158_ENS" "$DEV/imagesTs"
for M in C0 C2 ENS; do
  python -u scripts/evaluate_nnunet_adrenal.py --dataset-dir "$DEV" \
    --pred-dir "$OUT/dev158_$M" --output "$OUT/metrics_dev158_$M.csv" | tail -1
done

# ---- amos100 (held-out AMOS) -------------------------------------------------
predict "amos100 C0" 701 "$D701/imagesTs" "$OUT/amos100_C0"
predict "amos100 C2" 702 "$D701/imagesTs" "$OUT/amos100_C2"
ensemble "amos100 ENS" "$OUT/amos100_C0" "$OUT/amos100_C2" "$OUT/amos100_ENS" "$D701/imagesTs"
for M in C0 C2 ENS; do
  python -u scripts/evaluate_nnunet_adrenal.py --dataset-dir "$D701" \
    --pred-dir "$OUT/amos100_$M" --output "$OUT/metrics_amos100_$M.csv" | tail -1
done

# ---- ts76 (frozen external, unreviewed labels) -------------------------------
# Predict on the geometry-repaired CTs, then restore to the native CT grid
# with the existing, tested restore step before scoring.
predict "ts76 C0" 701 "$EXT_PREP/imagesTs" "$OUT/ts76_C0"
predict "ts76 C2" 702 "$EXT_PREP/imagesTs" "$OUT/ts76_C2"
ensemble "ts76 ENS" "$OUT/ts76_C0" "$OUT/ts76_C2" "$OUT/ts76_ENS" "$EXT_PREP/imagesTs"
for M in C0 C2 ENS; do
  python -u scripts/repair_totalseg_external_ct_geometry.py restore \
    --original "$EXT_ORIG" --prepared "$EXT_PREP" \
    --predictions "$OUT/ts76_$M" --restored "$OUT/ts76_${M}_native" --resume | tail -1
  python -u scripts/evaluate_totalseg_external_adrenal.py --dataset-dir "$EXT_ORIG" \
    --pred-dir "$OUT/ts76_${M}_native" --output "$OUT/metrics_ts76_$M.csv" | tail -2
done

# ---- one comparison report ---------------------------------------------------
python -u scripts/compare_c0_c2_ensemble.py --runs "$OUT" \
  --labels dev158="$DEV/labelsTs" amos100="$D701/labelsTs" ts76="$EXT_ORIG/labelsTs" \
  | tee "$OUT/comparison_report.txt"
note "SUCCESS: report at $OUT/comparison_report.txt"
