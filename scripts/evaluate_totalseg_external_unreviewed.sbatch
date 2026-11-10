#!/bin/bash -l
#SBATCH --job-name=nnunet_ext_unreviewed
#SBATCH --partition=k2-gpu-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:mi300x:1
#SBATCH --output=nnunet_ext_unreviewed.%j.out
#SBATCH --error=nnunet_ext_unreviewed.%j.err

# EXPLORATORY / UNREVIEWED external test: 76 original TotalSegmentator v3
# OFFICIAL TEST CTs; no manual annotation QC is required or implied.
# Original AMOS-trained nnU-Net 3d_fullres fold0; no retraining or model selection.
# Results are not an externally clinically validated benchmark and cannot be
# used for selecting/refining architecture, thresholds, postprocessing or SNN.
# The stricter reviewed script remains unchanged.
# This script checks the frozen dataset manifest and saves explicit provenance.
set -Eeuo pipefail
ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"

module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

CACHE_BASE="$ROOT/.hpc_cache/nnunet"
mkdir -p "$CACHE_BASE"
export TRITON_HOME="$CACHE_BASE"
export TRITON_CACHE_DIR="$CACHE_BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor"
export XDG_CACHE_HOME="$CACHE_BASE/xdg"
export TORCH_HOME="$CACHE_BASE/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_BASE/torch_extensions"
export MIOPEN_USER_DB_PATH="$CACHE_BASE/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_BASE/miopen_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" \
         "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
         "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
export nnUNet_compile=F
export nnUNet_raw="$ROOT/nnunet_raw"
export nnUNet_preprocessed="$ROOT/nnunet_preprocessed"
export nnUNet_results="$ROOT/nnunet_results"

DATA="$ROOT/nnunet_raw/ExternalTotalSegV3AdrenalTest_v1"
PRED="$ROOT/runs/nnunet3d_external_tsv3_predictions_fold0_unreviewed"
METRICS="$ROOT/runs/nnunet3d_external_tsv3_metrics_fold0_unreviewed.csv"
MODEL="$nnUNet_results/Dataset701_AMOSAdrenalCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

[[ -s "$MODEL" && -s "$DATA/external_test_manifest.json" ]] || {
  echo "ERROR: existing AMOS-trained fold-0 weights or frozen external manifest missing" >&2
  exit 2
}
python - "$DATA/external_test_manifest.json" <<'PY'
import json,sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text())
if len(manifest.get("case_ids",[]))!=76 or manifest.get("source_partition")!="test":
    raise SystemExit("External cohort is not the expected 76 official test cases")
print("Frozen 76-case official external test cohort loaded.", flush=True)
PY

# Pre-inference data-integrity checks: the 76 original case IDs, official
# test split, fixed left/right mapping, source-inventory SHA-256 and symlinks.
# The provenance JSON is kept locally with the experiment outputs, and
# is intentionally NOT an attestation of annotation quality.
python - "$DATA" "$MODEL" "$ROOT/runs/nnunet3d_external_tsv3_unreviewed_provenance.json" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, subprocess, sys

data, checkpoint, target = map(Path, sys.argv[1:])
manifest_path = data / "external_test_manifest.json"
m = json.loads(manifest_path.read_text())
ids = m.get("case_ids", [])
assert m.get("source_partition") == "test", "Only official test cases allowed"
assert len(ids) == 76 and len(set(ids)) == 76, "Expected 76 unique external test cases"
assert m.get("qc_status") == "PRELIMINARY_LABEL_QUALITY_REVIEW_PENDING", "Changed QC provenance"
assert m.get("label_mapping") == {
    "0": "background", "1": "left_adrenal", "2": "right_adrenal"
}, "Unexpected gland label mapping"
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fd:
        for chunk in iter(lambda: fd.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()
inventory = Path("runs/totalsegmentator_v3_qc_plan/external_test_bilateral_inventory.csv")
assert inventory.is_file(), "External inventory missing"
assert sha(inventory) == m.get("source_inventory_sha256"), "Source inventory changed"
assert sorted(p.name for p in (data / "labelsTs").glob("*.nii.gz")) == sorted(
    cid + ".nii.gz" for cid in ids), "External label inventory mismatch"
assert sorted(p.name for p in (data / "imagesTs").glob("*_0000.nii.gz")) == sorted(
    cid + "_0000.nii.gz" for cid in ids), "External CT inventory mismatch"
source = Path(m["source_dataset_root"]).resolve()
for cid in ids:
    image = data / "imagesTs" / (cid + "_0000.nii.gz")
    label = data / "labelsTs" / (cid + ".nii.gz")
    assert image.is_symlink() and image.is_file() and label.is_file(), f"Missing CT/label {cid}"
    assert source in image.resolve().parents, f"CT link escapes dataset {cid}"
try:
    commit = subprocess.check_output(["git","rev-parse","HEAD"], text=True,
                                     timeout=10).strip()
except (OSError, subprocess.SubprocessError):
    commit = None
report = {
    "experiment": "AMOS-only nnU-Net 3D external inference",
    "scope": "Exploratory, UNREVIEWED TotalSegmentator v3 reference annotations",
    "original_test_partition": "test",
    "test_cases": len(ids),
    "dataset_manifest_sha256": sha(manifest_path),
    "source_inventory_sha256": sha(inventory),
    "model_checkpoint_sha256": sha(checkpoint),
    "git_commit": commit,
    "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "warnings": [
      "Masks have passed automated basic integrity checks, not independent manual anatomy QC.",
      "No performance-based case exclusion, test-set tuning, or GT-derived cropping.",
      "76 of 109 official test cases have two nonempty adrenal reference masks; selection is label-dependent.",
      "Image overlap screen did not establish patient independence."
    ]
}
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    old = json.loads(target.read_text())
    for key in ("dataset_manifest_sha256", "model_checkpoint_sha256"):
        if old.get(key) != report[key]:
            raise SystemExit(f"Refusing to reuse changed exploratory cohort/model: {key}")
    print("Reusing matching provenance record, no overwrite.", flush=True)
else:
    target.write_text(json.dumps(report, indent=2)+"\n")
print("PASS: frozen external cohort; exploratory UNREVIEWED evaluation only.", flush=True)
PY

mkdir -p "$PRED"
if python - "$DATA" "$PRED" <<'PY'
import json,sys
from pathlib import Path
data, pred = map(Path, sys.argv[1:])
ids = json.loads((data/"external_test_manifest.json").read_text())["case_ids"]
complete = len(ids)==76 and all((pred/(cid+".nii.gz")).is_file()
                               and (pred/(cid+".nii.gz")).stat().st_size>0
                               for cid in ids)
raise SystemExit(0 if complete else 1)
PY
then
  echo "All 76 exploratory predictions present; skip inference."
else
  echo "Predicting from ORIGINAL AMOS-trained nnU-Net 3d_fullres fold0; NO training."
  nnUNetv2_predict -i "$DATA/imagesTs" -o "$PRED" \
    -d 701 -c 3d_fullres -f 0 -device cuda
fi

python -u scripts/evaluate_totalseg_external_adrenal.py \
  --dataset-dir "$DATA" --pred-dir "$PRED" --output "$METRICS"

echo "SUCCESS: exploratory external benchmark scored after signed sampled-label review."
echo "Summary: ${METRICS%.csv}.summary.json"
