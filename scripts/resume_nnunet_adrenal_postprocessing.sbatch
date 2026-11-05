#!/bin/bash -l
#SBATCH --job-name=adrenal_pp_resume
#SBATCH --partition=k2-medpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=nnunet_postproc_resume.%j.out
#SBATCH --error=nnunet_postproc_resume.%j.err

# Resume the interrupted 2h postprocessor without any nnU-Net inference.
# 1. Do not rerun the 100 complete largest-component masks.
# 2. Score them first (exploratory heldout analysis, no model selection).
# 3. Verify/reuse existing relative-15% masks; compute missing masks.
# 4. Score all 100 relative-15% masks and compare matched case IDs.
set -Eeuo pipefail

ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

RAW="runs/nnunet3d_holdout_predictions_fold0"
LARGEST="runs/nnunet3d_holdout_predictions_cc_largest"
RELATIVE="runs/nnunet3d_holdout_predictions_cc_relative15"
DATASET="nnunet_raw/Dataset701_AMOSAdrenalCT"

validate_cases() {
    python - "$RAW" "$1" <<'PY'
import sys
from pathlib import Path
raw = {p.name for p in Path(sys.argv[1]).glob("*.nii.gz")}
processed = {p.name for p in Path(sys.argv[2]).glob("*.nii.gz")}
# Interrupted NIfTI writes use a filename ending .partial.nii.gz and
# therefore must not count as completed cases.
processed = {name for name in processed if not name.endswith(".partial.nii.gz")}
print(f"Comparing: originals={len(raw)} processed={len(processed)} "
      f"missing={len(raw - processed)} extra={len(processed - raw)}", flush=True)
if len(raw) != 100 or raw != processed:
    raise SystemExit("Not all 100 expected case masks are present")
PY
}

echo "===== ENVIRONMENT ====="
python - <<'PY'
import sys, nibabel, scipy, numpy
print("Python:", sys.executable, flush=True)
print("nibabel:", nibabel.__version__, "scipy:", scipy.__version__, flush=True)
PY

echo "===== [1/4] Verify complete largest-component output ====="
if [[ -f "$LARGEST/postprocessing_audit.csv" ]] && validate_cases "$LARGEST"; then
    echo "All 100 largest-component masks already complete; skipping expensive recomputation."
else
    python -u scripts/postprocess_nnunet_adrenal.py \
      --input-dir "$RAW" --output-dir "$LARGEST" --resume
    validate_cases "$LARGEST"
fi

echo "===== [2/4] Score largest-component masks ====="
python -u scripts/evaluate_nnunet_adrenal.py \
  --pred-dir "$LARGEST" --dataset-dir "$DATASET" \
  --output runs/nnunet3d_holdout_metrics_cc_largest.csv

echo "===== [3/4] Safely resume relative-15% masks ====="
python -u scripts/postprocess_nnunet_adrenal.py \
  --input-dir "$RAW" --output-dir "$RELATIVE" \
  --mode relative --min-relative-size 0.15 --resume
validate_cases "$RELATIVE"

echo "===== [4/4] Score relative-15% masks ====="
python -u scripts/evaluate_nnunet_adrenal.py \
  --pred-dir "$RELATIVE" --dataset-dir "$DATASET" \
  --output runs/nnunet3d_holdout_metrics_cc_relative15.csv

echo "===== EXPLORATORY PAIRED COMPARISON ====="
python - <<'PY'
import csv
from pathlib import Path
from statistics import mean
files = {
    "raw": Path("runs/nnunet3d_holdout_metrics_fold0.csv"),
    "largest": Path("runs/nnunet3d_holdout_metrics_cc_largest.csv"),
    "relative15": Path("runs/nnunet3d_holdout_metrics_cc_relative15.csv"),
}
rows = {}
for method, path in files.items():
    with path.open(newline="") as fp:
        records = list(csv.DictReader(fp))
    rows[method] = {r["case_id"]: r for r in records}
ids = set(rows["raw"])
if len(ids) != 100 or any(set(group) != ids for group in rows.values()):
    raise SystemExit("ERROR: comparison files have different case identifiers")
raw_dice = {
    cid: (float(r["dice_left"]) + float(r["dice_right"])) / 2
    for cid, r in rows["raw"].items()
}
for method, group in rows.items():
    current = {
        cid: (float(r["dice_left"]) + float(r["dice_right"])) / 2
        for cid, r in group.items()
    }
    diffs = [current[cid] - raw_dice[cid] for cid in sorted(ids)]
    improved = sum(d > 1e-7 for d in diffs)
    worsened = sum(d < -1e-7 for d in diffs)
    unchanged = len(ids) - improved - worsened
    print(f"{method}: mean bilateral Dice={mean(current.values()):.5f}; "
          f"mean difference={mean(diffs):+.5f}; "
          f"improved={improved}, worsened={worsened}, unchanged={unchanged}",
          flush=True)
print("WARNING: These are exploratory held-out comparisons, not independent "
      "hyperparameter selection.", flush=True)
PY

echo "SUCCESS: postprocessing resumed, both complete and evaluated; raw masks preserved."
