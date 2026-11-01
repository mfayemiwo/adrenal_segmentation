#!/bin/bash -l
#SBATCH --job-name=delta_phase1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=phase1.%j.out
#SBATCH --error=phase1.%j.err
#SBATCH --mem=64G
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1

# Phase 1: matched baseline full-volume evaluation + paired refiner statistics.
# Run from repository root: sbatch scripts/run_delta_snn_phase1.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

BASE_CKPT="${BASE_CKPT:-runs/sb_positives/best_model.pt}"
DATA_ROOT="${DATA_ROOT:-../data/amos22}"
CACHE_DIR="${CACHE_DIR:-../cache}"
OUT="runs/delta_phase1"
mkdir -p "$OUT"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "No MI300X GPU detected; stopping before evaluation."
print("PyTorch:", torch.__version__, "ROCm:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0), flush=True)
PY

echo "[1/3] Matched baseline full-volume evaluation"
python -u scripts/evaluate_segmenter.py \
    --checkpoint "$BASE_CKPT" \
    --data-root "$DATA_ROOT" \
    --cache-dir "$CACHE_DIR" \
    --output-dir runs/delta_baseline_full_volume \
    --keep-largest 2 \
    --min-component-voxels 30 \
    --device cuda

echo "[2/3] Paired patient-level statistics (three refiners vs baseline)"
python -u scripts/compare_delta_snn_runs.py \
    --baseline-csv runs/delta_baseline_full_volume/per_case.csv \
    --runs-dir runs \
    --output "$OUT/statistics.txt"

echo "[3/3] Case-level failure and improvement ranking"
python - <<'PY'
from pathlib import Path
import pandas as pd
R = Path("runs")
O = R / "delta_phase1"
metric_cols = ["dice_left_postprocessed", "dice_right_postprocessed"]
baseline = pd.read_csv(R / "delta_baseline_full_volume/per_case.csv")
assert baseline["case_id"].is_unique
baseline = baseline.set_index("case_id")
base_ids = set(baseline.index.astype(str))
output = pd.DataFrame(index=baseline.index)
output.index.name = "case_id"
output["baseline_dice"] = baseline[metric_cols].mean(axis=1)
for variant in ("delta_snn", "raw_snn", "delta_analog"):
    scores = []
    for seed in range(1, 6):
        p = R / f"delta_refiner_{variant}_s{seed}" / "full_volume" / "per_case.csv"
        if not p.is_file():
            raise FileNotFoundError(f"Missing full-volume evaluation: {p}")
        df = pd.read_csv(p)
        if not df["case_id"].is_unique:
            raise ValueError(f"Duplicate case_id: {p}")
        df = df.set_index("case_id")
        if set(df.index.astype(str)) != base_ids:
            raise ValueError(f"Patient-set mismatch in {p}; do not pool results")
        scores.append(df[metric_cols].mean(axis=1).rename(f"s{seed}"))
    aligned = pd.concat(scores, axis=1).reindex(output.index)
    output[f"{variant}_dice_mean"] = aligned.mean(axis=1)
    output[f"{variant}_dice_seed_sd"] = aligned.std(axis=1, ddof=1)
    output[f"{variant}_minus_baseline"] = output[f"{variant}_dice_mean"] - output["baseline_dice"]
output["delta_snn_minus_delta_analog"] = output["delta_snn_dice_mean"] - output["delta_analog_dice_mean"]
output = output.sort_values("delta_snn_minus_baseline")
output.to_csv(O / "patient_comparisons.csv")
output.head(20).to_csv(O / "worst_delta_snn_vs_baseline.csv")
output.tail(20).sort_values("delta_snn_minus_baseline", ascending=False).to_csv(O / "best_delta_snn_vs_baseline.csv")
with (O / "summary.txt").open("w") as f:
    f.write(f"Matched patients: {len(output)}\n")
    f.write(f"Baseline mean Dice: {output['baseline_dice'].mean():.6f}\n")
    for variant in ("delta_snn", "raw_snn", "delta_analog"):
        diff = output[f"{variant}_minus_baseline"]
        f.write(f"{variant}: mean Dice={output[f'{variant}_dice_mean'].mean():.6f}; delta from baseline={diff.mean():+.6f}; improved cases={(diff > 0).sum()}/{len(diff)}\n")
print((O / "summary.txt").read_text())
print("Phase 1 outputs:", O, flush=True)
PY

echo "DONE: runs/delta_baseline_full_volume and runs/delta_phase1"
