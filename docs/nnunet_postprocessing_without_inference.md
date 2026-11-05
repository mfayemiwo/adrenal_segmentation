# Mask-only postprocessing for adrenal nnU-Net (no rerun of inference)

**Purpose:** evaluate low-risk cleanup of existing bilateral nnU-Net segmentation NIfTI files, NOT retraining, checkpoint reruns, ensembling, TTA or use of probabilities. This can run on CPU only after `runs/nnunet3d_holdout_predictions_fold0/*.nii.gz` exist.

**Important: existing job safety.** Do not pull new repository commits into the shared HPC working tree while `nnunet3d_all.<jobid>` runs. Check out the updated script after the current nnU-Net job reaches terminal success, or use a separate worktree. The launcher and already-submitted job are not modified by this new standalone tool.

## Scientific premise and risk

- Left adrenal=1, right adrenal=2. **Never** run largest-component filtering jointly across these two classes.
- These masks were generated on original full abdominal CT; postprocessing acts only on hard predicted label masks, without original CT or ground-truth information.
- 3D connected-component cleanup is established nnU-Net postprocessing and was used in prior adrenal segmentation literature.
- Smaller, disconnected components can be legitimate thin gland tissue, especially at anisotropic spacing; removal can **lower Dice, NSD or gland recall**. 3D connectivity=2 (18-connected) is a transparent preliminary choice, **not validated for this cohort**.
- A single connected-component filtering rule is not capable of restoring entirely missed gland voxels or correcting boundaries where the predicted mask is connected but too small. Therefore improvement to 0.90 Dice must NOT be assumed.
- The raw predictions are never overwritten; each variant gets a separate output path and `postprocessing_audit.csv`.

## Run after fold 0 held-out predictions exist

From the working project root:

```bash
cd /mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation
# Only after job 10650749 has finished, update this existing branch:
git pull origin research/adrenaspike3d
source .venv/bin/activate

# Confirm that original full-volume segmentation masks exist:
ls runs/nnunet3d_holdout_predictions_fold0/*.nii.gz | head

# Variant 1: largest 3D component independently per adrenal gland:
python scripts/postprocess_nnunet_adrenal.py \
  --input-dir runs/nnunet3d_holdout_predictions_fold0 \
  --output-dir runs/nnunet3d_holdout_predictions_cc_largest

# Variant 2: retain large secondary components (>=15% of largest size):
python scripts/postprocess_nnunet_adrenal.py \
  --input-dir runs/nnunet3d_holdout_predictions_fold0 \
  --output-dir runs/nnunet3d_holdout_predictions_cc_relative15 \
  --mode relative --min-relative-size 0.15
```

Both variants work on saved NIfTI masks, no GPU, no torch import, no NN forward. Choose the rule **on training-fold validation predictions**, not by maximizing scores on held-out `labelsTs`. Keep raw predictions as the baseline and report absolute score differences with per-case scatter/outliers.

## One-off evaluation of a PRE-CHOSEN rule (exploratory holdout if no validation selection)

The existing `evaluate_nnunet_adrenal.py` runs the optional energy hook if the metrics filename ends in `_fold0`. For mask-cleanup-only evaluation, choose filenames that do **not** end in `_fold<digit>` so it does not mistakenly launch GPU benchmarks:

```bash
python scripts/evaluate_nnunet_adrenal.py \
  --pred-dir runs/nnunet3d_holdout_predictions_cc_largest \
  --dataset-dir nnunet_raw/Dataset701_AMOSAdrenalCT \
  --output runs/nnunet3d_holdout_metrics_cc_largest.csv

python scripts/evaluate_nnunet_adrenal.py \
  --pred-dir runs/nnunet3d_holdout_predictions_cc_relative15 \
  --dataset-dir nnunet_raw/Dataset701_AMOSAdrenalCT \
  --output runs/nnunet3d_holdout_metrics_cc_relative15.csv
```

**DO NOT cherry-pick the best-performing postprocessing variant using the held-out test labels and call that a final unbiased test score.** If the original training fold's `validation/` predicted masks and its GT labels are available, preselect the single postprocessor on that **development-only** validation cohort and then evaluate that frozen choice on `imagesTs/labelsTs` once. If no independent validation predictions exist, report the two postprocessing variants as explicitly exploratory/sensitivity analyses and seek an external confirmatory set.

## What to inspect

- `postprocessing_audit.csv`: gland-wise component count, retained/removed voxels, fraction deleted, warnings for unusually aggressive removal.
- Compare patient-level Dice_left/Dice_right, HD95_left/HD95_right (mm), NSD@2 mm before/after, **including worsened cases** and missed glands.
- Negative/failure diagnostics: predicted gland missing, larger false-positive component than the true gland, disconnected thin anatomy, incorrect laterality or tumour distortion.
- Verify raw and cleaned masks have identical shape, affine, qform/sform, left/right semantic labels.
- Do not silently make additional changes to HU intensities or z crops based on test labels. Hole filling and morphological closing are candidates requiring their own internal-validation decision, not included in this deliberately conservative version.

**CPU-only tests:** `python -m unittest discover -s tests -p 'test_postprocess_nnunet_adrenal.py' -v`.
