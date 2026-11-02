# AdrenaSpike-3D: Phase 3 implementation and research decision plan

## Current status and scientific limitations

**Prototype implemented, not yet trained or validated on MI300X.** No improvement
is guaranteed. Work is isolated on `research/adrenaspike3d`.

**Important correction to historical evaluation:** `prepare_case` in the older
Stage B pipeline selects the z-range using ground-truth masks, even during
evaluation. Thus its reported "full-volume" scores are on a label-derived z
crop, *not the entire abdominal CT scan*. They remain useful for comparisons
under the same protocol but must not be presented as unassisted full-scan
performance. This branch creates true full-z inputs without mask-based crops.

## Code audit and revised run decision (2 November 2026)

**Proceed:** finish an already-running 3D nnU-Net job and collect its unrestricted
scan metrics. Keep completed Phase 2 mechanism ablations.

**Pause:** additional full AdrenaSpike-3D ROI training jobs until the
underlying feature extractor is evaluated, ROI coverage is audited, and
out-of-fold backbone predictions are available. The original SNN classification
gate had only a very small PR-AUC difference from its static control and has
not demonstrated safe full-scan gland recall. Do not introduce a hard
classification gate into inference on that evidence.

**Code corrections implemented:** (a) Stage B's best checkpoint saves its
threshold under `threshold`, not `best_threshold`; the cache now reads
the exact key and rejects absent values; (b) attention settings are
reconstructed from the checkpoint configuration; (c) cached training-patient
predictions are explicitly flagged as *in-sample*; (d) refiner training
refuses to consume these predictions without the intentional
`--allow-in-sample-backbone` exploratory switch; (e) evaluation has a
`--no-correction` exact identity control.

**Smoke tests:** `SMOKE=1` explicitly enables the exploratory override.
For a full *exploratory* refiner experiment only, set `EXPLORATORY=1` in
the `sbatch --export=ALL,...` line. Without that switch, training
correctly stops. An exploratory run is not suitable for publication claims.

**Proposed decision gate:** first examine original baseline on true full-z
scans, 3D nnU-Net full-scan Dice/HD95/NSD, gland ROI GT coverage, and
computational cost. Then train at most one seed of the SNN and analog
using out-of-fold training inputs. If spike/event ablation, paired
improvement, and boundary metrics do not support an advantage,
stop using spiking as a primary segmentation contribution.

**Potential classification role:** revisit Stage A only as a
high-sensitivity *soft proposal/triage* mechanism, not a hard negative
filter. Evaluate patient-level and per-gland recall, classification
calibration, rare missed positives and end-to-end Dice compared with
a matched CNN/ANN and a 3D baseline. A hard gate is unsafe for segmentation
when a false-negative slice can entirely remove part of an adrenal gland.

## Three independently testable objectives

1. **Mechanistic audit (Phase 2 checkpoints, no retraining):** same patient
   inputs, no events, event order reversed, and disabled confidence gate.
   A zero-events result matching the original would undermine the claim that
   anatomical event timing is useful.
2. **Strong 3D baseline:** convert original CT NIfTI with separate left and
   right labels to nnU-Net v2, preserving the entire volume. Train 3d_fullres
   with internal folds, predict imagesTs, evaluate against labelsTs.
   Do not use imagesTs for architecture choice.
3. **Gland-focused temporal ROI prototype:** a shared frozen 2.5D U-Net
   produces full-z per-gland probabilities. Prediction-only ROI localisation
   selects a 3D crop for **each gland independently** (default 24x96x96).
   A bidirectional recurrent Conv+GroupNorm+LIF encoder processes true
   successive z slices, with uncertainty, spatial boundary and positive/negative
   probability transitions as inputs. A small 3D decoder gives a gated
   residual correction. The matched `analog` network replaces only LIF with
   a leaky continuous unit. The same weights are **not** shared across
   alternative trained models.

**Scope:** objective 2 establishes a stronger benchmark. The first ROI prototype
(objective 3) currently refines the frozen **2.5D Stage B backbone**, not
nnU-Net predictions. Integrating nnU-Net out-of-fold predictions as the
refiner backbone is a *subsequent* implementation step, not complete.

## Reproducible cluster commands

On HPC in repository root:

```bash
git fetch origin
git checkout research/adrenaspike3d
git pull origin research/adrenaspike3d
python -m compileall -q scripts/ src/models/adrenaspike3d.py src/data/adrenaspike_roi.py
python -m pytest tests/test_adrenaspike3d.py -q
```

### A. Mechanistic diagnostic array — 8 jobs, eight-hour limit each

```bash
sbatch scripts/run_phase3_diagnostics.sbatch
```

Uses existing `runs/phase2_anatomy_snn_s1` and analog checkpoints.
Look at `runs/phase3_diagnostics/{anatomy_snn,anatomy_analog}/{none,zero_events,reverse_events,no_gate}`.

### B. True 3D nnU-Net baseline: **single GPU allocation (recommended)**

This pipeline automatically runs preparation, preprocessing, 3D full-resolution
nnU-Net fold training, heldout prediction, and per-gland Dice/HD95/NSD
evaluation **in one SLURM allocation**, requesting **24 hours**. There is
no reason to submit each stage separately. It skips finished stages on
re-entry and resumes the training checkpoint if it exists.

Before submitting, confirm that the cluster's queue permits 24h and that
`nnunetv2` is installed inside `.venv`. Verify the cluster's permitted
wall time using `sinfo -p k2-gpu-amd -o "%P %l"`. Some partitions cap jobs
below 24 hours; if so, adjust the `#SBATCH --time` value to the actual limit.

```bash
git fetch origin
git checkout research/adrenaspike3d
git pull origin research/adrenaspike3d
bash -n scripts/run_adrenaspike_nnunet_all.sbatch

# One submission: all stages, in order
sbatch scripts/run_adrenaspike_nnunet_all.sbatch
```

If a job reaches its permitted wall time before the full training is
finished, run the **same command** again. The launcher checks preparation,
preprocessing, `checkpoint_latest` / `checkpoint_final`, and completed
predictions. nnU-Net typically saves periodic checkpoints; any progress
after the last checkpoint may be lost on time limit. A 24h request does not
guarantee a completed 3D training run and does not change nnU-Net's original
training schedule.

A successful fold-0 run generates:

- `runs/nnunet3d_holdout_metrics_fold0.csv`
- `runs/nnunet3d_holdout_predictions_fold0/`
- `nnunet3d_all.<jobid>.out` and `.err`

For later cross-validation folds, set `FOLD=1` (etc.). For a future
out-of-fold teacher ensemble, save predicted probabilities with
`SAVE_PROBABILITIES=1`; this consumes more disk storage.

These are true-full-z nnU-Net baseline metrics. The older Stage B scores
use a ground-truth-derived z crop and must not be compared as if evaluated
under the same full-scan input condition. The imagesVa/labelsVa cohort is
exploratory development validation, not a genuinely independent external
test set. Do not use it to repeatedly tune the model and call it a final
unbiased test.

### C. AdrenaSpike-3D ROI pilot (2 matched models)

```bash
# Full-z baseline prediction cache, from original raw images (not old z-cropped cache)
sbatch --export=ALL,STAGE=cache,SMOKE=1 scripts/run_adrenaspike_inputs.sbatch
# Once finished:
sbatch --export=ALL,STAGE=tiles,SMOKE=1 scripts/run_adrenaspike_inputs.sbatch
# Two-epoch model smoke test for both arms:
sbatch --export=ALL,SMOKE=1,STAGE=train scripts/run_adrenaspike_roi.sbatch
# Then build all cases and ROI tiles:
sbatch --export=ALL,STAGE=cache scripts/run_adrenaspike_inputs.sbatch
sbatch --export=ALL,STAGE=tiles scripts/run_adrenaspike_inputs.sbatch
# Train 2 models on patient-disjoint internal development sets:
sbatch --export=ALL,STAGE=train,SEED=1,EXPLORATORY=1 scripts/run_adrenaspike_roi.sbatch
# After training:
sbatch --export=ALL,STAGE=evaluate,SEED=1 scripts/run_adrenaspike_roi.sbatch
```

The resulting `runs/adrenaspike3d_{snn,analog}_s1/full_volume/per_case.csv`
includes paired true-full-z Dice/HD95/NSD and the **ROI coverage** measured
against labels. If coverage is poor, improve localisation first; the refiner
cannot recover masks outside its selected crop.

**Critical training limitation:** the frozen Stage B backbone was itself trained
using original AMOS training patients. Therefore Stage B predictions of those
patients are **in-sample**; any ROI refiner trained on these is initially
a feasibility experiment. For publication-grade training, supply
patient-out-of-fold predictions from a separately trained backbone (e.g.
nnU-Net five-fold cross-validation) and reserve untouched external patients
for final claims. The prototype's internal patient split alone does not make
the backbone predictions out-of-fold.

## Acceptance / stopping criteria

- **Mechanistic:** temporal-event ablation significantly changes performance;
  event order matters and no-event output is not essentially unchanged.
- **Accurate and honest:** compare against original Stage B on *exactly the
  same newly constructed full-z scans*, and against the nnU-Net 3D baseline
  on the same case IDs. Do not juxtapose old GT-cropped Dice with true-full-z Dice.
- **Specific SNN contribution:** spiking outperforms matched analog by more
  than variation across seeds and patients, or else no specific SNN advantage
  may be claimed.
- **Reproducibility:** 2 pilot seeds first, then 5, paired bootstrap patient
  confidence intervals, external CT testing, runtime/energy on MI300X.
- **Novelty:** 'AdrenaSpike-3D' is a provisional name, not proof of originality.

