# START HERE — AdrenaSpike3D research handoff and manuscript map

**Date:** 2026-11-10. Branch: `research/adrenaspike3d`. Repository: `mfayemiwo/adrenal_segmentation`.

**If a future conversation loses historical context, read these two files FIRST:**
1. `docs/research_evidence_register_2026-11-10.md` — numerical results, dataset cohorts, provenance, postprocessing failures, energy telemetry and the exact per-case and raw-trial file paths.
2. `docs/adrenaspike3d_prospective_experiment_protocol_2026-11-10.md` — predeclared next experiments, fairness rules, external data safeguards and hypotheses for CNN vs matched 3D analog vs SNN.

**Additional methods:** `docs/nnunet_postprocessing_without_inference.md`; `docs/energy_efficiency_evaluation.md`; `docs/anatomy_phase2_plan.md`. Older Word reports in `docs/` are historical and may reflect older 2.5D or experimental settings; do not supersede the validated register.

## Headline measured results, NOT targets

- AMOS 100 independent full-volume CTs, 3D nnU-Net (fold 0, Dataset701): **mean bilateral Dice 0.806184**, left **0.811935**, right **0.800433**, original hard cases retained.
- Measured MI300X whole-predict GPU-socket energy: **49.319 ± 0.157 kJ/CT**, end-to-end **76.563 ± 0.057 s/CT** over **3 runs × 100 CTs**; AMD-SMI power counter agreed with independently integrated sampled power. VRAM **6.165 GiB** after correcting MB reporting. These are repeated GPU inference executions of **one checkpoint**, not independent training seeds.
- Postprocessing **negative**: largest-component Dice **0.80363**; relative15 Dice **0.80606**; raw **0.80618**. Anatomy/laterality mixup explained in `amos_0346`. Do not silently omit these.
- TotalSegmentator v3 from 1,939 CTs: **1,053** bilateral training candidates, **76** bilateral official test cases, 59 unilateral and 751 empty. **Not clinically validated.**
- Image overlap screen: 300 AMOS × 1129 eligible TS CTs, 0 exact and 0 high-similarity matches; not proof of no patient overlap.
- SNN historical 2.5D GT-z-cropped exploratory ~0.7313 and analog ~0.7311 **cannot** be compared to the nnU-Net full CT benchmark. A fully 3D matched CNN/SNN model has not yet been trained/evaluated to demonstrate benefit.
- >0.90 patient-level Dice is the **target**, not an achieved result; DOI `10.1007/s10278-025-01449-y` reported .91-.93 using a different 40-case AMOS evaluation protocol.

## Kelvin2 project and archival protection

`/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation`

All primary original result files remain on Kelvin2. The GitHub register is **not** a backup of patient-derived NIfTIs or large model weights. Preserve scripts + raw evidence with:

```bash
cd /mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation
git checkout research/adrenaspike3d
git pull origin research/adrenaspike3d
sbatch scripts/archive_adrenal_manuscript_results.sbatch
```

This OPTIONAL job freezes original metrics/power traces and hashes; creates a local tar with the model checkpoint and prediction masks; the tar + SHA256 must then be copied to approved **independent** institutional storage. **Do not upload to a public repo or unrelated cloud storage.**

## Next execution: independent source-annotation review, without GPU training

```bash
sbatch scripts/run_totalseg_v3_adrenal_qc.sbatch
```

Outputs local `runs/totalsegmentator_v3_qc_plan/plan_summary.json`, 60 selected official-training review rows and 15 official-test review rows, plus CT/reference-label montages in two subfolders. Requires human annotation QC review; nothing is automatically marked as accepted.

Next, prep the completely separate 76-case **external test only** input/label dataset:

```bash
sbatch scripts/prepare_totalseg_external_adrenal_test.sbatch
```

This step does not run the model and does not touch the AMOS trained checkpoint. Evaluate it later with `scripts/evaluate_totalseg_external_adrenal.sbatch` **only after** independent source-label/provenance review and a valid signed QC gate from `scripts/validate_totalseg_v3_qc_review.py`. The GPU job now verifies the gate and blocks without it. See `docs/totalseg_v3_adrenal_annotation_qc_rubric.md` for the completed review CSV, attestation and correct commands. Sampled QC does not approve all 76 labels; external results remain exploratory until limitations are addressed.

Do NOT mix official TotalSegmentator test CTs into training. Do NOT change original 100 AMOS heldout cohort. For Paper II's ACTS-3D, work on `research/acts3d-algorithm` separately; no branch-switching on a busy Paper I HPC checkout.

## Recordkeeping obligations

Every next model run needs commit SHA, source CT cohort and frozen IDs, split manifest, training seed/hyperparameters, SLURM ID, model checkpoint hash, test protocol, full-case score CSV, energy JSON/CSV and failure/anatomical QC notes. Update the evidence register and snapshot after each run, regardless of whether the result helps the hypothesis. Never invent a high Dice or SNN efficiency gain to support publication.

## Current external CNN strategy update — 2026-11-10

The investigator cancelled the signoff-gated external job and chose to **skip human source-label review** for the upcoming external test. Treat it as **exploratory unreviewed**. The original QC-gated script remains unchanged for later independent validation. To run the distinct frozen-cohort unreviewed job:

```bash
cd /mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation
git pull --ff-only origin research/adrenaspike3d
sbatch scripts/evaluate_totalseg_external_unreviewed.sbatch
```

Outputs: `runs/nnunet3d_external_tsv3_metrics_fold0_unreviewed.summary.json` and per-patient CSV, predictions `runs/nnunet3d_external_tsv3_predictions_fold0_unreviewed`, provenance `runs/nnunet3d_external_tsv3_unreviewed_provenance.json`. Do not imply manual source QA occurred. Do not tune architecture or postprocessing on this heldout cohort. Evaluate all 76 frozen cases regardless of score; after reading its results, continue CNN+SNN development on a separate internal training/validation cohort.

See the latest `E11` in `docs/research_evidence_register_2026-11-10.md`.

## Troubleshooting external CT NIfTI geometry — 2026-11-11

The first `evaluate_totalseg_external_unreviewed.sbatch` **FAILED**, not scored:
SimpleITK/ITK did not support orthonormal direction cosines in at least one original TotalSegmentator CT. Only 3/76 prediction masks exist and no summary was produced. Preserve these partial results and log, do not claim external Dice.

**Repaired, tested 2-stage workflow:**

```bash
cd /mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation
git pull --ff-only origin research/adrenaspike3d
J1=$(sbatch --parsable scripts/prepare_totalseg_external_ct_geometry.sbatch)
J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/run_totalseg_external_ortho_inference.sbatch)
echo "CT-only geometry repair job $J1; subsequent MI300X inference job $J2"
```

Read geometry summary `nnunet_raw/ExternalTotalSegV3AdrenalOrtho_v1/geometry_manifest.json`; restored native predictions `runs/nnunet3d_external_tsv3_native_predictions_fold0`; final summary `runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.summary.json`.

Both newly authored scripts preserve the original 76 source CT/labels and AMOS model. Only CTs with nonorthogonal/ITK-incompatible headers are regridded into world RAS; output gland masks are mapped back to **original native CT geometry** with nearest-neighbor before scoring. Label masks do not influence CT-space inference preparation. Synthetic end-to-end CI test passed. This is still exploratory **unreviewed** external test evaluation with no architecture selection on its results. Detailed methods and original error are in E12 of the evidence register.

## Completed external test and next experimental phase — 2026-11-16

The original AMOS-only nnU-Net CNN completed external testing **76/76** TotalSegmentator v3 official test CTs after CT-only affine fixes for **nine** incompatible CT inputs. **Mean bilateral Dice 0.6682536367697085** versus original AMOS holdout 100-case **0.8061840313892723** (different cohorts, unpaired). Original external mask annotations are **not manually checked**; full left/right HD95/NSD and bootstrap summary contents still need to be inspected at `runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.summary.json`. Source result, partial original failed inference and nine-case geometry evidence preserved in E12/E13 of the research register. No SNN efficiency claim yet.

**New expanded-data CNN preparatory stage (no training submitted):**

```bash
cd /mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation
git pull --ff-only origin research/adrenaspike3d
python -u scripts/report_external_tsv3_unreviewed.py
sbatch scripts/prepare_expanded_amos_tsv3_dataset702.sbatch
```

CPU preparation creates `runs/expanded_dataset702/plan.json` and `nnunet_raw/Dataset702_AMOSTotalSegAdrenalCT`, **never** overwriting Dataset701 or reading official test CTs for training. The frozen split is original AMOS train **200**, TS official train candidates **1053** split **895 train + 158 internal validation**, total **1253 raw imagesTr, 1095 fold0 training and 158 fold0 validation**. Original 100 AMOS holdout and all 109 official TotalSegmentator test cases remain untouched and excluded. The source masks are validated mechanically but **no clinical annotation quality review** has been completed.

After the Dataset702 preparation log reports SUCCESS and separate folder sizes/storage use are verified, submit:

```bash
sbatch scripts/preprocess_expanded_amos_tsv3_dataset702.sbatch
```

This CPU-only job runs nnU-Net fingerprint/preprocessing and installs the frozen `splits_final.json`; it does not train. Only once that is confirmed, separately submit the MI300X job:

```bash
sbatch scripts/train_expanded_amos_tsv3_dataset702.sbatch
```

Model weights are isolated to `nnunet_results/Dataset702_AMOSTotalSegAdrenalCT`. Training may need SLURM resubmission to resume checkpoint. Do not use the observed external Dice to choose any model hyperparameters. Prepare matched C3 analog/C4 genuine 3D SNN refiners after developing a credible C2 expanded-data CNN using internal validation; preserve full-volume energy protocol and all failures.

## Current queued jobs and external failure distribution — 2026-11-16

Completed unreviewed external AMOS-only nnU-Net on 76/76 TotalSegmentator official-test bilateral cases:
- **mean bilateral Dice 0.6682536367697084; sample SD 0.3021735302897953; median 0.8109285387468024**, left mean 0.6676954722695573, right mean 0.6688118012698596.
- 16/76 bilateral Dice below 0.50; 11/76 below 0.20; 6/76 bilateral Dice exactly zero; 6/76 at least 0.90; 8 empty predicted left masks, 9 empty predicted right masks. Counts overlap. 9 regridded CTs had mean bilateral Dice 0.6671142014744535 (not a causal comparison).
- Source complete `runs/nnunet3d_external_tsv3_failure_audit_unreviewed.json`, original 76-row `runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.csv`, and `runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.summary.json`. Data unreviewed, old evaluator physical HD95/NSD approximate for 9 sheared native grids. Full register E15 details.

**Submitted SLURM jobs, completion unverified:**
- `11026503`: `scripts/archive_adrenal_manuscript_results.sbatch` (back up raw paper evidence, including external Dice).
- `11026504`: `scripts/prepare_expanded_amos_tsv3_dataset702.sbatch`, `afterok:11026503` dependency.

Check `squeue -j 11026503,11026504` and `sacct -j 11026503,11026504 --format=JobID,JobName,State,ExitCode,Elapsed`. The full new dataset is NOT prepared until Dataset702 output reports SUCCESS. Only then submit CPU-only `scripts/preprocess_expanded_amos_tsv3_dataset702.sbatch`. The final GPU training job `scripts/train_expanded_amos_tsv3_dataset702.sbatch` must wait for fold0 preprocessing and installation. External observed scores must not drive model selection.

Original 3-run GPU socket energy baseline remains **49.319 ± 0.157 kJ/full AMOS CT**, latency **76.563 ± 0.057 s/full AMOS CT**; no matched full-volume SNN energy study yet.
