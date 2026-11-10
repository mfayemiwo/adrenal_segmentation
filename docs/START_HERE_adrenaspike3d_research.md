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
