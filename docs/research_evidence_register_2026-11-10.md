# AdrenaSpike3D Research Evidence Register — frozen state, 10 November 2026

**Status:** living, version-controlled evidence register for manuscript preparation. This document is an index of measurements that must be retained, not a substitute for original machine-generated logs. **Do not delete, overwrite or relabel negative results.** The Kelvin2 filesystem contains the original NIfTI predictions, per-patient CSVs, SLURM logs, reference data and checkpoint. The repository stores experiment scripts and this register; it does not itself contain the multi-gigabyte primary evidence files.

**Publication tracks**
- **Paper I — AdrenaSpike3D:** 3D CNN segmentation against a matched 3D CNN–SNN/refiner and a compute-matched analog refiner. Same original full CT, split and score protocol; energy empirically measured on the same device.
- **Paper II — ACTS-3D:** distinct *algorithmic* contribution: tri-action risk/budget-aware conditional routing (skip/analog/spiking), with counterfactual utility and safety analysis; develop independently on `research/acts3d-algorithm`. Do not conflate the Paper II CPU-only router proof-of-concept with an experimentally validated 3D SNN.

Repository: `mfayemiwo/adrenal_segmentation`, Paper I branch `research/adrenaspike3d`. HPC root: `/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation`. Date in this record is **2026-11-10**. Unless expressly labelled otherwise, scores are from held-out 100 original AMOS CT patients, using both adrenals. Preserve raw original files and split manifests.

## E01. Complete AMOS-derived CT baseline and methods

- Data conversion: `scripts/prepare_nnunet_adrenal.py` and `nnunet_raw/Dataset701_AMOSAdrenalCT/split_manifest.json`. AMOS CT case numeric ID <500; original full abdominal CT; *no ground-truth-derived crop*; independent 100 `imagesVa` CTs represented as `imagesTs/labelsTs` in nnU-Net. Existing training split 200 CT, heldout 100 CT. Original AMOS labels right=11 -> nnU-Net 2, left=12 -> nnU-Net 1. Verify provenance from the manifest.
- Model: nnU-Net v2 `3d_fullres`, dataset 701, fold 0, 1000 epochs, 3D full-volume, trained using ROCm 6.3.3 and an AMD Instinct MI300X. Completed SLURM job `10650749` on 2026-11-05; reported walltime ~2d 7h 51min.
- Model weights: `nnunet_results/Dataset701_AMOSAdrenalCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`; original predictions `runs/nnunet3d_holdout_predictions_fold0/*.nii.gz`; original per-patient metrics `runs/nnunet3d_holdout_metrics_fold0.csv` (100 rows).
- **Mean Dice**: left **0.8119347161134376**, right **0.800433346665107**, bilateral arithmetic mean **0.8061840313892723**.
- **Mean HD95**: left **4.096472572338184 mm**, right **3.7099032553406164 mm**.
- **Mean NSD@2 mm**: left **0.927869112911467**, right **0.927095339099297**.
- HD95 and NSD evaluator returns NaN for empty predicted/GT gland pairs, and mean calculations exclude non-finite values; manuscript must report valid denominators and handle missing values transparently. The `scripts/evaluate_nnunet_adrenal.py` surface metric uses voxel spacing with axis order (z,x,y); non-orthogonal/sheared affines are not accommodated. For external data require matching affines and explicit geometry QA.
- Do not confuse best training EMA pseudo-Dice (~0.7692) with held-out Dice.

## E02. Mask-only postprocessing ablation, **negative result**

CPU-only connected component filtering applied separately to class 1 and 2, 18-connectivity (3D), original CT predictions unchanged. `scripts/postprocess_nnunet_adrenal.py`; resumable SLURM batch `scripts/resume_nnunet_adrenal_postprocessing.sbatch` after initial job `10718766` was cancelled at 2h wall limit; follow-up `10726574` completed.

| Variant | Mean bilateral Dice (100 CT) | Δ to raw | Improved cases | Worsened | Unchanged | Evaluation |
|---|---:|---:|---:|---:|---:|---|
| Original | **0.80618** | 0 | 0 | 0 | 100 | `runs/nnunet3d_holdout_metrics_fold0.csv` |
| Largest component per class | **0.80363** | **−0.00255** | 13 | 7 | 80 | `runs/nnunet3d_holdout_metrics_cc_largest.csv` |
| Relative ≥15% of class-largest component | **0.80606** | **−0.00012** | 12 | 6 | 82 | `runs/nnunet3d_holdout_metrics_cc_relative15.csv` |

Keep both postprocessing audits at `runs/nnunet3d_holdout_predictions_cc_largest/postprocessing_audit.csv` and `runs/nnunet3d_holdout_predictions_cc_relative15/postprocessing_audit.csv`. **Neither improved mean Dice.** Postprocessing comparison on the already-inspected holdout is **exploratory, not a valid tuning set**. Do not drop low-performance cases or select postprocessing based on these results.

## E03. Confirmed hard failure and stratified analysis target

`amos_0346`, CT 512×512×78 with approx 0.621×0.621×5 mm voxels; raw bilateral Dice **0.27734** (left class Dice 0; right class Dice approx 0.5547). The *predicted right-class* mask had two disconnected regions:

| Region | Predicted class | Voxels | Centroid world mm (x,y,z) | Voxels overlapping GT-left | Voxels overlapping GT-right |
|---|---|---:|---|---:|---:|
| Component 2, largest | 2 (right) | **1,677** | (−42.2,118.0,−389.8) | **815** | 0 |
| Component 1 | 2 (right) | **1,411** | (+36.7,129.5,−372.9) | 0 | **1,268** |

Largest-component filtering retained the *wrongly laterality-labelled* left-side anatomical component and removed the correctly overlapping right-adrenal region, yielding **zero bilateral Dice** (change −0.27734). A hypothetical relabelling based on GT overlaps would give about 0.75 bilateral Dice, but that is an **oracle calculation, not a measured algorithm or published result**. Relabeling requires patient orientation and midline verification; the sign of NIfTI x-coordinate alone is not universal left/right.

Additional original hard cases to keep and inspect: `amos_0286` bilateral .6081 (left .3892/right .8269, 2mm); `amos_0333` .6415 (left .4469/right .8361, 2mm); `amos_0356` .6269 (left .7836/right .4703, 2.5mm); `amos_0189` .6265 (left .4934/right .7596, 5mm). Do **not** exclude on performance.

Largest-component patient examples:
- `amos_0346`: −0.27734, `amos_0087`: −0.01411, `amos_0132`: −0.01287, `amos_0136`: −0.00821, `amos_0372`: −0.00283.
- Positive examples: `amos_0228`: +0.03407, `amos_0150`: +0.01039.
- Relative15 worst: `amos_0087`: −0.01411, `amos_0132`: −0.01287, `amos_0136`: −0.00821; best `amos_0150`: +0.01039.
- Reported diffs derive from full paired CSVs; do not regard as independent repeat samples.

## E04. Full-volume MI300X GPU energy and latency baseline

Hardware telemetry preflight job `10719469` confirmed MI300X physical GPU identity (HIP PCI BDF `0000:9d:00.0` matches AMD-SMI), ROCm 6.3.3, Python AMD-SMI bindings matching the ROCm install. The earlier 2026-11-05 evaluation energy hooks were **skipped** because bindings were initially unavailable; do not count them as successful trials.

Dedicated repeat inference benchmark: `scripts/run_nnunet_energy_repro.sbatch`, `scripts/run_nnunet_energy_suite.py`, `scripts/measure_gpu_energy.py`, with pre-trained fold-0 model and original 100 complete AMOS held-out CTs, no retraining. Original full-volume masks and scored metrics were not overwritten. A run measures the entire `nnUNetv2_predict` subprocess and GPU socket energy, **including periods of GPU idle while preprocessing/writing**, excluding CPU/node power.

Evidence `runs/energy_nnunet_fullvolume_repro/`:
- `nnunet_fullvolume_r1.json`, `r2.json`, `r3.json` and per-run 0.5s-sampled power/counter CSVs.
- `nnunet_energy_trials.csv` with all 3 independent command executions; `nnunet_energy_summary.json` with mean and sample SD; `telemetry_probe.json`.
- The energy counter is independently cross-checked against integrated power; observed counter / integrated-power ratios = 1.0002416011, 1.0003405392, 1.0004017981, respectively.
- AMD-SMI ROCm 6.3 returns VRAM usage **in MB**. Trial 1 logged the raw MB value (6620) with a mislabelled `_bytes` key; **corrected** in the summary using `6620×10^6/2^30 = 6.165355443954468 GiB`. Preserve raw logs and the corrected analysis; subsequent trials have corrected values.

| Run | GPU socket energy total J | GPU socket energy J/CT | Seconds/CT | Peak sampled VRAM GiB | Mean sampled socket W |
|---|---:|---:|---:|---:|---:|
| 1 | 4,913,783.328889012 | 49,137.83328889012 | 76.62312034490053 | 6.165355443954468 | 641.0647401874058 |
| 2 | 4,941,546.725238919 | 49,415.46725238919 | 76.50988690927159 | 6.165355443954468 | 645.5832786454915 |
| 3 | 4,940,419.85129261 | 49,404.1985129261 | 76.55568877551937 | 6.165355443954468 | 644.9842602308499 |

**Mean ± sample standard deviation:** **49,319.16635140181 ± 157.14008334022103 J/CT**, **76.56289867656383 ± 0.05695998368053296 seconds/CT**, 6.165355443954468 GiB sampled peak. Three repeats, same dataset and checkpoint, **not** three separately trained networks. Energy coefficient of variation ~0.319%.

Energy source: AMD-SMI cumulative counter; agrees with numerical integration of GPU socket power to <0.05%. GPU socket measurement should NOT be presented as total-system energy or model kernel-only energy. Hardware-dependent values cannot be compared to cropping-conditional alternative methods without matching input workloads. Per-CT figures divide whole-command 100-CT totals by 100, so they include amortised startup and I/O.

## E05. TotalSegmentator dataset v3 audit, **not approved training yet**

Original release: Jakob Wasserthal, Zenodo 2026-09-10, `https://doi.org/10.5281/zenodo.22688904`, v3.0.0. Downloaded ZIP MD5 `0728beea76475e92cdb3160e228265d1`, contains 1,939 3D CT case folders and `meta.csv` with semicolon delimiter and official train/test flags. Dataset directory:
`../data/totalsegmentator_v3/extracted/Totalsegmentator_dataset_v300`

Audit script `scripts/audit_totalsegmentator_v3.py`, job `scripts/audit_totalsegmentator_v3.sbatch`, outputs `runs/totalsegmentator_v3_audit/case_audit.csv`, `summary.json`, `candidates_not_train_ready.csv`.

| Item | Count |
|---|---:|
| Total cases | 1939 |
| Official training split | 1830 |
| Official test split | 109 |
| Both adrenal masks non-empty | **1129** |
| Exactly one gland mask non-empty | 59 |
| Both empty | 751 |
| Incomplete gland labels with at least one visible kidney (triage clue only) | 71 |
| Candidate *training* cases with bilateral nonempty, binary, geometry-aligned labels | **1053** |
| Bilateral cases in official test split, reserved for external test only | **76** |
| Failed case geometry/metadata reader | 0 |

A valid NIfTI mask file or binary shape/affine does **not** mean the gland annotation is clinically accurate. Blank labels should not be assumed true negatives. Although CTs appear resampled to 1.5 mm, original scan slice thickness varies and is stored in `meta.csv`. V3 includes pediatric scans. Source annotations may contain model-assisted bias. Preserve official split, review label provenance/quality, secure approval before mixing.

## E06. AMOS–TotalSegmentator image-overlap screening

Script: `scripts/screen_amos_totalseg_ct_overlap.py`, job `scripts/screen_amos_totalseg_ct_overlap.sbatch`, full log `amos_ts_overlap.10764621.out`, report `runs/ct_overlap_screen/summary.json`, candidate pairs `runs/ct_overlap_screen/candidate_overlap_pairs.csv` (header only).

- `amos_cases_screened: 300`
- `totalseg_bilateral_cases_screened: 1129`
- `exact_canonical_voxel_matches: 0`
- `high_similarity_pairs_flagged: 0`
- `candidate_pairs_for_manual_review: 0`
- `cutoff_is_unvalidated: 0.985`, `complete_screen: true`.

The 338,700 possible pairs produced no flagged matches. **This is a heuristic screen, not proof of no shared patient.** Exact decoded canonical voxel checksums cannot detect different acquisitions, regridding or major crop changes; coarse normalised 16³ similarity threshold is unvalidated. Dataset provenance review remains a condition before final claims of independent external generalisation.

## E07. Historical Phase 2 2.5D study and leakage comparability caveat

Existing historical Phase 2 anatomy-conditional SNN refinement reached about **0.7313** Dice; matched analog around **0.7311** on a GT-derived z-cropped task. These are distinct *historical exploratory* results, **not full-volume 3D CNN–SNN results** and **not directly comparable** to the full-image 0.80618 nnU-Net. Small difference alone does not demonstrate SNN advantage. The existing Phase 2 energy study had **0 valid energy trials** until AMD telemetry was fixed. Preserve all prior logs; do not erase or mislabel this study as successful full-volume SNN energy optimisation.

## E08. Above-0.9 literature benchmarking is a research target, not an achieved score

Relevant literature: *A Novel Pipeline for Adrenal Gland Segmentation: Integration of a Hybrid Post-Processing Technique with Deep Learning*, Journal of Digital Imaging, DOI `10.1007/s10278-025-01449-y`, `https://doi.org/10.1007/s10278-025-01449-y`. The paper reports **0.91–0.93** adrenal Dice for certain models after postprocessing on **40 AMOS CTs**, with separate gland analyses and test-time augmentation/postprocessing. **Do not directly rank this against our 100 full-volume CT Dice** without verifying identical cases, volume-vs-slice metric aggregation, GT access, cropping or other task characteristics. Other reported adrenal values up to ~0.90 in recent external/multicentre CT studies have different patient cohorts and protocols. The objective is *honest* >0.90 **full-volume bilateral patient-level Dice** on a clearly defined independent cohort with accompanying boundary metrics; extra data does not guarantee achieving it.

## E09. Required next experiments and design gates

1. Frozen, predeclared **annotation QA** of stratified 60 TotalSegmentator *train* candidates and independent review of ~15 official test cases. Sample seed, IDs, metadata, reviewer judgements, revision history, explicit annotation inclusion criteria archived. Incomplete labels and extreme volumes undergo focused inspection. Do NOT select test cases based on model errors.
2. **External generalisation baseline first**: existing AMOS-trained fold-0 CNN, no finetuning, no postprocessing selected on AMOS holdout, predict the separately prepared quality-controlled official TotalSegmentator v3 test cases and score left/right volumetric Dice, HD95 and NSD@2mm, including failure distributions, valid-case denominators and confidence intervals. Keep 76 test cases reserved, do not inspect external test accuracy to adjust architecture.
3. **Expanded-data ablation**: original CNN as fixed reference; train CNN from scratch on AMOS + approved TotalSegmentator **training** cases (not test), explicit validation subset from development-only pool; an AMOS-only model run with same protocol, plus optional TotalSegmentator-only control. Freeze case IDs and label mapping.
4. **True full-volume 3D SNN vs matched analog refiner**: same CNN backbone, ROI proposal, parameters/FLOPs estimates, training/compute constraints, inputs and splits; compare 3D SNN and non-spiking analog on unseen validation, external test once. Document cross-laterality error rate, per-gland Dice, HD95, NSD, false-positive volume, CT slice-thickness subgroup, latency, GPU energy and peak VRAM.
5. **Conditional compute**: test skip/analog/spiking with strict no-ground-truth gating, predefined utility vs budget, overhead measured; do not assume sparse spiking saves energy on GPU. Keep this as a separate algorithm paper or clearly labelled secondary experiment.
6. Run paired patient bootstrap confidence intervals, robust multi-seed training where feasible, predeclared primary comparison and explicit failure analysis. Do **not** use heldout data to tune augmentation, postprocessing, thresholds or radiological QC exclusion.

## Reproducibility / anti-context-loss protocol

1. Keep this document and source scripts in Git with commit history; future chat context can be re-established from this file without relying on chat memory.
2. Keep **original machine-generated artefacts** under Kelvin2 `runs/`, `nnunet_results/`, with immutable copies/manifest and checksums. **Do not check in private CT volumes, NIfTI predictions or model checkpoints** into a public repo, or raw patient/pathology metadata.
3. Use `scripts/snapshot_adrenal_research_evidence.py` to capture per-case metrics, audit summaries, versions, file SHA-256 hashes and the *three raw energy trials* into a versioned local manifest. Back it up to approved institutional storage. A textual paper register does not replace this verified copy.
4. Update this register after *every* experiment—successful or failed—with hypothesis, branch+commit, SLURM ID, dataset manifest ID, baseline/control, numerical outputs, limitations and go/no-go decision.
5. Manuscript calculations should be regenerated directly from preserved CSV/JSON, not recollected from summaries. Preserve source of each number and distinctions between computed, illustrative/oracle and measured results.

### Manuscript-ready baseline statement

> On 100 held-out, uncropped AMOS abdominal CT examinations, a 3D full-resolution nnU-Net attained left and right adrenal Dice scores of 0.8119 and 0.8004 (mean bilateral Dice 0.8062). The mean HD95 was 4.10 mm and 3.71 mm for left and right glands, respectively; NSD@2mm was 0.9279 and 0.9271. Across three independently repeated full-volume inference commands on one AMD Instinct MI300X, measured GPU socket energy was 49.319 ± 0.157 kJ per scan and end-to-end nnU-Net prediction wall time was 76.563 ± 0.057 s per scan (mean ± sample SD). Counter-derived energy agreed with independently integrated power; measurements cover the whole prediction subprocess and exclude CPU/node power. Class-wise connected-component postprocessing yielded no overall improvement (largest-only 0.8036, relative 15% 0.8061). All reported energy repeats use the same model checkpoint and the same 100 CT cases.

**Open questions:** annotation QC evidence; provenance-derived patient independence; externally validated full-volume Dice; effectiveness and GPU energy of fully 3D SNN; fully matched analog comparison; architecture-specific improvements above 0.90.


## E10. Frozen external-test dataset preparation + QA sampling — 10 November 2026 (NEW)

On Kelvin2 under the established project root, the three sequential CPU workflows completed far enough to produce the following verified-by-user artefacts:

- **Institutional research evidence archive**: `runs/research_evidence_archives/adrenaspike3d_evidence_20261110T161746Z.tar.sha256` exists (reported 16:18, 10 Nov). The corresponding `.tar` and its SHA256 verification against that file **still require checking** with `sha256sum -c`. A copy on the same Kelvin2 filesystem is **not** an independent archival backup.
- **Frozen TotalSegmentator QA selection**: `runs/totalsegmentator_v3_qc_plan/plan_summary.json` with seed **20261110**, 1053 eligible train, 76 eligible external-test, sampled training QC **60**, sampled external-test QC **15**, deterministic greedy metadata diversity with no model predictions.
- QA plan source audit SHA-256 `efa5f858a129d9475f2ea5ab168592f00d0fb6920d5b5633736582eb8a4e9f55`; metadata SHA-256 `b9ca9a5ce9407f9caad1d039c7574cba154efe93f3a5802bc1aed2e05a732922`.
- **Separate external-test-only NIfTI cohort**: `nnunet_raw/ExternalTotalSegV3AdrenalTest_v1/external_test_manifest.json` contains exactly **76** `tsv3_sNNNN` official test-case identifiers, reference-only image symlinks and labels; source inventory SHA-256 `995a314a4638d8356367c468d4bd3a585738ebddf770e766f3ea748dd9a8644f`.
- External source path remains `/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/data/totalsegmentator_v3/extracted/Totalsegmentator_dataset_v300`, **within user-approved parent path** `/mnt/userapps/isrcdl_apps/mfayemiwo/`.
- **Status: NO HUMAN QC SIGNOFF YET.** Plan explicitly states `UNREVIEWED; no automatic approval` and manifest `PRELIMINARY_LABEL_QUALITY_REVIEW_PENDING`. It is scientifically wrong to claim these source annotations were approved based on successful file conversion or baseline model training. Montage counts and manual annotation decisions are not yet reported.
- New `scripts/validate_totalseg_v3_qc_review.py` plus automated regression tests require a completed, distinct 15-case external-test review copy, all ACCEPT/PASS, human reviewer/date, original frozen fields unchanged, human attestation acknowledging source provenance and unreviewed-label limitations. The GPU external evaluation SBATCH now **verifies this signed, hash-checked local gate** instead of accepting only an environment variable. This authorises *exploratory* evaluation, not clinical certification or patient-independence.
- Commands/precise rubric: `docs/totalseg_v3_adrenal_annotation_qc_rubric.md`. Preserved originals are mandatory, including all negative/QUERY/REJECT judgements. Further decisions about expanded-data CNN training or SNN comparisons remain pending.

**Decision:** Finish source-only image and label reviews and archive checksum verification before opening the exploratory external-test Dice. Do not add any of the 76 official test cases to training.

## E11. Protocol deviation authorised — exploratory external evaluation without manual QA (10 November 2026)

The investigator explicitly elected to **skip manual source-annotation QC for the next external CNN assessment** and cancelled the previously submitted signoff-gated Slurm job. Do not imply any human reviewer inspected or approved the 76 gland masks. This is a transparent change in evaluation scope, not proof of source label accuracy.

New standalone job: `scripts/evaluate_totalseg_external_unreviewed.sbatch` (separate from and does NOT modify the reviewed job). It uses original immutable AMOS-only nnU-Net Dataset701 fold-0 checkpoint, the exact official TotalSegmentator v3 76-case bilateral external-test manifest, left=1/right=2, and no model retraining, tuning or postprocessing. Frozen inventory checksum, exact case count, CT symlink source, label ID mapping and original model SHA-256 are checked and recorded in:
`runs/nnunet3d_external_tsv3_unreviewed_provenance.json`.

Predictions: `runs/nnunet3d_external_tsv3_predictions_fold0_unreviewed/*.nii.gz`. Per-case metrics: `runs/nnunet3d_external_tsv3_metrics_fold0_unreviewed.csv`. Summary: `runs/nnunet3d_external_tsv3_metrics_fold0_unreviewed.summary.json`. **At time of writing no results are available.** Reconcile the actual Slurm job ID, Git SHA, timing, scores, failure denominators and uncertainties when output is received.

Dataset annotations were mechanically audited but **not independently adjudicated**. This external test selection includes only 76/109 official test cases with bilateral nonempty reference masks; clinical/source annotation quality and true patient independence are unverified. The results are exploratory evidence of domain generalisation, *not* claims of clinical validation. As this external test is now being examined, it must remain a frozen benchmark: new architectural and hyperparameter selections must use development-only data, not the 76 external Dice scores. Preserve all 76, including low-Dice cases; no post hoc drops or cherry picking.

The original signed-QC workflow remains available independently. The manual-QC bypass may not be described later as completed human quality review, nor should it erase the prior audit documentation.

## E12. External TotalSegmentator v3 inference failure and orthogonal-NIfTI repair — 11 November 2026

**Observed failure (not a Dice result):** the first exploratory `scripts/evaluate_totalseg_external_unreviewed.sbatch` job started nnU-Net inference for all 76 prepared TotalSegmentator v3 official test CTs but terminated in nnU-Net's SimpleITK NIfTI loader with:

```text
ITK ERROR: ITK only supports orthonormal direction cosines. No orthonormal definition found!
RuntimeError: Background workers died
```

The user found **3 completed NIfTI predictions** in `runs/nnunet3d_external_tsv3_predictions_fold0_unreviewed/`, while the log had reached `tsv3_s0040`. The fourth scan's export may not have completed; the error involves CT spatial header geometry. The GPU job **did not complete**, the external metrics summary **does not exist**, and **no external Dice can be claimed**. Background-worker termination alone is not evidence of OOM. Do not re-submit the same unmodified failing script; keep its partial predictions and failure logs as experimental evidence.

**Reproducible corrective protocol:** `scripts/repair_totalseg_external_ct_geometry.py`:
- Preserve original 76 CT NIfTIs as supplied and the original 76 two-gland reference masks. No GT-derived bounding boxes, cropping or label-based interpolation to generate model inputs.
- Run SimpleITK/ITK metadata-read preflight and orthogonality check on every CT. Symlink directly if the original header is readable and orientation orthogonal; otherwise linearly interpolate the **CT only** to an orthonormal RAS+ world grid at the physical original voxel-size magnitudes. Record every changed case and its before/after shape/affine in a local `geometry_manifest.json`.
- Use unchanged original AMOS-only nnU-Net checkpoint Dataset701 fold0 to predict all 76 prepared CTs. Predictions and metadata are stored in **new directories**, never mixed with the failed run's 3 partial files.
- Restore each prediction to the **original CT voxel grid** with nearest-neighbour label resampling (order 0) using physical NIfTI affines. For inputs that were symlinked unchanged, do not interpolate prediction masks at all.
- Score the restored native-grid masks on all 76 original class-1-left/class-2-right reference labels with the same Dice, HD95, NSD evaluator as the original AMOS benchmark. No human source-label QC is claimed. Report geometry preparation protocol and number of affected cases in the manuscript's external-generalisation methods.

CPU-only preparation: `scripts/prepare_totalseg_external_ct_geometry.sbatch` (64G, no GPU), output `nnunet_raw/ExternalTotalSegV3AdrenalOrtho_v1/geometry_manifest.json`.

GPU inference and restoration: `scripts/run_totalseg_external_ortho_inference.sbatch` (MI300X), output:
- prepared CT-space predictions `runs/nnunet3d_external_tsv3_ortho_predictions_fold0/`;
- native CT-space predictions `runs/nnunet3d_external_tsv3_native_predictions_fold0/`;
- external patient metrics `runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.csv`;
- external summary `runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.summary.json`;
- native geometry restoration evidence `runs/external_geometry_restore_summary.json`.

Until both successor jobs finish and all cases are accounted for, no external performance has been measured. This is an engineering fix to ensure valid medical-image physical geometry, **not a new model architecture** and **not proof of SNN contribution**.


## E13. Completed external TotalSegmentator v3 generalisation evaluation — reported 16 Nov 2026

**CONFIRMED external result:** initial AMOS-only `Dataset701_AMOSAdrenalCT`, 3D fullres nnU-Net fold 0, unchanged trained checkpoint, was applied to **all 76** preselected official TotalSegmentator v3 test volumes with bilateral nonempty masks; the reference annotations were **not manually reviewed**. Mean patient-level bilateral Dice **0.6682536367697085**. The prior AMOS holdout 100-case mean was **0.8061840313892723**, an **unpaired** cross-cohort difference of **−0.1379303946195638 Dice**, **not** a paired model-improvement measure.

Source evidence reported by investigator from completed Kelvin2 job:
- `runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.csv` — all 76 per-patient Dice left/right/bilateral, HD95/NSD (not yet transmitted/independently opened in the chat).
- `runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.summary.json` — summary location reported by job; **full contents have not yet been seen**, so left/right means, HD95/NSD, confidence intervals and empty-prediction counts are currently NOT KNOWN.
- `runs/nnunet3d_external_tsv3_native_predictions_fold0/` — complete native-reference-grid prediction NIfTI directory (success log confirms 76/76 scored); `runs/nnunet3d_external_tsv3_ortho_predictions_fold0/` — preprocessing-grid outputs.
- `nnunet_raw/ExternalTotalSegV3AdrenalOrtho_v1/geometry_manifest.json`: `status: COMPLETE`, `case_count: 76`, `regridded_count: 9`. Regridded CT IDs: `tsv3_s0235`, `tsv3_s0236`, `tsv3_s0311`, `tsv3_s0561`, `tsv3_s0735`, `tsv3_s0802`, `tsv3_s0829`, `tsv3_s1174`, `tsv3_s1276` (9/76 = 11.84%). Original CT/masks were not overwritten; regridded images used linear interpolation, and returned predictions were mapped to native grid with nearest neighbour. Source gland GT not consulted when preparing model inputs.
- Job stdout ends: `Scored 76/76 tsv3_s2137: bilateral Dice=0.2354`, JSON `test_cases:76`, `mean_bilateral_dice:0.6682536367697085`, `cohort_status: PRELIMINARY_LABEL_QUALITY_REVIEW_PENDING`, `SUCCESS: 76 full-volume exploratory external CTs scored on ORIGINAL reference grid.` Last stderr lines show nnU-Net tqdm and `Collecting results: 100% | 76/76`; no final exception. Actual Slurm job ID still not provided.
- Examples from user-provided tail (cases 55–76, **not** a population estimate): `tsv3_s1773`: bilateral Dice **0.0625**; `tsv3_s1827`: **0.0000**; `tsv3_s2123`: **0.9226**; `tsv3_s1854`: **0.8934**; `tsv3_s2137`: **0.2354**. These are descriptive failure examples, NOT permission to change test eligibility or tune on test labels.

**Interpretation and boundaries:** Generalisation of the original AMOS-only CNN is substantially worse on the distinct TotalSegmentator external cohort; causes not established (scanner, patient mix, annotation conventions, gland visibility, geometry effects possible). Nine geometry-repaired cases MUST be retained in whole-cohort reporting. The TotalSegmentator test subset contains **76/109** official test cases selected by presence of bilateral source annotations: report the parent denominator and any selection bias, never silently represent the 76 as all official tests. External labels were not manually quality checked and source overlap screen is heuristic, not proof of patient independence. Do not fit architecture/postprocessing/thresholds or select the best training run using these 76 external Dice scores. A Dice exceeding 0.90 has **not** been achieved by this model.

**Action:** Preserve every raw per-patient value, summary JSON, both 76-mask directories, geometry manifest, Slurm logs and failed first run as protected on-HPC evidence plus approved institutional backup. Internal training/validation decisions for the next enlarged-data CNN and matched analog/SNN models must be fixed using training-split-only data, independent of these 76 scores.


## E14. Prospective exploratory expanded-data CNN development, 16 November 2026 — code ready; no new model trained

After completion of the 76-case external test with bilateral Dice **0.6682536367697085**, the investigator and team maintain the *previously stated* plan to add TotalSegmentator **official training** images to improve CT-domain generalisation and aim for full-volume Dice >0.90. **This is a new planned experiment, NOT evidence of a performance gain.** The external test has already been observed and must not be treated as model-development validation.

**New independent nnU-Net dataset namespace:** `Dataset702_AMOSTotalSegAdrenalCT` under `/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation/nnunet_raw/`. The original `Dataset701_AMOSAdrenalCT` and its 100 CT heldout scans and checkpoint are untouched.

Freeze script `scripts/freeze_expanded_amos_tsv3_manifest.py`: creates a non-overwriting local `runs/expanded_dataset702/plan.json`, seeded **20261116**. Frozen split expected:
- Original **200 AMOS `imagesTr`**: all 200 used in fold0 training; **none** of the 100 original `imagesVa` included in this dataset.
- TotalSegmentator official **train** with both nonempty binary affine-aligned adrenal masks: **1053** candidate CTs, of which **895** used in fold0 development training and **158** held out as fold0 internal validation, deterministic site-stratified sample at 15% using no external model scores.
- Overall **1253** imagesTr (train+internal validation), fold0 **1095 train** and **158 validation**.
- All **109 TotalSegmentator official test** cases excluded from training, including the fixed 76 bilateral-labelled external test and 33 other official test cases. All original 100 AMOS heldout excluded.

**Automated source-label QC ONLY:** the 1053 TotalSegmentator masks are candidate annotations from the dataset audit, not clinically verified. The user elected to skip manual review; any manuscript accuracy statements must declare annotation provenance and limitations. No label-derived crop, no test-case removal, no training from external labels.

**Geometry and binary label conversion:** `scripts/prepare_expanded_amos_tsv3_dataset702.py` reuses AMOS original training image/label symlinks. For each TS candidate it checks left/right binary masks, source CT/label alignment, nonzero gland masks and no class overlap; symlinks a valid CT or linearly regrids a sheared CT to orthonormal physical coordinates, applying **nearest-neighbour** spatial resampling to both source adrenal masks on that same grid. Neither original source CT nor annotation files are modified. Any invalid mask or impossible geometry leads to an explicit failure; no cases silently excluded. Intermediate outputs resume after walltime; a manifest is frozen.

**Preparation and training scripts (code committed, no results yet):**
- `scripts/prepare_expanded_amos_tsv3_dataset702.sbatch`: CPU-only freeze/prepare 1253 cases, `runs/expanded_dataset702/plan.json`, `nnunet_raw/Dataset702_AMOSTotalSegAdrenalCT/geometry_manifest.json`, aligned labels and original CT symlinks/resampled files.
- `scripts/preprocess_expanded_amos_tsv3_dataset702.sbatch`: CPU-only nnU-Net 3D fullres preparation and `scripts/install_dataset702_frozen_split.py` to write immutable `nnunet_preprocessed/Dataset702_AMOSTotalSegAdrenalCT/splits_final.json`.
- `scripts/train_expanded_amos_tsv3_dataset702.sbatch`: independent MI300X checkpoint-resumable 3D nnU-Net fold0, no automatic external test evaluation.
- `scripts/report_external_tsv3_unreviewed.py`: descriptive 76-case result audit (mean/SD Dice sides, zero/low scores, 9 geometry subgroup) with all denominators and no test tuning. This is a diagnostic manuscript artifact, not a training criterion.

**Benchmark methodology:** compare C0 AMOS-only nnU-Net to C2 new expanded CNN on the same frozen full-volume evaluation data; assess whether new data helps AMOS as well as TotalSegmentator and look for domain imbalance. Must report that C2 has *more training cases*, not architectural superiority. Use the internal 158 TotalSegmentator development validation CTs for model selection only. Then develop C3 matched analog refiner and C4 genuine 3D SNN refiner with identical C2 backbone, full-volume CT and same ROI proposals/input information; energy protocol applies whole-subprocess MI300X GPU socket Joules, latency and VRAM. Do not claim useful SNN until matched-control quantitative results exist.

**Current status:** scripts are version-controlled, CPU synthetic tests verify split disjointness and sheared source CT-mask resampling. **No Dataset702 HPC preparation/preprocessing/train job has been reported as run yet, and no expanded-data Dice or SNN results are available.**


## E15. Completed 76-case external Dice distribution and queued data-preparation jobs — 16 November 2026

The user executed `python -u scripts/report_external_tsv3_unreviewed.py` against the original full 76-case CSV and the original CT geometry manifest. It returned `DESCRIPTIVE_COMPLETE_NO_MODEL_TUNING` and wrote `runs/nnunet3d_external_tsv3_failure_audit_unreviewed.json`. **Actual source findings, no synthetic examples:**

| Measure | n | Mean | Sample SD | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Per-patient bilateral Dice | 76 | **0.6682536367697084** | **0.3021735302897953** | **0.8109285387468024** | 0.0 | 0.93683284457478 |
| Left adrenal Dice | 76 | **0.6676954722695573** | **0.3174164430486637** | **0.8313004175532428** | 0.0 | 0.9381818181818182 |
| Right adrenal Dice | 76 | **0.6688118012698596** | **0.3156895449131599** | **0.8175679034754972** | 0.0 | 0.9425562470081379 |

Failure and high-score distribution: 11/76 bilateral Dice below 0.20; 16/76 below 0.50; 6/76 exactly zero; 6/76 at least 0.90; 8 left-gland prediction masks empty; 9 right-gland prediction masks empty. **These counts overlap** (do not sum into a total failure count). Larger median than mean indicates a severe lower tail of failures rather than uniformly middling predictions. Differential causes are **unproven** (missing glands, laterality/coverage, label source and acquisition variation may contribute). No case has been removed or selected for model tuning.

Nine original nonorthonormal/sheared CTs repaired for inference: bilateral Dice within that subgroup **n=9, mean 0.6671142014744535, sample SD 0.35284947825308444, median 0.840556223534947, minimum 0, maximum 0.8710121647669904**. The subgroup average resembles the overall average but cannot establish equivalence or a causal conclusion. All 9 included in 76-case mean. For the sheared cases, the original evaluator's HD95/NSD voxel-spacing metric does not fully model sheared world-space distances: disclose this limitation and pursue physically affine-correct surface metrics before publication.

The complete 76-case patient audit remains in `runs/nnunet3d_external_tsv3_failure_audit_unreviewed.json`, the underlying immutable per-patient CSV in `runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.csv` and the original full scores with bootstrap confidence intervals in `runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.summary.json` (still to be copied to chat for precise HD95/NSD/CI reporting).

**Research archive and Dataset702 preparation accepted by Kelvin2 SLURM:**
- Evidence backup submission **job 11026503**, `scripts/archive_adrenal_manuscript_results.sbatch`.
- Dataset702 candidate input preparation submission **job 11026504**, `scripts/prepare_expanded_amos_tsv3_dataset702.sbatch`, dependency `afterok:11026503`.
- The user confirmed *submission*, **NOT completion** of either job. Do not call Dataset702 prepared, trained, or evaluated until verified by output logs and frozen manifests. All results remain under `/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation`; user-controlled backup transfer to approved independent storage still necessary.

**Design consequence:** Dataset702 training may help, but Dice >0.90 is unachieved, and no matched spiking/analog benefit has been measured. These external results are descriptive; they must not drive parameter selection or retroactive test exclusions. Primary next analysis is development-only generalisation/accuracy plus balanced analog/SNN comparison.


## E16. Complete original-CNN external metrics and 95% bootstrap intervals — 17 November 2026

**SOURCE:** Investigator supplied full contents of Kelvin2 file `runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.summary.json` (generated by `scripts/evaluate_totalseg_external_adrenal.py`). This supplants the previously unresolved detail in E13–E15; the raw source JSON/CSV on Kelvin2 remains authoritative. Results are from *the same frozen AMOS-only Dataset701 nnU-Net 3D fullres fold0 model*, **76 original TotalSegmentator v3 official test CTs with both gland reference masks nonempty**, no training or postprocessing, with previously documented 9 sheared CTs preprocessed and predictions restored to native geometry. Manual annotation review not performed; reported original cohort state `PRELIMINARY_LABEL_QUALITY_REVIEW_PENDING`. No cases have been removed.

| Patient-level metric | External mean | Valid cases | Total cases |
|---|---:|---:|---:|
| Left Dice | **0.6676954722695573** | 76 | 76 |
| Right Dice | **0.6688118012698595** | 76 | 76 |
| Bilateral average Dice | **0.6682536367697085** | 76 | 76 |
| Left HD95 (mm) | **9.42723013920646** | **68** | 76 |
| Right HD95 (mm) | **3.24714968569406** | **67** | 76 |
| Left NSD@2mm | **0.8308720085785088** | **68** | 76 |
| Right NSD@2mm | **0.8741154443036228** | **67** | 76 |

**Bootstrap 95% percentile confidence intervals for 76-case patient-level mean Dice**; 10,000 draws, random seed **20261110**, bootstrap case resampling in the original evaluator:

| Target | Reported mean | 95% bootstrap percentile CI | Cases |
|---|---:|---|---:|
| Left Dice | 0.6676954722695573 | [**0.5939310002957708**, **0.7370892077674296**] | 76 |
| Right Dice | 0.6688118012698595 | [**0.5967355321176755**, **0.7364344924941189**] | 76 |
| Bilateral mean Dice | 0.6682536367697085 | [**0.5989608291793763**, **0.7332477808211579**] | 76 |

Predicted gland-empty counts: **left 8**, **right 9**; these constitute the missing-surface HD95/NSD entries (NaN), **not omitted Dice cases**. The 76-case Dice average includes zeros, as required for honest failure accounting. Since boundary metrics are defined only for 68 and 67 positive-prediction cases, comparing their means directly against full-cohort Dice or against AMOS without checking denominators can mislead. Larger left HD95 vs right is a feature of the *evaluable masks*, not evidence of a gland-specific causal defect; assess high-distance outliers, per-patient HD95 and point-cloud geometry.

**Important geometry limitation**: source CT NIfTI affines for nine cases were sheared/non-orthogonal. Original evaluator calculates HD95/NSD with only header zooms/spacings in an axis-transposed voxel grid (does not use full 4×4 world affine). These nine-case reported physical-distance values are **approximate** even though predictions and reference labels have matching original voxel grids. Before manuscript-ready boundary claims, independently implement or verify affine/world-coordinate-aware boundary distances for those 9 cases, including empty prediction handling; preserve these original preliminary values as historical rather than overwrite.

Historical cross-dataset comparison (different populations, unpaired): AMOS 100-CT bilateral mean Dice **0.8061840313892723**, TotalSegmentator 76-CT bilateral mean Dice **0.6682536367697085**. Numerical difference is approximately **−0.13793** but must NOT be described as a paired statistical effect, and a paired bootstrap CI of this difference is not available. Existing C0 measured GPU socket energy **49.319 ± 0.157 kJ/full AMOS CT**; this 76-case external test did **not** yield a new matched energy measurement. No expanded-data CNN or 3D SNN result is claimed here.

**Retain for Paper I:** report the 76-case parent cohort from original 109 official tests, 9/76 CT geometry corrections, full patient-level Dice distribution (including 6 bilateral zero-Dice and 16 below 0.50), uncertainty intervals, boundary valid denominators, and original source-annotation quality limitation. Explicitly reserve all 76 cases from model selection. The next scientific experiment is development-only Dataset702 CNN improvement and matched analog/SNN refiners, not test-case tuning.


## E17. Dataset702 preparation succeeded; non-destructive preprocessing safeguard added — 17 Nov 2026

**Kelvin2 user-confirmed:** `dataset702_cpu_prep.11026504.out` ends `SUCCESS: Dataset702 has 1253 development cases; 1095 fold0 train and 158 fold0 val.` Error log showed module loading messages only, no failure. This confirms the CPU image/label **preparation** stage completed for 200 AMOS source-train CTs and 1053 TotalSegmentator *official training split* candidates, with prospective 895/158 TotalSegmentator train/internal validation partition. **This is not yet nnU-Net fingerprinting, preprocessing, GPU training or model evaluation.** Preserve the input plan `runs/expanded_dataset702/plan.json` and `nnunet_raw/Dataset702_AMOSTotalSegAdrenalCT/geometry_manifest.json`. Source mask quality remains automated-screen only, without human review; original AMOS heldout 100 and 109 TS official test cases remain excluded.

**Critical correction to earlier planned CPU preprocess script:** Investigation of the current upstream nnU-Net v2 `DefaultPreprocessor.run` found that it **unconditionally removes an existing fullres configuration output directory with `shutil.rmtree(output_directory)`** before processing all training cases. Therefore the earlier `scripts/preprocess_expanded_amos_tsv3_dataset702.sbatch`, which invoked `nnUNetv2_plan_and_preprocess -d 702 ...` directly if partial data existed, would **not be genuinely resumable** across an expired 24h SLURM job. This was caught **before any preprocess job was reported to have run**.

**Corrected robust workflow now committed:**
- `scripts/preprocess_expanded_amos_tsv3_dataset702.sbatch` retains a 24h CPU-only/192GB allocation but first runs `nnUNetv2_plan_and_preprocess -d 702 --verify_dataset_integrity --no_pp -c 3d_fullres -npfp 4 --no_pbar` **only if fingerprint/plans do not yet exist**. `--no_pp` prevents default case preprocessing from wiping prior results. No model/training settings are changed.
- `scripts/resume_nnunet702_preprocessing.py` uses installed nnU-Net `configuration.preprocessor_class.run_case_save` exactly as upstream for missing cases; per-case binary image, segmentation and properties are staged under a case-isolated temporary directory and atomically committed with `.pkl` LAST. Both modern Blosc2 (`.b2nd` + `_seg.b2nd` + `.pkl`) and legacy `.npz` + `.pkl` files are detected. Frozen development split, geometry report and plans/fingerprint source hashes, plus installed nnU-Net version, are enforced in a non-overwriting method manifest. Completed cases are **never** overwritten by resumed job.
- 4 CPU preprocessor worker processes use a 16-core node, 192GB RAM, single-threaded ITK/BLAS per process; jobs checkpoint between small batches and stop requesting new cases by 20h into 24h allocation, leaving ~4h margin. Resubmitting **the same sbatch file** resumes missing cases. Status code **3** explicitly means incomplete but saved; no automatic training is triggered. System/policy preemption can still interrupt active cases, but only their incomplete temporary staging must be redone.
- `scripts/finalize_nnunet702_preprocessing.py` checks every one of 1253 cases and atomically copies the original training gland masks to `nnUNet_preprocessed/Dataset702_AMOSTotalSegAdrenalCT/gt_segmentations`, verifying SHA-256. Then the stage builds foreground sampling-location metadata with `nnUNetv2_extract_sampling_locations` if available (or validates legacy per-case `class_locations`), and installs immutable `splits_final.json` with **1095 train + 158 val**.
- Regression tests `tests/test_dataset702_preprocessing_resume.py` cover legacy/current case output formats, checkpoint/atomic commit, plan overwrite rejection and 1253-mask GT copy. GitHub CI syntax/tests cover remaining Dataset702 scripts.

**Rationale for CPU vs GPU:** standard nnU-Net v2 fingerprint extraction, planning and preprocessing use CPU-oriented ITK/NumPy/resampling/compression and multiprocessing; requesting MI300X would not automatically accelerate these functions. GPU-specific preprocessing would require a separately validated alternative numerical pipeline and would not be the unmodified baseline. 24h limit may still require multiple CPU job allocations; now work survives time limits without restarting from case 1.

**Research status at this update:** Dataset702 source dataset prepared, no completed preprocessing or trained expanded-data CNN reported, no 3D SNN/analog matched-control results. Original full-volume AMOS and 76-case external results and failure-count evidence remain preserved in E01–E16.


## E18. Dataset702 checkpointed full-resolution preprocessing COMPLETE — 17 November 2026

**Investigator-reported Kelvin2 log evidence:** the checkpoint-safe `scripts/preprocess_expanded_amos_tsv3_dataset702.sbatch` job finished with all **1,253/1,253** original Dataset702 `gt_segmentations` reference masks copied and verified; `legacy class_locations` checked for all **1,253** nnU-Net per-case properties; `scripts/install_dataset702_frozen_split.py` returned `status: SUCCESS`, `action: installed`, `fold: 0`, `train_count: 1095`, `val_count: 158`.

Final stdout explicitly reported: `SUCCESS: all 1253 Dataset702 fullres cases preprocessed, original GT copied, foreground sampling locations ready, and 1095/158 fold0 installed.` There was no final error in the supplied log. Frozen split location:

`/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation/nnunet_preprocessed/Dataset702_AMOSTotalSegAdrenalCT/splits_final.json`

Source images and labels: `nnunet_raw/Dataset702_AMOSTotalSegAdrenalCT/`, as frozen before preprocessing. Original Dataset701 trained model and 100-case holdout remain untouched. All 109 TotalSegmentator official test CTs remain outside Dataset702. The 1,053 TotalSegmentator training reference masks underwent automatic integrity checks only, with **no human adjudication**.

**Stage status: completed CPU preprocessing and validation split installation; Dataset702 MI300X training NOT YET REPORTED.** The next action is separate GPU job `scripts/train_expanded_amos_tsv3_dataset702.sbatch`. It uses `nnUNetv2_train 702 3d_fullres 0 -device cuda` (CUDA compatibility under AMD ROCm), GPU partition `k2-gpu-amd`, MI300X 1x, 16 CPU cores, 96GB host memory, 72-hour allocation, checkpoint-latest resume with `--c`, and isolated output under `nnunet_results/Dataset702_AMOSTotalSegAdrenalCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/`. Training should **not** yet be described as producing improved Dice or any SNN advantage. Training/validation metrics must be evaluated before optional frozen test-set benchmark comparison. Model selection and architecture tuning must rely on the frozen internal 158 development cases, not previously inspected 76 external-test Dice.

**Handoff instructions:** `cd /mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation && git pull --ff-only origin research/adrenaspike3d && sbatch scripts/train_expanded_amos_tsv3_dataset702.sbatch`. Preserve training logs, checkpoint hashes, configuration, training plots, nnU-Net validation predictions and all failures. Do not run original `nnUNetv2_plan_and_preprocess` without `--no_pp` on completed Dataset702.
