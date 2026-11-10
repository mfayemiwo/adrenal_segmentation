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
