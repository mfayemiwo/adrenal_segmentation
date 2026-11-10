# AdrenaSpike3D — prospective next-phase protocol (10 November 2026)

**Paper I** only. This is a design plan, NOT completed experimental evidence. Do not rewrite it after viewing external-test scores without clearly versioning changes and marking analyses post-hoc. For all achieved results and machine file paths see `docs/research_evidence_register_2026-11-10.md`.

## A. Research objective, primary claim and benchmarks

1. **Accuracy target:** exceed 0.90 mean bilateral patient-level 3D Dice on truly held-out full abdominal CT, **if observed**; lower boundary error and robust worst-case performance also matter. The frozen AMOS-only full-volume 100-CT baseline is 0.806184. No posthoc case dropping or non-CT label leakage.
2. **Scientific SNN question:** Does temporally persistent/spiking refinement add meaningful accuracy, generalisation, or measured GPU-efficiency value over a truly matched *analog* 3D refiner at equivalent information and compute budgets? A sparse spike count is not evidence of power savings. A finding that SNN fails to help is also scientifically interpretable but must not be marketed as success.
3. **Literature comparator:** DOI 10.1007/s10278-025-01449-y reports ~0.91–0.93 adrenal Dice after postprocessing on a different AMOS 40-case evaluation protocol. The numerical goal is ambitious, but claims of outperformance require identical patient cohort, prediction unit, averaging convention, and fair handling of crop, two separate gland networks, test-time augmentation and negatives. A ~0.90 Dice on another data domain is not an apples-to-apples ranking.

## B. Frozen datasets and review gates

| Cohort | Purpose | Rule |
|---|---|---|
| AMOS original full CT training (200 CT) | Development | Never admit 100 prior `imagesVa` into training |
| AMOS original 100 `imagesVa` CT | Historical heldout and fixed score | Already inspected; any newly designed improvements on it are **exploratory** until independently validated |
| TotalSegmentator v3 official `train`, 1053 bilateral-annotation candidates | Potential new training data | Label QA + provenance check; preserve `train` split; sample internal development validation only from here/training pool |
| TotalSegmentator v3 official `test`, 76 bilateral-annotation candidates | Frozen external test | Source-only label-quality QC, independent identity/provenance review, existing original CNN evaluated *before* any training on TotalSegmentator; never tune to test outcomes |

- Reproducible audit: 1,939 source CT, 1,129 bilateral populated masks, 59 unilateral, 751 empty, 0 image/geometry read failures. 71 incomplete-labeled scans show a kidney-coverage clue; treat as uncertain, not certified true negatives.
- Exact canonical-voxel match and coarse thumbnail overlap screen across 300 AMOS CTs × 1129 TotalSegmentator bilateral CTs found 0 flagged pairs, `complete_screen=true`. The 0.985 threshold is unvalidated; no match is not proof of distinct patients.
- Version 3 of TotalSegmentator includes paediatric images and updated labels. Review original scanner slice-thickness metadata (not only distributed 1.5mm spacing).
- Before training/predicting, save `scripts/plan_totalseg_v3_adrenal_qc.py` deterministic QA plan (seed 20261110), 60 train and 15 independent test samples; `scripts/render_totalseg_v3_adrenal_qc.py` makes **reference-only** CT label overlays. Have an appropriately experienced reviewer document laterality, gland completeness, boundaries, scan coverage and possible disease/anatomic variants without exposure to model predictions. Document sampling limitations. Any exclusion reason must be predeclared and applied independent of prediction score.
- All new nnU-Net datasets must use **new IDs** and approved case manifests; **do not modify Dataset701**.
- Split IDs and checkpoint hashes must be frozen before external model comparison. CT image symlinks only, no anatomy-derived crop from held-out GT.

## C. Experiment sequence and ablations

**C0 — Existing, frozen control:** original AMOS-only `Dataset701_AMOSAdrenalCT`, nnU-Net v2 3d_fullres fold 0, 1000 epochs. Dice .806184 on 100. Measured MI300X GPU socket energy 49.319±0.157 kJ per full CT, 76.563±0.057 s/CT across three repeats. Preserve three original trial traces and their configuration.

**C1 — No-training external generalisation:** same original C0 checkpoint, no fine-tuning, no model selection, no component filtering, same whole-volume input, evaluate predeclared 76 TotalSegmentator v3 test scans with source-only QA and 95% patient bootstrap CIs. This is the external control and defines the domain gap; all 76 are reserved for evaluation only.

**C2 — Expanded-data CNN:** train nnU-Net v2 3d_fullres from scratch (or explicitly report transfer pretraining separately) on original AMOS train + approved TotalSegmentator *train* only, with stratified development validation from the training pool. Hold training compute and nnU-Net plans explicit. Additional ablation to disambiguate source/quantity: equalised-size AMOS/TS mixture versus all approved data where feasible. Never choose hyperparameters using either 100 AMOS holdout or 76 TS test scores.

**C3 — Analog refiner control:** end-to-end inference uses C2's *frozen* coarse CNN output and a 3D ROI proposal computed entirely from CT/model probabilities, not GT. Add a non-spiking 3D residual/refinement module with the same inputs, ROI/candidate locations, receptive field, effective depth and comparable parameter/train/compute budget as C4. Include refinement decision overhead in energy/latency.

**C4 — Spiking refiner:** use the same coarse C2 segmentation and *identical* data/ROIs as C3. Replace a predeclared part of the C3 refiner with genuine spiking state dynamics (e.g., surrogate-gradient LIF neurons with measured spike rates). Document time steps, resets, surrogate gradient, membrane time constant and updates; avoid cross-patient hidden-state leakage. Evaluate both **always-on** (C4a) and **selectively routed** (C4b) with the same quality gate given to C3. Record absolute mean Dice and paired difference C4-C3.

**C5 — Ablations:** (i) no refiner C2, (ii) analog vs SNN C3/C4, (iii) matched gating on/off, (iv) time-step/spiking memory ablations, (v) anatomy consistency using only predictions and CT geometry, (vi) parameter/FLOPs, training budget and GPU energy, (vii) thick vs thin original slice metadata and left/right class-confusion subgroup. A non-spiking conditional-refinement baseline is mandatory: gains from routing alone cannot be ascribed to spiking.

**ACTS-3D Paper II boundary:** tri-action utility routing algorithm on `research/acts3d-algorithm` is distinct and must not be described as achieved by C4 automatically. Keep its counterfactual reward and compute budget analysis independent; coordinate shared data manifests but do not mix source branches into active HPC jobs.

## D. A predeclared anatomically targeted challenge

From the existing holdout, `amos_0346` contains two predicted *right-class* components; largest is physically left and overlaps left reference (1677 voxels, 815 overlapping GT-left), smaller is physically right (1411 voxels, 1268 overlapping GT-right). Largest-component filtering reduced bilateral Dice .27734 to zero. A developing lateralisation consistency refiner is scientifically motivated, but **DO NOT tune on amos_0346**. Select error categories using *development* predictions and assess genuine generalisation on external heldout test. Anatomical side determined from verified orientation, patient midline and adjacent structures—not hard-coded sign of scanner x coordinate. Report false lateralisation rate per gland even if average Dice improves.

## E. Outcome definitions and statistical analyses

- Primary: per-patient bilateral `(Dice_left+Dice_right)/2`, mean over all eligible independent patient exams; report left/right separately.
- Secondary: HD95 mm and NSD@2mm each gland, false-positive empty slices, gland-volume error, laterality swaps, false-negative glands, failure quantiles, per-institution/original slice-thickness/age stratification where sample sizes support.
- Report metric denominators; HD95/NSD undefined for missing predicted surfaces must be counted explicitly and treated as failures in interpretation. No silent case removal.
- Use same per-patient CT cohort for paired C2–C4 comparisons. 95% patient-level paired bootstrap CIs and robust number of training seeds where feasible; distinguish uncertainty across 3 benchmark repeats from between-patient uncertainty.
- Pre-register acceptable effect size before model selection. Accuracy >.90, and **SNN gain over analog** (not merely over the unrefined CNN) are separate outcomes.
- Efficiency benchmark all candidates with identical CTs, batch, hardware, preprocessing, checkpoint load/caching, power counter method and ≥3 repeated executions. Compare measured GPU socket joules/CT, seconds/CT, sampled VRAM, operations and spike activity; evaluate full-volume end-to-end for a clinically useful SNN comparison. GPU socket energy excludes CPU/node power. Report overhead of conditional gating. No direct comparison to earlier GT-cropped 2.5D evaluation.

## F. Freeze points and execution authorization

1. [ ] Preserve final baseline evidence snapshot + approved institutional backup of original scans/weights/predictions.
2. [ ] Generate immutable QA sampling plan, annotation review overlays and adjudication log.
3. [ ] Prepare separate frozen official 76-case external test NIfTI labels (class 1 left / class 2 right; no training directory).
4. [ ] Document source annotation quality, provenance, human reviewer sign-off **before** opening C1 test scores.
5. [ ] Run C1 with original checkpoint, lock external summary as historical baseline.
6. [ ] Finalize enlarged training manifest and **development-only** split with no overlap to test.
7. [ ] Train C2 expanded CNN and label all code/SLURM commits.
8. [ ] Train and compare C3 analog and C4 SNN refiners on matched conditions; report all failures.
9. [ ] Assemble Paper I Tables: data demographics + QC, CNN metrics, postprocessing negative ablations, external generalisation, CNN+SNN/analog, energy/power/counter agreement, error stratification.
10. [ ] Keep final external model claims blind until evaluation protocol frozen; never iterate multiple designs using test results.

No results of C1–C5 are available yet. A manuscript may state C0/E01–E06 findings precisely but must not imply future CNN+SNN improvement or >.90 Dice has been achieved.
