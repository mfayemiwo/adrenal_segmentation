# ACTS-3D provisional preregistration and statistical analysis plan
**Created:** 3 November 2026. **Status: DRAFT; NOT FROZEN.** To freeze BEFORE training the Paper II policy / inspecting its confirmatory cohort, replace the TBDs with a signed commit/tag and final exact patient hashes, seeds, budget thresholds, sample counts, cost units, tests and multiplicity rules. Do not retroactively present this draft as a preregistration.

## A. Exact scientific hypotheses

**H_mechanism (spike-specific necessity):**
For a fixed set of gland/3D blocks, the same fitted SNN weights run with genuine ordered physical-z feature events outperform the same SNN with zeroed events, after applying identical ROI proposals, spatial context, postprocessing and forward passes. Secondary temporal interventions: reversed and permuted z-event order. Criterion is **measured held-out improvement**, not nonzero spikes.

**H_decision (new algorithm):**
At an internally calibrated, matched **total GPU energy per full CT** budget and the same frozen CNN, ACTS-3D's tri-action policy (skip, analog, SNN) improves mean bilateral full-volume segmentation Dice compared with (i) best analog-only adaptive router and (ii) ErrorRoute-style two-action utility router, while respecting a prespecified limit on the frequency of meaningful gland-score degradation. If event-specific quality gain cannot be demonstrated, the work must not claim SNN benefit.

**H_efficiency (secondary):**
The net pipeline may reduce GPU joules per CT relative to all-ROI refinement at an equivalent accuracy target by actually skipping selected network forwards. Spike sparsity does not count as energy evidence; if the hardware gives no trustworthy joules, classify this hypothesis as *untested*.

## B. Locked cohorts and information barriers

- Paper II development: full abdomen CT data from original AMOS `imagesTr`, with **patient-level** partition, no identity/slice leakage, grouping repeated studies of same patient when IDs permit.
- Meta-training and intervention-label construction: cross-fitting with backbone/refiner **trained on patients disjoint** from those used to generate utility labels. Calibrate decision and harm thresholds on separate internal validation.
- Original AMOS `imagesVa`: provenance audit required; because previous Paper I models/results examined it, **do not describe it as prospective untouched Paper II test**. It may be reported as an explicitly reused benchmark only.
- Confirmatory external CT cohort: TotalSegmentator adrenal masks is a candidate, conditional on usable annotated labels, access rights, body coverage and ID/site-overlap audit. A separate institution is preferable when feasible. Document selection/exclusion before running predictions.
- Include both difficult and easy patients and bilateral gland failures. Do not use GT to filter z ranges or to select ROIs at inference.
- No direct development use of new locked test GT (even for postprocessing choice, ROI threshold, energy budget, calibration or qualitative debugging).

## C. Prespecified experimental arms

1. Full-volume 3D nnU-Net, identical freeze/checkpoint for all relevant arms.
2. CNN + analog refiner on all candidate ROIs (same proposal and recurrent steps).
3. CNN + SNN refiner on all candidate ROIs.
4. CNN + SNN refiner with anatomical events clamped to zero, *same S weights*.
5. CNN + analog-only conditional routing (risk/budget matched).
6. CNN + SNN-only conditional routing (risk/budget matched).
7. CNN + two-choice utility router (ErrorRoute-style baseline, reimplemented or verified code).
8. **ACTS-3D** with three actions and event-necessity supervision.
9. ACTS without event-necessity head; without risk penalty; without routing budget; with shuffled events.
10. Random, high-uncertainty and equal-active-ROI controls (same actual cost as feasible).

Keep model parameters, training steps, input features, ROI shape, augmentation, normalization, seed, label remapping, backbone features and evaluation consistent. Report mismatches, do not hide them.

## D. Endpoints

**Primary endpoint (draft, to freeze):** patient-level bilateral macro-average whole-volume Dice for ACTS-3D vs strongest predeclared matched-energy analog/utility router on locked external CT cohort, selecting budget `B*` *solely on development*. Analyze paired patient differences. Report 95% confidence interval, effect size and two-sided paired p-value if sample size supports it. The fixed operating point and comparator selection must be locked prior to test; do not choose the best comparator post hoc.

**Mechanistic gatekeeper (prior to primary):** held-out development signed per-ROI `ell_S0 - ell_S` and corresponding patient full-volume Dice change under fixed routing and identical S weights. Require positive, reproducible event effect across predeclared seeds with uncertainty interval excluding zero for primary mechanism assertion.

**Secondary endpoints:** per-gland Dice, HD95 (mm), NSD@2 mm, gland presence recall, z-presence Dice, false-positive volume, missed-gland count, conditional ROI recall and harm rate, per-case p50/p95 latency, GPU socket joules/CT, peak GPU VRAM, throughput, route percentages, spike/event rates, expected-minus-realised energy model error, regret to oracle router (development only).

**Safety/harm endpoint:** fraction of cases with a predeclared clinically meaningful gland Dice deterioration relative to CNN baseline (example eps=0.02, **must freeze/justify with domain expert**), plus severe missed-gland cases. Any statistical risk control is empirical on development/test and does NOT provide a formal patient-safety guarantee.

**Uncertainty:** patient-clustered bootstrap CI with >=2000 resamples when sample sizes permit, fixed patient set for paired methods; seed-to-seed variation reported separately and hierarchical variance as sensitivity. Missing HD95 due to empty masks must be explicitly counted/handled, not dropped to inflate results.

## E. Multiple comparisons and operating-point selection

- Define one primary baseline from development ONLY (e.g., best analog-only learned utility router), and lock threshold and budget. Alternative comparisons are secondary/exploratory.
- Sequential gatekeeping: test event mechanism on development; only if supported, proceed to the locked external primary test. Avoid pretending retrospective selection has nominal false-positive control.
- For multiple secondary methods and budgets, use Holm adjustment or labelled exploratory findings; report all budget settings, not just best.
- Do not decide after looking at external test which weight, `tau_events`, harm cut-off or z spacing method is 'best'.

## F. Runtime/energy measurement protocol

- AMD MI300X with ROCm build and exact device BDF mapping documented; no fabricated energy when AMD-SMI APIs missing, incompatible or sampling noise is substantial.
- Fixed CTA volume order, case IDs, run environment, same batch size and resolution, explicit warmup and `torch.cuda.synchronize` for model-only timings.
- Record both **model-only** and **whole-pipeline** energy/time, which have different scope; at least 3–5 interleaved runs with repeat confidence intervals.
- Account for energy from B, proposal generator, Q, CPU/GPU transfers, kernels, resizing, postprocessing and output encoding. If energy is only GPU socket, report GPU socket rather than total node energy.
- Test whether the conditional route truly skips S and A kernels (profiler trace/forward-call assertions); sparsity within dense convolution does not prove cheaper inference.
- Compare energy–accuracy Pareto curves across prescribed budgets, with no cross-protocol cropped-vs-whole-CT energy comparison.
- If compilation state differs (e.g. `nnUNet_compile=F`), record that as a limitation; compare equal optimization levels when assessing an architectural efficiency claim.

## G. Compute/resource and stop rules

Before scheduling heavy 3D CNN OOF training: require patient-provenance audit, verified no GT use at inference, basic policy reference tests, an exploratory event-specific positive signal and confirmed feasible external-data access.

**Stop/reframe if:** real events indistinguishable from zero events; matched analog provides equal quality and lower joules; no measured on-GPU saving because of dispatch overhead; confidence/harm prediction too poorly calibrated; or a credible newer study already demonstrates the exact method. A well-conducted null study is publishable scientifically, but not under an unsubstantiated SNN efficiency title.

## H. Documentation and audit trail to complete before a claim

Freeze: `paper2_split_manifest.json`, training/prediction CLI hashes and seeds, CNN A S S0 checkpoint provenance, `policy_config.json`, SNN event encoder specification, ROCm/driver/software versions, actual energy monitor output, published comparisons, patient-level CSV, calibration curves, CI generation scripts, sample counts and exclusion log, ethics/data licence, preprint and nearest prior art audit.

**Current implementation coverage:** `src/algorithms/acts3d_policy.py` implements deterministic choice semantics and optional conservative quantized knapsack reference. All cross-fitted expert training, label generation, ROI proposals, event dynamics, real conditional GPU dispatch, calibration and testing remain to be built.
