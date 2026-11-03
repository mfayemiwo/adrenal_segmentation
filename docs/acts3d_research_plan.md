# Two-publication research programme and ACTS-3D execution plan
**Programme date:** 3 November 2026.
**Paper II status:** algorithm concept + initial CPU-only routing reference; **no model training, no experimental validation**.
**Paper I branch:** `research/adrenaspike3d`.
**Paper II branch:** `research/acts3d-algorithm`.

## Executive decision

Start a *second, independent research contribution* focused on **event-attributable tri-action decision learning**, not on 'a CNN combined with an SNN'. Retain all validated work from Paper I but do not rebrand the same gated refiner as Paper II. No new GPU batch job should be started until the existing nnU-Net baseline finishes and the low-cost design/data gates pass.

### Publication separation: what is and is not reused

| | Paper I: SNN mechanism and 3D baseline | Paper II: ACTS-3D algorithm |
|---|---|---|
| Problem | Does anatomy-aware signed event recurrence improve segmentation over matched analog and no-event controls? | Which optional refiner should be executed, and where, based on measured event-specific utility/harm/cost? |
| Primary novelty | Empirical/mechanistic findings about SNN refinement and the limits of evaluation/energy claims; strong full-volume CNN baseline | A **learned, risk-aware three-action policy** supervising SNN's advantage over **both** analog and zero-event SNN under physical CT event encoding |
| Method | Frozen shared 2.5D U-Net + SNN/analog; nnU-Net whole-CT comparison | Frozen/finetuned 3D CNN + learned mm-normalized feature events + cross-fitted counterfactual triage for skip/A/S |
| Experiment | Earlier 5-seed Phase 1, 1-seed Phase 2, eight diagnostic ablations; ongoing Fold 0 nnU-Net | New split manifests, OOF CNN features and intervention utility labels, 3+ trained seed comparisons, hard conditional kernel skipping |
| Strength of claims | Does not claim spike efficacy when zero events do not hurt; note GT z-cropped legacy | Claim only if *event-specific mechanism* adds measurable quality or energy-Pareto value versus analog and strong adaptive routers |
| Dataset policy | Disclose historical label-dependent crop; report new full-volume nnU-Net on original holdout | Use **development-only AMOS**, strict OOF training, locked test external to repeated architecture decisions |
| Manuscript overlap | Background/data descriptions may be cited, not duplicated text/novelty/results | Cite Paper I transparently; new algorithm, studies, independent controls and failure analysis |

**Important leakage warning:** The AMOS original validation split used in Paper I cannot automatically be designated an untouched Paper II test. Audit the patient IDs in `runs` and `split_manifest.json` first. If same patient IDs contributed to architecture decisions or comparisons, treat them as development or a previously examined benchmark; seek a *new independent* external dataset (e.g. TotalSegmentator) for locked confirmatory testing. Any data terms/licence requirements must be checked before downloading or redistributing.

## Novel contribution hierarchy

**Primary novelty candidate (algorithmic):** supervised *within-model event intervention* plus cross-expert analog-vs-spike advantage estimation, learned risk-aware tri-action selection, and actual dynamic GPU execution. This is an algorithm because it changes **how training counterfactuals are generated, how the action objective is learned, and how inference computes actions**.

**Supporting, not standalone, innovations:**
- anatomy-interpretable signed CNN feature deltas normalized per mm;
- dual-direction recurrent anatomical context;
- default baseline-preserving residual blending;
- explicit calibration of harmful corrections by gland.

**Prior-art limit:** ErrorRoute already learns utility per refined block under a budget; SpikeMoE routes spiking experts. Neither 'utility routing' nor 'mixture of experts' nor 'hybrid SNN' alone is claimable novelty. See `acts3d_literature_review.md`.

## Work packages and acceptance criteria

### WP0 — Baseline, data and code freeze (CPU/admin while existing GPU job runs)
- Record Paper I branch/commit, ongoing nnU-Net job ID, configuration, dataset patient IDs, CT orientation, original CT affine and bilateral label convention; no dataset movement.
- Export split manifest and hash arrays. Compare Phase 1/2/3 patient lists against nnU-Net `imagesTr/imagesTs` so no 'untouched' claim rests on assumption.
- Audit present Paper I SNN/analog checkpoints and whether their predictions came from in-sample backbone outputs.
- Catalogue prospective TotalSegmentator class labels (left/right adrenal), data availability, licence, annotation provenance, overlap risks and computational/storage cost. Do not count model-generated pseudo-labels as independent GT.
- **Gate:** documented patient provenance and no accidental GT crop; otherwise fix before any Paper II training.
- Output: `docs/acts3d_data_protocol.md` and split-manifest scripts (future implementation).

### WP1 — Minimal falsification pilot (low GPU cost, **only after current job completes**)
- Reuse frozen CNN and old trained SNN/analog checkpoints as a **mechanistic pilot**, not a formal full-CT performance result.
- Collect matched real-vs-zero event `loss_S0-loss_S` and SNN-vs-analog `loss_A-loss_S` on development cases, stratified by gland and high/low uncertainty.
- Confirm event effect distribution contains a reproducible positive subgroup. Measure overhead from candidate detection and routing on CPU first; use short matched GPU inference later.
- **Gate:** if real events consistently fail to help vs zero events, do not spend resources training an SNN routing policy without an explicit new event-encoding ablation that recovers signal.
- Deliverable: one patient-level exploratory CSV/report, clearly separated from final test.

### WP2 — Algorithms and mathematical baselines (CPU-first)
- Implement mm-normalized learned event encoder and matched analog/LIF recurrence with identical input and dimensionality.
- Implement 4-way training intervention table B/A/S/S0 and policy-target builder. All target generation uses GT only during development training; inference code rejects labels.
- Implement utility/harm estimator with calibration diagnostics (MAE, sign recall, AUROC, reliability/calibration curves); cost model learned from real device timing/energy, not spikes.
- Implement exact or near-exact multiple-choice budget solver; the current router contains greedy and a quantized exact DP reference; benchmark runtime/scalability on synthetic data before connecting to GPU.
- Add integration unit tests: no GT dependence, event benefit zero means S ineligible, no-refiner identity exact, no more than one expert per ROI, budget respected, ROI/gland label consistency, deterministic selection, no execution of an unselected expert.
- **Gate:** all tests pass, verified true conditional skip in mock forward execution, proxy benefit on held-out synthetic/dev controls.

### WP3 — Leak-free 3D feature/prediction development set (heavy GPU, prioritise incremental resource use)
- Train K patient-level CNN backbone folds on *training-side AMOS* only (K=3 pilot; upgrade to 5 if justified). Current Fold 0 nnU-Net may provide partial OOF output if IDs and training provenance are correct; do not treat training examples' predictions as OOF.
- Generate CT/full-z backbone logits, features and ROI candidates with fold ID and original spacing.
- Patient-stratified development splits for refiner training, intervention target generation, advantage-estimator training and calibration; calibrators must not see their own refiner's training patients.
- Freeze original nnU-Net preprocessing to ensure credible model comparisons.
- **Cost:** standard ~189 s/epoch x 1000 ~= 52.5 GPU-hours *per 3D nnU-Net fold* observed in Paper I, with variability and inference overhead. 3 extra folds may imply 150+ additional GPU-hours unless amortised/simplified. Obtain allocations only after WP1/WP2 merit it.

### WP4 — Matched model training and counterfactual labels (GPU when justified)
- Pre-register architecture capacity, recurrent steps, event threshold parametrisation, optimizer/batch sizes, seeds 0/1/2 and validation/early stopping.
- Train analog A, spike S, and SNN-zero-events as ablation (same S checkpoint for intervention). Compare against a temporal CNN/GRU with matched parameters and same spatial context.
- Out-of-fold intervention labels on per-ROI *training-development* patients only: base/analog/spike/zero.
- Train `Q_phi` and calibrate harm and cost on internal validation. Do not tune any threshold on external test.
- **Gate:** positive, reproducible event necessity and improved utility/regret over analog-only at matched compute. If negative, report an honest negative mechanistic result; do not force an SNN novelty claim.

### WP5 — Prospective evaluation and MI300X hardware energy (GPU)
- Complete-volume bilateral Dice, HD95 mm, NSD@2 mm, gland-wise sensitivity and case-level failures; report even absent-gland prediction cases. Use identical case IDs, spacing, thresholds, TTA and postprocessing across models.
- Budget sweep from zero refinement to 100% regions, plus strict equal-energy comparisons; report non-inferiority/superiority only if supported.
- Compare random patches, uncertainty-based routing, two-action utility routing (ErrorRoute-style), tri-action without event supervision, analog-only, SNN-only, nnU-Net and full ACTS-3D.
- Split model-forward vs whole-pipeline GPU socket energy; include CNN, routing predictor, ROI extraction/transfer and both refiners when executed; 3-5 interleaved trials, power measurement source and per-patient joules, latency, VRAM, overhead.
- Independent external CT cohort: TotalSegmentator adrenal left/right is a *candidate*, conditional on licence, patient overlap checks, annotation validity and storage/compute.
- **Gate:** pre-registered primary comparison plus positive event necessity; if metrics fail, write the negative or methodological finding rather than claim a win.

### WP6 — Manuscript, code and reproducibility
- Confirm final contribution statement against all 2026 nearest prior art, including ErrorRoute, SpikeMoE and SPiKe-Med.
- Paper II structure: Abstract; anatomy/need; related work and novelty; algorithm (math, pseudocode and intervention protocol); leakage-free training; statistical/energy protocol; results and mechanism ablations; failures/limitations; reproducibility and ethics.
- Supply split hashes, seed configs, trained checkpoints subject to rights, calibration curves, all-cases metrics, paired bootstrap CIs, actual energy logs and a **public exact-branch commit**.
- Appropriate venues to assess when results exist: *Medical Image Analysis*, IEEE TMI, *Neural Networks* or MICCAI (methodology/rigour fit, not promise of acceptance).
- Do not duplicate Paper I figures, tables, novelty claims or uncredited writing. Cross-reference overlapping data/method sections.

## GPU job protection policy

- **Do not** submit a Paper II GPU job until the existing 3D nnU-Net Fold 0 `research/adrenaspike3d` job finishes and is collected.
- **Do not** change the running job's branch, scripts, dataset folders, outputs, GPU allocations or SLURM queue state. Paper II branch may be checked out in a *different worktree* or after the job finishes; its existing running script references a shared filesystem at runtime.
- CPU-only unit tests and design review are safe from a separate local checkout. Avoid switching the HPC working directory's branch while its GPU job is running.
- All Paper II results go to a different namespace, e.g. `runs/acts3d/`, and all GPU jobs use explicitly new names and output paths to prevent overwriting Paper I.

## Milestone/decision dashboard

| Milestone | Evidence required | Status at plan creation |
|---|---|---|
| M0: literature + own null results reviewed | prior-art matrix with DOI, ErrorRoute and SpikeMoE | Done for targeted audit |
| M1: policy algorithm specified and basic CPU tests | precise outcomes, routing and safety constraints | Reference implementation committed; CI pending |
| M2: data provenance audit | patient-level no-leakage manifest | Not started |
| M3: event necessity pilot | real vs zero events and vs analog | Not started |
| M4: leak-free OOF features | patient-level OOF and cost budget | Not started |
| M5: trained A/S experts, utility prediction | per-case gains/harm calibration | Not started |
| M6: locked whole-volume + external evaluation | paired Dice/HD95/NSD, failure tables | Not started |
| M7: physical GPU energy comparison | actual joules, time, memory, repeatability | Not started |
| M8: Paper II submission candidate | criteria passed + manuscript and archived code | Not started |

## Risk register

1. **Novelty conflict:** ErrorRoute and SpikeMoE overlap; addressed by interventional event supervision vs analog and joint selection, but still subject to full-text audit. *High*.
2. **Mechanism null:** historical zero-event ablation did not degrade Dice; spiking value may be zero. Pre-register hard stop. *High*.
3. **GPU resource:** 1000 epochs/fold may exceed available 72-hour allocations; staged pilot and reusing OOF fold0 appropriately. *High*.
4. **Data leakage:** OOF refiner target/held-out mixups and previously viewed AMOS validation patients. Explicit provenance/hashes, disjoint folds. *High*.
5. **No actual savings:** dense SNN kernels and routing overhead offset sparse spikes. Test full MI300X joules and latency. *High*.
6. **External validation:** TotalSegmentator availability/licensing, annotation reliability, annotation shift and overlap. Document and validate. *Medium-high*.
7. **Small anatomy:** missed glands and mismatched laterality from mirroring/ROI selection. Per-gland recall and failure audit. *High*.

## Immediate next research actions with no GPU reservation

1. Review/reference-test the new policy and read nearest prior art full texts; maintain a novelty-claim table.
2. Add a non-GPU patient-split/provenance audit; check orientation and left/right label invariants.
3. Draft a frozen pre-registration and experimental matrix.
4. When Paper I nnU-Net results arrive, audit baseline full-volume failure modes before committing to expensive OOF training.
