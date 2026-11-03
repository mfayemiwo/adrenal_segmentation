# ACTS-3D: algorithm specification (Research Paper II)
**Working title:** *ACTS-3D: Event-Attributable, Risk-Constrained Allocation of Analog and Spiking Refinement for Volumetric Adrenal Segmentation*.
**State:** research proposal / reference policy prototype; NOT a trained model or a demonstrated performance improvement.
**Branch:** `research/acts3d-algorithm`, derived from `research/adrenaspike3d`. Paper I and its running SLURM jobs are untouched.

## 0. Core scientific distinction

A CNN–SNN *architecture* merely adds a spiking layer. ACTS-3D is a **learning-and-decision algorithm**: it observes training-time controlled interventions, predicts which operation would actually improve segmentation, then chooses **exactly one** of skip, matched analog, or spiking refinement for each selected spatial block subject to risk and hardware-budget constraints.

Do NOT claim event attribution is patient-level causal inference. It is a controlled **within-model computational intervention** (real events vs clamped zero events with identical SNN weights) on held-out development data.

## 1. Data and inputs

- Patient CT volume `X`, true voxel spacing `(s_z,s_y,s_x)` in mm; segmentation GT `Y` only for training/development metrics.
- Frozen, properly trained full-volume CNN backbone `B` computes `P_B, F_B` (left/right/background probabilities and intermediate feature volume). Predictions never depend on labels at test time.
- Region proposals `R = {r_i}` from **predictions/image/feature geometry only**. Cover both gland candidates, low-confidence regions and plausible high-confidence failures; include negative/background blocks. No ground-truth-dependent crop or slice-range selection. Verify proposal recall against GT *offline*, across patients.
- Each region contains centered physical-mm context; use voxel affine and resampling; never assume all cases have identical z spacing.
- Leave all voxels outside processed ROIs exactly equal to the whole-volume CNN output. Use overlap-aware, baseline-preserving fusion and evaluate both false-positive and false-negative gland errors.
- Acquisition order along z represents *anatomical spatial progression*, **not physiological time**. State this in all figures/text.

## 2. Event representation (trainable, physically normalized)

Let `F_z` be CNN features sampled at successive anatomical slice positions with physical displacement `dz_z>0` in mm. Define a learned two-polarity event candidate:

```
D_z = project(LN(F_z) - LN(F_{z-1})) / max(dz_z, epsilon)
E_z^(+) = ReLU( D_z - softplus(tau))
E_z^(-) = ReLU(-D_z - softplus(tau))
```

Choose smooth/straight-through stochastic spikes during training and explicit deterministic LIF spikes at evaluation; log event density, firing rates and state-reset behaviour. Contrast against raw CT deltas, raw feature deltas, shuffled z, reversed z, fixed thresholds and **zero events**. Two event steps are not automatically informative.

Pre-register anatomical orientation and normalization. Bidirectional inferior-to-superior/superior-to-inferior processing may be used, but both the analog and spiking controls must receive exactly the same feature context and number of steps.

## 3. Three actions and fourth counterfactual

For each patient-derived region i, run the following **only during training-label generation**:

- `a=0` BASE: output unmodified `P_B`; incremental refiner cost = 0.
- `a=A` ANALOG: matched recurrent continuous-valued refiner `A(P_B,F_B,E)`.
- `a=S` SPIKE: recurrent LIF refiner `S(P_B,F_B,E)` with genuine anatomical events.
- `a=S0` intervention/control: **same exact S weights** and states as S, but all anatomical events clamped to zero. S0 is never an inference routing option; it is a training/validation test of event necessity.

The CNN backbone is always included in end-to-end inference energy and latency. The analog model needs matched trainable parameter counts and input pathways; report any unavoidable parameter difference. All four teacher outputs are created on non-overlapping patient folds, NEVER using test labels.

Define region loss `ell_a(i)`: a pre-declared weighted sum of soft Dice or Tversky error, boundary error in mm, false-positive penalty and false-negative penalty. Define it to handle absent foreground correctly. Report per-gland and per-case metrics *independently* of this training objective.

### Interventional learning targets

```
G_A(i)      = ell_0(i)  - ell_A(i)        # analog benefit over baseline
G_S(i)      = ell_0(i)  - ell_S(i)        # spiking benefit over baseline
G_events(i) = ell_S0(i) - ell_S(i)        # actual benefit of anatomical events
G_S_vs_A(i) = ell_A(i) - ell_S(i)         # spiking benefit over analog
H_A(i)      = 1[ ell_A(i) - ell_0(i) > eps_harm ]
H_S(i)      = 1[ ell_S(i) - ell_0(i) > eps_harm ]
```

`G_events>0` is **necessary but not sufficient** for choosing S: S must also beat baseline and the analog under estimated cost and harm. Histograms must show if events are useful anywhere, rather than trusting a nonzero spike-rate number.

The estimator `Q_phi` sees only features accessible before refinement (baseline logits, uncertainty, mm-normalized differences, anatomical location descriptors, potential region size). It predicts the **four expected potential losses** `(ell_0_hat, ell_A_hat, ell_S_hat, ell_S0_hat)` and harm probabilities `P(H_A),P(H_S)`; **all utility comparisons are derived as algebraic differences of these four predictions**. This prevents logically inconsistent independent heads (for example, a predicted SNN-vs-analog benefit disagreeing with the predicted baseline gains). If full CNN feature access is expensive, use cached low-resolution backbone outputs and measure this routing cost.

Use cross-fitting: generate target outcomes from refiners not trained on the target cases; then fit and calibrate the advantage/harm estimator on disjoint development patients. This breaks the optimistic bias of labeling a model's training examples with its own apparent performance.

## 4. Conditional inference decision: multiple-choice risk- and budget-constrained routing

For each independent candidate block i choose `a_i in {0,A,S}`.
Baseline is always admissible, no refiner execution, utility 0. A/S allowed only if estimated harmful-correction probability is below a development-calibrated threshold `alpha`. A spiking choice also requires:

```
predicted_G_events(i) > tau_events
predicted_G_S_vs_A(i) > tau_spike_vs_analog
predicted_G_S(i) > 0
```

These are predeclared selection rules, **not hard proof** of event benefit or safety.

Let `c_a(i)` be measured *incremental* device-energy or time cost over B, including host dispatch and model batching overhead. Conservative mm/J estimates and risk bounds must be calibrated on separate development patients and actual MI300X measurement. Do not use spike rate as a cost estimator without validating against joules. An implementation can instead use measured latency until energy telemetry works, but then must label its budget as time rather than joules.

A conceptual constrained objective is:

```
maximize     sum_i ( predicted_G_{a_i}(i) - lambda_h * predicted_harm_{a_i}(i) )
subject to   sum_i c_{a_i}(i) <= B
             a_i in {0,A,S} and action-specific eligibility
             observed development-cohort gland-level recall/harm constraints
```

The final selection is a **multiple-choice knapsack**, not a soft gate that computes all three branches. The initial reference implements a conservative greedy policy with deterministic tie-breaking. Later exact dynamic programming or solver-based alternatives must be compared on representative batch sizes, as greedy selection is not globally optimal.

Actual implementation must:
1. compute B once on every CT;
2. compute cheap routing features;
3. score candidate actions *without executing* A or S;
4. group selected A patches into an analog batch, S patches into an SNN batch;
5. **skip** all unchosen refiner forwards;
6. blend corrections only in their selected regions, preserving the baseline elsewhere;
7. return decisions, intervention-score predictions, actual compute time, full-volume masks and audit logs.

Selection and voxel blending for overlapping proposals must be specified and deterministic. Policy must avoid contradictory left/right labels; report overlap conflicts and label-swap errors. No hard slice-classification gate may remove baseline probability outside a candidate ROI.

## 5. Learning procedure (cross-fitting and controls)

Stage 0 — Finish the existing fold-0 full-volume 3D nnU-Net baseline; verify train/test IDs and preprocessing. Audit its error modes. Its current heldout AMOS cohort has been used during development of Paper I and should not be called untouched Paper II test data by default.

Stage 1 — On AMOS *development* patients only, create out-of-fold CNN predictions/features with K patient-level backbone folds (target K=3 initially, evaluate 5 if resources permit). Store only predictions, CT, affine/spacing, fold IDs and labels, with a cryptographic split manifest; enforce no in-sample probability generation.

Stage 2 — Train A and S refiners from matching initialization/protocol/capacity on this development set; predefine 3 seeds. Use a matched non-spiking temporal CNN or GRU as additional control. Capture both positive and negative proposals. Bootstrap no-refinement identity and zero-event intervention.

Stage 3 — Build **out-of-fold counterfactual outcome labels** using held-aside refiner-training folds. Fit `Q_phi` on only training-meta-folds, calibrate risk and the cost model on held-out internal development patients. This may require nested/grouped folds; do not evaluate the final estimator on examples whose refiner fitted their GT.

Stage 4 — Freeze the refiner and router. Calibrate `tau_events, tau_spike_vs_analog, alpha, B` using internal validation only, not the prospective external test. Pre-register primary and secondary settings.

Stage 5 — Measure full-scan performance with fixed metrics and exact same CT patients, preproc, batching, TTA/postprocessing policy and hardware. Zero-event SNN and matched analog must be evaluated under identical settings.

Stage 6 — Freeze all choices and run the locked final test and external set, e.g. appropriately licensed TotalSegmentator adrenal annotations **only after** checking any train/test subject overlap, label consistency and CT distribution shifts.

## 6. Ablations required to substantiate novelty

- Full-volume nnU-Net identity, nnU-Net + analog, nnU-Net + spike, nnU-Net + SNN-zero-events.
- Analog-versus-spiking forced all ROIs, same proposal/feature context/decoder and parameter budget.
- Event-order reversal; random event permutation; event-free SNN; learned vs fixed physical-mm thresholds.
- Single-branch confidence gate vs utility-only router (ErrorRoute-style control) vs ACTS three-way router.
- ACTS without event-necessity supervision; without harm calibration; without budget; 2-way analog/skip; 2-way spike/skip.
- Fixed random or uncertainty-based selection at equal actual energy and equal selected-ROI count.
- LIF vs matched leaky analog neuron at multiple temporal steps and fixed high-rate dense simulation.
- Region size and budget sweep (e.g. 0, 5, 10, 20, 40, 100% eligible patches) with all operating points selected on internal validation.
- Same seeds, patient split and test-time pipeline for all methods. Mechanistic ablations must not alter preprocessing or GT access.

## 7. Key failure modes and falsification

- Events no better than zero-events: **no event-specific spiking contribution**; stop describing events as helping.
- Spiking never selected or beaten by matched analog: three-way algorithm may be a valid negative finding, but cannot claim SNN necessity or energy improvement.
- Policy predicted utility poorly calibrated: no honest risk guarantee; fallback to baseline and report failure.
- Routing overhead exceeds saved computations: no hardware efficiency claim, even if many regions skipped.
- Low ROI proposal recall: small refined regions do not compensate for missed glands; inspect gland recall, volumes, false positives and external shift.
- Claims based only on foreground crops or in-sample backbone probabilities are invalid.
- Main baseline alone may underperform other standard 3D segmentation methods; do not compare to weak or incomplete nnU-Net.

## 8. Formal deliverables / acceptance gates

**Gate A: data integrity** — verified no train/val/test overlap, physical spacing/affine preserved, no GT crop, ROI localisation audited per gland. Stop if violated.

**Gate B: mechanistic necessity** — across >=3 seeds, predeclared event-effect metric is consistently positive on held-out development patients and a matched analog control does not explain all gains. If not, reframe as negative study or stop Paper II SNN claims.

**Gate C: utility policy** — compared at equal cost with confidence, random, ErrorRoute-style utility router and analog-only router; report confidence intervals, regret vs oracle, utility prediction calibration and harm false-discovery.

**Gate D: hardware efficiency** — 3–5 interleaved MI300X inference repetitions with power counters correctly mapped, model-only and end-to-end scope separately, variance and overhead; distinguish joules and GPU socket vs whole system.

**Gate E: external validity** — locked external patient CT set with independent expert reference annotation where possible, 95% paired patient bootstrap CIs, Dice/HD95/NSD and per-gland failure analysis. Do not tune on the external set.

## 9. Algorithm pseudocode (inference)

```text
ACTS3D_INFER(CT, spacing, CNN B, analog A, spike S, calibrated Q, budget):
  P0, F = B(CT)                           # complete abdomen
  R = propose_rois(P0, F, spacing)        # NO test GT
  Z = cheap_routing_descriptors(P0, F, R, spacing)
  ga, gs, ge, gsa, ha, hs, cA, cS = Q(Z)
  options = []
  for each ROI i:
      options[i] = {SKIP}
      if ga[i] > 0 and ha[i] <= alpha:
          options[i] += {ANALOG}
      if gs[i] > 0 and hs[i] <= alpha and ge[i] > tau_events
                    and gsa[i] > tau_spike_vs_analog:
          options[i] += {SPIKE}
  selected = budgeted_multiple_choice(options, expected_gain, expected_cost)
  output = copy(P0)
  for expert in (ANALOG, SPIKE):
      batch = gather_rois(selected_for_expert(expert))
      refined = execute(expert, batch)    # NO forward on unselected expert
      merge_residuals(output, refined, selected_for_expert(expert))
  return output, selected, measured_inference_diagnostics
```

The implementation in `src/algorithms/acts3d_policy.py` is an **early CPU-only reference for decision semantics**, not end-to-end network training, event encoding, frozen CNN or GPU kernel speedups.
