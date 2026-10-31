# DeltaSpike-AG research plan

## Working title

**DeltaSpike-AG: Inter-Slice Event-Driven Spiking Refinement for Adrenal Gland Segmentation in CT**

This document is the pre-specified research plan for the
`research/delta-snn-refiner` branch. It should be updated with experiment
identifiers and final results, not rewritten after seeing the test set.

## 1. Why a new direction is needed

The repository already tests a sequence-aware SNN slice gate and a matched
CNN-LSTM control. The first true-sequence versus static-repeat SNN comparison
showed only a very small PR-AUC difference, while a synthetic reversal task
showed that the architecture can represent order. That makes a simple
"slices-as-timesteps" claim too weak for a new paper.

Recent work also means that "an SNN for medical segmentation" is no longer
sufficient novelty. Analog Spiking U-Net (Neural Networks, 2025) and newer
hybrid spiking segmentation systems already establish dense medical-image
prediction with SNN components.

The new hypothesis is therefore about **representation**:

> Adjacent CT slices are highly redundant. A spiking temporal pathway should
> receive sparse inter-slice changes, not repeated dense anatomy. A conventional
> segmentation backbone can preserve absolute anatomy while a small LIF
> pathway learns where through-plane change should correct the mask.

This combines two established observations without simply reproducing either:
inter-slice residual/context learning is useful for volumetric segmentation,
and SNNs are naturally event-driven. The proposed contribution is to make
**inter-slice change itself the event stream** for small-organ CT segmentation.

## 2. Proposed method

The base model is the strongest defensible Stage-B 2.5D U-Net selected before
the DeltaSpike experiments.

For a T-slice input stack x:

1. Base U-Net predicts segmentation logits from the original T slices.
2. Temporal input is constructed from T-1 adjacent pairs.
3. Proposed delta encoding uses d_t = x_t - x_(t-1).
4. Positive and negative changes are split into two polarity channels after a
   small magnitude threshold.
5. A compact Conv-GroupNorm-LIF temporal encoder integrates the T-1 events.
6. A lightweight decoder maps the final temporal state to a dense logit
   correction.
7. Final logits = base logits + learned residual scale * temporal correction.

The base branch provides absolute anatomy. The SNN branch is therefore not
asked to reconstruct the gland from sparse changes alone.

## 3. Strict core ablation

All three refiner arms start from the **same baseline checkpoint**, inherit the
same preprocessing/geometry/patient split, freeze the baseline, and use the same
training budget. Five training seeds are pre-specified.

| Arm | Temporal representation | Dynamics | Question |
| --- | --- | --- | --- |
| Baseline | none | none | Reference |
| delta_snn | adjacent-slice deltas | LIF threshold/reset | Proposed method |
| raw_snn | raw slices, T-1 steps | same LIF | Does delta representation matter? |
| delta_analog | same deltas | continuous leaky state | Do spiking dynamics matter? |

The core launcher fixes neuron beta/threshold. This keeps the spiking and analog
temporal pathways parameter-matched as closely as possible and avoids conflating
spiking with extra learnable dynamics.

Only after the strict comparison is complete should secondary variants be
tested: learnable beta/threshold, learnable event threshold, rate readout,
event-threshold sweep, longer slice windows, or end-to-end fine-tuning.

## 4. Primary and secondary outcomes

### Primary segmentation outcomes

- Mean per-patient Dice, reported separately for left and right adrenal glands.
- HD95 in millimetres.
- NSD at 2 mm.
- Worst-case and lower-quartile Dice, not only the cohort mean.

### Through-plane outcomes

These directly test the temporal claim.

- z-presence Dice: overlap of the binary vector saying which axial slices
  contain the gland.
- z-fragment error: difference in the number of contiguous positive slice runs.
- 3D connected-component count after thresholding.
- Error stratified by gland volume and number of gland-bearing slices.

### SNN diagnostics

- Event density and mean event magnitude.
- Hidden spike rate.
- Trainable and total parameter counts.
- Wall-clock inference latency on the same hardware and batch size.

Spike rate or SynOps can support a computational-efficiency discussion, but
**real energy savings must not be claimed from GPU measurements**. Energy claims
require neuromorphic deployment or an explicitly labelled analytical estimate.

## 5. Statistical plan

The patient is the unit of statistical inference.

For each model variant:

1. Run five training seeds while keeping the patient split fixed.
2. Average each patient's metric across seeds.
3. Compare methods with paired Wilcoxon signed-rank tests.
4. Report paired mean/median differences and 95% bootstrap confidence intervals.
5. Report rank-biserial effect size.
6. Apply Holm correction across the pre-specified model comparisons.

Do not pool patient-seed rows as independent observations; that is
pseudoreplication.

The three pre-specified comparisons are:

1. delta_snn vs baseline.
2. delta_snn vs raw_snn.
3. delta_snn vs delta_analog.

## 6. Claim decision table

The paper claim must follow the evidence.

- If delta_snn > baseline, raw_snn and delta_analog: support an
  **event-driven spiking refinement** contribution.
- If delta_snn > baseline and raw_snn, but not delta_analog: support
  **inter-slice delta representation**, not a specifically spiking advantage.
- If delta_snn > baseline but not raw_snn: support temporal residual refinement,
  but not delta-event novelty.
- If all refiners are equivalent: publishable value would need to come from
  robustness/efficiency or a carefully framed negative result; do not claim an
  SNN accuracy contribution.
- If delta_snn is worse: stop architecture expansion and analyse failure modes
  before further tuning.

## 7. Dataset strategy for a Q1 submission

### Development dataset

AMOS22 CT is the development dataset because the repository already has
patient-level preprocessing, bilateral adrenal labels and prior published
results on it.

### External validation

A strong submission should include at least one dataset not used for model
selection. Candidate public abdominal CT datasets with adrenal labels include:

- BTCV / Beyond the Cranial Vault.
- WORD (left/right adrenal labels can be harmonized).
- TotalSegmentator v2.
- A-Eval provides a cross-dataset label harmonization framework spanning AMOS,
  WORD and TotalSegmentator, including bilateral adrenal glands.

Preferred design: train/select on AMOS only, freeze the method, then evaluate
without retraining on BTCV and one larger external dataset. If scanner/protocol
shift is severe, report zero-shot performance first and any adaptation
experiment separately.

## 8. Reproducibility rules

- Baseline checkpoint must be selected before DeltaSpike test results are seen.
- The checkpoint configuration is inherited automatically by refiner runs.
- Training seed and patient split seed are separate.
- All reported model comparisons use identical patient IDs.
- Full-volume evaluation reconstructs HU window and CLAHE from each checkpoint.
- Every table reports n, mean/median where appropriate, spread, and confidence
  intervals.
- Failed/degenerate runs are reported, not silently discarded.
- Model-selection sweeps and final confirmatory experiments are labelled
  separately.

## 9. Publication gate

A Q1-targeted submission should not be written around a marginal single-run
gain. Before drafting the main claims, aim for all of the following:

- Reproducible improvement across seeds.
- Patient-level confidence interval supporting a non-trivial benefit.
- Improvement in at least one boundary/through-plane metric in addition to Dice.
- A clean win over the raw-SNN or analog control that identifies *why* the
  method works.
- External validation on at least one independent adrenal-labelled CT dataset.
- Failure-case analysis showing where the method helps and where it does not.

## 10. Key related work to position against

- Fayemiwo et al. A Novel Pipeline for Adrenal Gland Segmentation: Integration
  of a Hybrid Post-Processing Technique with Deep Learning. Journal of Imaging
  Informatics in Medicine, 2025. DOI: 10.1007/s10278-025-01449-y.
- Li et al. MCNet: A multi-level context-aware network for the segmentation of
  adrenal gland in CT images. Neural Networks, 2024.
  DOI: 10.1016/j.neunet.2023.11.028.
- Meng et al. A deep learning algorithm for automated adrenal gland segmentation
  on non-contrast CT images. BMC Medical Imaging, 2025.
  DOI: 10.1186/s12880-025-01682-5.
- Zhang et al. Inter-Slice Context Residual Learning for 3D Medical Image
  Segmentation. IEEE Transactions on Medical Imaging, 2021.
  DOI: 10.1109/TMI.2020.3034995.
- Ma et al. Analog Spiking U-Net integrating CBAM&ViT for medical image
  segmentation. Neural Networks, 2025.
  DOI: 10.1016/j.neunet.2024.106765.
- Luo et al. WORD: A large scale dataset, benchmark and clinical applicable
  study for abdominal organ segmentation from CT image. Medical Image Analysis,
  2022. DOI: 10.1016/j.media.2022.102642.

The literature search should be refreshed immediately before manuscript
submission and the novelty wording revised if a directly overlapping
inter-slice-event SNN method appears.
