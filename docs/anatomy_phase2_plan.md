# Phase 2: anatomy-aware temporal memory (prototype)

This experiment **does not** claim a new best segmenter or spiking superiority.
Phase 1 found that the analog delta refiner marginally outperformed the SNN.

## Hypothesis

Inter-slice changes in predicted adrenal *anatomy* carry a more useful
temporal signal than raw changes in CT intensity. A LIF temporal memory may
help refine uncertainty-localised errors and improve z-axis continuity.

## Implementation

- Base 2.5D U-Net is frozen and shared, loaded from the identical checkpoint.
- Run the base on three overlapping slice windows centred at z-1,z,z+1.
- Convert predicted gland probability changes into signed event streams.
- Feed the two transitions to a compact spiking recurrent conv refiner.
- Add a bounded residual to centre logits, gated by centre uncertainty and
  disagreement with neighbouring predictions.
- Matched control `anatomy_analog` shares the exact architecture/input
  protocol/gating, except for its analog leaky temporal blocks.

**Costs:** three base-model forwards per input. This is a research prototype;
do not claim lower inference energy. The model is local window memory rather
than a full-volume tracking model. Segmentation loss is unchanged in this first
prototype; continuity-specific loss is a subsequent controlled ablation.

## Evaluation discipline

Run a smoke test, then both variants on an identical fixed validation split.
Report full-volume per-case Dice, HD95, NSD, z-presence and fragment error,
along with mean and worst-case Dice. Compare against the frozen baseline and
phase 1 delta-analog. Do not treat checkpoint-selection validation as an
external test. Obtain independent CT holdout before confirmatory inference.
