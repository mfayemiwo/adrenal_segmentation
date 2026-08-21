# Adrenal Gland & Tumour Segmentation — Sequence-Aware Spiking Gate + 2.5D Cascade

This repository implements a hybrid classification-then-segmentation pipeline for
adrenal gland (and adrenocortical tumour) segmentation from abdominal CT volumes.

It extends a previously published 2D U-Net + TTA + connected-component
post-processing pipeline (Fayemiwo et al., *Journal of Imaging Informatics in
Medicine*, 2025) with a **sensitivity-controlled slice-selection gate** that
removes irrelevant slices *before* segmentation, reducing false-positive masks
on slices that contain no adrenal tissue.

The central research contribution is **Stage A**: instead of treating the gate
as "a CNN classifier that happens to run per slice," we present each slice in a
craniocaudal neighbourhood (e.g. 5 consecutive slices) to a **spiking neural
network as one discrete time step per slice**. This uses the actual temporal
integration dynamics of a spiking network (leaky integrate-and-fire membrane
potential accumulating across time) to model *anatomical progression through
the volume*, rather than the common (and scientifically weaker) practice of
repeating one static image across several SNN time steps.

## Why this design, not just "SNN instead of CNN"

Two-stage localisation-then-segmentation systems for small organs already
exist (e.g. Xu et al., *Computers in Biology and Medicine*, 2021 — cascade
localisation network + boundary-attention Small-organNet for the adrenal
gland). Swapping the classifier for an SNN is not, by itself, a publishable
contribution. Because of that, this codebase treats a **matched-capacity
CNN-LSTM gate** (`src/models/cnn_lstm_gate.py`) as a *mandatory* baseline,
not an optional ablation — see `scripts/run_ablation.py`. The SNN gate is only
a genuine contribution if it matches or beats that baseline on sensitivity /
NPV / F2 / PR-AUC, not merely on simulated energy/latency proxies (which, per
the SNN efficiency literature, are unreliable on GPUs and only become
meaningful on neuromorphic hardware).

## Repository layout

```
configs/                  YAML experiment configs
src/data/                  patient-level splitting, HU windowing/normalisation, slice-window dataset
src/models/
    snn_gate.py            Spiking ResNet-18 gate, LIF neurons, temporal slice encoding (snnTorch)
    cnn_lstm_gate.py        matched-capacity CNN+LSTM gate baseline
    unet25d.py              2.5D U-Net segmenter (swappable encoder backbone)
    losses.py               focal / sensitivity-weighted / Dice losses
src/training/
    uncertainty_buffer.py   uncertainty-driven adaptive slice buffering (novel contribution #2)
    train_gate.py            training loop for either gate model
    train_segmenter.py       training loop for the 2.5D segmenter with hard-negative mining
src/postprocessing/         TTA and connected-component pruning (carried over from prior pipeline)
src/evaluation/             Dice/IoU/HD95/NSD, sensitivity/NPV/F2/PR-AUC/calibration, empty-slice FP rate
src/inference/pipeline.py   end-to-end: gate -> uncertainty buffer -> segment -> post-process
scripts/run_ablation.py     orchestrates the 8-arm ablation matrix (see docs/methodology.docx)
tests/                      unit tests against synthetic tensors (no patient data required)
```

## Status

This is a **runnable methodological scaffold**, verified with unit tests on
synthetic tensors (see `tests/`). It does not yet contain patient data or
trained weights — plug in your NIfTI/DICOM cohort via `src/data/dataset.py`
and run `scripts/run_ablation.py` to produce the results tables described in
the accompanying methodology document.

## Installation

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Novel contributions (map to code)

1. **Sequence-aware spiking gate** — `src/models/snn_gate.py::SpikingSliceGate`
2. **Uncertainty-driven adaptive slice buffering** — `src/training/uncertainty_buffer.py`
3. **Matched-capacity CNN-LSTM ablation** (required, not optional) — `src/models/cnn_lstm_gate.py`
4. **Hard-negative-aware 2.5D segmenter** — `src/training/train_segmenter.py`
5. **Full-volume evaluation with empty-slice false-positive rate** as a primary
   outcome — `src/evaluation/evaluate_pipeline.py`

See `docs/methodology.docx` for the full write-up, related work, statistical
validation plan, and target-journal positioning.
