# Adrenal Gland Segmentation — Sequence-Aware Spiking Gate + 2.5D Cascade

This repository implements a hybrid classification-then-segmentation pipeline for
left and right adrenal gland segmentation from abdominal CT volumes. Tumour
(adrenocortical carcinoma) segmentation is explicitly out of scope for this project.

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
scripts/
    train_slice_gate.py          Stage A gate training - arms D / E / G (see below)
    evaluate_slice_gate.py       Stage A on whole volumes - the deployment number
    probe_order_sensitivity.py   control: can the architecture use slice order at all?
    sweep_snn_gate.sbatch        spiking-gate configuration sweep
    train_adrenal_segmenter.py   standalone Stage B training run (see "Training" below)
    evaluate_segmenter.py        full-volume evaluation, per-case CSV + report
    inspect_failures.py          headless diagnosis of the cases that score near zero
    run_ablation.py              orchestrates the 8-arm ablation matrix (docs/methodology.docx)
notebooks/
    amos22_adrenal_experiments.ipynb  data inspection, label checks, overlay visualisation
    inspect_failure_cases.ipynb       why individual validation cases score near zero
docs/                       methodology.docx + architecture figures
tests/                      unit tests against synthetic tensors (no patient data required)
```

## Status

The methodology and the full pipeline scaffold are in place and verified with
unit tests on synthetic tensors (see `tests/`). Stage B (the 2.5D segmenter)
has a complete training entry point; the gate stages and the ablation matrix
are wired but not yet trained on a patient cohort. No trained weights are
included.

## Installation

```bash
pip install -r requirements.txt          # everything except torch
# then install torch for your platform - see below - and finally:
pytest tests/ -v
```

**PyTorch is installed separately, and last.** `requirements.txt` deliberately
omits it: `pip install torch` resolves to the CUDA build on PyPI, which on an
AMD GPU silently falls back to CPU. Training still runs, just far slower, and
the only symptoms are a version ending in `+cuXXX` and `NNPACK` warnings.

On the AMD MI300X cluster (ROCm 6.3.3), with the venv **activated**:

```bash
module load amd-rocm/rocm-6.3.3 apps/python3/3.12.4/gcc-14.1.0
source .venv/bin/activate

python -m pip --no-cache-dir install torch==2.6.0 torchvision==0.21.0 \
    torchaudio==2.6.0 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.3/

python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

The last line must print `2.6.0+rocm6.3`, a HIP version, and `True`. For CUDA
or CPU-only machines see https://pytorch.org/get-started/locally/.

Never put `pip install -r requirements.txt` inside a job script: compute nodes
often have no network, it spends allocated GPU time on downloads, and it gives
pip a chance to replace the platform torch on every run.

## Running on SLURM

`scripts/train_amos.sbatch` is a ready-to-edit submission script for the
`k2-gpu-amd` partition. It loads the modules, activates the venv, asserts that
a GPU is actually visible (aborting in seconds rather than wasting the
allocation on CPU), and launches training with unbuffered output.

```bash
sbatch scripts/train_amos.sbatch
squeue -u $USER
tail -f runs/run1/train.log        # live regardless of SLURM's output buffering
```

If the wall-clock limit stops a run that was still improving, resubmit with
`--resume runs/run1/last_model.pt` appended to the python call; it continues
from the last epoch and keeps appending to the same `metrics.csv`.

## Training the segmenter (Stage B)

```bash
# ~1 minute end-to-end check of paths, data and pretrained weights
python scripts/train_adrenal_segmenter.py --data-root /path/to/amos22 --smoke-test

# the real run
python scripts/train_adrenal_segmenter.py --data-root /path/to/amos22 --run-name run1
```

Each run writes to `runs/<run-name>/`:

| file | purpose |
| --- | --- |
| `progress.txt` | rewritten every epoch — best/latest Dice, precision/recall, epochs since improvement, ETA, ASCII trend chart. Open this to see whether the model is improving. |
| `train.log` | append-only, flushed per line — `tail -f` it |
| `metrics.csv` | one flushed row per epoch, for plotting |
| `best_model.pt` / `last_model.pt` | best checkpoint by validation Dice; `last` supports `--resume` |

Notes that matter for results:

- The encoder is imagenet-pretrained by default. Training from scratch does not
  work on a structure occupying ~0.3% of pixels in any reasonable budget.
- Volumes are resampled to a fixed physical spacing so that a k-slice window
  spans the same distance for every patient. AMOS22 mixes 1.25/2/5 mm slice
  thickness, which otherwise confounds any slice-sequence model.
- Validation Dice is reported at the best of 19 thresholds as well as at 0.5;
  a fixed 0.5 cutoff reads exactly 0.0 until predictions happen to cross it.
- Expect whole-gland Dice around **0.60–0.75**. Adrenal glands are small and
  genuinely hard; be suspicious of anything above ~0.85.
- On ROCm the script sets `MIOPEN_USER_DB_PATH` / `MIOPEN_CUSTOM_CACHE_DIR`
  before importing torch, working around the read-only MIOpen kernel database
  that otherwise fails as `miopenStatusInternalError`.

Run `--help` for the full option list (spacing, image size, encoder, batch
size, max epochs, early-stopping patience, augmentation).

## Training the gate (Stage A)

Stage A decides which slices are worth segmenting; Stage B segments only those.
`scripts/train_slice_gate.py` trains it, and reads the **same volume cache** as
the segmenter, so nothing needs rebuilding and the two stages cannot drift.

```bash
python scripts/train_slice_gate.py --data-root ../data/amos22 --smoke-test   # ~2 min

sbatch scripts/train_slice_gate.sbatch      # all three arms, one after another
tail -f runs/gate_snn/progress.txt
```

Three arms, differing by one flag each and identical in everything else -
sampler, augmentation, schedule, seed, metric. That is the point: the claim is
that the spiking gate beats the recurrent baseline *given the same windows*.

| arm | flags | what it tests |
| --- | --- | --- |
| E | `--model snn` | k consecutive slices as k SNN time steps - the contribution |
| D | `--model cnn_lstm --lstm-hidden 32` | matched-capacity recurrent baseline |
| G | `--model snn --static-repeat` | the centre slice repeated k times: conventional rate coding with no anatomy in the time axis |

Arm G is the one that tests the framing. If the sequence-aware gate does not
beat the static-repeat control, the temporal axis is carrying no anatomical
information and the paper's central claim needs restating.

**On capacity matching.** `CNNLSTMSliceGate` at its original `lstm_hidden=128`
carries 417,762 parameters against the spiking gate's 285,666 - 46% more, from
the LSTM's four gate matrices, for which LIF membrane integration has no
counterpart. "Matched-capacity baseline" is therefore only accurate at
`--lstm-hidden 32` (281,826, within 2%). The trainer prints both counts at
startup and warns when they diverge; either match them or report the confound.

**What is optimised.** Not accuracy. A missed positive slice removes adrenal
tissue from the segmenter's input permanently; a false positive costs one wasted
segmentation. The operating threshold is therefore chosen post hoc as the
highest threshold retaining `--target-sensitivity` (default 99%) of
gland-bearing slices, and the headline is how many slices the gate removes at
that point. Checkpoint selection uses PR-AUC, which is threshold-free and stable
before the model separates the classes at all.

**On negatives, and the number you actually report.** The shared cache keeps
`--z-margin` slices either side of the gland, so a negative during training is a
*near-gland* slice - the hard-negative regime, right for learning to discriminate
but the wrong denominator for the paper. In use the gate runs over a whole
abdominal CT where most slices are trivially rejectable, so the training-time
slice reduction *understates* it.

`scripts/evaluate_slice_gate.py` measures the real thing. It streams each case
through the same preprocessing at full z-extent and discards it, so it needs no
cache and no disk:

```bash
python scripts/evaluate_slice_gate.py --checkpoint runs/gate_snn/best_model.pt \
    --data-root ../data/amos22
cat runs/gate_snn/full_volume/gate_evaluation_report.txt
```

The two are comparable because `prepare_case` z-scores over the whole resampled
volume *before* cropping in z - changing `--z-margin` changes which slices you
see, not what any one of them looks like. The report gives slices removed at the
retention target, metrics at both the training threshold and one recalibrated on
full volumes, the fixed-width buffer trade-off that the uncertainty-adaptive
buffer must beat, and where lost gland slices sit (an end slice trims the gland;
an interior one splits it, and Stage B recovers neither).

## Can the spiking gate use slice order at all?

The first ablation found arm E beating arm G by 0.0035 PR-AUC - not at all.
That has two explanations calling for opposite responses: the architecture
cannot represent slice order (a defect), or it can but the order carries little
signal for gland presence (a finding). Real data cannot tell them apart.

```bash
python scripts/probe_order_sensitivity.py
```

A blob translates through a stack of k slices; the ascending class is the
descending class reversed, so both contain exactly the same slices and only the
order separates them. A model that cannot represent order sits at 50% forever.

**Arm E, exactly as run, scores 99.5%.** The architecture is not the problem, so
the null result is a statement about the data rather than a bug - and the probe
is what lets that be claimed instead of assumed.

`scripts/sweep_snn_gate.sbatch` is aimed accordingly: at generalisation (the
spiking arms overfit badly, the CNN-LSTM does not) rather than at the readout,
plus a k = 9 run of both arms, since a longer window should help a true sequence
more than a control that gains only redundancy.

The spiking gate's options (`--readout`, `--norm`, `--learn-beta`,
`--learn-threshold`, `--surrogate`, `--surrogate-slope`) all default to the
original behaviour, verified bit-identical. Note that `--readout integrator` is
a non-spiking output layer: the network becomes a hybrid, which weakens any
neuromorphic-energy argument and must be stated.

## Inspecting failures

Mean Dice hides the cases that matter. `scripts/evaluate_segmenter.py` writes a
per-case table; `notebooks/inspect_failure_cases.ipynb` explains the rows at the
bottom of it.

```bash
python scripts/evaluate_segmenter.py --checkpoint runs/run5/best_model.pt \
    --data-root ../data/amos22 --cache-dir ../cache
jupyter lab notebooks/inspect_failure_cases.ipynb     # edit the paths in cell 1
```

If you cannot run Jupyter - no desktop session, a full home quota, or you simply
prefer batch - `scripts/inspect_failures.py` performs the same analysis headlessly
and writes a text report plus a directory of PNGs:

```bash
python scripts/inspect_failures.py --checkpoint runs/run5/best_model.pt \
    --data-root ../data/amos22 --cache-dir ../cache

cat runs/run5/inspection/inspection_report.txt     # tables and per-gland verdicts
ls  runs/run5/inspection/figures/                  # overlays to copy down and view
```

Add `--no-figures` for the tables alone (seconds), `--cases amos_0346 amos_0333`
to pick specific patients, or `--top 8` to widen the automatic selection.

For each failing case, both separate the five explanations that a mean cannot:
an annotation error, an anatomical variant, a gland cropped out of the field of
view, a lateralisation error (right gland written to the left channel), and a
genuine miss. Both reuse `GeometryConfig` / `prepare_case` from the training
script, so what they draw is what the network actually saw, and both run on CPU.

## Novel contributions (map to code)

1. **Sequence-aware spiking gate** — `src/models/snn_gate.py::SpikingSliceGate`
2. **Uncertainty-driven adaptive slice buffering** — `src/training/uncertainty_buffer.py`
3. **Matched-capacity CNN-LSTM ablation** (required, not optional) — `src/models/cnn_lstm_gate.py`
4. **Hard-negative-aware 2.5D segmenter** — `src/training/train_segmenter.py`
5. **Full-volume evaluation with empty-slice false-positive rate** as a primary
   outcome — `src/evaluation/evaluate_pipeline.py`

See `docs/methodology.docx` for the full write-up, related work, statistical
validation plan, and target-journal positioning.
