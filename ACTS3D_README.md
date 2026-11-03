# ACTS-3D — Publication II research workspace

**Research programme:** two related but distinct publications on adrenal CT segmentation.
**Working title of Paper II:** *ACTS-3D: Event-Attributable, Risk-Constrained Allocation of Analog and Spiking Refinement for Volumetric Adrenal Segmentation*.

**Research hypothesis (not yet experimentally demonstrated):**
A learned decision procedure can select *no refinement, a matched analog refiner, or an SNN refiner* using the estimated benefit of **genuine ordered z-events over both matched analog computation and the identical SNN with its events disabled**, while respecting a hardware-energy budget and calibrated harm threshold.

## Status at branch creation

- Branch created independently from `research/adrenaspike3d`.
- Formal method specification, literature/novelty audit, programme plan and provisional preregistration committed.
- CPU-only, standard-library reference routing implementation created in `src/algorithms/acts3d_policy.py`.
- Unit tests and dedicated ACTS workflow added.
- **NOT YET DONE:** event encoder, trained experts, cross-fitted teacher labels, calibrated policy estimator, conditional GPU dispatch, real performance benchmark or validated clinical/energy improvement.
- Paper I experiments and the shared HPC working copy must not be modified by this new effort.

## Start here

1. [Targeted literature review and claims to avoid](docs/acts3d_literature_review.md).
2. [Formal algorithm and pseudocode](docs/acts3d_algorithm_spec.md).
3. [Two-publication work packages, decision gates and resources](docs/acts3d_research_plan.md).
4. [Provisional, **not frozen**, preregistration](docs/acts3d_preregistration_draft.md).
5. [Reference tri-action router](src/algorithms/acts3d_policy.py).
6. [Conditional expert-execution contract](src/algorithms/acts3d_execution.py).
7. [Reference unit tests](tests/test_acts3d_policy.py).

### CPU-only local verification (no GPU allocation)

```bash
python -m py_compile src/algorithms/acts3d_policy.py src/algorithms/acts3d_execution.py tests/test_acts3d_policy.py
python -m unittest discover -s tests -p 'test_acts3d_policy.py' -v
```

### Safe branch access on an HPC cluster

If a Paper I nnU-Net job is still running from the original project directory,
**do not check out the ACTS branch in that same directory**: the running batch
script imports source files from the working tree at runtime.

Instead, from the **parent directory** of the repository:

```bash
cd /mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation
git fetch origin research/acts3d-algorithm
git worktree add -b research/acts3d-algorithm \
  ../adrenal_segmentation_acts3d origin/research/acts3d-algorithm
cd ../adrenal_segmentation_acts3d
python -m unittest discover -s tests -p 'test_acts3d_policy.py' -v
```

This separate worktree requires no GPU, does not move datasets/checkpoints
and does not affect the current `research/adrenaspike3d` working tree.
The branch may be viewed on GitHub without any local checkout at all.

## Core scientific question

Unlike a simple hybrid CNN–SNN, ACTS-3D asks:

> For this gland and this image region, is executing an SNN better than (a)
> leaving the CNN prediction alone, (b) using a matched nonspiking recurrent
> refiner, and (c) running the *same* SNN with no real anatomical events?
> Is the difference worth the physical GPU cost and error risk?

These are **controlled computational interventions**, not claims about
clinical causality. If spikes do not demonstrate a benefit, the algorithm must
select the analog or identity alternative and the manuscript must report
the negative finding.

Nearest related algorithms include **ErrorRoute** (utility-based 3D routing,
bioRxiv 2026) and **SpikeMoE** (spiking expert routing, arXiv 2026);
they limit which parts can honestly be claimed novel.

## Publication II data discipline

Original AMOS validation cases may have appeared in Paper I experiments. Do
not claim they are untouched for Paper II. Plan a patient-disjoint
development/cross-fitting process and confirmatory external abdominal CT test
set. Use full-volume CT scoring and audit right/left adrenal labels and
reference annotation quality.

## Planned output namespace

Use `runs/acts3d/` for all Paper II artefacts, experiment metadata, audit
manifests and plots, and unique `acts3d_*.out/.err` for future jobs.
No Paper II GPU submission is in this branch's initial deliverable.
