# Hardware energy and computational-efficiency evaluation

## Scope and reason for measurement

Our hybrid Phase 2 model has very sparse event/spike activity, but those
numbers are **not hardware energy measurements**. Dense U-Net inference,
dense Conv2d kernels, memory traffic, PyTorch overhead, and the number of
slice steps remain significant. There is no expectation that sparsity must
save joules on a MI300X unless the executed operations are actually skipped
or an event-aware implementation is provided.

Current position: do **not** claim that Phase 1, Phase 2, or AdrenaSpike-3D
is more energy efficient based on spike density alone.

## Two complementary efficiency studies

### Study A: model-only (to implement once pilot checkpoints are ready)

Use the same real patient CT/segmentation inputs in GPU memory, matched
batch size, same ROI dimensions, identical warmup and number of repetitions.
Call CUDA/HIP synchronize before and after timed inference. Record p50/p95
latency, throughput, peak torch allocated/reserved VRAM, and SNN spike/event
rate. Compare frozen baseline, analog refiner, SNN refiner and the no-refiner
identity control. This captures the additional cost of the refinement model.

### Study B: end-to-end GPU socket energy (measurement wrapper implemented)

Measure a complete inference command under an identical input and evaluation
protocol using AMD-SMI's physical GPU power and energy counters:

`scripts/measure_gpu_energy.py`

The wrapper records time, power in W, an AMD-SMI cumulative counter converted
to joules where supported, and a fallback trapezoidal W*seconds estimate.
It logs total **GPU socket** energy and an approximate idle-subtracted
secondary estimate; it does **not** measure CPU, storage, node-level, or
facility energy. The wrapper does not run without AMD-SMI bindings or an
explicit physical GPU index when more than one device is visible.

**Check on the allocated GPU node, not a login node:**

```bash
which amd-smi || true
amd-smi list
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
python -c "import amdsmi; print('AMD SMI python binding available')"
python scripts/measure_gpu_energy.py --help
```

Identify which **physical AMD-SMI device index** belongs to your allocated
HIP device. `torch.cuda:0` can differ from `amd-smi -g 0`, particularly
with a multi-GPU compute node or partitioned GPU. Measuring the wrong
device will invalidate the experiment. Use your cluster team's mapping
documentation if uncertain.

**Example Phase 2 matched evaluation (existing checkpoints):**

```bash
python scripts/measure_gpu_energy.py --gpu-index 0 --label phase2_snn_01 \
    --output-dir runs/energy_phase2 --require-energy -- \
    python -u scripts/evaluate_segmenter.py \
      --checkpoint runs/phase2_anatomy_snn_s1/best_model.pt \
      --data-root ../data/amos22 --cache-dir ../cache \
      --output-dir runs/energy_phase2/phase2_snn_01_eval \
      --keep-largest 2 --min-component-voxels 30 --device cuda

python scripts/measure_gpu_energy.py --gpu-index 0 --label phase2_analog_01 \
    --output-dir runs/energy_phase2 --require-energy -- \
    python -u scripts/evaluate_segmenter.py \
      --checkpoint runs/phase2_anatomy_analog_s1/best_model.pt \
      --data-root ../data/amos22 --cache-dir ../cache \
      --output-dir runs/energy_phase2/phase2_analog_01_eval \
      --keep-largest 2 --min-component-voxels 30 --device cuda
```

Adjust --gpu-index to the physical device. The existing Phase 2
`evaluate_segmenter.py` evaluates the previously cached label-derived
z-neighbourhood, **not genuine unrestricted whole-scan inference**; label
its throughput/energy as per evaluated z-cropped cohort. These runs measure
evaluation process energy and include model loading and CPU-side metrics,
so they do not alone isolate inference GPU energy. Future ROI and nnU-Net
runs should use the new true full-z caches and match processing boundaries.

Benchmark at least 3-5 times in interleaved order (SNN / analog /
analog / SNN); warm filesystem/cache equally, maintain identical
post-processing and batch size, avoid other workloads on the physical
GPU, and report mean, standard deviation and uncertainty. Do not compare
a training job against an inference job, or single-slice vs whole-volume
workloads. If power counters return unsupported/error or nonsensical
values, report no hardware energy result rather than a guessed saving.

**Reporting table to complete after measurement:**

| Method | Dice | HD95 mm | NSD@2mm | GPU energy J/case | Time s/case | Cases/s | Peak VRAM GiB | Spike rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-volume 3D nnU-Net | TBD | TBD | TBD | TBD | TBD | TBD | TBD | N/A |
| Phase 2 analog (matched window) | 0.73110* | 16.764* | 0.83504* | TBD | TBD | TBD | TBD | N/A |
| Phase 2 SNN (matched window) | 0.73128* | 16.338* | 0.83508* | TBD | TBD | TBD | TBD | 0.05588%* |
| AdrenaSpike-3D analog | TBD | TBD | TBD | TBD | TBD | TBD | TBD | N/A |
| AdrenaSpike-3D SNN | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

`*` Historical results were on the label-dependent z-cropped protocol, not
fully unrestricted CT scans. Do not compare with nnU-Net full-scan numbers
without matching the evaluation input and case definitions.

### Efficiency-sensitive architecture changes for *future* experiments

1. Replace three repeated U-Net evaluations per middle slice (Phase 2)
   with a single 3D feature extractor or cached volumetric backbone features.
2. Keep gland ROIs small, but **measure missed gland ROI coverage** and
   false-positive localisation; reducing compute by excluding glands is not
   a valid improvement.
3. Use a truly sparse/event-aware inference implementation only if
   actual ROCm kernels can exploit sparsity. The current snnTorch
   convolution path does not automatically avoid dense Conv2d compute.
4. Consider quantised/pruned analog baselines; benchmark quality-adjusted
   joules, not an SNN only against an unoptimised analog network.
5. Compare accuracy/energy/latency tradeoffs at equal resolution,
   batch size, and image context. Maintain no-correction and analog controls.

## References and measurement implementation

- AMD SMI CLI: https://rocm.docs.amd.com/projects/amdsmi/en/docs-6.3.0/how-to/amdsmi-cli-tool.html
- AMD SMI counter: https://rocmdocs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-api.html
- Zeus ROCm caveats: https://ml.energy/zeus/measure/
- Yan, Bai and Wong (2024), *Reconsidering the energy efficiency of spiking neural networks*, https://arxiv.org/abs/2409.08290
- Shen et al. (CVPR 2024), *Are Conventional SNNs Really Efficient? A Perspective from Network Quantization*.

Do not allocate an extra GPU job *only* for telemetry while baseline experiments
are awaiting resources. Run the wrapper in a later existing allocation, after
training and evaluation have finished, or within the same batch job if enough
wall time remains. Never run two measurement targets concurrently on the GPU.
