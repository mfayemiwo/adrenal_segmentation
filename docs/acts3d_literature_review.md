# ACTS-3D — targeted literature and novelty audit
**Snapshot:** 3 November 2026. **Status:** targeted, source-verified narrative review; not a systematic or exhaustive search. Update immediately before any manuscript submission.

## Motivation from our existing evidence
Earlier Phase 1 and Phase 2 ANN/SNN paired experiments found very small changes in segmentation accuracy and no demonstrated spike-specific superiority. In Phase 3, **zeroing events did not degrade mean Dice**, whereas disabling the confidence gate caused a large decline. Prior results used GT-dependent z cropping and are *not* complete abdominal-volume evidence. The new work must overturn this mechanistic null with held-out, leakage-free data, not simply repackage a U-Net-plus-LIF head.

## Source-based prior-art matrix

| Work / status | What it establishes | Claim we MUST NOT make / remaining gap relevant to ACTS-3D |
|---|---|---|
| Isensee et al., **nnU-Net**, *Nature Methods* 2021, https://doi.org/10.1038/s41592-020-01008-z | Strong automatic baseline with full preprocessing/training configuration. | A custom encoder/decoder alone is not adequate novelty; always use an honestly tuned nnU-Net control. |
| Isensee et al., **nnU-Net Revisited**, arXiv 2024, https://arxiv.org/abs/2404.09556 | Numerous purported architecture improvements collapse under strong baselines and fair resource accounting. | Do not claim new = superior; require matched evaluation, seeds, compute. |
| Ji et al., **AMOS**, NeurIPS 2022, https://proceedings.neurips.cc/paper_files/paper/2022/hash/ee604e1bedbd069d9fc9328b7b9584be-Abstract-Datasets_and_Benchmarks.html | Multi-centre, annotated abdominal CT/MRI including both adrenal glands. | No GT-derived ROI at inference; patient-level splits and external testing. |
| Ma et al., **Analog Spiking U-Net with CBAM/ViT**, *Neural Networks* 2025, https://doi.org/10.1016/j.neunet.2024.106765 | ANN/SNN mixing for medical dense prediction and efficiency already explored. | Hybrid CNN–SNN, attention, surrogate learning and spiking refinement are NOT individually novel. |
| Kumar & Krishnamoorthy, **Hybrid CNN–SNN ischemic stroke**, *Scientific Reports* 21 Sep 2026, https://doi.org/10.1038/s41598-026-70960-1 | Lightweight hybrid segmentation with subject-level validation; hybrid and CNN boundary differences were statistically indistinguishable in reported study. | A smaller network or spike layer alone is not evidence of higher segmentation quality. |
| Sujatha & Aarthi, **SPiKe-Med**, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC13598077/ | CNN/transformer feature encoding followed by SNN refinement for medical segmentation. | Feature-to-spike encoding and a spiking refiner are explicitly prior art. Distinguish *learning which expert is beneficial*, event interventions and physically meaningful z sequence. |
| Zhang et al., **SpikingEDN**, *IEEE TNNLS* 2025, https://doi.org/10.1109/TNNLS.2024.3437415 | Adaptive thresholds, multimodal dual-path SNNs for event-camera semantic segmentation. | Adaptive thresholds / dual paths are not new by themselves; event-camera real time is NOT CT anatomical z. |
| **SSPT**, MICCAI 2026, https://papers.miccai.org/miccai-2026/0995-Paper2435.html | SNN applied to sparse 3D anatomical topology in liver-segment point clouds. | Spiking 3D anatomy is not unique to our work; point-cloud hardware energy figures cannot be transferred to dense CT on MI300X. |
| **Inter-Slice Dual Cross Attention**, IEEE AVSS 2025, https://ieeexplore.ieee.org/document/11149935/ | Inter-slice context improves 2.5D medical segmentation without SNNs. | Slice-sequence consistency is not unique to spikes; compare temporal CNN, GRU and analog recurrence. |
| **RARE-UNet**, arXiv Jul 2025, https://arxiv.org/abs/2507.15524 | Adaptive resolution-based inference routing in medical segmentation. | Spatially adaptive compute is prior art. |
| **Dynamic Dual Gating**, ICCV 2021, https://mlanthology.org/iccv/2021/li2021iccv-dynamic/ | Actual runtime spatial/channel compute skipping with gating. | Do not claim conditional execution generally as novel. |
| **UDEL**, *Signal Processing: Image Communication* 2026, https://www.sciencedirect.com/science/article/pii/S1051200425007456 | Learning uncertainty and correcting ambiguous medical-image regions. | Uncertainty gating and correction already exist. |
| **ErrorRoute**, Chen & Chen, **bioRxiv preprint**, Sep 2026, https://doi.org/10.64898/2026.09.21.753373 | **Closest algorithmic prior art**: trains a marginal-refinement-utility predictor and uses budget-aware 3D block selection. | **Counterfactual utility, ROI selection and budgeted routing alone are NOT novel**. ACTS-3D must beat/reproduce an ErrorRoute-style utility router and add a distinctly testable event-specific tri-action mechanism. This item is a preprint; independently reproduce before asserting superiority. |
| Liu et al., **SpikeMoE**, **arXiv preprint**, Oct 2026, https://arxiv.org/abs/2610.01418 | Conditional spiking mixture-of-experts with spike-count routing. | Choosing among spiking experts is also prior art; our router is *before expert computation* and uses event-intervention advantage vs matched analog with gland-specific harm calibration. |
| Neftci, Mostafa & Zenke, **Surrogate gradient learning in SNNs**, *IEEE Signal Processing Magazine* 2019, https://doi.org/10.1109/MSP.2019.2931595 | Direct gradient training of spiking models. | Do not present surrogate-gradient learning as new. |
| Yan, Bai & Wong, **Reconsidering SNN energy efficiency**, arXiv 2024, https://arxiv.org/abs/2409.08290 | Memory traffic, time steps and sparsity determine whether hardware savings are realised. | Spike count is NOT measured joules; dense PyTorch Conv can still execute at zero spikes. |
| Voudaskas et al., **SNNs in Imaging survey**, *Sensors* 2025, https://doi.org/10.3390/s25216747 | Structured imaging SNN literature and limited actual hardware deployment. | Always separate theoretical operation count from real MI300X GPU socket energy. |
| Wasserthal et al., **TotalSegmentator**, *Radiology: AI* 2023, https://doi.org/10.1148/ryai.230024 and https://github.com/wasserth/TotalSegmentator | External CT set with right/left adrenal labels; candidate independent validation dataset, subject to licence and overlap audit. | External data access and no subject overlap must be verified; model-produced masks alone are not independent human-verified ground truth. |

## Where ACTS-3D can still be an *algorithm*, not only an architecture

Our proposed scientific unit is **an event-attributable, risk-bounded choice between three actions** for each gland/3D region:
0. trust the whole-volume CNN prediction (no refiner);
1. execute a strictly matched analog recurrent refiner;
2. execute a spiking recurrent refiner **only where it is predicted to outperform the analog and the same spiking network with its genuine anatomical events disabled**.

Training explicitly collects *four counterfactual outcomes* from each development ROI: baseline, analog, SNN-real-events, and SNN-zero-events. A cross-fitted advantage model learns:
- `gain_analog = loss_baseline - loss_analog`;
- `gain_spike = loss_baseline - loss_spike`;
- `event_effect = loss_spike_zero_events - loss_spike_real_events`;
- probability that each correction **harms** a gland, and physical-device incremental inference cost.

At inference the algorithm optimises a risk-constrained, cost-budgeted **multiple-choice assignment** (skip / analog / spiking), using only predicted counterfactual advantage and fixed CT-derived inputs. A spike action additionally requires positive expected advantage against the analog and the zero-event control. It must **really skip execution** of unselected modules, not merely multiply dense outputs by zero. Predictions outside selected regions exactly equal the baseline.

**Novelty hypothesis, NOT established fact:** the combination of *event-necessity supervision*, *three-way cross-expert intervention comparisons*, *gland-harm-calibrated resource allocation*, *physically spaced z-feature events*, and *actual on-device conditional execution* has not been identified as the same integrated method in this targeted search. The individual building blocks are known. Further prior-art/patent review and experimental ablations are required before claiming originality.

## Reproducibility and review limitations

- Checked academic publisher records and DOI/official proceedings pages where available. ErrorRoute, SpikeMoE and RARE-UNet are **preprints**, not peer-reviewed validation.
- This review searched title/abstract-level and selected methods information, not every full text, repository, appendix or patent. No categorical 'first ever' claim is warranted.
- Publication-1 and Publication-2 dataset and model provenance must be disclosed. A previously evaluated AMOS `imagesVa` set **cannot** automatically be called untouched for Publication 2.
- Review the state of the art again before choosing the final title and submission venue; 2026 developments are moving quickly.

## Next systematic-review expansion before submission

Search PubMed, Scopus/Web of Science, IEEE Xplore, MICCAI proceedings, arXiv and Google Scholar with date bounds and tracked exclusions. Search combinations of `(spiking OR SNN) AND (medical segmentation OR volumetric CT) AND (dynamic routing OR energy OR uncertainty)`, plus `(counterfactual utility OR adaptive refinement OR budgeted inference) AND segmentation`. Track publication stage, data, metrics, open code, exact novelty, replicated baselines, energy instrumentation and algorithmic components in a PRISMA-style screening sheet; do not imply this current targeted audit is systematic.
