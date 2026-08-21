import numpy as np
import torch

from src.data.dataset import PatientVolume
from src.inference.pipeline import AdrenalSegmentationPipeline, PipelineConfig
from src.models.snn_gate import SpikingSliceGate
from src.models.unet25d import build_unet25d
from src.training.uncertainty_buffer import BufferConfig


def _synthetic_volume(n_slices=20, size=32):
    rng = np.random.RandomState(0)
    image = rng.normal(0, 1, size=(n_slices, size, size)).astype(np.float32)
    mask = np.zeros_like(image, dtype=np.int16)
    mask[8:12, 10:20, 10:20] = 1
    return PatientVolume(patient_id="p0", image=image, mask=mask)


def test_end_to_end_pipeline_runs_and_respects_gate():
    volume = _synthetic_volume()
    gate = SpikingSliceGate(slice_window=5, hidden_dim=16)
    segmenter = build_unet25d(encoder="resnet18", in_channels=5, encoder_weights=None)

    cfg = PipelineConfig(slice_window=5, buffer_cfg=BufferConfig(fixed_buffer=1, max_expansion=0),
                          use_tta=False, remove_small_components=False)
    pipeline = AdrenalSegmentationPipeline(gate_model=gate, segmenter_model=segmenter, cfg=cfg)

    result = pipeline.run(volume)

    assert result["pred_stack"].shape == (20, 32, 32)
    assert result["total_slices"] == 20
    # An untrained gate won't necessarily flag the right slices, but the
    # pipeline must still only run the segmenter on the slices the gate (or
    # its buffer) actually included.
    assert result["slices_segmented"] == int(result["inclusion_mask"].sum())


def test_no_gate_baseline_segments_every_slice():
    volume = _synthetic_volume()
    segmenter = build_unet25d(encoder="resnet18", in_channels=5, encoder_weights=None)
    cfg = PipelineConfig(slice_window=5, use_tta=False, remove_small_components=False)
    pipeline = AdrenalSegmentationPipeline(gate_model=None, segmenter_model=segmenter, cfg=cfg)

    result = pipeline.run(volume)
    assert result["slices_segmented"] == volume.num_slices
