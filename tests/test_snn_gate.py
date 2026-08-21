import torch

from src.models.snn_gate import SpikingSliceGate


def test_forward_shape_and_grad():
    model = SpikingSliceGate(slice_window=5, in_channels=1, hidden_dim=32)
    x = torch.randn(2, 5, 1, 64, 64, requires_grad=True)

    logits, spike_stats = model(x)

    assert set(logits.keys()) == {"left_present", "right_present", "tumour_present"}
    for name, tensor in logits.items():
        assert tensor.shape == (2,), f"{name} has shape {tensor.shape}"

    assert len(spike_stats["per_timestep_spike_count"]) == 5
    assert spike_stats["total_spike_count"] >= 0

    loss = sum(logits.values()).sum()
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "surrogate gradient did not flow back through the LIF neurons"


def test_rejects_wrong_time_dimension():
    model = SpikingSliceGate(slice_window=5)
    x = torch.randn(2, 3, 1, 64, 64)
    try:
        model(x)
        assert False, "expected an assertion error for mismatched slice_window"
    except AssertionError:
        pass
