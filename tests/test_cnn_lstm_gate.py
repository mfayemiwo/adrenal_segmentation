import torch

from src.models.cnn_lstm_gate import CNNLSTMSliceGate, count_trainable_params
from src.models.snn_gate import SpikingSliceGate


def test_forward_shape():
    model = CNNLSTMSliceGate(slice_window=5, hidden_dim=32)
    x = torch.randn(2, 5, 1, 64, 64)
    logits, _ = model(x)
    assert set(logits.keys()) == {"left_present", "right_present", "tumour_present"}
    for tensor in logits.values():
        assert tensor.shape == (2,)


def test_capacity_is_comparable_to_snn_gate():
    snn_model = SpikingSliceGate(slice_window=5, hidden_dim=256)
    cnn_lstm_model = CNNLSTMSliceGate(slice_window=5, hidden_dim=256)

    n_snn = count_trainable_params(snn_model)
    n_cnn_lstm = count_trainable_params(cnn_lstm_model)

    # Not required to be identical, but should be within the same order of
    # magnitude so a capacity difference can't explain away a performance gap.
    ratio = max(n_snn, n_cnn_lstm) / min(n_snn, n_cnn_lstm)
    assert ratio < 3.0, f"parameter counts diverge too much for a fair comparison: {n_snn} vs {n_cnn_lstm}"
