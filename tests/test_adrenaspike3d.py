"""AdrenaSpike-3D ROI geometry and paired temporal model smoke tests."""
import numpy as np
import pytest

from src.data.adrenaspike_roi import locate,crop,insert


def test_prob_based_roi_and_reconstruction():
    p=np.zeros((12,36,40),dtype=np.float32)
    p[3:6,14:19,20:25]=0.9
    centre=locate(p)
    assert 3<=centre[0]<=5
    patch=crop(p,centre,(8,24,24))
    assert patch.max()>0.8
    out=np.zeros_like(p)
    insert(out,patch,centre,blend=False)
    assert np.allclose(p,out)


def test_boundary_padded_roi():
    vol=np.ones((4,5,6),dtype=np.float32)
    patch=crop(vol,(0,0,0),(8,12,12))
    assert patch.shape==(8,12,12)
    assert patch.sum()>0
    rec=np.zeros_like(vol)
    insert(rec,patch,(0,0,0),blend=False)
    assert np.allclose(rec,vol)


@pytest.mark.parametrize("variant",["snn","analog"])
def test_model_backward(variant):
    torch=pytest.importorskip("torch")
    pytest.importorskip("snntorch")
    from src.models.adrenaspike3d import AdrenaSpike3D
    model=AdrenaSpike3D(variant)
    ct=torch.randn(1,1,5,16,16)
    prob=torch.rand(1,1,5,16,16)*0.4+0.1
    y=model(ct,prob,operating_threshold=0.03)
    assert y.shape==(1,1,5,16,16)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert model.decoder[0].weight.grad is not None


def test_zero_event_diagnostic():
    torch=pytest.importorskip("torch")
    pytest.importorskip("snntorch")
    from src.models.adrenaspike3d import AdrenaSpike3D
    model=AdrenaSpike3D("snn")
    p=torch.full((1,1,5,16,16),0.1)
    y=model(torch.zeros_like(p),p,zero_events=True)
    assert y.shape==p.shape
    assert model.last_event_density.item()==0
