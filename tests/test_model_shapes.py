import pytest
import torch

pytest.importorskip("mamba_ssm")
from huoshaoyun_mambahsi.model import MambaClassifier


@pytest.mark.parametrize("patch", [1, 3, 5, 7])
def test_direct_patch_sizes(patch):
    model = MambaClassifier(in_channels=217)
    x = torch.randn(2, 217, patch, patch)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 2)
