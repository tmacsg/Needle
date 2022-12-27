import sys
sys.path.append('./python')
import numpy as np
from needle import backend_ndarray as nd
import needle as ndl
from needle.data import *
from needle import nn
import pytest

test_path = './model_saved/'


A = [1, 5, 10]
B = [2, 6, 11]
_DEVICES = [ndl.cpu(), ndl.cuda()]
@pytest.mark.parametrize("a", A)
@pytest.mark.parametrize("b", B)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_model_save(a, b, device):
    model1 = nn.Sequential(
        nn.Linear(a,b, device=device),
        nn.ReLU(),
        nn.BatchNorm1d(b, device=device),
        nn.Linear(b,a, device=device)
    )

    model2 = nn.Sequential(
        nn.Linear(a,b, device=device),
        nn.ReLU(),
        nn.BatchNorm1d(b, device=device),
        nn.Linear(b,a, device=device)
    )

    for param in model1.parameters():
        param.cached_data = param.cached_data * a + b + np.random.rand()
    ndl.save(model1, test_path + 'test.pkl')
    ndl.load(model2, test_path + 'test.pkl')
    params1 = model1.parameters()
    params2 = model2.parameters()
    for i in range(len(params1)):
        np.testing.assert_allclose(params1[i].numpy(),
        params2[i].numpy(), rtol=1e-08, atol=1e-08)
