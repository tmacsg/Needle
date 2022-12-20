import sys
sys.path.append('./python')
sys.path.append('./apps')
from needle import ops
import needle as ndl
import numpy as np
from needle import backend_ndarray as array_api

from tests.test_sequence_models import *
from tests.test_conv import *

_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

if __name__ == '__main__':
    # H0 -> (H0-1)*S + K - 2*P 
    # stride = 1
    # padding = 0
    # a = ndl.Tensor(np.ones((1,5,5,1)))
    # b = ndl.Tensor(np.ones((3,3,1,1)))
    # c = ops.conv_transposed(a, b, stride=stride, padding=padding)
    # c.backward()
    # print((5-1)*stride + 3 - 2*padding)

    # model = nn.Conv(3,6,3, stride=3)
    # a = ndl.Tensor(np.ones((3,3,5,5))) # NCWH
    # b = model(a)
    # print(a.shape, b.shape)

    # model = torch.nn.ConvTranspose2d(5,10,3)
    # print(model.weight.shape) 

    s, cin, cout, k, stride, device = 4, 1, 1, 3, 1, _DEVICES[0]
    test_nn_conv_transposed_backward(s, cin, cout, k, stride, device)

    # a = ndl.Tensor(np.ones((1,5,5,3)))
    # b = ops.transpose(a, (3,1,2,0))
    # print(b.shape)