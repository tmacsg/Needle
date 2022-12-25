import sys
sys.path.append('./python')
sys.path.append('./apps')
from needle import ops
import needle as ndl
import numpy as np
from needle import backend_ndarray as array_api
from needle.data import *

from tests.test_sequence_models import *
from tests.test_conv import *
import yaml
import os

if __name__ == '__main__':
    # test_up_down_sampling()
    # Z_shape = ((4,16,392,392),(4,16,392,392))
    # test_op_concat(Z_shape, 1, False, _DEVICES[0])
    # a = ndl.Tensor(np.random.rand(4,128,8,8), device=ndl.cuda())
    # # b = ndl.Tensor(np.random.rand(3,3,128,64), device=ndl.cuda())
    # c = nn.Conv(128, 64, kernel_size=3, stride=1, padding=0, device=ndl.cuda())(a)
    # d = nn.Maxpool(2)(c)
    # e = nn.Conv_transposed(64, 32, kernel_size=2, stride=2, padding=0, device=ndl.cuda())(d)
    # f = nn.Concat(1)([c,e])
    # g = ops.unpad(a, ((0,0),(0,0),(1,1),(1,1)))
    # h = nn.Concat(1)([f,g])

    # h.backward()
    # print('hello')
    test_up_down_sampling()


    
