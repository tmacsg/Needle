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
    Z_shape = ((2,3,4,5),(2,3,4,5))
    axis = 0
    test_op_concat(Z_shape, axis, False, _DEVICES[0])

