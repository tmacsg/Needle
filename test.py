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
    shape = (3, 8, 14, 14)
    kernel_size = 7
    test_op_maxpool(shape, kernel_size, True, _DEVICES[0])

