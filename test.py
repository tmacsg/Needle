import sys
sys.path.append('./python')
sys.path.append('./apps')
from needle import ops
import needle as ndl
import numpy as np
import pickle
from simple_training import inference
# from needle import backend_ndarray as array_api

# from tests.test_sequence_models import *
# from tests.test_conv import *
# from tests.test_memory import *
# from tests.test_nn_and_optim import *


if __name__ == '__main__':
    # test_toy_dataset()
    # inference('007_HC.PNG')
    test_nn_modules()
    # test_nn_modules_torch()


    
