import sys
sys.path.append('./python')
sys.path.append('./apps')
from needle import ops
import needle as ndl
import numpy as np
# from needle import backend_ndarray as array_api

# from tests.test_sequence_models import *
from tests.test_conv import *
from tests.test_memory import *


if __name__ == '__main__':
    test_toy_dataset()



    
