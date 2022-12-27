import sys
sys.path.append('./python')
sys.path.append('./apps')
from needle import ops
import needle as ndl
import numpy as np
import pickle
# from needle import backend_ndarray as array_api

# from tests.test_sequence_models import *
from tests.test_conv import *
from tests.test_memory import *
from tests.test_nn_and_optim import *


if __name__ == '__main__':
    # test_toy_dataset()
    model = nn.Sequential(
        nn.Linear(2,2),
        nn.ReLU(),
        nn.BatchNorm1d(2),
        nn.Linear(2,1)
    )
    
    opt = ndl.optim.Adam(model.parameters())

    for param in opt.params:
        print(param)

    ndl.save(model, 'test_.pkl')

    print('------------------------------')

    new_model = nn.Sequential(
        nn.Linear(2,2),
        nn.ReLU(),
        nn.BatchNorm1d(2),
        nn.Linear(2,1)
    )
    ndl.load(new_model, 'test_.pkl')
    for param in new_model.parameters():
        print(param)


    
