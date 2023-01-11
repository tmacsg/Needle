import sys
sys.path.append('./python')
sys.path.append('./apps')
from needle import ops
import needle as ndl
import numpy as np
import pickle
from simple_training import inference
from models import *
from utils import *
# from needle import backend_ndarray as array_api

# from tests.test_sequence_models import *
# from tests.test_conv import *
# from tests.test_memory import *
# from tests.test_nn_and_optim import *


if __name__ == '__main__':
    # model = unet(feature_scale=1, in_channels=1, n_classes=2, 
    #             device=ndl.cuda(), dtype="float32", is_batchnorm=True)

    model = ResNet9()
    # model1 = nn.Sequential(
    #     nn.Conv(1, 2, 3),
    #     nn.ReLU(),
    #     nn.BatchNorm2d(2)
    # )
    # model2 = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(100,200),
    #     nn.Conv(1, 2, 3),
    #     nn.ReLU(),
    #     nn.BatchNorm2d(2),
    #     nn.Flatten()
    # )

    # model = nn.Sequential(
    #     model1,
    #     model2,
    #     nn.Conv(1,1,1)
    # )

    ndl.print_model(model)

    
