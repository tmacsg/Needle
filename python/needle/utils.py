import pickle
from needle import nn
import needle as ndl
from typing import List, Optional


def save(model: nn.Module, filename: str):
    """
    Save model parameters 
    """
    params = model.parameters()
    with open(filename, 'wb') as f:
        for param in params:
            pickle.dump(param.numpy(), f)    
    f.close()

def load(model: nn.Module, filename: str):
    """
    load model parameters 
    """
    model_params = model.parameters() 
    with open(filename, 'rb') as f:
        for param in model_params:
            param.cached_data = ndl.NDArray(pickle.load(f), device=param.device)
    f.close()
 