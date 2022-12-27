import pickle
from needle import nn
import needle as ndl
from typing import List, Optional
import matplotlib.pyplot as plt


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
 

def draw_loss(epochs, train_loss, test_loss, index):
    plt.figure(figsize=(10,6))
    plt.xlabel('epoch', fontsize=15, loc='right')
    plt.ylabel('loss', fontsize=15, loc='top')
    plt.plot(epochs,train_loss, 'k', label="train loss")
    plt.plot(epochs,test_loss, 'r', label="test loss")
    plt.legend()
    plt.grid()
    plt.savefig(f'fetal_result_{index}.png')