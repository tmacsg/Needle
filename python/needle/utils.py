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
 

def print_model(model: nn.Module, indent=0):
    def indent2str(count):
        str = ''
        while count > 0:
            str += '    '
            count -= 1
        return str

    if hasattr(model, 'modules'):
        print(f'{indent2str(indent)}{model.__class__.__name__}')
        indent += 1
        for module in model.modules:
            print_model(module, indent)
        indent -= 1
    else:
        desc = ''
        class_name = model.__class__.__name__
        desc += f'{class_name} '
        if class_name == 'Conv' or class_name == 'Conv_transposed':
            desc += f'(c_in: {model.in_channels}, c_out: {model.out_channels}, kernel_size: {model.kernel_size}, stride: {model.stride}, padding: {model.padding}, bias: {bool(model.bias)})'
        elif class_name == 'BatchNorm2d':
            desc += f'(dim: {model.dim})'
        elif class_name == 'Linear':
            desc += f'(in: {model.in_features}, out: {model.out_features}, bias: {bool(model.bias)})'
        print(f'{indent2str(indent)}{desc}')      

        for k, v in model.__dict__.items():
            if isinstance(v, nn.Module):
                indent += 1
                print_model(v, indent)
                indent -= 1

def draw_loss(epochs, train_loss, test_loss, index):
    plt.figure(figsize=(10,6))
    plt.xlabel('epoch', fontsize=15, loc='right')
    plt.ylabel('loss', fontsize=15, loc='top')
    plt.plot(epochs,train_loss, 'k', label="train loss")
    plt.plot(epochs,test_loss, 'r', label="test loss")
    plt.legend()
    plt.grid()
    plt.savefig(f'fetal_result_{index}.png')