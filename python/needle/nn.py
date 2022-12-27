"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = None
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose((1,0)))        
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.bias:
            temp = self.bias.reshape((1, self.out_features)).broadcast_to((X.shape[0], self.out_features))
            return X @ self.weight + temp
        else:
            return X @ self.weight
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = X.shape
        dim = 1
        for i in range(1,len(shape)):
            dim *= shape[i]
        return ops.reshape(X, (X.shape[0], dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.sigmoid(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    """
    logits: Tensor of shape (batch_size, classes)
    y: Tensor of shape (batch_size,) of integers in [0, classes)
    """
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        if len(logits.shape) == 1:
            logits = ops.reshape(logits, (1, logits.shape[0])) 
        z_logSumExp = ops.logsumexp(logits, axes=(1,))
        y_one_hot = init.one_hot(logits.shape[1], y.data, device=logits.device)
        z_y = ops.summation(ops.multiply(logits, y_one_hot), (1,))
        return ops.summation(z_logSumExp - z_y) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True)) # self.weight -> (dim,)
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True)) # self.bias -> (dim,)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False) # self.running_mean -> (dim,)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False) # self.ruuning_var -> (dim,)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert self.dim == x.shape[1]
        w = self.weight.reshape((1,self.dim)).broadcast_to(x.shape)
        b = self.bias.reshape((1,self.dim)).broadcast_to(x.shape)
        if self.training:
            batch_mean_ = ops.summation(x, (0,)) / x.shape[0]
            batch_mean = batch_mean_.reshape((1,self.dim)).broadcast_to(x.shape)
            batch_dev = x - batch_mean
            batch_var = ops.summation(batch_dev * batch_dev, (0,)) / x.shape[0]             
            batch_std_ = (batch_var + self.eps) ** 0.5 
            batch_std = batch_std_.reshape((1,self.dim)).broadcast_to(x.shape)
            temp = (x - batch_mean) / batch_std
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean_.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data
            return w * temp + b
        else:
            running_mean = self.running_mean.reshape((1,self.dim)).broadcast_to(x.shape)
            running_var = self.running_var.reshape((1,self.dim)).broadcast_to(x.shape)
            running_std = (running_var + self.eps) ** 0.5
            temp = (x - running_mean) / running_std
            return w * temp + b
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype)) 
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert self.dim == x.shape[1]
         
        x_mean = ops.summation(x, (1,)) / x.shape[1]  # x -> (B,)
        x_dev = x - ops.broadcast_to(ops.reshape(x_mean, (x.shape[0], 1)), x.shape)  # x_dev -> (B, D)
        x_var = ops.summation(x_dev * x_dev, (1,)) / x.shape[1] # x_var -> (B,)
        x_var = ops.broadcast_to(ops.reshape(x_var, (x_var.shape[0],1)), x.shape) # x_var -> (B,D)
        x_var = (x_var + self.eps) ** 0.5 # x_var -> (B,D)
        temp = x_dev / x_var # temp -> (B,D)
        w = ops.broadcast_to(ops.reshape(self.weight, (1,self.dim)), x.shape)
        b = ops.broadcast_to(ops.reshape(self.bias, (1,self.dim)), x.shape)       
        return w * temp + b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            prob_mat = init.randb(*x.shape, p=1-self.p)
            mask_mat = prob_mat / (1 - self.p)
            return x * mask_mat
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = None, bias=None, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if padding is None:
            self.padding = kernel_size // 2 
        else:
            self.padding = padding

        ### BEGIN YOUR SOLUTION
        receptive_field_size = kernel_size ** 2
        shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.weight = Parameter(init.kaiming_uniform(in_channels * receptive_field_size, 
                                                     out_channels * receptive_field_size, 
                                                     shape=shape,
                                                     device=device, dtype=dtype))

        self.bias = None
        if bias:
            val = 1.0/((in_channels * (kernel_size**2))**0.5)
            self.bias = Parameter(init.rand(out_channels, low=-val, high=val, 
                device=device, dtype=dtype))

        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N,C,H,W = x.shape

        # padding = (self.kernel_size + self.stride * H - H - self.stride) / 2
        # padding = math.ceil(padding)
        padding = self.padding

        x = x.transpose((1,2)).transpose((2,3)) # NC_inHW -> NHWC_in

        if self.bias is None:
            result =  ops.conv(x, self.weight, stride=self.stride, padding=padding) # NHWC_out
            result = result.transpose((2,3)).transpose((1,2))
            return result
        else:
            bias_ = self.bias.reshape((1,1,1,self.out_channels))
            
            result =  ops.conv(x, self.weight, stride=self.stride, padding=padding) 
            bias_ = bias_.broadcast_to(result.shape)
            result = result + bias_
            result = result.transpose((2,3)).transpose((1,2))
            return result

        ### END YOUR SOLUTION

class Conv_transposed(Module):
    """
    Multi-channel 2D transposed convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = None, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if padding is None: 
            self.padding = kernel_size // 2
        else:
            self.padding = padding

        ### BEGIN YOUR SOLUTION
        receptive_field_size = kernel_size ** 2
        shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.weight = Parameter(init.kaiming_uniform(in_channels * receptive_field_size, 
                                                     out_channels * receptive_field_size, 
                                                     shape=shape,
                                                     device=device, dtype=dtype))

        self.bias = None
        if bias:
            val = 1.0/((in_channels * (kernel_size**2))**0.5)
            self.bias = Parameter(init.rand(out_channels, low=-val, high=val, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION 
        N,C,H,W = x.shape
        padding = self.padding

        x = x.transpose((1,2)).transpose((2,3)) # NC_inHW -> NHWC_in

        if self.bias is None:
            result =  ops.conv_transposed(x, self.weight, stride=self.stride, padding=padding) # NHWC_out
            result = result.transpose((2,3)).transpose((1,2))
            return result
        else:
            bias_ = self.bias.reshape((1,1,1,self.out_channels))
            
            result =  ops.conv_transposed(x, self.weight, stride=self.stride, padding=padding) 
            bias_ = bias_.broadcast_to(result.shape)
            result += bias_
            result = result.transpose((2,3)).transpose((1,2))
            return result
        ### END YOUR SOLUTION

class Maxpool(Module):
    """
    Maxpool2D, input shape NCHW, kernel_size = stride
    """
    def __init__(self, kernel_size, device=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.maxpool(x, self.kernel_size)
        ### END YOUR SOLUTION

class Concat(Module):
    """
    Concat a list of tensors, input shape NCHW
    """
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.concat(X, self.axis)
        ### END YOUR SOLUTION



class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.bias_ih = None
        self.bias_hh = None

        val = (1 / hidden_size) ** 0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-val, high=val, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-val, high=val, device=device, dtype=dtype))

        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-val, high=val, device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-val, high=val, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        activation = None
        if self.nonlinearity == 'tanh':
            activation = Tanh()
        else:
            activation = ReLU()

        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype="float32")

        result = X @ self.W_ih + h @ self.W_hh
        if self.bias:
            bias1 = self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(result.shape)
            bias2 = self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(result.shape)
            result += (bias1 + bias2)
        
        return activation(result)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.device = device

        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, 
                nonlinearity, device, dtype)]

        for _ in range(num_layers):
            self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, 
                nonlinearity, device, dtype))
            

        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # assert isinstance(X, Tensor)
        # X = Tensor(X, device=X.device)

        seq_len, bs, input_size = X.shape
      

        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype="float32")

        X_split_seq = list(ops.split(X, 0)) 

        H = init.zeros(self.num_layers, seq_len, bs, self.hidden_size, device=self.device, dtype=X.dtype)  
        H_split_layer = list(ops.split(H, 0)) 
        H_split_layer_seq = [list(ops.split(H_split_layer[i], 0)) for i in range(self.num_layers)]
 
        # h0_split_layer = list(ops.split(h0, 0))
        # for i in range(self.num_layers):
        #     for k in range(seq_len):
        #         if i == 0:
        #             H_split_layer_seq[i][k] = self.rnn_cells[i](X_split_seq[k], h0_split_layer[i]) # -> bs, hidden_size
        #         else:
        #             H_split_layer_seq[i][k] = self.rnn_cells[i](H_split_layer_seq[i-1][k], h0_split_layer[i])


        # # H shape: num_layers, seq_len, bs, hidden_size
        # return ops.stack(H[self.num_layers-1][:],0), ops.stack(H[:][seq_len-1],0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.bias = bias
        self.bias_ih = None
        self.bias_hh = None

        val = (1 / hidden_size) ** 0.5
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-val, high=val, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-val, high=val, device=device, dtype=dtype))

        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-val, high=val, device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-val, high=val, device=device, dtype=dtype))
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h0 = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype="float32")
            c0 = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype="float32")
            h = ops.make_tuple(h0,c0)
      
        i_f_g_o = X @ self.W_ih + h[0] @ self.W_hh
               
        if self.bias:
            bias1 = self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to(i_f_g_o.shape)
            bias2 = self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to(i_f_g_o.shape)
            i_f_g_o += (bias1 + bias2)

        result = ops.split_by_batch(i_f_g_o, axis=1, batch=4)
        for i in range(4):
            if isinstance(result[i].realize_cached_data(), Tensor):
                result[i].cached_data = result[i].realize_cached_data().realize_cached_data()

        i = Sigmoid()(result[0])
        f = Sigmoid()(result[1])
        g = Tanh()(result[2])
        o = Sigmoid()(result[3])

        c_out = f * h[1] + i * g
        h_out = Tanh()(c_out) * o 
        
        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
