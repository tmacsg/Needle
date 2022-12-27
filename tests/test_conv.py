import sys
sys.path.append('./python')
import numpy as np
import pytest
from needle import backend_ndarray as nd
import needle as ndl
from needle import nn
import mugrade
import itertools


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

def backward_check(f, *args, **kwargs):
    eps = 1e-3
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    is_stacked = False
    if isinstance(args[0], list):
        args = args[0]
        is_stacked = True
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            if is_stacked:
                f1 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            if is_stacked:
                f2 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(ndl.Tensor(c, device=args[0].device), out)
    if isinstance(backward_grad[0], ndl.TensorTuple): # TODO keep this?
        backward_grad = backward_grad[0].tuple()
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 1e-2
    return [g.numpy() for g in backward_grad]


stack_back_params = [
    ( (3, 4), 3, 0),
    ( (3, 4), 3, 1),
    ( (3, 4), 3, 2),
    ( (3, 4), 5, 2),
    ( (3, 4), 1, 2),
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("shape, n, axis", stack_back_params)
def test_stack_backward(shape, n, axis, device):
    np.random.seed(0)
    get_tensor = lambda shape: ndl.Tensor(np.random.randn(*shape)*5, device=device)
    backward_check(ndl.stack, [get_tensor(shape) for _ in range(n)], axis=axis)


stack_params = [
    {"shape": (10,3),    "n": 4, "axis": 0},
    {"shape": (4, 5, 6), "n": 5, "axis": 0},
    {"shape": (4, 5, 6), "n": 3, "axis": 1},
    {"shape": (4, 5, 6), "n": 2, "axis": 2}
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", stack_params)
def test_stack_forward(params, device):
    np.random.seed(0)
    shape, n, axis = params['shape'], params['n'], params['axis']
    to_stack_ndl = []
    to_stack_npy = []
    for i in range(n):
        _A = np.random.randn(*shape)
        to_stack_ndl += [ndl.Tensor(_A, device=device)]
        to_stack_npy += [_A]

    lhs = np.stack(to_stack_npy, axis=axis)
    rhs = ndl.stack(to_stack_ndl, axis=axis)


pad_params = [
    {"shape": (10, 32, 32, 8), "padding": ( (0, 0), (2, 2), (2, 2), (0, 0) )},
    {"shape": (10, 32, 32, 8), "padding": ( (0, 0), (0, 0), (0, 0), (0, 0) )},
]
@pytest.mark.parametrize("device", [nd.cpu()])
@pytest.mark.parametrize("params", pad_params)
def test_pad_forward(params, device):
    np.random.seed(0)
    shape, padding = params['shape'], params['padding']
    _A = np.random.randn(*shape)
    _B = np.pad(_A, padding)
    A = nd.NDArray(_A, device=device)
    B = A.pad(padding)

    assert np.linalg.norm(B.numpy() - _B) < 1e-4


flip_forward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0,1)},
    {"shape": (10, 32, 32, 8), "axes": (0,1)},
    {"shape": (3, 3, 6, 8), "axes": (0,1)},
    {"shape": (10, 32, 32, 8), "axes": (1,2)},
    {"shape": (3, 3, 6, 8), "axes": (1,2)},
    {"shape": (10, 32, 32, 8), "axes": (2,3)},
    {"shape": (3, 3, 6, 8), "axes": (2,3)},
    {"shape": (10, 32, 32, 8), "axes": (0,1,2,3)},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", flip_forward_params)
def test_flip_forward(params, device):
    np.random.seed(0)
    shape, axes = params['shape'], params['axes']
    _A = np.random.randn(*shape)
    _B = np.flip(_A, axes)
    A = ndl.Tensor(_A, device=device)
    B = ndl.flip(A, axes=axes)

    assert np.linalg.norm(B.numpy() - _B) < 1e-4


flip_backward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0,1)},
    {"shape": (2, 3, 3, 8), "axes": (0,1)},
    {"shape": (3, 3, 6, 4), "axes": (0,1)},
    {"shape": (2, 3, 3, 4), "axes": (1,2)},
    {"shape": (3, 3, 6, 4), "axes": (1,2)},
    {"shape": (2, 3, 3, 4), "axes": (2,3)},
    {"shape": (3, 3, 6, 4), "axes": (2,3)},
    {"shape": (2, 3, 3, 4), "axes": (0,1,2,3)},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", flip_backward_params)
def test_flip_backward(params, device):
    np.random.seed(0)
    shape, axes = params['shape'], params['axes']
    backward_check(ndl.flip, ndl.Tensor(np.random.randn(*shape), device=device), axes=axes)


# @pytest.mark.parametrize("device", _DEVICES)
# def test_init_calculate_fans(device):
#     _A = np.random.randn(3, 3, 16, 8)
#     A = ndl.Tensor(_A, device=device)
#     assert ndl.init._calculate_fans(A) == (144, 72)

#     _A = np.random.randn(3, 3, 16, 8)
#     A = ndl.Tensor(_A, device=device)
#     assert ndl.init._calculate_fans(A) == (144, 72)


#     _A = np.random.randn(16, 8)
#     A = ndl.Tensor(_A, device=device)
#     assert ndl.init._calculate_fans(A) == (16, 8)


@pytest.mark.parametrize("device", _DEVICES)
def test_init_kaiming_uniform(device):
    _A = np.random.randn(3, 3, 16, 8)
    A = ndl.Tensor(_A, device=device)
    np.random.seed(0)
    A = ndl.init.kaiming_uniform(16*9, 8*9, shape=A.shape)
    assert abs(A.sum().numpy() - -2.5719218) < 1e-4


@pytest.mark.parametrize("device", _DEVICES)
def test_resnet9(device):
    def num_params(model):
        return np.sum([np.prod(x.shape) for x in model.parameters()])

    from apps.models import ResNet9
    np.random.seed(0)
    model = ResNet9(device=device)

    assert num_params(model) == 431946

    _A = np.random.randn(2, 3, 32, 32)
    A = ndl.Tensor(_A, device=device)
    y = model(A)

    assert np.linalg.norm(np.array([[-1.8912625 ,  0.64833605,  1.9400386 ,  1.1435282 ,  1.89777   ,
         2.9039745 , -0.10433993,  0.35458302, -0.5684191 ,  2.6178317 ],
       [-0.2905612 , -0.4147861 ,  0.90268034,  0.46530387,  1.3335679 ,
         1.8534894 , -0.1867125 , -2.4298222 , -0.5344223 ,  4.362149  ]]) - y.numpy()) < 1e-2



@pytest.mark.parametrize("device", _DEVICES)
def test_dilate_forward(device):
    np.random.seed(0)
    device = ndl.cpu()

    _A = np.random.randint(1, 10, size=(2, 5))
    A = ndl.Tensor(_A, device=device)
    assert np.linalg.norm(ndl.dilate(A, dilation=0, axes=(0,)).numpy() - np.array([[6., 1., 4., 4., 8.],
       [4., 6., 3., 5., 8.]])) < 1e-5 

    _A = np.random.randint(1, 10, size=(2, 5))
    A = ndl.Tensor(_A, device=device)
    assert np.linalg.norm(ndl.dilate(A, dilation=1, axes=(0,)).numpy() - np.array([[7., 9., 9., 2., 7.],
       [0., 0., 0., 0., 0.],
       [8., 8., 9., 2., 6.],
       [0., 0., 0., 0., 0.]])) < 1e-5

    _A = np.random.randint(1, 10, size=(2, 5))
    A = ndl.Tensor(_A, device=device)
    assert np.linalg.norm(ndl.dilate(A, dilation=1, axes=(1,)).numpy() - np.array([[9., 0., 5., 0., 4., 0., 1., 0., 4., 0.],
       [6., 0., 1., 0., 3., 0., 4., 0., 9., 0.]])) < 1e-5

    _A = np.random.randint(1, 10, size=(2, 5))
    A = ndl.Tensor(_A, device=device)
    assert np.linalg.norm(ndl.dilate(A, dilation=1, axes=(0,1)).numpy() - np.array([[2., 0., 4., 0., 4., 0., 4., 0., 8., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 2., 0., 1., 0., 5., 0., 8., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])) < 1e-5

    _A = np.random.randint(1, 10, size=(2, 2))
    A = ndl.Tensor(_A, device=device)
    assert np.linalg.norm(ndl.dilate(A, dilation=2, axes=(0,1)).numpy() - np.array([[4., 0., 0., 3., 0., 0.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.],
       [8., 0., 0., 3., 0., 0.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.]])) < 1e-5

    _A = np.random.randint(1, 10, size=(2, 2, 2, 2))
    A = ndl.Tensor(_A, device=device)
    assert np.linalg.norm(ndl.dilate(A, dilation=1, axes=(1,2)).numpy() - np.array([[[[1., 1.],
         [0., 0.],
         [5., 6.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]],

        [[6., 7.],
         [0., 0.],
         [9., 5.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]]],


       [[[2., 5.],
         [0., 0.],
         [9., 2.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]],

        [[2., 8.],
         [0., 0.],
         [4., 7.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]]]])) < 1e-5


dilate_backward_params = [
    {"shape": (2, 5),          "d": 1, "axes": (0,)},
    {"shape": (2, 5),          "d": 2, "axes": (1,)},
    {"shape": (2, 5),          "d": 1, "axes": (0,1)},
    {"shape": (2, 5),          "d": 0, "axes": (0,1)},
    {"shape": (2, 3, 3, 4),     "d": 2, "axes": (0,1)},
    {"shape": (3, 3, 6, 4),     "d": 3, "axes": (0,1)},
    {"shape": (2, 3, 3, 4),     "d": 0, "axes": (1,2)},
    {"shape": (2, 3, 3, 4),     "d": 1, "axes": (1,2)},
    {"shape": (3, 3, 6, 4),     "d": 1, "axes": (1,2)},
    {"shape": (2, 3, 3, 4),     "d": 1, "axes": (2,3)},
    {"shape": (3, 3, 6, 4),     "d": 1, "axes": (2,3)},
    {"shape": (2, 3, 3, 4),     "d": 1, "axes": (0,1,2,3)},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", dilate_backward_params)
def test_dilate_backward(params, device):
    np.random.seed(0)
    shape, d, axes = params['shape'], params['d'], params['axes']
    backward_check(ndl.dilate, ndl.Tensor(np.random.randn(*shape), device=device), dilation=d, axes=axes)


def test_stack_vs_pytorch():
    np.random.seed(0)
    import torch
    A = np.random.randn(5, 5)
    B = np.random.randn(5, 5)
    C = np.random.randn(5, 5)
    D = np.random.randn(15, 5)

    Andl = ndl.Tensor(A, requires_grad=True)
    Bndl = ndl.Tensor(B, requires_grad=True)
    Cndl = ndl.Tensor(C, requires_grad=True)
    Dndl = ndl.Tensor(D, requires_grad=True)

    Atch = torch.tensor(A, requires_grad=True)
    Btch = torch.tensor(B, requires_grad=True)
    Ctch = torch.tensor(C, requires_grad=True)
    Dtch = torch.tensor(D, requires_grad=True)

    Xndl = ndl.stack([Andl, Cndl @ Bndl, Cndl], axis=1)
    Xtch = torch.stack([Atch, Ctch @ Btch, Ctch], dim=1)

    assert Xndl.shape == Xtch.shape
    assert np.linalg.norm(Xndl.numpy() - Xtch.detach().numpy()) < 1e-3

    Yndl = (Dndl @ Xndl.reshape((5, 15)) @ Dndl).sum()
    Ytch = (Dtch @ Xtch.reshape(5, 15) @ Dtch).sum()

    assert np.linalg.norm(Yndl.numpy() - Ytch.detach().numpy()) < 1e-3

    Yndl.backward()
    Ytch.backward()

    assert np.linalg.norm(Andl.grad.cached_data.numpy() - Atch.grad.detach().numpy()) < 1e-3
    assert np.linalg.norm(Bndl.grad.cached_data.numpy() - Btch.grad.detach().numpy()) < 1e-3
    assert np.linalg.norm(Cndl.grad.cached_data.numpy() - Ctch.grad.detach().numpy()) < 1e-3



conv_forward_params = [
    (4, 8, 16, 3, 1),
    (32, 8, 16, 3, 2),
    (32, 8, 8, 3, 2),
    (32, 16, 8, 3, 1),
    (32, 16, 8, 3, 2)
]
@pytest.mark.parametrize("s,cin,cout,k,stride", conv_forward_params)
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_conv_forward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = ndl.nn.Conv(cin, cout, k, stride=stride, device=device, bias=True)
    x = ndl.init.rand(10, cin, s, s, device=device)
    g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)

    g.weight.data = torch.tensor(f.weight.realize_cached_data().numpy().transpose(3, 2, 0, 1))
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy())  
    assert np.linalg.norm(f(x).realize_cached_data().numpy() - g(z).data.numpy()) < 1e-3


conv_back_params = [
    (4, 1, 1, 3, 1),
    (14, 8, 16, 3, 1),
    (14, 8, 16, 3, 2),
    (14, 8, 8, 3, 1),
    (14, 8, 8, 3, 2),
    (14, 16, 8, 3, 1),
    (14, 16, 8, 3, 2),
]
@pytest.mark.parametrize("s,cin,cout,k,stride", conv_back_params)
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_conv_backward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = ndl.nn.Conv(cin, cout, k, stride=stride, device=device, bias=True)
    x = ndl.init.rand(1, cin, s, s, device=device, requires_grad=True)

    g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
    g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy(), requires_grad=True)
    z.requires_grad = True

    res1 = f(x)
    y1 = res1.sum()

    y2 = g(z).sum()

    y1.backward()
    y2.backward()

    assert np.linalg.norm(g.weight.grad.data.numpy() - f.weight.grad.realize_cached_data().numpy().transpose(3, 2, 0, 1)) < 1e-3, "weight gradients match"
    assert np.linalg.norm(g.bias.grad.data.numpy() - f.bias.grad.realize_cached_data().numpy()) < 1e-3, "bias gradients match"
    assert np.linalg.norm(z.grad.data.numpy() - x.grad.realize_cached_data().numpy()) < 1e-3, "input gradients match"

conv_transposed_forward_params = [
    (4, 8, 16, 3, 1),
    (32, 8, 16, 3, 2),
    (32, 8, 8, 3, 2),
    (32, 16, 8, 3, 1),
    (32, 16, 8, 3, 2)
]
@pytest.mark.parametrize("s,cin,cout,k,stride", conv_transposed_forward_params)
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_conv_tranposed_forward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = ndl.nn.Conv_transposed(cin, cout, k, stride=stride, device=device)
    x = ndl.init.rand(10, cin, s, s, device=device)
    g = torch.nn.ConvTranspose2d(cin, cout, k, stride=stride, padding=k//2)

    g.weight.data = torch.tensor(f.weight.realize_cached_data().numpy().transpose(2, 3, 0, 1))  # k ,k ,in, out -> in, out, k, k 
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy())  
    assert np.linalg.norm(f(x).realize_cached_data().numpy() - g(z).data.numpy()) < 1e-3


conv_transposed_back_params = [
    (4, 1, 1, 3, 1),
    (14, 8, 16, 3, 1),
    (14, 8, 16, 3, 2),
    (14, 8, 8, 3, 1),
    (14, 8, 8, 3, 2),
    (14, 16, 8, 3, 1),
    (14, 16, 8, 3, 2),
]
@pytest.mark.parametrize("s,cin,cout,k,stride", conv_transposed_back_params)
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_conv_transposed_backward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = ndl.nn.Conv_transposed(cin, cout, k, stride=stride, device=device)
    x = ndl.init.rand(1, cin, s, s, device=device, requires_grad=True)

    g = torch.nn.ConvTranspose2d(cin, cout, k, stride=stride, padding=k//2)
    g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(2, 3, 0, 1))
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy(), requires_grad=True)
    z.requires_grad = True

    res1 = f(x)
    y1 = res1.sum()

    y2 = g(z).sum()

    y1.backward()
    y2.backward()

    assert np.linalg.norm(g.weight.grad.data.numpy() - f.weight.grad.realize_cached_data().numpy().transpose(2, 3, 0, 1)) < 1e-3, "weight gradients match"
    assert np.linalg.norm(g.bias.grad.data.numpy() - f.bias.grad.realize_cached_data().numpy()) < 1e-3, "bias gradients match"
    assert np.linalg.norm(z.grad.data.numpy() - x.grad.realize_cached_data().numpy()) < 1e-3, "input gradients match"

maxpool_forward_params = [
    ((1, 1, 4, 4), 2),
    ((3, 8, 14, 14), 7),
    ((3, 8, 20, 20), 2),
    ((3, 8, 20, 20), 4),

    ((5, 10, 15, 15), 3),
    ((5, 10, 15, 15), 5),
    ((5, 10, 16, 16), 4),
    ((5, 10, 16, 16), 8),
]
@pytest.mark.parametrize("Z_shape,kernel_size", maxpool_forward_params) #NCHW
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_maxpool_forward(Z_shape, kernel_size, device):
    np.random.seed(0)
    import torch
    f = ndl.nn.Maxpool(kernel_size, device=device)
    x = ndl.init.rand(*Z_shape, device=device, requires_grad=True)
    g = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)

    z = torch.tensor(x.cached_data.numpy())  
    assert np.linalg.norm(f(x).realize_cached_data().numpy() - g(z).data.numpy()) < 1e-3



maxpool_back_params = [
    ((1, 1, 4, 4), 2),
    ((3, 8, 14, 14), 7),
    ((3, 8, 20, 20), 2),
    ((3, 8, 20, 20), 4),

    ((5, 10, 15, 15), 3),
    ((5, 10, 15, 15), 5),
    ((5, 10, 16, 16), 4),
    ((5, 10, 16, 16), 8),
]
@pytest.mark.parametrize("Z_shape,kernel_size", maxpool_back_params) #NCHW
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_maxpool_backward(Z_shape, kernel_size, device):
    np.random.seed(0)
    import torch
    f = ndl.nn.Maxpool(kernel_size, device=device)
    x = ndl.init.rand(*Z_shape, device=device, requires_grad=True)

    g = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)

    z = torch.tensor(x.cached_data.numpy(), requires_grad=True)
    z.requires_grad = True

    res1 = f(x)
    y1 = res1.sum()

    y2 = g(z).sum()

    y1.backward()
    y2.backward()

    assert np.linalg.norm(z.grad.data.numpy() - x.grad.realize_cached_data().numpy()) < 1e-3, "input gradients match"


op_conv_shapes = [
    ( (3, 14, 14, 8), (3, 3, 8, 16), 1, 0 ),
    ( (3, 14, 14, 8), (3, 3, 8, 16), 1, 1 ),
    ( (3, 16, 16, 8), (3, 3, 8, 16), 1, 2 ),
    ( (3, 16, 16, 8), (3, 3, 8, 14), 1, 0 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 1, 0 ),

    ( (3, 14, 14, 8), (3, 3, 8, 16), 2, 0 ),
    ( (3, 14, 14, 8), (3, 3, 8, 16), 2, 1 ),
    ( (3, 16, 16, 8), (3, 3, 8, 16), 2, 2 ),
    ( (3, 16, 16, 8), (3, 3, 8, 14), 2, 0 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 2, 0 ),

    ( (3, 16, 16, 24), (3, 3, 24, 14), 1, 0 ),
    ( (3, 14, 14, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 1), (5, 5, 1, 16) ,  1, 0),
    ( (3, 17, 17, 16), (5, 5, 16, 1),  1, 0 ),
    ( (3, 17, 17, 16), (1, 1, 16, 1),  1, 0 ),
    ( (1, 14, 14, 2), (3, 3, 2, 2),    1, 0 ),
]
@pytest.mark.parametrize("Z_shape, W_shape, stride, padding", op_conv_shapes)
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("backward", [True, False], ids=["backward", "forward"])
def test_op_conv(Z_shape, W_shape, stride, padding, backward, device):
    np.random.seed(0)
    import torch
    _Z = np.random.randn(*Z_shape)*5
    _Z = _Z.astype(np.float32)
    _W = np.random.randn(*W_shape)*5
    _W = _W.astype(np.float32)
    Z = ndl.Tensor(_Z, device=device)
    W = ndl.Tensor(_W, device=device)
    y = ndl.conv(Z, W, padding=padding, stride=stride)
    y2 = y.sum()
    if backward:
        y2.backward()
    Ztch = torch.Tensor(_Z).float()
    Ztch.requires_grad=True
    Wtch = torch.Tensor(_W).float()
    Wtch.requires_grad=True
    out = torch.nn.functional.conv2d(Ztch.permute(0, 3, 1, 2), Wtch.permute(3, 2, 0, 1), padding=padding, stride=stride)
    out2 = out.sum()
    if backward:
        out2.backward()
    if backward:
        err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
        err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
    err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
    if backward:
        assert err1 < 1e-2, "input grads match"
        assert err2 < 1e-2, "weight grads match"
    assert err3 < 1e-1, "outputs match %s, %s" % (y2, out2)


op_conv_transposed_shapes = [
    ( (3, 14, 14, 8), (1, 1, 8, 16), 1, 0 ),
    ( (3, 14, 14, 8), (1, 1, 8, 16), 1, 1),
    ( (1, 196, 196, 1), (2, 2, 1, 1), 1, 0),
    ( (3, 16, 16, 8), (1, 1, 8, 14), 1, 1 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 1, 0 ),

    ( (3, 14, 14, 8), (2, 2, 8, 16), 2, 0 ),
    ( (3, 7, 7, 8), (2, 2, 8, 16), 2, 0 ),
    ( (3, 16, 16, 8), (2, 2, 8, 16), 2, 0 ),
    ( (3, 16, 16, 8), (3, 3, 8, 14), 2, 0 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 2, 0 ),

    ( (3, 16, 16, 24), (3, 3, 24, 14), 1, 2 ),
    ( (3, 14, 14, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 1), (5, 5, 1, 16) ,  1, 2),
    ( (3, 17, 17, 16), (5, 5, 16, 1),  1, 0 ),
    ( (3, 17, 17, 16), (1, 1, 16, 1),  1, 2 ),
    ( (1, 14, 14, 2), (3, 3, 2, 2),    1, 0 ),
]
@pytest.mark.parametrize("Z_shape, W_shape, stride, padding", op_conv_transposed_shapes)
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("backward", [True, False], ids=["backward", "forward"])
def test_op_conv_tranposed(Z_shape, W_shape, stride, padding, backward, device):
    np.random.seed(0)
    import torch
    _Z = np.random.randn(*Z_shape)*5
    _Z = _Z.astype(np.float32)
    _W = np.random.randn(*W_shape)*5
    _W = _W.astype(np.float32)
    Z = ndl.Tensor(_Z, device=device)
    W = ndl.Tensor(_W, device=device)
    y = ndl.conv_transposed(Z, W, padding=padding, stride=stride)
    y2 = y.sum()
    if backward:
        y2.backward()
    Ztch = torch.Tensor(_Z).float()
    Ztch.requires_grad=True
    Wtch = torch.Tensor(_W).float()
    Wtch.requires_grad=True
    out = torch.nn.functional.conv_transpose2d(Ztch.permute(0, 3, 1, 2), Wtch.permute(2, 3, 0, 1), padding=padding, stride=stride)
    out2 = out.sum()
    if backward:
        out2.backward()
    if backward:
        err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
        err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
    err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
    if backward:
        assert err1 < 1e-2, "input grads match"
        assert err2 < 1e-2, "weight grads match"
    assert err3 < 1e-1, "outputs match %s, %s" % (y2, out2)

op_maxpool_shapes = [
    ((1, 1, 4, 4), 2),
    ((3, 8, 14, 14), 7),
    ((3, 8, 20, 20), 2),
    ((3, 8, 20, 20), 4),

    ((5, 10, 15, 15), 3),
    ((5, 10, 15, 15), 5),
    ((5, 10, 16, 16), 2),
    ((5, 10, 16, 16), 8),
]
@pytest.mark.parametrize("Z_shape, kernel_size", op_maxpool_shapes)  ### NCHW
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("backward", [True, False], ids=["backward", "forward"])
def test_op_maxpool(Z_shape, kernel_size, backward, device):
    np.random.seed(0)
    import torch
    _Z = np.random.randn(*Z_shape)*5
    _Z = _Z.astype(np.float32)
    Z = ndl.Tensor(_Z, device=device)
    y = ndl.maxpool(Z, kernel_size)
    y2 = y.sum()
    if backward:
        y2.backward()
    Ztch = torch.Tensor(_Z).float()
    Ztch.requires_grad=True

    out = torch.nn.functional.max_pool2d(Ztch, kernel_size=kernel_size, stride=kernel_size)
    out2 = out.sum()
    if backward:
        out2.backward()
    if backward:
        err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
    err2 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
    if backward:
        assert err1 < 1e-2, "input grads match"
    assert err2 < 1e-1, "outputs match %s, %s" % (y2, out2)

op_concat_shapes = [
    (((1,1,1,1),(1,1,1,1)), 2),
    (((2,3,4,5),(2,3,4,5)), 1),
    (((2,3,4,5),(2,3,4,5)), 2),
    (((2,3,4,5),(2,3,4,5)), 3),

    (((4,16,392,392),(4,16,392,392)), 1),
    (((2,3,4,5),(2,3,4,5)), 1),
    (((2,3,4,5),(2,3,4,5)), 2),
    (((2,3,4,5),(2,3,4,5)), 3),

    (((2,2,4,5),(2,3,4,5), (2,4,4,5)), 1),
    (((2,3,4,5),(3,3,4,5), (4,3,4,5)), 0),
    (((2,3,4,5),(2,3,5,5), (2,3,6,5)), 2),
    (((2,3,4,5),(2,3,4,6), (2,3,4,7)), 3),
]
@pytest.mark.parametrize("Z_shape, axis", op_concat_shapes)  ### NCHW
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("backward", [True, False], ids=["backward", "forward"])
def test_op_concat(Z_shape, axis, backward, device):
    np.random.seed(0)
    import torch
    input_arrays = []
    input_tch = []
    input_ndl = []
    for shape in Z_shape:
        a = np.random.randn(*shape) * 5
        a = a.astype(np.float32)
        Z = ndl.Tensor(a, device=device)
        Ztch = torch.Tensor(a).float()
        Ztch.requires_grad=True
        input_ndl.append(Z)
        input_tch.append(Ztch) 

    y = ndl.concat(input_ndl, axis=axis)
    y2 = y.sum()
    if backward:
        y2.backward()

    out = torch.cat(input_tch, dim=axis)
    out2 = out.sum()
    if backward:
        out2.backward()
    if backward:
        err1= 0
        for i in range(len(input_arrays)):
            err1 += np.linalg.norm(input_tch[i].grad.numpy() - input_ndl[i].grad.numpy())
    err2 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
    if backward:
        assert err1 < 1e-2, "input grads match"
    assert err2 < 10, "outputs match %s, %s" % (y2, out2)

nn_concat_forward_shapes = [
    (((2,3,4,5),(2,3,4,5)), 0),
    (((2,3,4,5),(2,3,4,5)), 1),
    (((2,3,4,5),(2,3,4,5)), 2),
    (((2,3,4,5),(2,3,4,5)), 3),

    (((2,2,4,5),(2,3,4,5), (2,4,4,5)), 1),
    (((2,3,4,5),(3,3,4,5), (4,3,4,5)), 0),
    (((2,3,4,5),(2,3,5,5), (2,3,6,5)), 2),
    (((2,3,4,5),(2,3,4,6), (2,3,4,7)), 3),
]
@pytest.mark.parametrize("Z_shape, axis", nn_concat_forward_shapes)  ### NCHW
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_concat_forward(Z_shape, axis, device):
    np.random.seed(0)
    import torch
    input_arrays = []
    input_tch = []
    input_ndl = []
    for shape in Z_shape:
        a = np.random.randn(*shape) * 5
        a = a.astype(np.float32)
        Z = ndl.Tensor(a, device=device)
        Ztch = torch.Tensor(a).float()
        Ztch.requires_grad=True
        input_ndl.append(Z)
        input_tch.append(Ztch) 

    f = ndl.nn.Concat(axis)
    y = f(input_ndl).realize_cached_data().numpy()
    y2 = torch.cat(input_tch, dim=axis).data.numpy()
    assert np.linalg.norm(y - y2) < 1e-3

nn_concat_backward_shapes = [
    (((2,3,4,5),(2,3,4,5)), 0),
    (((2,3,4,5),(2,3,4,5)), 1),
    (((2,3,4,5),(2,3,4,5)), 2),
    (((2,3,4,5),(2,3,4,5)), 3),

    (((2,2,4,5),(2,3,4,5), (2,4,4,5)), 1),
    (((2,3,4,5),(3,3,4,5), (4,3,4,5)), 0),
    (((2,3,4,5),(2,3,5,5), (2,3,6,5)), 2),
    (((2,3,4,5),(2,3,4,6), (2,3,4,7)), 3),
]
@pytest.mark.parametrize("Z_shape, axis", nn_concat_backward_shapes)  ### NCHW
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_concat_backward(Z_shape, axis, device):
    np.random.seed(0)
    import torch
    input_tch = []
    input_ndl = []
    for shape in Z_shape:
        a = np.random.randn(*shape) * 5
        a = a.astype(np.float32)
        Z = ndl.Tensor(a, device=device)
        Ztch = torch.Tensor(a).float()
        Ztch.requires_grad=True
        input_ndl.append(Z)
        input_tch.append(Ztch) 

    f = ndl.nn.Concat(axis)
    
    y1 = f(input_ndl).sum()
    y2 = torch.cat(input_tch, dim=axis).sum()
    y1.backward()
    y2.backward()

    err = 0
    for i in range(len(input_ndl)):
        err += np.linalg.norm(input_tch[i].grad.data.numpy() - input_ndl[i].grad.realize_cached_data().numpy())
    assert err < 1e-3, "input gradients match"

# @pytest.mark.parametrize("device", _DEVICES)
def test_up_down_sampling(device=ndl.cuda()):
    import os
    import yaml
    np.random.seed(0)
    configs = {}
    with open('./config/fetal.yaml') as f:
        configs = configs | yaml.safe_load(f)

    li = os.listdir(configs['data_path'] + '/all_images/')

    data_loader = ndl.data.DataLoader(ndl.data.FetalHeadDataset(configs['data_path'], li), batch_size=2)

    down_sample = nn.Sequential(
        nn.Conv(1,1,3,padding=0, device=device),
        nn.ReLU(),
        nn.Conv(1,1,3,padding=0, device=device), 
        nn.ReLU(),
        nn.Maxpool(2),
        nn.Conv(1,1,3,padding=0, device=device),
        nn.ReLU(),
        nn.Conv(1,1,3,padding=0, device=device),        
        nn.ReLU()        
    )
    up_sample = nn.Sequential(
        nn.Conv(1,1,3,padding=0, device=device),
        nn.ReLU(),
        nn.Conv(1,1,3,padding=0, device=device),
        nn.ReLU(),
        nn.Conv_transposed(1,1, kernel_size=2, stride=2, padding=0, device=device),
        nn.ReLU(),
        nn.Conv(1,1,3,padding=0, device=device),
        nn.ReLU(),
        nn.Conv(1,1,3,padding=0, device=device),
        # nn.Conv(1,2,1,padding=0, device=device)
    )
    
    class FCN(nn.Module):
        def forward(self, x):
            x = down_sample(x)
            x = ndl.ops.unpad(x, ((0,0),(0,0),(40,40),(40,40)))
            x = up_sample(x)
            x = nn.ops.concat([x,x], axis=1)
            return x

    model = FCN()
    optimizer = ndl.optim.Adam(model.parameters())
    for x, y in data_loader:
        optimizer.reset_grad()
        x, y = ndl.Tensor(x, device=device), ndl.Tensor(y, device=device) 
        x = model(x)
        B,C,H,W = x.shape
        loss = nn.SoftmaxLoss()(x.transpose((1,2)).transpose((2,3)).reshape((B*H*W, C)), 
            y.reshape((B*H*W,)))
        loss.backward()
        optimizer.step()
        print(loss)
        del x, y, loss

    # out = one_iter_of_cifar10_training(dataloader, model, opt=ndl.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001), device=device)
    # assert np.linalg.norm(np.array(list(out)) - np.array([0.09375, 3.5892258])) < 1e-2


@pytest.mark.parametrize("device", _DEVICES)
def test_train_cifar10(device):
    np.random.seed(0)
    dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    dataloader = ndl.data.DataLoader(\
             dataset=dataset,
             batch_size=128,
             shuffle=False
             # collate_fn=ndl.data.collate_ndarray,
             # drop_last=False,
             # device=device,
             # dtype="float32"
             )
    from apps.models import ResNet9
    np.random.seed(0)
    model = ResNet9(device=device, dtype="float32")
    out = one_iter_of_cifar10_training(dataloader, model, opt=ndl.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001), device=device)
    assert np.linalg.norm(np.array(list(out)) - np.array([0.09375, 3.5892258])) < 1e-2


def one_iter_of_cifar10_training(dataloader, model, niter=1, loss_fn=ndl.nn.SoftmaxLoss(), opt=None, device=None):
    np.random.seed(4)
    model.train()
    correct, total_loss = 0, 0
    i = 1
    for batch in dataloader:
        opt.reset_grad()
        X, y = batch
        X,y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
        out = model(X)
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())       
        loss = loss_fn(out, y)
        total_loss += loss.data.numpy() * y.shape[0]
        loss.backward()
        opt.step()
        if i >= niter:
            break
        i += 1
    # print('loss: ', correct/(y.shape[0]*niter), total_loss/(y.shape[0]*niter))
    return correct/(y.shape[0]*niter), total_loss/(y.shape[0]*niter)


######################    |    ######################
###################### MUGRADE ######################
######################    v    ######################

def Prepare(A):
    return (A.numpy().flatten()[:64], A.shape)


def Rand(*shape, device=ndl.cpu(), entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    _A = np.random.randint(low=1, high=10, size=shape)
    return ndl.Tensor(_A, device=device)


def RandC(*shape, entropy=1):
    if ndl.cuda().enabled():
        return Rand(*shape, device=ndl.cuda(), entropy=2)
    else:
        raise NotImplementedError("You need a GPU to run these tests.")


def MugradeSubmit(things):
    mugrade.submit(Prepare(things))
    # print(Prepare(things))


def submit_conv_forward():
    def DoConvOp(batches, cin, cout, n, k=3, stride=1, padding=0, device=ndl.cpu()):
        X = Rand(batches, n, n, cin, device=device)
        W = Rand(k, k, cin, cout, device=device)
        y = ndl.conv(X, W, stride=stride, padding=padding)
        return y

    def DoConvLayer(batches, cin, cout, n, k=3, stride=1, bias=True, device=ndl.cpu()):
        X = Rand(batches, cin, n, n, device=device)
        f = ndl.nn.Conv(cin, cout, k, stride=stride, bias=bias, device=device)
        return f(X)

    MugradeSubmit(DoConvOp(2, 1, 2, 4, k=1, stride=1, padding=0))
    MugradeSubmit(DoConvOp(2, 1, 2, 4, k=1, stride=1, padding=2))
    MugradeSubmit(DoConvOp(2, 3, 1, 6, k=1, stride=2, padding=2))


    MugradeSubmit(DoConvOp(2, 1, 2, 4, k=3, stride=1, padding=0))
    MugradeSubmit(DoConvOp(3, 1, 2, 4, k=3, stride=1, padding=2))
    MugradeSubmit(DoConvOp(1, 1, 3, 6, k=5, stride=2, padding=2))

    MugradeSubmit(DoConvLayer(3, 2, 4, 6, k=3, stride=1, bias=True))
    MugradeSubmit(DoConvLayer(3, 4, 2, 6, k=3, stride=1, bias=True))
    MugradeSubmit(DoConvLayer(1, 1, 1, 12, k=3, stride=2, bias=True))
    MugradeSubmit(DoConvLayer(1, 1, 1, 12, k=1, stride=1, bias=False))
    MugradeSubmit(DoConvLayer(1, 2, 1, 12, k=7, stride=1, bias=False))
    MugradeSubmit(DoConvLayer(1, 1, 3, 12, k=7, stride=4, bias=False))


    if ndl.cuda().enabled():
        MugradeSubmit(DoConvLayer(3, 2, 4, 6, k=3, stride=1, bias=False, device=ndl.cuda()))
        MugradeSubmit(DoConvLayer(3, 4, 2, 6, k=3, stride=1, bias=False, device=ndl.cuda()))
    else:
        print('You need a GPU to run these tests!')


def submit_conv_backward():

    def DoConvOpBackward(batches, cin, cout, n, k=3, stride=1, padding=0, device=ndl.cpu(), wrtX=True):
        X = Rand(batches, n, n, cin, device=device)
        X.requires_grad = True
        W = Rand(k, k, cin, cout, device=device)
        W.requires_grad = True
        y = ndl.conv(X, W, stride=stride, padding=padding).sum()
        y.backward()
        if wrtX:
            return W.grad
        else:
            return X.grad

    def DoConvLayerBackward(batches, cin, cout, n, k=3, stride=1, bias=True, device=ndl.cpu(), wrtX=True):
        X = Rand(batches, cin, n, n, device=device)
        X.requires_grad = True
        f = ndl.nn.Conv(cin, cout, k, stride=stride, bias=bias, device=device)
        y = f(X).sum()
        y.backward()
        if wrtX:
            return f.weight.grad
        else:
            return X.grad

    MugradeSubmit(DoConvOpBackward(2, 1, 2, 4, k=1, stride=1, padding=0, wrtX=True))
    MugradeSubmit(DoConvOpBackward(2, 3, 1, 6, k=1, stride=2, padding=0, wrtX=True))
    MugradeSubmit(DoConvOpBackward(2, 1, 2, 10, k=3, stride=1, padding=1, wrtX=True))
    MugradeSubmit(DoConvOpBackward(2, 3, 1, 8, k=3, stride=2, padding=2, wrtX=True))
    MugradeSubmit(DoConvOpBackward(2, 1, 3, 8, k=5, stride=1, padding=2, wrtX=True))

    MugradeSubmit(DoConvOpBackward(2, 1, 2, 4, k=1, stride=1, padding=0, wrtX=False))
    MugradeSubmit(DoConvOpBackward(2, 3, 1, 6, k=1, stride=2, padding=0, wrtX=False))
    MugradeSubmit(DoConvOpBackward(2, 1, 2, 6, k=3, stride=1, padding=1, wrtX=False))
    MugradeSubmit(DoConvOpBackward(2, 3, 1, 6, k=3, stride=2, padding=2, wrtX=False))
    MugradeSubmit(DoConvOpBackward(2, 1, 3, 8, k=5, stride=1, padding=2, wrtX=False))

    MugradeSubmit(DoConvLayerBackward(3, 2, 4, 6, k=3, stride=1, bias=True, wrtX=True))
    MugradeSubmit(DoConvLayerBackward(1, 2, 1, 12, k=7, stride=1, bias=False, wrtX=True))
    MugradeSubmit(DoConvLayerBackward(1, 1, 3, 12, k=7, stride=4, bias=False, wrtX=True))
    MugradeSubmit(DoConvLayerBackward(3, 2, 4, 6, k=3, stride=1, bias=True, wrtX=False))
    MugradeSubmit(DoConvLayerBackward(1, 2, 1, 12, k=7, stride=1, bias=False, wrtX=False))
    MugradeSubmit(DoConvLayerBackward(1, 1, 3, 12, k=7, stride=4, bias=False, wrtX=False))

    if ndl.cuda().enabled():
        MugradeSubmit(DoConvLayerBackward(3, 2, 4, 6, k=3, stride=1, bias=False, wrtX=True, device=ndl.cuda()))
        MugradeSubmit(DoConvLayerBackward(3, 4, 2, 6, k=3, stride=1, bias=False, wrtX=False, device=ndl.cuda()))
    else:
        print('You need a GPU to run these tests!')


def submit_new_ops():
    # pad
    np.random.seed(1337)
    _A = np.random.randint(low=1, high=10, size=(2, 2, 2, 2))
    A  = nd.NDArray(_A, device=nd.cpu())
    MugradeSubmit(A.pad(( (0, 0), (1, 1), (2, 2), (0, 0))))

    def DoFlip(shape, axes, backward=False, device=ndl.cpu()):
        X = Rand(*shape, device=device)
        X.requires_grad = True
        Y = ndl.flip(X, axes=axes)
        if backward:
            V = Rand(*shape, device=device, entropy=2)
            Z = (V*Y).sum()
            Z.backward()
            
            return X.grad
        else:
            return Y

    def DoDilate(shape, axes, dilation, backward=False, device=ndl.cpu()):
        X = Rand(*shape, device=device)
        X.requires_grad = True
        Y = ndl.dilate(X, dilation=dilation, axes=axes)
        if backward:
            V = Rand(*Y.shape, device=device, entropy=2)
            Z = (V*Y).sum()
            Z.backward()
            return X.grad
        else:
            return Y

    # flip
    MugradeSubmit(DoFlip((2, 2, 3, 1), (1,2)))
    MugradeSubmit(DoFlip((2, 1, 3, 2), (0,1,2,3)))
    MugradeSubmit(DoFlip((8, 4), (1,)))
    MugradeSubmit(DoFlip((4, 8), (0,)))
    MugradeSubmit(DoFlip((2, 2, 3, 1), (2,3), backward=True))
    MugradeSubmit(DoFlip((2, 1, 3, 2), (1,2,3), backward=True))

    # dilate
    MugradeSubmit(DoDilate((2, 2, 3, 1), (1,2), 1))
    MugradeSubmit(DoDilate((2, 2), (2,), 1))
    MugradeSubmit(DoDilate((2, 2, 3, 1), (1,2), 1, backward=True))
    MugradeSubmit(DoDilate((2, 2), (2,), 1, backward=True))



def submit_resnet9():
    def num_params(model):
        return np.sum([np.prod(x.shape) for x in model.parameters()])

    device = ndl.cpu()
    import sys
    sys.path.append('.')
    from apps.models import ResNet9
    np.random.seed(1)
    model = ResNet9(device=device)

    MugradeSubmit(ndl.Tensor(num_params(model)))

    np.random.seed(1)
    dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    dataloader = ndl.data.DataLoader(\
             dataset=dataset,
             batch_size=128,
             shuffle=True
             )
    np.random.seed(1)
    model = ResNet9(device=device, dtype="float32")
    out = one_iter_of_cifar10_training(dataloader, model, niter=2, opt=ndl.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001), device=device)
    print(out)
    MugradeSubmit(ndl.Tensor(list(out)))


if __name__ == "__main__":
    submit_conv_forward()
    submit_conv_backward()
    submit_new_ops()
    submit_resnet9()
