"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return self.scalar * out_grad * a ** (self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs * rhs)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if (len(a.shape) == 1):
            return a
        ax = list(range(len(a.shape)))
        if self.axes is not None:
            temp = ax[self.axes[0]]
            ax[self.axes[0]] = ax[self.axes[1]]
            ax[self.axes[1]] = temp
        else:
            temp = ax[-2]
            ax[-2] = ax[-1]
            ax[-1] = temp
        return a.permute(new_axes=ax)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.broadcast_to(self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        dim_diff = len(self.shape) - len(input_shape)
        for i in range(dim_diff):
            out_grad = summation(out_grad, 0)
        for j in range(len(input_shape)):
            if input_shape[j] != out_grad.shape[j]:
                out_grad = out_grad.transpose((0,j))
                out_grad = summation(out_grad, 0)
                out_grad = out_grad.reshape((1, *out_grad.shape))
                out_grad = out_grad.transpose((0,j))
        return out_grad
        # input_shape = node.inputs[0].shape
        # output_shape = out_grad.shape
        # axes = []
        # for i in range(len(output_shape)):
        #     if input_shape[i] != output_shape[i]:
        #         axes.append(i)

        # temp_shape = output_shape
        # for ax in axes:            
        #     temp_shape[ax] = 1
        #     out_grad = summation(out_grad, axes=ax).reshape(temp_shape)
        # return out_grad

        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):        
        ### BEGIN YOUR SOLUTION
        operator = node.inputs[0]
        if self.axes is None:
            # return Tensor(operator.device.full(operator.shape, out_grad.numpy()[0]))
            return init.constant(*operator.shape, c=out_grad.numpy()[0], device=out_grad.device)
        else:
            old_axes = operator.shape
            new_axes = list(out_grad.shape)

            if isinstance(self.axes, tuple):
                index = self.axes[0]
            else:
                index = self.axes

            new_axes.insert(index, 1)
            return out_grad.reshape(new_axes).broadcast_to(old_axes)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        temp_1 = out_grad @ rhs.transpose()
        temp_2 = lhs.transpose() @ out_grad

        def reduce_shape(shape):
            dim = 1
            for i in range(len(shape)):
                dim *= shape[i]
            return dim

        dim1 = reduce_shape(temp_1.shape) //  reduce_shape(lhs.shape) 
        dim2 = reduce_shape(temp_2.shape) //  reduce_shape(rhs.shape) 

        temp_1 = reshape(temp_1, (dim1, *lhs.shape))
        temp_2 = reshape(temp_2, (dim2, *rhs.shape))
        return summation(temp_1, 0), summation(temp_2,0)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.log()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0]
        ones = init.ones(*inp.shape, device=out_grad.device)
        return out_grad * (ones / inp)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0]
        result =  out_grad * Tensor(self.compute(inp), device=out_grad.device)
        return result
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        zeros = a.device.zeros(a.shape)
        return a.maximum(zeros)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0]
        zeros = init.zeros(*inp.shape, device=inp.device)
        mask = inp.realize_cached_data() > zeros.realize_cached_data()

        return out_grad * Tensor(mask, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z1_ = Z.max(self.axes)
        if self.axes is None:
            Z1 = Z.device.full(Z.shape, Z1_.numpy()[0])
        else:
            indexes = list(Z1_.shape)

            if isinstance(self.axes, tuple):
                index = self.axes[0]
            else:
                index = self.axes

            indexes.insert(index, 1)
            Z1 = Z1_.reshape(indexes).broadcast_to(Z.shape)
        Z2 = Z - Z1
        Z3 = Z2.exp()
        Z4 = Z3.sum(self.axes)
        Z5 = Z4.log() + Z1_
        return Z5
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        Z_array = Z.realize_cached_data()
        Z_shape = list(Z.shape)
        if self.axes is not None:
            if isinstance(self.axes, tuple):
                index = self.axes[0]
            else:
                index = self.axes
            Z_shape[index] = 1
        else:
            Z_shape = [1 for _ in range(len(Z.shape))]
        
        c = Tensor(Z_array.max(axis=self.axes), device=out_grad.device).reshape(Z_shape).broadcast_to(Z.shape)
        sum_of_exp = summation(exp(Z-c), axes=self.axes).reshape(Z_shape).broadcast_to(Z.shape)
        softmax = exp(Z-c) / sum_of_exp
        return out_grad.reshape(Z_shape).broadcast_to(Z.shape) * softmax

        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = Tensor(self.compute(node.inputs[0]), device=out_grad.device)
        ones = init.ones(*a.shape, device=a.device)
        return out_grad * (ones - a * a)
        ### END YOUR SOLUTION

def tanh(a):
    return Tanh()(a)

class Sigmoid(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION 
        return a.sigmoid()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = Tensor(self.compute(node.inputs[0], device=out_grad.device))
        ones = init.ones(*a.shape, device=a.device)
        return out_grad * a * (ones - a)
        ### END YOUR SOLUTION

def sigmoid(a):
    return Sigmoid()(a)

class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        size = 1
        for i in range(len(args[0].shape)):
            size *= args[0].shape[i]
        result = args[0].device.empty((len(args), size))
        for i, arg in enumerate(args):
            result[i,:] = arg.flat[:]
        result = result.reshape((len(args), *args[0].shape))
        if self.axis != 0:
            new_axes = list(range(0, len(result.shape)))
            del new_axes[0]
            new_axes.insert(self.axis, 0)
            result = result.permute(new_axes) 

        return result
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION        
        args = node.inputs[0]
        size = 1
        for i in range(len(out_grad.shape)):
            size *= out_grad.shape[i]
        out = out_grad.numpy()
        new_axes = list(range(len(out.shape)))
        new_axes.remove(self.axis)
        new_axes.insert(0, self.axis)    
        out = out.transpose(new_axes)
        out = out.reshape((len(args), size//len(args)))
        result = []
        for i in range(len(args)):
            result.append(Tensor(out[i,:].reshape(args[0].shape)),)
        return MakeTensorTuple()(*result)
        # return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))

class Stack_by_batch(TensorOp):
    def __init__(self, axis: int, batch):
        self.axis = axis
        self.batch = batch

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert self.batch == len(args)

        size = 1
        for i in range(len(args[0].shape)):
            size *= args[0].shape[i]

        batch_size = args[0].shape[self.axis]

        result = args[0].device.empty((batch_size*len(args), size // batch_size))

        indexes = list(range(len(args[0].shape)))
        del indexes[self.axis]
        indexes.insert(0, self.axis)

        new_shape = args[0].shape
        del new_shape[self.axis]

        for i, arg in enumerate(args):
            arg = arg.permute(indexes).reshape((batch_size, size // batch_size))
            result[i*batch_size:(i+1)*batch_size,:] = arg
        result = result.reshape((batch_size*len(args), *new_shape))

        new_indexes =  list(range(len(result.shape)))
        del new_indexes[0]
        new_indexes.insert(self.axis, 0)
        result = result.permute(new_indexes) 

        return result
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION        

        return split_by_batch(out_grad, self.axis, self.batch)
        ### END YOUR SOLUTION


def stack_by_batch(args, axis, batch):
    return Stack_by_batch(axis, batch)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION        
        size = 1
        for i in A.shape:
            size *= i
        
        new_shape = list(A.shape)
        del new_shape[self.axis]
        new_shape.insert(0, A.shape[self.axis])

        indexes = list(range(len(A.shape)))
        del indexes[self.axis]
        indexes.insert(0, self.axis)

        num_childs = new_shape[0]
        B = A.permute(indexes).reshape((num_childs, size // num_childs))

        result = []

        for i in range(num_childs):
            child = B[i,:]
            child = child.reshape(tuple(new_shape[1:]))
            result.append(Tensor(child, device=A.device))

        return MakeTensorTuple()(*result)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION        
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)

class Split_by_batch(TensorTupleOp):
    def __init__(self, axis: int, batch: int=1):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis
        self.batch = batch

    def compute(self, A):
        ### BEGIN YOUR SOLUTION      

        size = 1
        for i in A.shape:
            size *= i
        
        new_shape = list(A.shape)
        del new_shape[self.axis]
        new_shape.insert(0, A.shape[self.axis])

        indexes = list(range(len(A.shape)))
        del indexes[self.axis]
        indexes.insert(0, self.axis)

        num_childs = new_shape[0] // self.batch
        B = A.permute(indexes).reshape((new_shape[0], size // new_shape[0]))

        result = []

        new_indexes = list(range(len(A.shape)))
        del new_indexes[0]
        new_indexes.insert(self.axis, 0)

        for i in range(self.batch):
            child = B[i*num_childs:(i+1)*num_childs, :]
            child = child.reshape((num_childs, *new_shape[1:]))
            child = child.permute(new_indexes)
            result.append(Tensor(child, device=A.device))

        ret = make_tuple(*result)    
        return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION        
        return stack_by_batch(out_grad, self.axis, self.batch)
        ### END YOUR SOLUTION


def split_by_batch(a, axis, batch):
    return Split_by_batch(axis, batch)(a)



class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.flip(self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Pad(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.pad(self.axes)

    def gradient(self, out_grad, node):
        array = out_grad.realize_cached_data().unpad(self.axes)
        return Tensor(array, device=out_grad.device)

def pad(a, axes):
    return Pad(axes)(a)


class UnPad(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.unpad(self.axes)

    def gradient(self, out_grad, node):
        array = out_grad.realize_cached_data().pad(self.axes)
        return Tensor(array, device=out_grad.device)

def unpad(a, axes):
    return UnPad(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        slices = []
        for axis in range(len(a.shape)):
            if axis in self.axes:
                new_shape[axis] *= (1 + self.dilation)
                slices.append(slice(0,new_shape[axis],self.dilation+1))
            else:
                slices.append(slice(0,new_shape[axis],1))          
        result = a.device.zeros(new_shape)
        result[tuple(slices)] = a
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.undilate(self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        slices = []
        for axis in range(len(a.shape)):
            if axis in self.axes:
                new_shape[axis] //= (1 + self.dilation)
                slices.append(slice(0,a.shape[axis],self.dilation+1))
            else:
                slices.append(slice(0,a.shape[axis],1))          
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.dilate(self.axes, self.dilation)
        ### END YOUR SOLUTION

def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION

        A_padded = A.pad(((0,0),(self.padding, self.padding), 
                    (self.padding, self.padding), (0,0)))
                
        N,H,W,C_in = A_padded.shape
        K,_,_,C_out = B.shape

        Ns,Hs,Ws,Cs = A_padded.strides
        Hs_ = Hs * self.stride
        Ws_ = Ws * self.stride
        
        H_new = (H - K) // self.stride + 1
        W_new = (W - K) // self.stride + 1

        new_shape = (N, H_new, W_new, K, K, C_in)
        new_strides = (Ns, Hs_, Ws_, Hs, Ws, Cs)
        Z = A_padded.as_strided(new_shape, new_strides).reshape((N * H_new * W_new, K * K * C_in))
        Z2 = B.reshape((K * K * C_in, C_out))
        out = Z @ Z2

        return out.reshape((N,H_new,W_new,C_out))

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs 
        N,H0,W0,C1 = A.shape
        K,_,_,C2 = B.shape
        mod = (H0 + 2 * self.padding - K) % self.stride
        A1 = A.realize_cached_data().pad(((0,0),(self.padding,self.padding),
                                         (self.padding,self.padding),(0,0)))
                                
        _,H,W,_ = A1.shape
        A1 = A1[:,0:H-mod,0:W-mod,:]                                 

        out1 = dilate(out_grad, axes=(1,2), dilation=self.stride-1).realize_cached_data()
        a,b,c,d = out1.shape
        out1 = out1[:,0:b-self.stride+1,0:c-self.stride+1,:]

        B1 = flip(B, axes=(0,1)).realize_cached_data().permute((0,1,3,2))
        result1 = conv(Tensor(out1), Tensor(B1), padding=K-1).realize_cached_data()

        a,b,c,d = result1.shape

        if result1.shape[1] >= A.shape[1]:
            result1 = result1[:,self.padding:self.padding+H0,
                            self.padding:self.padding+W0,:]
        else:
            pad = A.shape[1] - result1.shape[1]
            result1 = result1.pad(((0,0),(0,pad),(0,pad),(0,0)))

        out2 = dilate(out_grad, axes=(1,2), dilation=self.stride-1).realize_cached_data()
        a,b,c,d = out2.shape
        out2 = out2[0:,0:b-self.stride+1,0:c-self.stride+1,0:]
        out2 = out2.permute((1,2,0,3))
        A2 = A1.permute((3,1,2,0))
        result2 = conv(Tensor(A2), Tensor(out2),padding=0).realize_cached_data()
        result2 = result2.permute((1,2,0,3))

        return Tensor(result1, device=out_grad.device), Tensor(result2, device=out_grad.device)
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


class Conv_transposed(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # dilate A with strides-1, pad A with K-padding-1, conv with B_flipped
        # H0 -> (H0-1)*S + K - 2*P 
        N,H0,W0,C1 = A.shape
        K,_,_,C2 = B.shape
        s_ = self.stride - 1
        p_ = K - self.padding - 1
        A1 = A.dilate((1,2), s_)
        A1 = A1[:,0:A1.shape[1]-s_,0:A1.shape[2]-s_,:].pad(((0,0),(p_,p_),(p_,p_),(0,0)))
        B1 = B.flip(axes=(0,1))
        return conv(Tensor(A1,device=A1.device), Tensor(B1,device=B1.device), padding=0).realize_cached_data()
        
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs    
        N,H0,W0,C1 = A.shape
        K,_,_,C2 = B.shape
        s_ = self.stride - 1
        p_ = K - self.padding - 1
        A1 = dilate(A, axes=(1,2), dilation=s_)  
        A2 = unpad(A1, ((0,0),(0,s_),(0,s_),(0,0)))
        A3 = pad(A2, ((0,0),(p_,p_),(p_,p_),(0,0)))
        B1 = flip(B, axes=(0,1))
        out = conv(A3, B1, padding=0)
        out.backward()
        return A.grad, B.grad
        ### END YOUR SOLUTION


def conv_transposed(a, b, stride=1, padding=0):
    return Conv_transposed(stride, padding)(a, b)


class Maxpool(TensorOp):
    def __init__(self, kernel_size: Optional[int] = 2):
        self.kernel_size = kernel_size

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        return A.maxpool(self.kernel_size)        
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        N,C,H,W = node.inputs[0].shape
        _,_,H_new,W_new = out_grad.shape
        k = self.kernel_size

        input_array = node.inputs[0].realize_cached_data()
        output_array = self.compute(input_array)

        output_array = output_array.reshape((*out_grad.shape, 1, 1)).broadcast_to((*out_grad.shape, k, k))
        output_array = output_array.permute((0,1,2,4,3,5)).reshape((N,C,H,W))

        mask = (output_array == input_array)
        mask2 = mask.as_strided((N,C,H_new,W_new,k,k), strides=[C*H*W, H*W, W*k, k, W, 1])
        mask2 = mask2.reshape((N, C, H_new, W_new, k * k)).sum(axis=4)
        mask2 = mask2.reshape((*mask2.shape, 1, 1)).broadcast_to((*mask2.shape, k, k))
        mask2= mask2.permute((0,1,2,4,3,5)).reshape((N,C,H,W))

        ratio = mask / mask2
        grad = out_grad.realize_cached_data().reshape((*out_grad.shape, 1, 1)).broadcast_to((*out_grad.shape, k, k))
        grad = grad.permute((0,1,2,4,3,5)).reshape((N,C,H,W))

        return Tensor(grad * ratio, device=out_grad.device)
        ### END YOUR SOLUTION


def maxpool(a, kernel_size=2):
    return Maxpool(kernel_size)(a)



class Concat(TensorOp):
    def __init__(self, axis):
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        shapes = [list(arg.shape).remove(self.axis) for arg in args]
        assert len(set(shapes)) == 1, "All tensors must have the same shape except for the axis to concatenate on"
        
        new_shape = list(args[0].shape)
        new_axis_size = 0
        for arg in args:
            new_axis_size += arg.shape[self.axis]
        new_shape[self.axis] = new_axis_size
        result = args[0].device.zeros(new_shape)

        sl = [slice(0,args[0].shape[i],1) for i in range(len(args[0].shape))]

        offset = 0
        for arg in args:
            sl[self.axis] = slice(0+offset,arg.shape[self.axis]+offset,1)            
            result[tuple(sl)] = arg
            offset += arg.shape[self.axis]

        return result
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION        
        output_array = out_grad.realize_cached_data()
        args = node.inputs[0]
        sl = [slice(0,args[0].shape[i],1) for i in range(len(args[0].shape))]
        result = []
        offset = 0
        for arg in args:
            temp = arg.realize_cached_data().device.zeros(arg.shape)
            sl[self.axis] = slice(0+offset, arg.shape[self.axis]+offset, 1)          
            temp = output_array[tuple(sl)]
            result.append(Tensor(temp, device=out_grad.device))
            offset += arg.shape[self.axis]
        return make_tuple(*result)
        ### END YOUR SOLUTION


def concat(args, axis):
    return Concat(axis)(make_tuple(*args))