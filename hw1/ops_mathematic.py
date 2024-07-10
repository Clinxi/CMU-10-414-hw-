"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad, out_grad)


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad, )


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a*b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return (out_grad * rhs, out_grad * lhs)


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        return (out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1)), )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return (grad_a, grad_b)

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        grad_a = out_grad / rhs
        grad_b = -out_grad * lhs / (rhs * rhs)
        return (grad_a, grad_b)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar, )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes: return array_api.swapaxes(a, *self.axes)
        # default is the last two axes
        else: return array_api.swapaxes(a, -1, -2)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        return (transpose(out_grad, self.axes), )
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        a = node.inputs[0]
        return (reshape(out_grad, a.shape), )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # the most important thing is that we need to find the
        # broadcasted dimension
        a = node.inputs[0]
        if a.shape == self.shape:
          return out_grad
        axes = []
        # we need to take care of the added dimension and the '1' dimension
        new_shape = [1] * (len(self.shape)-len(a.shape)) + list(a.shape)
        for i in range(len(self.shape)):
          if new_shape[i] != self.shape[i]:
            axes.append(i)
        grad = summation(out_grad, axes=tuple(axes))
        grad = reshape(grad, a.shape)
        return (grad, )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # change out_grad's shape to node.inputs[0].shape
        
        shape = node.inputs[0].shape
        out_shape = [1] * len(shape)
        if self.axes is None:
          axes_set = set(range(len(shape)))
        else:
          axes_set = set(self.axes)
        
        index = 0
        for i in range(len(shape)):
          if i not in axes_set:
            out_shape[i] = out_grad.shape[index]
            index+=1
        return (broadcast_to(reshape(out_grad, tuple(out_shape)), shape), )
        

        # raise NotImplementedError()
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        try :
           ans = array_api.matmul(a, b)
        except:
          print(f"a is {a} b is {b.shape}")
        return array_api.matmul(a, b)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # lhs : (?) * m * n, rhs :(?) * n * d, outputs :(?) * m * d
        # need to consider (?)'s dimension with broadcast situation
        # lhs, rhs = node.inputs
        # # if there is no bradcast
        # grad_a = out_grad @ transpose(rhs, axes = (-1, -2))
        # grad_b = transpose(lhs, axes=(-1, -2)) @ out_grad

        # # if there exists broadcast, and grad_a.shape is larger
        # if grad_a.shape != lhs.shape:
        #   # how to compress?, using summation(since the gradient...)
        #   grad_a = summation(grad_a, axes = tuple(range(len(grad_a.shape)-len(lhs.shape))))
        # if grad_b.shape != rhs.shape:
        #   grad_b = summation(grad_b, axes = tuple(range(len(grad_b.shape)-len(rhs.shape))))
        
        # return (grad_a, grad_b)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

        a, b = node.inputs
        grad_a, grad_b = (matmul(out_grad, transpose(b)), matmul(transpose(a), out_grad))
        if a.shape != grad_a.shape:
            grad_a = summation(grad_a, axes=tuple(range(len(grad_a.shape)-len(a.shape))))
        if b.shape != grad_b.shape:
            grad_b = summation(grad_b, axes=tuple(range(len(grad_b.shape)-len(b.shape))))
        return (grad_a, grad_b)


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1 * a
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-1 * out_grad, )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR 
        a = node.inputs[0]
        return (out_grad / a, )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * exp(node.inputs[0]), )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        
        return array_api.maximum(a, 0) 
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        # a = node.inputs[0]
        return out_grad * Tensor(a>0)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
