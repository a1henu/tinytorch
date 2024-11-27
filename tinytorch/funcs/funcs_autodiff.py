from __future__ import annotations

from .funcs import (
    relu_forward, relu_backward,
    sigmoid_forward, sigmoid_backward,
    fc_forward, fc_backward,
    conv2d_forward, conv2d_backward,
    max_pool2d_forward, max_pool2d_backward,
    softmax_forward,
    cross_entropy_forward, cross_entropy_backward
)
from ..tensor import TensorOp, Tensor

class ReLU(TensorOp):
    def compute(self, x):
        return relu_forward(x)
    
    def gradient(self, out_grad, node):
        return Tensor(relu_backward(node.inputs[0].get_cached_data(), out_grad.get_cached_data()))

class Sigmoid(TensorOp):
    def compute(self, x):
        return sigmoid_forward(x)
    
    def gradient(self, out_grad, node):
        return Tensor(sigmoid_backward(node.inputs[0].get_cached_data(), out_grad.get_cached_data()))

class FC(TensorOp):
    def __init__(self):
        self.output = None
        
    def compute(self, x, weight, bias):
        self.output = fc_forward(x, weight, bias)
        return self.output
    
    def gradient(self, out_grad, node):
        x, w, b = node.inputs
        x, w, b = x.get_cached_data(), w.get_cached_data(), b.get_cached_data()
        grads = fc_backward(x, w, b, self.output, out_grad.get_cached_data())
        return tuple(Tensor(grad) for grad in grads)
    
class Conv2d(TensorOp):
    def __init__(self, padding, stride):
        self.padding = padding
        self.stride = stride
        
    def compute(self, x, weight, bias):
        return conv2d_forward(x, weight, bias, self.padding, self.stride)
    
    def gradient(self, out_grad, node):
        x, w, _ = node.inputs
        x, w = x.get_cached_data(), w.get_cached_data()
        grads = conv2d_backward(x, w, out_grad.get_cached_data(), self.padding, self.stride)
        return tuple(Tensor(grad) for grad in grads)
    
class MaxPool2d(TensorOp):
    def __init__(self, kernel_size, padding, stride):
        self.mask = None
        self.shape = None
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
    def compute(self, x):
        self.shape = x.shape
        output, self.mask = max_pool2d_forward(
            x, 
            self.kernel_size, 
            self.padding, 
            self.stride
        )
        return output
    
    def gradient(self, out_grad, node):
        return Tensor(max_pool2d_backward(
            out_grad.get_cached_data(), 
            self.mask, 
            self.kernel_size, 
            self.padding, 
            self.stride,
            self.shape
        ))
    
class Softmax(TensorOp):
    def compute(self, x):
        return softmax_forward(x)
    def gradient(self, out_grad, node):
        return out_grad

class CrossEntropy(TensorOp):
    def __init__(self, target):
        self.target = target.get_cached_data()
    
    def compute(self, x):
        return cross_entropy_forward(x, self.target)
    
    def gradient(self, out_grad, node):
        x = node.inputs[0].get_cached_data()
        return Tensor(cross_entropy_backward(x, self.target))
    
    