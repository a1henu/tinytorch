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
from ..tensor import TensorOp

class ReLU(TensorOp):
    def compute(self, x):
        return relu_forward(x)
    
    def gradient(self, out_grad, node):
        return relu_backward(node.inputs[0].get_cached_data(), out_grad)

class Sigmoid(TensorOp):
    def compute(self, x):
        return sigmoid_forward(x)
    
    def gradient(self, out_grad, node):
        return sigmoid_backward(node.inputs[0].get_cached_data(), out_grad)

class FC(TensorOp):
    def compute(self, x, weight, bias):
        return fc_forward(x, weight, bias)
    
    def gradient(self, out_grad, node):
        x, w, b = node.inputs
        x, w, b = x.get_cached_data(), w.get_cached_data(), b.get_cached_data()
        return fc_backward(x, w, b, FC.compute(x, w, b), out_grad)
    
class Conv2D(TensorOp):
    def compute(self, x, weight, bias, padding, stride):
        return conv2d_forward(x, weight, bias, padding, stride)
    
    def gradient(self, out_grad, node):
        x, w, b, padding, stride = node.inputs
        return conv2d_backward(x, w, out_grad, padding, stride)
    
# TODO: Check the input order
class MaxPool2D(TensorOp):
    def compute(self, x, kernel_size, stride):
        return max_pool2d_forward(x, kernel_size, stride)
    
    def gradient(self, out_grad, node):
        return max_pool2d_backward(out_grad, node.inputs[0].get_cached_data(), node.inputs[1].get_cached_data(), node.inputs[2].get_cached_data())
    
class Softmax(TensorOp):
    def compute(self, x):
        return softmax_forward(x)
    def gradient(self, out_grad, node):
        return out_grad

class CrossEntropy(TensorOp):
    def compute(self, x, target):
        return cross_entropy_forward(x, target)
    def gradient(self, out_grad, node):
        return cross_entropy_backward(out_grad, node.inputs[0].get_cached_data(), node.inputs[1].get_cached_data())