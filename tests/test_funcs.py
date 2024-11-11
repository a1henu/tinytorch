from __future__ import annotations
from typing import Callable, Tuple, List, Any

import pytest
import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_allclose

import torch
import torch.nn.functional as F

from tinytorch import DeviceType, Tensor
from tinytorch.funcs import (
    relu_forward, relu_backward,
    sigmoid_forward, sigmoid_backward
)

from utils import skip_if_no_cuda

def _test_activation_forward_backward(
    x: NDArray,
    forward_fn: Callable[[Tensor], Tensor],
    backward_fn: Callable[[Tensor, Tensor], Tensor],
    torch_fn: Callable[[torch.Tensor], torch.Tensor],
    rtol: float = 1e-5
) -> None:
    """Helper function for testing activation forward/backward"""
    x_tensor = Tensor.from_numpy(x)
    x_torch = torch.tensor(x, requires_grad=True)
    
    output = forward_fn(x_tensor)
    output_torch = torch_fn(x_torch)
    
    assert_allclose(
        output.to_numpy(),
        output_torch.detach().numpy(),
        rtol=rtol
    )
    
    grad = np.random.randn(*x.shape)
    grad_tensor = Tensor.from_numpy(grad)
    
    output_torch.backward(torch.tensor(grad))
    grad_output = backward_fn(x_tensor, grad_tensor)
    
    assert_allclose(
        grad_output.to_numpy(),
        x_torch.grad.numpy(),
        rtol=rtol
    )

def test_relu_cpu():
    """Test ReLU on CPU"""
    # Basic functionality test
    _test_activation_forward_backward(
        x=np.random.randn(2, 3),
        forward_fn=relu_forward,
        backward_fn=relu_backward,
        torch_fn=F.relu
    )
    
    # Edge cases test
    x = np.array([[-1, 0, 1], [1e3, -1e3, 0]])
    x_tensor = Tensor.from_numpy(x)
    output = relu_forward(x_tensor)
    
    assert output.device() == DeviceType.CPU
    assert_allclose(
        output.to_numpy(),
        np.maximum(x, 0)
    )
    
    # Different shapes test
    shapes = [(1, 1), (2, 3), (4, 5, 6)]
    for shape in shapes:
        x = np.random.randn(*shape)
        x_tensor = Tensor.from_numpy(x)
        output = relu_forward(x_tensor)
        assert output.shape() == list(shape)
        assert output.device() == DeviceType.CPU

@skip_if_no_cuda
def test_relu_gpu():
    """Test ReLU on GPU"""
    x = np.random.randn(2, 3)
    x_tensor = Tensor.from_numpy(x)
    x_tensor.to_gpu()
    
    # Forward test
    output = relu_forward(x_tensor)
    assert output.device() == DeviceType.GPU
    
    # Backward test
    grad = Tensor.from_numpy(np.random.randn(2, 3))
    grad.to_gpu()
    grad_output = relu_backward(x_tensor, grad)
    assert grad_output.device() == DeviceType.GPU

def test_sigmoid_cpu():
    """Test Sigmoid on CPU"""
    # Basic functionality test
    _test_activation_forward_backward(
        x=np.random.randn(3, 4),
        forward_fn=sigmoid_forward,
        backward_fn=sigmoid_backward,
        torch_fn=torch.sigmoid
    )
    
    # Edge cases test
    x = np.array([[0, 10, -10]])
    x_tensor = Tensor.from_numpy(x)
    output = sigmoid_forward(x_tensor)
    
    assert output.device() == DeviceType.CPU
    expected = 1 / (1 + np.exp(-x))
    assert_allclose(output.to_numpy(), expected)
    
    # Different shapes test
    shapes = [(1, 1), (2, 3), (4, 5, 6)]
    for shape in shapes:
        x = np.random.randn(*shape)
        x_tensor = Tensor.from_numpy(x)
        output = sigmoid_forward(x_tensor)
        assert output.shape() == list(shape)
        assert output.device() == DeviceType.CPU

@skip_if_no_cuda
def test_sigmoid_gpu():
    """Test Sigmoid on GPU"""
    x = np.random.randn(2, 3)
    x_tensor = Tensor.from_numpy(x)
    x_tensor.to_gpu()
    
    # Forward test
    output = sigmoid_forward(x_tensor)
    assert output.device() == DeviceType.GPU
    
    # Backward test
    grad = Tensor.from_numpy(np.random.randn(2, 3))
    grad.to_gpu()
    grad_output = sigmoid_backward(x_tensor, grad)
    assert grad_output.device() == DeviceType.GPU

if __name__ == "__main__":
    pytest.main([__file__])