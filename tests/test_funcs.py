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

def _test_activation_forward_backward(
    x: NDArray,
    forward_fn: Callable[[Tensor], Tensor],
    backward_fn: Callable[[Tensor, Tensor], Tensor],
    torch_fn: Callable[[torch.Tensor], torch.Tensor],
    rtol: float = 1e-5
) -> None:
    """
    Generic test utility for activation function's forward and backward propagation
    
    Parameters:
        x: Input data
        forward_fn: Forward propagation function in tinytorch
        backward_fn: Backward propagation function in tinytorch
        torch_fn: Corresponding PyTorch activation function
        rtol: Relative tolerance for numerical comparison
    """
    x_tensor = Tensor.from_numpy(x)
    x_torch = torch.tensor(x, requires_grad=True)
    
    # Test forward propagation
    output = forward_fn(x_tensor)
    output_torch = torch_fn(x_torch)
    
    assert_allclose(
        output.to_numpy(),
        output_torch.detach().numpy(),
        rtol=rtol
    )
    
    # Test backward propagation
    grad = np.random.randn(*x.shape)
    grad_tensor = Tensor.from_numpy(grad)
    
    output_torch.backward(torch.tensor(grad))
    grad_output = backward_fn(x_tensor, grad_tensor)
    
    assert_allclose(
        grad_output.to_numpy(),
        x_torch.grad.numpy(),
        rtol=rtol
    )
    
    # Check shape preservation
    assert output.shape() == x_tensor.shape()
    assert grad_output.shape() == x_tensor.shape()
    
    # Check device preservation
    assert output.device() == x_tensor.device()
    assert grad_output.device() == x_tensor.device()

def _test_activation_edge_cases(
    forward_fn: Callable[[Tensor], Tensor],
    edge_cases: List[Tuple[NDArray, Any]],
    rtol: float = 1e-5
) -> None:
    """
    Test activation function behavior on edge cases
    
    Parameters:
        forward_fn: Activation function
        edge_cases: List of (input, expected_output) tuples for edge cases
        rtol: Relative tolerance for numerical comparison
    """
    for x, expected in edge_cases:
        x_tensor = Tensor.from_numpy(x)
        output = forward_fn(x_tensor).to_numpy()
        
        if isinstance(expected, np.ndarray):
            assert_allclose(output, expected, rtol=rtol)
        else:
            assert_allclose(output[x > 0], expected[0], rtol=rtol)
            assert_allclose(output[x < 0], expected[1], rtol=rtol)

def _test_activation_shapes(
    forward_fn: Callable[[Tensor], Tensor],
    shapes: List[Tuple[int, ...]]
) -> None:
    """
    Test activation function handling of different input shapes
    
    Parameters:
        forward_fn: Activation function
        shapes: List of shapes to test
    """
    for shape in shapes:
        x = np.random.randn(*shape)
        x_tensor = Tensor.from_numpy(x)
        output = forward_fn(x_tensor)
        assert output.shape() == list(shape)

def _test_activation_device(
    forward_fn: Callable[[Tensor], Tensor],
    shape: Tuple[int, ...] = (2, 3)
) -> None:
    """
    Test device consistency of activation function
    
    Parameters:
        forward_fn: Activation function
        shape: Tensor shape for testing
    """
    x = np.random.randn(*shape)
    x_tensor = Tensor.from_numpy(x)
    
    if hasattr(x_tensor, "gpu"):
        x_tensor.to_gpu()
        output = forward_fn(x_tensor)
        assert output.device() == DeviceType.GPU
        x_tensor.to_cpu()
    
    output = forward_fn(x_tensor)
    assert output.device() == DeviceType.CPU

# ReLU tests
def test_relu():
    """Test ReLU in various scenarios"""
    # Basic functionality test
    _test_activation_forward_backward(
        x=np.random.randn(2, 3),
        forward_fn=relu_forward,
        backward_fn=relu_backward,
        torch_fn=F.relu
    )
    
    # Edge cases test
    edge_cases = [
        (np.zeros((2, 2)), np.zeros((2, 2))),
        (np.array([[-1, 0, 1]]), np.array([[0, 0, 1]])),
        (np.array([[1e3, -1e3]]), np.array([[1e3, 0]]))
    ]
    _test_activation_edge_cases(relu_forward, edge_cases)
    
    # Different shapes test
    shapes = [(1, 1), (2, 3), (4, 5, 6), (2, 3, 4, 5)]
    _test_activation_shapes(relu_forward, shapes)
    
    # Device consistency test
    _test_activation_device(relu_forward)

# Sigmoid tests
def test_sigmoid():
    """Test Sigmoid in various scenarios"""
    # Basic functionality test
    _test_activation_forward_backward(
        x=np.random.randn(3, 4),
        forward_fn=sigmoid_forward,
        backward_fn=sigmoid_backward,
        torch_fn=torch.sigmoid
    )
    
    # Edge cases test
    edge_cases = [
        (np.zeros((2, 2)), 0.5 * np.ones((2, 2))),
        (np.array([[1e3, -1e3]]), (1.0, 0.0))  # Saturation test
    ]
    _test_activation_edge_cases(sigmoid_forward, edge_cases)
    
    # Different shapes test
    shapes = [(1, 1), (2, 3), (4, 5, 6), (2, 3, 4, 5)]
    _test_activation_shapes(sigmoid_forward, shapes)
    
    # Device consistency test
    _test_activation_device(sigmoid_forward)

if __name__ == "__main__":
    pytest.main([__file__])