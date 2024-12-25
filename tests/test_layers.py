from __future__ import annotations
from typing import Callable, Tuple, List, Any

import pytest
import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_allclose

import torch
import torch.nn.functional as F

from tinytorch import DeviceType, TensorBase
from tinytorch.funcs import (
    fc_forward, fc_backward,
    conv2d_forward, conv2d_backward,
    max_pool2d_forward, max_pool2d_backward,
    softmax_forward,
    cross_entropy_forward, cross_entropy_backward
)

from utils import skip_if_no_cuda

def _test_fc_forward_backward(
    batch_size: int,
    in_features: int,
    out_features: int,
    device: DeviceType,
    rtol: float = 1e-5
) -> None:
    """
    Test fully connected layer forward and backward propagation
    
    Args:
        batch_size: Number of samples in batch
        in_features: Input feature dimension
        out_features: Output feature dimension
        device: Device type (CPU or GPU)
        rtol: Relative tolerance for numerical comparison
    """
    # Initialize input and parameters
    x = np.random.randn(batch_size, in_features)
    w = np.random.randn(in_features, out_features) / np.sqrt(in_features)
    b = np.random.randn(out_features)
    
    # Create tensors
    x_tensor = TensorBase.from_numpy(x, device=device)
    w_tensor = TensorBase.from_numpy(w, device=device)
    b_tensor = TensorBase.from_numpy(b, device=device)
    
    # Create PyTorch tensors
    x_torch = torch.tensor(x, requires_grad=True)
    w_torch = torch.tensor(w, requires_grad=True)
    b_torch = torch.tensor(b, requires_grad=True)
    
    # Forward pass
    output = fc_forward(x_tensor, w_tensor, b_tensor)
    output_torch = F.linear(x_torch, w_torch.T, b_torch)
    
    # Check forward results
    assert_allclose(
        output.to_numpy(),
        output_torch.detach().cpu().numpy(),
        rtol=rtol
    )
    
    # Backward pass
    grad = np.random.randn(batch_size, out_features)
    grad_tensor = TensorBase.from_numpy(grad, device=device)
    
    output_torch.backward(torch.tensor(grad))
    grad_x, grad_w, grad_b = fc_backward(
        x_tensor, w_tensor, b_tensor, output, grad_tensor
    )
    
    # Check backward results
    assert_allclose(grad_x.to_numpy(), x_torch.grad.cpu().numpy(), rtol=rtol)
    assert_allclose(grad_w.to_numpy(), w_torch.grad.cpu().numpy(), rtol=rtol)
    assert_allclose(grad_b.to_numpy(), b_torch.grad.cpu().numpy().reshape(1, -1), rtol=rtol)

def _test_conv2d_forward_backward(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    device: DeviceType,
    rtol: float = 1e-5
) -> None:
    """
    Test 2D convolution forward and backward propagation
    
    Args:
        batch_size: Number of samples in batch
        in_channels: Number of input channels
        out_channels: Number of output channels
        height: Input height
        width: Input width
        kernel_size: (kernel_height, kernel_width)
        stride: (stride_height, stride_width)
        padding: (padding_height, padding_width)
        device: Device type (CPU or GPU)
        rtol: Relative tolerance for numerical comparison
    """
    kernel_h, kernel_w = kernel_size
    
    # Initialize input and parameters
    x = np.random.randn(batch_size, in_channels, height, width)
    w = np.random.randn(out_channels, in_channels, kernel_h, kernel_w) / np.sqrt(in_channels * kernel_h * kernel_w)
    b = np.random.randn(out_channels)
    
    # Create tensors
    x_tensor = TensorBase.from_numpy(x, device=device)
    w_tensor = TensorBase.from_numpy(w, device=device)
    b_tensor = TensorBase.from_numpy(b, device=device)
    
    # Create PyTorch tensors
    x_torch = torch.tensor(x, requires_grad=True)
    w_torch = torch.tensor(w, requires_grad=True)
    b_torch = torch.tensor(b, requires_grad=True)
    
    # Forward pass
    output = conv2d_forward(x_tensor, w_tensor, b_tensor, padding, stride)
    output_torch = F.conv2d(x_torch, w_torch, b_torch, stride=stride, padding=padding)
    
    # Check forward results
    assert_allclose(
        output.to_numpy(),
        output_torch.detach().cpu().numpy(),
        rtol=rtol,
        err_msg="Forward pass values mismatch"
    )
    
    # Backward pass
    grad = np.random.randn(*output.shape)
    grad_tensor = TensorBase.from_numpy(grad, device=device)
    
    output_torch.backward(torch.tensor(grad))
    grad_x, grad_w, grad_b = conv2d_backward(
        x_tensor, w_tensor, grad_tensor,
        padding, stride
    )
    
    # Check backward results
    assert_allclose(
        grad_x.to_numpy(), x_torch.grad.cpu().numpy(),
        rtol=rtol, err_msg="Gradient input values mismatch"
    )
    assert_allclose(
        grad_w.to_numpy(), w_torch.grad.cpu().numpy(),
        rtol=rtol, err_msg="Gradient weight values mismatch"
    )
    assert_allclose(
        grad_b.to_numpy(), b_torch.grad.cpu().numpy(),
        rtol=rtol, err_msg="Gradient bias values mismatch"
    )
    
def _test_max_pool_forward_backward(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    device: DeviceType,
    rtol: float = 1e-5
) -> None:
    """
    Test max pooling forward and backward propagation
    
    Args:
        batch_size: Number of samples in batch
        channels: Number of channels
        height: Input height
        width: Input width
        kernel_size: (kernel_height, kernel_width)
        stride: (stride_height, stride_width)
        padding: (padding_height, padding_width)
        device: Device type (CPU or GPU)
        rtol: Relative tolerance for numerical comparison
    """
    # Initialize input
    x = np.random.randn(batch_size, channels, height, width)
    x_tensor = TensorBase.from_numpy(x, device=device)
    x_torch = torch.tensor(x, requires_grad=True)
    
    # Forward pass
    output, mask = max_pool2d_forward(
        x_tensor, kernel_size, padding, stride
    )
    output_torch = F.max_pool2d(
        x_torch,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    
    # Check forward results
    assert_allclose(
        output.to_numpy(),
        output_torch.detach().cpu().numpy(),
        rtol=rtol,
        err_msg="Forward pass values mismatch"
    )
    
    # Backward pass
    grad = np.random.randn(*output.shape)
    grad_tensor = TensorBase.from_numpy(grad, device=device)
    
    output_torch.backward(torch.tensor(grad))
    grad_x = max_pool2d_backward(
        grad_tensor, mask,
        kernel_size, padding, stride,
        x_tensor.shape
    )
    
    # Check backward results
    assert_allclose(
        grad_x.to_numpy(),
        x_torch.grad.cpu().numpy(),
        rtol=rtol,
        err_msg="Gradient values mismatch"
    )

def _test_softmax_cross_entropy(
    batch_size: int,
    num_classes: int,
    device: DeviceType,
    rtol: float = 1e-5
) -> None:
    """
    Test softmax and cross entropy loss
    
    Args:
        batch_size: Number of samples in batch
        num_classes: Number of classes
        device: Device type (CPU or GPU)
        rtol: Relative tolerance for numerical comparison
    """
    # Initialize input and target
    x = np.random.randn(batch_size, num_classes)
    t = np.random.randint(0, num_classes, size=batch_size)
    
    x_tensor = TensorBase.from_numpy(x, device=device)
    t_tensor = TensorBase.from_numpy(t, device=device)
    x_torch = torch.tensor(x, requires_grad=True)
    t_torch = torch.tensor(t)
    
    # Forward pass
    prob = softmax_forward(x_tensor)
    loss = cross_entropy_forward(x_tensor, t_tensor)
    
    prob_torch = F.softmax(x_torch, dim=1)
    loss_torch = F.cross_entropy(x_torch, t_torch)
    
    # Check forward results
    assert_allclose(
        prob.to_numpy(),
        prob_torch.detach().cpu().numpy(),
        rtol=rtol,
        err_msg="Softmax forward pass values mismatch"
    )
    assert_allclose(
        loss.to_numpy(),
        loss_torch.detach().cpu().numpy(),
        rtol=rtol,
        err_msg="Cross entropy forward pass values mismatch"
    )
    
    # Backward pass
    grad = cross_entropy_backward(x_tensor, t_tensor)
    loss_torch.backward()
    
    # Check backward results
    assert_allclose(
        grad.to_numpy(),
        x_torch.grad.cpu().numpy(),
        rtol=rtol,
        err_msg="Gradient values mismatch"
    )

def test_fc_cpu():
    """Test Fully Connected layer in various scenarios (CPU)"""
    # Basic functionality test
    _test_fc_forward_backward(
        batch_size=32,
        in_features=64,
        out_features=128,
        device=DeviceType.CPU
    )
    
    # Test different configurations
    configs = [
        (1, 10, 5),      # Single sample
        (100, 784, 10),  # MNIST-like
        (32, 512, 256),  # Large features
        (64, 3, 1),      # Single output
    ]
    
    for config in configs:
        _test_fc_forward_backward(*config, device=DeviceType.CPU)
    
@skip_if_no_cuda
def test_fc_gpu():
    """Test Fully Connected layer GPU operations"""
    # Basic functionality test
    _test_fc_forward_backward(
        batch_size=32,
        in_features=64,
        out_features=128,
        device=DeviceType.GPU
    )
    
    # Test different configurations
    configs = [
        (1, 10, 5),      # Single sample
        (100, 784, 10),  # MNIST-like
        (32, 512, 256),  # Large features
        (64, 3, 1),      # Single output
    ]
    
    for config in configs:
        _test_fc_forward_backward(*config, device=DeviceType.GPU)

def test_conv2d_cpu():
    """Test Conv2D layer in various scenarios (CPU)"""
    # Basic functionality test
    _test_conv2d_forward_backward(
        batch_size=2,
        in_channels=3,
        out_channels=16,
        height=32,
        width=32,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        device=DeviceType.CPU
    )
    
    # Test different configurations
    configs = [
        (1, 1, 1, 28, 28, (3, 3), (1, 1), (0, 0)),    # Single channel
        (2, 3, 6, 28, 28, (3, 3), (1, 1), (0, 0)),    # Multiple channels
    ]
    
    for config in configs:
        _test_conv2d_forward_backward(*config, device=DeviceType.CPU)
    
@skip_if_no_cuda
def test_conv2d_gpu():
    """Test Conv2D layer GPU operations"""
    # Basic functionality test
    _test_conv2d_forward_backward(
        batch_size=2,
        in_channels=3,
        out_channels=16,
        height=32,
        width=32,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        device=DeviceType.GPU
    )
    
    # Test different configurations
    configs = [
        (1, 1, 1, 28, 28, (3, 3), (1, 1), (0, 0)),    # Single channel
        (2, 3, 6, 28, 28, (3, 3), (1, 1), (0, 0)),    # Multiple channels
    ]
    
    for config in configs:
        _test_conv2d_forward_backward(*config, device=DeviceType.GPU)
    
def test_max_pool_cpu():
    """Test Max Pooling layer in various scenarios (CPU)"""
    # Basic functionality test
    _test_max_pool_forward_backward(
        batch_size=2,
        channels=16,
        height=32,
        width=32,
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=(0, 0),
        device=DeviceType.CPU
    )
    
    # Test different configurations
    configs = [
        # (batch_size, channels, h, w, kernel, stride, pad)
        (1, 1, 28, 28, (2, 2), (2, 2), (0, 0)),     # Single channel
        (8, 16, 14, 14, (2, 2), (2, 2), (0, 0)),   # Multiple channels
        (32, 3, 32, 32, (2, 2), (2, 2), (0, 0)),  # Large size
    ]
    
    for config in configs:
        _test_max_pool_forward_backward(*config, device=DeviceType.CPU)
    
@skip_if_no_cuda
def test_max_pool_gpu():
    """Test Max Pooling layer GPU operations"""
    # Basic functionality test
    _test_max_pool_forward_backward(
        batch_size=2,
        channels=16,
        height=32,
        width=32,
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=(0, 0),
        device=DeviceType.GPU
    )
    
    # Test different configurations
    configs = [
        # (batch_size, channels, h, w, kernel, stride, pad)
        (1, 1, 28, 28, (2, 2), (2, 2), (0, 0)),     # Single channel
        (8, 16, 14, 14, (2, 2), (2, 2), (0, 0)),   # Multiple channels
        (32, 3, 32, 32, (2, 2), (2, 2), (0, 0)),  # Large size
    ]
    
    for config in configs:
        _test_max_pool_forward_backward(*config, device=DeviceType.GPU)

def test_softmax_cross_entropy_cpu():
    """Test Softmax and Cross Entropy in various scenarios (CPU)"""
    # Basic functionality test
    _test_softmax_cross_entropy(
        batch_size=32,
        num_classes=10,
        device=DeviceType.CPU
    )
    
    # Test different configurations
    configs = [
        (1, 2),      # Binary classification
        (100, 100),  # Large number of classes
        (256, 10),   # Large batch size
        (16, 5),     # Small number of classes
    ]
    
    for config in configs:
        _test_softmax_cross_entropy(*config, device=DeviceType.CPU)
    
@skip_if_no_cuda
def test_softmax_cross_entropy_gpu():
    """Test Softmax and Cross Entropy GPU operations"""
    # Basic functionality test
    _test_softmax_cross_entropy(
        batch_size=32,
        num_classes=10,
        device=DeviceType.GPU
    )
    
    # Test different configurations
    configs = [
        (1, 2),      # Binary classification
        (100, 100),  # Large number of classes
        (256, 10),   # Large batch size
        (16, 5),     # Small number of classes
    ]
    
    for config in configs:
        _test_softmax_cross_entropy(*config, device=DeviceType.GPU)

if __name__ == "__main__":
    pytest.main([__file__])