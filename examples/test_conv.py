from __future__ import annotations
from typing import Tuple

import numpy as np
from numpy.testing import assert_allclose

import torch
import torch.nn.functional as F

from tinytorch import DeviceType, TensorBase
from tinytorch.funcs import (
    conv2d_forward, conv2d_backward,
)

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
    
if __name__ == "__main__":
    print("Running Fully Connected Layer tests")
    print("===== CPU =====")
    test_conv2d_cpu()
    print("===== GPU =====")
    test_conv2d_gpu()