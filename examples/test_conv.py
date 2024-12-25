from __future__ import annotations
from typing import Tuple

import numpy as np
from numpy.testing import assert_allclose

import torch
import torch.nn.functional as F

from tinytorch import DeviceType, TensorBase
from tinytorch.funcs import (
    max_pool2d_forward, max_pool2d_backward
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
    
if __name__ == "__main__":
    print("Running Fully Connected Layer tests")
    print("===== CPU =====")
    test_max_pool_cpu()
    print("===== GPU =====")
    test_max_pool_gpu()