from __future__ import annotations
from typing import List, Tuple

import numpy as np 
from numpy.typing import NDArray

from tinytorch import Tensor

from ._libfuncs import                                                                                  \
    relu_forward as _relu_forward, relu_backward as _relu_backward,                                     \
    sigmoid_forward as _sigmoid_forward, sigmoid_backward as _sigmoid_backward,                         \
    fc_forward as _fc_forward, fc_backward as _fc_backward,                                             \
    conv2d_forward as _conv2d_forward, conv2d_backward as _conv2d_backward,                             \
    max_pool_forward as _max_pool_forward, max_pool_backward as _max_pool_backward,                     \
    softmax_forward as _softmax_forward,                                                                \
    cross_entropy_forward as _cross_entropy_forward, cross_entropy_backward as _cross_entropy_backward
    
    
def relu_forward(
    input: Tensor
) -> Tensor:
    """
    ReLU activation function forward propagation.
    - ReLU(x) = max(0, x)
    
    Parameters:
        input: Input tensor.
    
    Returns:
        Tensor: Output tensor.
    """
    output = Tensor(input.shape(), input.device())
    _relu_forward(output, input)
    return output

def relu_backward(
    input: Tensor,
    grad: Tensor
) -> Tensor:
    """
    ReLU activation function backward propagation.
    - dL/dx = dL/dy * (x > 0)
    
    Parameters:
        input: Input tensor.
        grad: Gradient tensor.
    
    Returns:
        Tensor: Output tensor.
    """
    assert input.shape() == grad.shape(), (
        f"Input shape {input.shape()} doesn't match gradient shape {grad.shape()}"
    )
    
    output = Tensor(input.shape(), input.device())
    _relu_backward(output, input, grad)
    return output

def sigmoid_forward(
    input: Tensor
) -> Tensor:
    """
    Sigmoid activation function forward propagation.
    - Sigmoid(x) = 1 / (1 + exp(-x))
    
    Parameters:
        input: Input tensor.
    
    Returns:
        Tensor: Output tensor.
    """
    output = Tensor(input.shape(), input.device())
    _sigmoid_forward(output, input)
    return output

def sigmoid_backward(
    input: Tensor,
    grad: Tensor
) -> Tensor:
    """
    Sigmoid activation function backward propagation.
    - dL/dx = dL/dy * Sigmoid(x) * (1 - Sigmoid(x))
    
    Parameters:
        input: Input tensor.
        grad: Gradient tensor.
    
    Returns:
        Tensor: Output tensor.
    """
    assert input.shape() == grad.shape(), (
        f"Input shape {input.shape()} doesn't match gradient shape {grad.shape()}"
    )
    
    output = Tensor(input.shape(), input.device())
    _sigmoid_backward(output, input, grad)
    return output

def fc_forward(
    input: Tensor,    # (batch_size, in_features)
    weight: Tensor,   # (in_features, out_features)
    bias: Tensor      # (1, out_features)
) -> Tensor:
    """
    Fully connected layer forward propagation.
    - Y = XW + b
    
    Parameters:
        input: Input tensor.
        weight: Weight tensor.
        bias: Bias tensor.
    
    Returns:
        Tensor: Output tensor.
    """
    if input.dim() == 1:
        input = input.reshape([1, -1])
    if bias.dim() == 1:
        bias = bias.reshape([1, -1])
    
    batch_size, in_features =  input.shape()
    in_weight, out_features =  weight.shape()
    assert weight.dim() == 2, f"Weight tensor must be 2D, but got shape {weight.shape()}"
    assert in_features == in_weight, (
        f"Input feature dimension {in_features} doesn't match weight dimension {in_weight}"
    )
    assert bias.shape()[1] == out_features, (
        f"Bias feature dimension {bias.shape()[1]} doesn't match weight output dimension {out_features}"
    )
    
    output = Tensor([batch_size, out_features], input.device())
    _fc_forward(input, weight, bias, output)
    return output

def fc_backward(
    input: Tensor,    # (batch_size, in_features)
    weight: Tensor,   # (in_features, out_features)
    bias: Tensor,     # (1, out_features)
    output: Tensor,   # (batch_size, out_features)
    grad: Tensor      # (batch_size, out_features)
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Fully connected layer backward propagation.
    - dX = dY * W^T
    - dW = X^T * dY
    - db = \sum dY
    
    Parameters:
        input: Input tensor.
        weight: Weight tensor.
        bias: Bias tensor.
        output: Output tensor.
        grad: Gradient tensor.
    
    Returns:
        Tuple[
            Tensor,      // Gradients of input
            Tensor,      // Gradients of weight
            Tensor       // Gradients of bias
        ]: Gradients of input, weight, and bias.
    """
    if input.dim() == 1:
        input = input.reshape([1, -1])
    if bias.dim() == 1:
        bias = bias.reshape([1, -1])
    if output.dim() == 1:
        output = output.reshape([1, -1])
    
    batch_size, in_features  =  input.shape()
    in_weight, out_features  =  weight.shape()
    
    assert in_features == in_weight, (
        f"Input feature dimension {in_features} doesn't match weight dimension {in_weight}"
    )
    assert bias.shape()[1] == out_features, (
        f"Bias feature dimension {bias.shape()[1]} doesn't match weight output dimension {out_features}"
    )
    assert output.shape()[0] == batch_size, (
        f"Output batch size {output.shape()[0]} doesn't match input batch size {batch_size}"
    )
    assert output.shape()[1] == out_features, (
        f"Output feature dimension {output.shape()[1]} doesn't match weight output dimension {out_features}"
    )
    assert grad.shape() == output.shape(), (
        f"Gradient shape {grad.shape()} doesn't match output shape {output.shape()}"
    )
    
    grad_input = Tensor(input.shape(), input.device())
    grad_weight = Tensor(weight.shape(), weight.device())
    grad_bias = Tensor(bias.shape(), bias.device())
    
    _fc_backward(input, weight, bias, output, grad_input, grad_weight, grad_bias, grad)
    return grad_input, grad_weight, grad_bias

def conv2d_forward(
    input: Tensor,   # (batch_size, in_channels, height, width)
    weight: Tensor,  # (out_channels, in_channels, kernel_h, kernel_w)
    bias: Tensor,    # (out_channels)
    padding: Tuple[int, int],
    stride: Tuple[int, int]
) -> Tensor:
    """
    2D convolution forward propagation.
    - Y = W conv X + b
    
    Parameters:
        input: Input tensor (batch_size, in_channels, height, width)
        weight: Weight tensor (out_channels, in_channels, kernel_h, kernel_w)
        bias: Bias tensor (out_channels)
        padding: Tuple of height and width padding
        stride: Tuple of height and width stride
    
    Returns:
        Tensor: Output tensor (batch_size, out_channels, height_out, width_out)
    """    
    if input.dim() == 2:
        height, width = input.shape()
        input = input.reshape([1, 1, height, width])
    if input.dim() == 3:
        channels, height, width = input.shape()
        input = input.reshape([1, channels, height, width])
        
    batch_size, in_channels, height, width = input.shape()
    out_channels, in_channels_w, kernel_h, kernel_w = weight.shape()
    out_channels_b = bias.shape()[0]
    
    assert input.dim() == 4, f"Input tensor must be 2D, 3D or 4D, but got shape {input.shape()}"
    assert weight.dim() == 4, f"Weight tensor must be 4D, but got shape {weight.shape()}"
    assert bias.dim() == 1, f"Bias tensor must be 1D, but got shape {bias.shape()}"
    assert in_channels == in_channels_w, (
        f"Input channel dimension {in_channels} doesn't match weight channel dimension {in_channels_w}"
    )
    assert out_channels == out_channels_b, (
        f"Output channel dimension {out_channels} doesn't match bias dimension {out_channels_b}"
    )
    
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    
    height_out = (height + 2 * pad_h - kernel_h) // stride_h + 1
    width_out = (width + 2 * pad_w - kernel_w) // stride_w + 1
    
    output = Tensor([batch_size, out_channels, height_out, width_out], input.device())
    _conv2d_forward(input, weight, bias, output, pad_h, pad_w, stride_h, stride_w)
    return output

def conv2d_backward(
    input: Tensor,          # (batch_size, in_channels, height, width)
    weight: Tensor,         # (out_channels, in_channels, kernel_h, kernel_w)
    grad_output: Tensor,    # (batch_size, out_channels, height_out, width_out)
    padding: Tuple[int, int],
    stride: Tuple[int, int]
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    2D convolution backward propagation.
    - dX = dY conv W^T
    - dW = dY conv X
    - db = sum dY
    
    Parameters:
        input: Input tensor (batch_size, in_channels, height, width)
        weight: Weight tensor (out_channels, in_channels, kernel_h, kernel_w)
        grad_output: Output gradient tensor (batch_size, out_channels, height_out, width_out)
        padding: Tuple of height and width padding
        stride: Tuple of height and width stride
    
    Returns:
        Tuple[
            Tensor,      // Gradients for input
            Tensor,      // Gradients for weight
            Tensor       // Gradients for bias
        ]: Gradients for input, weight, and bias
    """
    if input.dim() == 2:
        height, width = input.shape()
        input = input.reshape([1, 1, height, width])
    if input.dim() == 3:
        channels, height, width = input.shape()
        input = input.reshape([1, channels, height, width])
    if grad_output.dim() == 2:
        height_out, width_out = grad_output.shape()
        grad_output = grad_output.reshape([1, 1, height_out, width_out])
    if grad_output.dim() == 3:
        batch_size, height_out, width_out = grad_output.shape()
        grad_output = grad_output.reshape([batch_size, 1, height_out, width_out])
    
    batch_size, in_channels, height, width = input.shape()
    out_channels, in_channels_w, kernel_h, kernel_w = weight.shape()
    batch_size_d, out_channels_d, height_out_d, width_out_d = grad_output.shape()
    
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    
    height_out = (height + 2 * pad_h - kernel_h) // stride_h + 1
    width_out = (width + 2 * pad_w - kernel_w) // stride_w + 1
    
    assert input.dim()       == 4, f"Input tensor must be 4D, but got shape {input.shape()}"
    assert weight.dim()      == 4, f"Weight tensor must be 4D, but got shape {weight.shape()}"
    assert grad_output.dim() == 4, f"Gradient tensor must be 4D, but got shape {grad_output.shape()}"
    assert in_channels       == in_channels_w, (
        f"Input channel dimension {in_channels} doesn't match weight channel dimension {in_channels_w}"
    )
    assert batch_size        == batch_size_d, (
        f"Input batch size {batch_size} doesn't match gradient batch size {batch_size_d}"
    )
    assert out_channels      == out_channels_d, (
        f"Output channel dimension {out_channels} doesn't match gradient channel dimension {out_channels_d}"
    )
    assert height_out        == height_out_d, (
        f"Output height {height_out} doesn't match gradient height {height_out_d}"
    )
    assert width_out         == width_out_d, (
        f"Output width {width_out} doesn't match gradient width {width_out_d}"
    )
    
    grad_input = Tensor(input.shape(), input.device())
    grad_weight = Tensor(weight.shape(), weight.device())
    grad_bias = Tensor([out_channels], weight.device())
    
    _conv2d_backward(
        input, weight, grad_input, grad_weight, grad_bias, grad_output,
        pad_h, pad_w, stride_h, stride_w
    )
    return grad_input, grad_weight, grad_bias

def max_pool2d_forward(
    input: Tensor,   # (batch_size, channels, height, width)
    kernel_size: Tuple[int, int],
    padding: Tuple[int, int],
    stride: Tuple[int, int]
) -> Tuple[Tensor, Tensor]:
    """
    Max pooling forward propagation
    
    Parameters:
        input: Input tensor (batch_size, channels, height, width)
        kernel_size: Tuple of height and width kernel size
        padding: Tuple of height and width padding
        stride: Tuple of height and width stride
    
    Returns:
        Tuple[
            Tensor,      // Output tensor
            Tensor       // Mask tensor
        ]: Output tensor and mask tensor
    """
    if input.dim() == 3:
        channels, height, width = input.shape()
        input = input.reshape([1, channels, height, width])
    
    batch_size, channels, height, width = input.shape()
    assert input.dim() == 4, f"Input tensor must be 3D or 4D, but got shape {input.shape()}"
    
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    
    height_out = (height + 2 * pad_h - kernel_h) // stride_h + 1
    width_out = (width + 2 * pad_w - kernel_w) // stride_w + 1
    
    assert height_out > 0 and width_out > 0, (
        f"Invalid output dimensions: ({height_out}, {width_out}). "
        f"Input: ({height}, {width}), "
        f"Kernel: ({kernel_h}, {kernel_w}), "
        f"Padding: ({pad_h}, {pad_w}), "
        f"Stride: ({stride_h}, {stride_w})"
    )
    
    output = Tensor([batch_size, channels, height_out, width_out], input.device())
    mask = Tensor([batch_size, channels, height_out, width_out], input.device())
    
    _max_pool_forward(
        input, mask, output,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w
    )
    return output, mask

def max_pool2d_backward(
    grad_output: Tensor,  # (batch_size, out_channels, height_out, width_out)
    mask: Tensor,         # (batch_size, out_channels, height_out, width_out)
    kernel_size: Tuple[int, int],
    padding: Tuple[int, int],
    stride: Tuple[int, int],
    input_shape: List[int]  # (batch_size, channels, height, width)
) -> Tensor:
    """
    Max pooling backward propagation
    
    Parameters:
        grad_output: Output gradient tensor (batch_size, channels, height_out, width_out)
        mask: Mask tensor from forward pass (batch_size, channels, height_out, width_out)
        kernel_size: Tuple of height and width kernel size
        padding: Tuple of height and width padding
        stride: Tuple of height and width stride
        input_shape: Shape of the input tensor
    
    Returns:
        Tensor: Input gradient tensor
    """
    if grad_output.dim() == 3:
        channels, height_out, width_out = grad_output.shape()
        grad_output = grad_output.reshape([1, channels, height_out, width_out])
    if mask.dim() == 3:
        channels, height_out, width_out = mask.shape()
        mask = mask.reshape([1, channels, height_out, width_out])
    
    assert grad_output.dim() == 4, f"Gradient tensor must be 4D, but got shape {grad_output.shape()}"
    assert mask.dim() == 4, f"Mask tensor must be 4D, but got shape {mask.shape()}"
    assert grad_output.shape() == mask.shape(), (
        f"Gradient shape {grad_output.shape()} doesn't match mask shape {mask.shape()}"
    )
    assert len(input_shape) == 4, f"Input shape must be 4D, but got shape {input_shape}"
    
    grad_input = Tensor(input_shape, grad_output.device())
    
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    
    _max_pool_backward(
        grad_input, mask, grad_output,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w
    )
    return grad_input

def softmax_forward(
    input: Tensor  # (batch_size, num_classes)
) -> Tensor:
    """
    Softmax forward propagation
    - Y = softmax(X)
    
    Parameters:
        input: Input tensor (batch_size, num_classes)
    
    Returns:
        Tensor: Output tensor (batch_size, num_classes)
    """
    if input.dim() == 1:
        input = input.reshape([1, -1])
    
    assert input.dim() == 2, f"Input tensor must be 1D or 2D, but got shape {input.shape()}"
    
    output = Tensor(input.shape(), input.device())
    _softmax_forward(input, output)
    return output

def cross_entropy_forward(
    input: Tensor,  # (batch_size, num_classes)
    target: Tensor  # (batch_size)
) -> Tensor:
    """
    Cross entropy loss forward propagation
    - loss = -sum(y_i * log(p_i))
    
    Parameters:
        input: Input tensor (batch_size, num_classes)
        target: Target tensor (batch_size)
    
    Returns:
        Tensor: Loss value (scalar)
    """
    if input.dim() == 1:
        input = input.reshape([1, -1])
    if target.dim() == 0:
        target = target.reshape([1])
    
    batch_size, num_classes = input.shape()
    assert input.dim() == 2, f"Input tensor must be 1D or 2D, but got shape {input.shape()}"
    assert target.dim() == 1, f"Target tensor must be scalar or 1D, but got shape {target.shape()}"
    assert target.shape()[0] == batch_size, (
        f"Batch size mismatch: input {batch_size}, target {target.shape()[0]}"
    )
    
    output = Tensor([1], input.device())
    _cross_entropy_forward(input, target, output)
    return output

def cross_entropy_backward(
    input: Tensor,  # (batch_size, num_classes)
    target: Tensor  # (batch_size)
) -> Tensor:
    """
    Cross entropy loss backward propagation (with softmax)
    - dX_i = p_i - y_i
    
    Parameters:
        input: Input tensor (batch_size, num_classes) or (num_classes,)
        target: Target tensor (batch_size,) or scalar
    
    Returns:
        Tensor: Input gradient tensor (batch_size, num_classes)
    """
    if input.dim() == 1:
        input = input.reshape([1, -1])
    if target.dim() == 0:
        target = target.reshape([1])
    
    batch_size, num_classes = input.shape()
    assert input.dim() == 2, f"Input tensor must be 1D or 2D, but got shape {input.shape()}"
    assert target.dim() == 1, f"Target tensor must be scalar or 1D, but got shape {target.shape()}"
    assert target.shape()[0] == batch_size, (
        f"Batch size mismatch: input {batch_size}, target {target.shape()[0]}"
    )
    
    grad = Tensor(input.shape(), input.device())
    _cross_entropy_backward(input, target, grad)
    return grad