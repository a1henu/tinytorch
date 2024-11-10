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
    output = Tensor(input.shape(), input.device())
    _sigmoid_backward(output, input, grad)
    return output

def fc_forward(
    input: Tensor,
    weight: Tensor,
    bias: Tensor
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
    batch_size = input.shape()[0]
    out_features = weight.shape()[1]
    
    if bias.dim() == 1:
        bias = bias.reshape([1, -1])
    
    output = Tensor([batch_size, out_features], input.device())
    _fc_forward(input, weight, bias, output)
    return output

def fc_backward(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    output: Tensor,
    grad: Tensor
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
    input_shape = input.shape()
    if len(input_shape) == 1:
        input = input.reshape([1, -1])
    elif len(input_shape) != 2:
        raise ValueError(
            f"Input tensor must be 1D or 2D, but got shape {input_shape}"
        )
    
    # Handle weight dimensions
    weight_shape = weight.shape()
    if len(weight_shape) != 2:
        raise ValueError(
            f"Weight tensor must be 2D, but got shape {weight_shape}"
        )
    
    # Handle bias dimensions
    if bias.dim() == 1:
        bias = bias.reshape([1, -1])
    elif len(bias.shape()) != 2 or bias.shape()[0] != 1:
        raise ValueError(
            f"Bias tensor must be 1D or (1, out_features), but got shape {bias.shape()}"
        )
    
    # Handle grad dimensions
    grad_shape = grad.shape()
    if len(grad_shape) != 2:
        raise ValueError(
            f"Gradient tensor must be 2D, but got shape {grad_shape}"
        )
    
    # Check dimensions match
    batch_size, in_features = input.shape()
    in_features_w, out_features = weight.shape()
    batch_size_g, out_features_g = grad.shape()
    
    if in_features != in_features_w:
        raise ValueError(
            f"Input features ({in_features}) doesn't match "
            f"weight input features ({in_features_w})"
        )
    
    if batch_size != batch_size_g:
        raise ValueError(
            f"Batch size mismatch: input {batch_size}, gradient {batch_size_g}"
        )
    
    if out_features != out_features_g:
        raise ValueError(
            f"Output features mismatch: weight {out_features}, gradient {out_features_g}"
        )
    
    grad_input = Tensor(input.shape(), input.device())
    grad_weight = Tensor(weight.shape(), weight.device())
    grad_bias = Tensor(bias.shape(), bias.device())
    
    _fc_backward(input, weight, bias, output, grad_input, grad_weight, grad_bias, grad)
    return grad_input, grad_weight, grad_bias

def conv2d_forward(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    pad_h: int,
    pad_w: int,
    stride_h: int,
    stride_w: int
) -> Tensor:
    """
    2D convolution forward propagation.
    - Y = W conv X + b
    
    Parameters:
        input: Input tensor (batch_size, in_channels, height, width)
        weight: Weight tensor (out_channels, in_channels, kernel_h, kernel_w)
        bias: Bias tensor (out_channels)
        pad_h: Height padding
        pad_w: Width padding
        stride_h: Height stride
        stride_w: Width stride
    
    Returns:
        Tensor: Output tensor (batch_size, out_channels, height_out, width_out)
    """
    input_shape = input.shape()
    if len(input_shape) == 3:
        # Add batch dimension
        input = input.reshape([1] + input_shape)
    elif len(input_shape) != 4:
        raise ValueError(
            f"Input tensor must be 3D or 4D, but got shape {input_shape}"
        )
    
    # Handle weight dimensions
    weight_shape = weight.shape()
    if len(weight_shape) != 4:
        raise ValueError(
            f"Weight tensor must be 4D, but got shape {weight_shape}"
        )
    
    # Handle bias dimensions
    bias_shape = bias.shape()
    if len(bias_shape) != 1:
        bias = bias.reshape([-1])
        
    batch_size, in_channels, height, width = input.shape()
    out_channels, _, kernel_h, kernel_w = weight.shape()
    
    height_out = (height + 2 * pad_h - kernel_h) // stride_h + 1
    width_out = (width + 2 * pad_w - kernel_w) // stride_w + 1
    
    output = Tensor([batch_size, out_channels, height_out, width_out], input.device())
    _conv2d_forward(input, weight, bias, output, pad_h, pad_w, stride_h, stride_w)
    return output

def conv2d_backward(
    input: Tensor,
    weight: Tensor,
    grad_output: Tensor,
    pad_h: int,
    pad_w: int,
    stride_h: int,
    stride_w: int
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
        pad_h: Height padding
        pad_w: Width padding
        stride_h: Height stride
        stride_w: Width stride
    
    Returns:
        Tuple[
            Tensor,      // Gradients for input
            Tensor,      // Gradients for weight
            Tensor       // Gradients for bias
        ]: Gradients for input, weight, and bias
    """
    # Handle input dimensions
    input_shape = input.shape()
    if len(input_shape) == 3:
        input = input.reshape([1] + input_shape)
    elif len(input_shape) != 4:
        raise ValueError(
            f"Input tensor must be 3D or 4D, but got shape {input_shape}"
        )
    
    # Handle weight dimensions
    weight_shape = weight.shape()
    if len(weight_shape) != 4:
        raise ValueError(
            f"Weight tensor must be 4D, but got shape {weight_shape}"
        )
    
    # Handle grad_output dimensions
    grad_shape = grad_output.shape()
    if len(grad_shape) != 4:
        raise ValueError(
            f"Gradient tensor must be 4D, but got shape {grad_shape}"
        )
    
    # Check dimensions match
    batch_size, in_channels, height, width = input.shape()
    out_channels, in_channels_w, kernel_h, kernel_w = weight.shape()
    batch_size_g, out_channels_g, height_out, width_out = grad_output.shape()
    
    if in_channels != in_channels_w:
        raise ValueError(
            f"Input channels ({in_channels}) doesn't match "
            f"weight input channels ({in_channels_w})"
        )
    
    if batch_size != batch_size_g:
        raise ValueError(
            f"Batch size mismatch: input {batch_size}, gradient {batch_size_g}"
        )
    
    if out_channels != out_channels_g:
        raise ValueError(
            f"Output channels mismatch: weight {out_channels}, gradient {out_channels_g}"
        )
    
    grad_input = Tensor(input.shape(), input.device())
    grad_weight = Tensor(weight.shape(), weight.device())
    grad_bias = Tensor([out_channels], weight.device())
    
    _conv2d_backward(
        input, weight, grad_input, grad_weight, grad_bias, grad_output,
        pad_h, pad_w, stride_h, stride_w
    )
    return grad_input, grad_weight, grad_bias

def max_pool_forward(
    input: Tensor,
    kernel_h: int,
    kernel_w: int,
    pad_h: int,
    pad_w: int,
    stride_h: int,
    stride_w: int
) -> Tuple[Tensor, Tensor]:
    """
    Max pooling forward propagation
    
    Parameters:
        input: Input tensor (batch_size, channels, height, width)
        kernel_h: Kernel height
        kernel_w: Kernel width
        pad_h: Height padding
        pad_w: Width padding
        stride_h: Height stride
        stride_w: Width stride
    
    Returns:
        Tuple[
            Tensor,      // Output tensor
            Tensor       // Mask tensor
        ]: Output tensor and mask tensor
    """
    # Handle input dimensions
    input_shape = input.shape()
    if len(input_shape) == 3:
        # Add batch dimension
        input = input.reshape([1] + input_shape)
    elif len(input_shape) != 4:
        raise ValueError(
            f"Input tensor must be 3D or 4D, but got shape {input_shape}"
        )
    
    batch_size, channels, height, width = input.shape()
    height_out = (height + 2 * pad_h - kernel_h) // stride_h + 1
    width_out = (width + 2 * pad_w - kernel_w) // stride_w + 1
    
    if height_out <= 0 or width_out <= 0:
        raise ValueError(
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

def max_pool_backward(
    grad_output: Tensor,
    mask: Tensor,
    kernel_h: int,
    kernel_w: int,
    pad_h: int,
    pad_w: int,
    stride_h: int,
    stride_w: int,
    input_shape: List[int]
) -> Tensor:
    """
    Max pooling backward propagation
    
    Parameters:
        grad_output: Output gradient tensor (batch_size, channels, height_out, width_out)
        mask: Mask tensor from forward pass (batch_size, channels, height_out, width_out)
        kernel_h: Kernel height
        kernel_w: Kernel width
        pad_h: Height padding
        pad_w: Width padding
        stride_h: Height stride
        stride_w: Width stride
        input_shape: Shape of the input tensor
    
    Returns:
        Tensor: Input gradient tensor
    """
    # Handle grad_output dimensions
    grad_shape = grad_output.shape()
    if len(grad_shape) != 4:
        raise ValueError(
            f"Gradient tensor must be 4D, but got shape {grad_shape}"
        )
    
    # Handle mask dimensions
    mask_shape = mask.shape()
    if len(mask_shape) != 4:
        raise ValueError(
            f"Mask tensor must be 4D, but got shape {mask_shape}"
        )
    
    # Check dimensions match
    if grad_shape != mask_shape:
        raise ValueError(
            f"Gradient shape {grad_shape} doesn't match mask shape {mask_shape}"
        )
    
    if len(input_shape) != 4:
        raise ValueError(
            f"Input shape must be 4D, but got shape {input_shape}"
        )
    
    grad_input = Tensor(input_shape, grad_output.device())
    
    _max_pool_backward(
        grad_input, mask, grad_output,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w
    )
    return grad_input

def softmax_forward(
    input: Tensor
) -> Tensor:
    """
    Softmax forward propagation
    - Y = softmax(X)
    
    Parameters:
        input: Input tensor (batch_size, num_classes)
    
    Returns:
        Tensor: Output tensor (batch_size, num_classes)
    """
    # Handle input dimensions
    input_shape = input.shape()
    if len(input_shape) == 1:
        # Add batch dimension
        input = input.reshape([1, -1])
    elif len(input_shape) != 2:
        raise ValueError(
            f"Input tensor must be 1D or 2D, but got shape {input_shape}"
        )
    
    output = Tensor(input.shape(), input.device())
    _softmax_forward(input, output)
    return output

def cross_entropy_forward(
    input: Tensor,
    target: Tensor
) -> Tensor:
    """
    Cross entropy loss forward propagation
    - loss = -sum(y_i * log(p_i))
    
    Parameters:
        input: Input tensor (batch_size, num_classes) or (num_classes,)
        target: Target tensor (batch_size,) or scalar
    
    Returns:
        Tensor: Loss value (scalar)
    """
    # Handle input dimensions
    input_shape = input.shape()
    if len(input_shape) == 1:
        # Add batch dimension
        input = input.reshape([1, -1])
    elif len(input_shape) != 2:
        raise ValueError(
            f"Input tensor must be 1D or 2D, but got shape {input_shape}"
        )
    
    # Handle target dimensions
    target_shape = target.shape()
    if len(target_shape) == 0:
        # Scalar target
        target = target.reshape([1])
    elif len(target_shape) > 1:
        raise ValueError(
            f"Target tensor must be scalar or 1D, but got shape {target_shape}"
        )
    
    if target.shape()[0] != input.shape()[0]:
        raise ValueError(
            f"Batch size mismatch: input {input.shape()[0]}, target {target.shape()[0]}"
        )
    
    output = Tensor([1], input.device())
    _cross_entropy_forward(input, target, output)
    return output

def cross_entropy_backward(
    input: Tensor,
    target: Tensor
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
    # Handle input dimensions
    input_shape = input.shape()
    if len(input_shape) == 1:
        # Add batch dimension
        input = input.reshape([1, -1])
    elif len(input_shape) != 2:
        raise ValueError(
            f"Input tensor must be 1D or 2D, but got shape {input_shape}"
        )
    
    # Handle target dimensions
    target_shape = target.shape()
    if len(target_shape) == 0:
        # Scalar target
        target = target.reshape([1])
    elif len(target_shape) > 1:
        raise ValueError(
            f"Target tensor must be scalar or 1D, but got shape {target_shape}"
        )
    
    if target.shape()[0] != input.shape()[0]:
        raise ValueError(
            f"Batch size mismatch: input {input.shape()[0]}, target {target.shape()[0]}"
        )
    
    grad = Tensor(input.shape(), input.device())
    _cross_entropy_backward(input, target, grad)
    return grad