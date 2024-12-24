import numpy as np
from numpy.testing import assert_allclose
import torch

from tinytorch import DeviceType, Tensor
from tinytorch.nn import Linear

def test_fc(device, batch_size, in_features, out_features):
    input = np.random.randn(batch_size, in_features)
    out_grad = np.random.randn(batch_size, out_features)
    
    tinytorch_fc = Linear(in_features, out_features)
    tinytorch_fc.to(device)
    
    weight_t = tinytorch_fc.weight
    bias_t = tinytorch_fc.bias
    
    weight = weight_t.to_numpy()
    bias = bias_t.to_numpy()
    
    # PyTorch
    
    # TinyTorch
    input_t = Tensor.from_numpy(input, device)
    out_grad_t = Tensor.from_numpy(out_grad, device)
    
    
    