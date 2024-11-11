import numpy as np

import torch
import torch.nn.functional as F

import tinytorch as tt
from tinytorch.funcs import max_pool_forward, max_pool_backward

im = np.array([
    [
        [-0.5752, 1.1023, 0.8327, -0.3337], 
        [-0.0532, 0.8745, 1.4135, -0.4422], 
        [-0.4538, 0.2952, 0.4086, -0.3135], 
        [0.6764, 0.3422, -0.1896, 0.3065]
    ], [
        [-0.3942, 1.3151, 0.5020, 0.7686], 
        [-1.7310, 0.8545, -1.3705, -0.3178], 
        [-2.5553, 1.1632, 0.4868, -0.1809],  
        [0.0281, 1.2346, 0.3800, 0.2100]
    ]
])

grad = np.array([
    [
        [1, 1],
        [1, 1],
    ],[
        [1, 1],
        [1, 1]
    ]
])

im_torch = torch.tensor(im, requires_grad=True)
im_max = F.max_pool2d(im_torch, kernel_size=2, stride=2, padding=0)
print(im_max)

im_max.backward(torch.tensor(grad))
print(im_torch.grad)

im_t = tt.Tensor.from_numpy(im)
grad_t = tt.Tensor.from_numpy(grad)

output_t, mask_t = max_pool_forward(im_t, 2, 2, 0, 0, 2, 2)
print(output_t)

grad_im_t = max_pool_backward(grad_t, mask_t, 2, 2, 0, 0, 2, 2, [1, 2, 4, 4])
print(grad_im_t)
