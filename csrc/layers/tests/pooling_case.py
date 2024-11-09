import torch
import torch.nn.functional as F

a = torch.randn(2, 4, 4, requires_grad=True)
b = F.max_pool2d(a, 2, 2)
print(a)
print(b)