import pytest
import torch

def is_cuda_available() -> bool:
    return torch.cuda.is_available()

skip_if_no_cuda = pytest.mark.skipif(
    not is_cuda_available(),
    reason="CUDA is not available"
)