from __future__ import annotations
from typing import List, overload

import numpy as np 
from numpy.typing import NDArray

from ._tensor import Tensor

from ._libfuncs import                                                                                  \
    fc_forward as _fc_forward, fc_backward as _fc_backward,                                             \
    conv2d_forward as _conv2d_forward, conv2d_backward as _conv2d_backward,                             \
    max_pool_forward as _max_pool_forward, max_pool_backward as _max_pool_backward,                     \
    softmax_forward as _softmax_forward,                                                                \
    cross_entropy_forward as _cross_entropy_forward, cross_entropy_backward as _cross_entropy_backward, \
    relu_forward as _relu_forward, relu_backward as _relu_backward,                                     \
    sigmoid_forward as _sigmoid_forward, sigmoid_backward as _sigmoid_backward
    
def re
    