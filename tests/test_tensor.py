from __future__ import annotations

import pytest
import numpy as np

from tinytorch import DeviceType, Tensor

def test_tensor_constructor():
    t1 = Tensor()
    t2 = Tensor([2, 3], DeviceType.CPU)
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t3 = Tensor([2, 3], DeviceType.CPU, data)
    
def test_tensor_from_numpy():
    arr1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    t1 = Tensor.from_numpy(arr1)
    assert t1.shape() == [6]
    
    arr2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    t2 = Tensor.from_numpy(arr2)
    assert t2.shape() == [2, 3]
    
def test_tensor_to_numpy():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t = Tensor([2, 3], DeviceType.CPU, data)
    t_np = t.to_numpy()
    assert np.array_equal(t_np, np.array(data).reshape([2, 3]))
    
def test_tensor_ops():
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = np.random.randn(3, 2)
    ta = Tensor.from_numpy(a)
    tb = Tensor.from_numpy(b)
    tc = Tensor.from_numpy(c)
    
    np.testing.assert_allclose((ta + tb).to_numpy(), a + b)
    np.testing.assert_allclose((ta - tb).to_numpy(), a - b)
    np.testing.assert_allclose((ta @ tc).to_numpy(), a @ c)
    
if __name__ == "__main__":
    test_tensor_constructor()
    test_tensor_from_numpy()
    test_tensor_to_numpy()
    test_tensor_ops()
    