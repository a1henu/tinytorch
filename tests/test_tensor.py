from __future__ import annotations

import pytest
import numpy as np
from numpy.testing import assert_allclose

from tinytorch import DeviceType, Tensor

def test_tensor_constructor():
    """Test tensor constructor"""
    t1 = Tensor()
    t2 = Tensor([2, 3], DeviceType.CPU)
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t3 = Tensor([2, 3], DeviceType.CPU, data)
    assert t3.device() == DeviceType.CPU
    
def test_tensor_numpy():
    """Test from_numpy method"""
    arr1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    t1 = Tensor.from_numpy(arr1)
    assert t1.shape() == [6]
    
    arr2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    t2 = Tensor.from_numpy(arr2)
    assert t2.shape() == [2, 3]
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t = Tensor([2, 3], DeviceType.CPU, data)
    t_np = t.to_numpy()
    assert np.array_equal(t_np, np.array(data).reshape([2, 3]))

def test_tensor_creation():
    """Test tensor creation methods"""
    # Test ones
    t = Tensor.ones([2, 3])
    assert_allclose(t.to_numpy(), np.ones((2, 3)))
    
    # Test randn
    t = Tensor.randn([2, 3])
    assert t.shape() == [2, 3]
    
    # Test from_numpy with different dtypes
    for dtype in [np.float32, np.float64, np.int32, np.int64]:
        arr = np.array([[1, 2], [3, 4]], dtype=dtype)
        t = Tensor.from_numpy(arr)
        assert t.shape() == [2, 2]
    
def test_tensor_ops():
    """Test tensor operations"""
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = np.random.randn(3, 2)
    ta = Tensor.from_numpy(a)
    tb = Tensor.from_numpy(b)
    tc = Tensor.from_numpy(c)
    
    assert_allclose((ta + tb).to_numpy(), a + b)
    assert_allclose((ta - tb).to_numpy(), a - b)
    assert_allclose((ta @ tc).to_numpy(), a @ c)

def test_tensor_chain_ops():
    """Test chaining multiple operations"""
    a = Tensor.randn([2, 3])
    b = Tensor.randn([3, 4])
    c = Tensor.randn([4, 2])
    
    # Test multiple matrix multiplications
    result = (a @ b @ c).to_numpy()
    expected = a.to_numpy() @ b.to_numpy() @ c.to_numpy()
    assert_allclose(result, expected)
    
    # Test mixed operations
    d = Tensor.randn([2, 2])
    result = ((a @ b @ c) + d).to_numpy()
    expected = (a.to_numpy() @ b.to_numpy() @ c.to_numpy()) + d.to_numpy()
    assert_allclose(result, expected)
    
def test_tensor_device_ops():
    """Test device operations"""
    t = Tensor([2, 3], DeviceType.CPU)
    assert t.device() == DeviceType.CPU
    assert t.in_cpu()
    assert not t.in_gpu()
    
    # Test device movement
    if hasattr(t, "gpu"):  # Only test if GPU is available
        t.to_gpu()
        assert t.device() == DeviceType.GPU
        t.to_cpu()
        assert t.device() == DeviceType.CPU

def test_tensor_shape_ops():
    """Test shape operations"""
    # Test dim
    t = Tensor.randn([2, 3, 4])
    assert t.dim() == 3
    assert t.shape() == [2, 3, 4]
    assert t.size() == 24
    
    # Test reshape
    t_reshaped = t.reshape([4, 6])
    assert t_reshaped.shape() == [4, 6]
    assert t_reshaped.size() == 24
    
    # Test transpose
    t = Tensor.randn([2, 3])
    t_t = t.transpose()
    assert t_t.shape() == [3, 2]
    np_t = t.to_numpy()
    assert_allclose(t_t.to_numpy(), np_t.T)

def test_tensor_special_methods():
    """Test special methods"""
    # Test __getitem__
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    t = Tensor.from_numpy(np.array(data))
    assert t[0, 0] == 1.0
    assert t[1, 2] == 6.0
    
    # Test __eq__
    t1 = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    t2 = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    t3 = Tensor.from_numpy(np.array([1.0, 2.0, 4.0]))
    assert t1 == t2
    assert t1 != t3
    assert t1 != "not a tensor"
    
    # Test __repr__ and __str__
    t = Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert isinstance(str(t), str)
    assert isinstance(repr(t), str)

if __name__ == "__main__":
    pytest.main([__file__])
    