from __future__ import annotations

import pytest
import numpy as np
from numpy.testing import assert_allclose

from tinytorch import DeviceType, Tensor

from utils import skip_if_no_cuda

def test_tensor_constructor():
    """Test tensor constructor (CPU)"""
    # Test empty constructor
    t1 = Tensor()
    
    # Test shape-only constructor
    t2 = Tensor([2, 3], DeviceType.CPU)
    assert t2.shape == [2, 3]
    assert t2.device == DeviceType.CPU
    
    # Test full constructor
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t3 = Tensor([2, 3], DeviceType.CPU, data)
    assert t3.shape == [2, 3]
    assert t3.device == DeviceType.CPU
    assert_allclose(t3.to_numpy(), np.array(data).reshape(2, 3))
    
    # Test zeros
    t4 = Tensor.zeros([2, 3])
    assert t4.shape == [2, 3]
    assert t4.device == DeviceType.CPU
    assert_allclose(t4.to_numpy(), np.zeros([2, 3]))
    
    # Test ones
    t5 = Tensor.ones([2, 3])
    assert t5.shape == [2, 3]
    assert t5.device == DeviceType.CPU
    assert_allclose(t5.to_numpy(), np.ones([2, 3]))

def test_tensor_numpy():
    """Test numpy conversion methods (CPU)"""
    arr1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    t1 = Tensor.from_numpy(arr1)
    assert t1.shape == [6]
    assert t1.device == DeviceType.CPU
    
    arr2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    t2 = Tensor.from_numpy(arr2)
    assert t2.shape == [2, 3]
    assert t2.device == DeviceType.CPU
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t = Tensor([2, 3], DeviceType.CPU, data)
    t_np = t.to_numpy()
    assert np.array_equal(t_np, np.array(data).reshape([2, 3]))

def test_tensor_ops():
    """Test tensor operations (CPU)"""
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = np.random.randn(3, 2)
    ta = Tensor.from_numpy(a)
    tb = Tensor.from_numpy(b)
    tc = Tensor.from_numpy(c)
    
    # Test basic operations
    result = ta + tb
    assert result.device == DeviceType.CPU
    assert_allclose(result.to_numpy(), a + b)
    
    result = ta - tb
    assert result.device == DeviceType.CPU
    assert_allclose(result.to_numpy(), a - b)
    
    # Test matrix multiplication
    result = ta @ tc
    assert result.device == DeviceType.CPU
    assert_allclose(result.to_numpy(), a @ c)
    
    # Test scalar multiplication
    scalar = 2.5
    result = ta * scalar
    assert result.device == DeviceType.CPU
    assert_allclose(result.to_numpy(), a * scalar)
    
    result = scalar * ta
    assert result.device == DeviceType.CPU
    assert_allclose(result.to_numpy(), scalar * a)

@skip_if_no_cuda
def test_tensor_ops_gpu():
    """Test tensor operations (GPU)"""
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = np.random.randn(3, 2)
    
    # Move tensors to GPU
    ta = Tensor.from_numpy(a)
    tb = Tensor.from_numpy(b)
    tc = Tensor.from_numpy(c)
    
    ta.to_gpu()
    tb.to_gpu()
    tc.to_gpu()
    
    # Test basic operations
    result = ta + tb
    assert result.device == DeviceType.GPU
    assert_allclose(result.to_numpy(), a + b)
    
    result = ta - tb
    assert result.device == DeviceType.GPU
    assert_allclose(result.to_numpy(), a - b)
    
    # Test matrix multiplication
    result = ta @ tc
    assert result.device == DeviceType.GPU
    assert_allclose(result.to_numpy(), a @ c)
    
    # Test scalar multiplication
    scalar = 2.5
    result = ta * scalar
    assert result.device == DeviceType.GPU
    assert_allclose(result.to_numpy(), a * scalar)
    
    result = scalar * ta
    assert result.device == DeviceType.GPU
    assert_allclose(result.to_numpy(), scalar * a)

def test_tensor_device_shape():
    """Test basic device operations (CPU)"""
    t = Tensor([2, 3], DeviceType.CPU)
    assert t.device == DeviceType.CPU
    assert t.in_cpu()
    assert not t.in_gpu()

@skip_if_no_cuda
def test_tensor_device_gpu():
    """Test GPU device operations"""
    t = Tensor([2, 3], DeviceType.CPU)
    
    # Test device movement
    t.to_gpu()
    assert t.device == DeviceType.GPU
    assert not t.in_cpu()
    assert t.in_gpu()
    
    # Test data persistence after device movement
    data = np.random.randn(2, 3)
    t = Tensor.from_numpy(data)
    t.to_gpu()
    t.to_cpu()
    assert_allclose(t.to_numpy(), data)

def test_tensor_shape_ops():
    """Test shape operations"""
    # Test dim
    t = Tensor.randn([2, 3, 4])
    assert t.dim == 3
    assert t.shape == [2, 3, 4]
    assert t.size == 24
    
    # Test reshape
    t_reshaped = t.reshape([4, 6])
    assert t_reshaped.shape == [4, 6]
    assert t_reshaped.size == 24
    
    # Test transpose
    t = Tensor.randn([2, 3])
    t_t = t.transpose()
    assert t_t.shape == [3, 2]
    np_t = t.to_numpy()
    assert_allclose(t_t.to_numpy(), np_t.T)

@skip_if_no_cuda
def test_tensor_shape_ops_gpu():
    """Test shape operations on GPU"""
    # Test reshape on GPU
    t = Tensor.randn([2, 3, 4])
    t.to_gpu()
    t_reshaped = t.reshape([4, 6])
    assert t_reshaped.device == DeviceType.GPU
    assert t_reshaped.shape == [4, 6]
    
    # Test transpose on GPU
    t = Tensor.randn([2, 3])
    t.to_gpu()
    t_t = t.transpose()
    assert t_t.device == DeviceType.GPU
    assert t_t.shape == [3, 2]

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
    
    # Test __repr__ and __str__
    t = Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert isinstance(str(t), str)
    assert isinstance(repr(t), str)

if __name__ == "__main__":
    pytest.main([__file__])